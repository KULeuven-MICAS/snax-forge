"""rowsum 2: each lane keeps its own running sum down a row, W rows at a time (lane_accumulate,
``lane_fold`` without a front op).

The recipe splits the row map by W, tags the column map temporal, puts the spatial row map
inside it (``interchange_map``) and binds that map: one job is one block of W rows, JOBS = M / W,
JOB_BEATS = N. The worked example, W 2, M 4, N 3: A's rows (1, 2, 3), (4, 5, 6), (7, 8, 9),
(10, 11, 12) give y = (6, 15, 24, 33), JOBS 2, JOB_BEATS 3.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from snax_forge.dfg import Graph, execute
from snax_forge.dfg.execute import against_reference
from snax_forge.sandbox import Recipe, SandboxError, apply_recipe, bind
from snax_forge.sdfg.loader import load
from tests.helpers import imported, recipe_path

from .helpers import graph, recipe_graphs
from .test_rowsum import rates

LANE = {"brm": "lane_accumulate", "implementation": "chisel_lane_accumulate", "instance": "acc"}


def steps(**params: int) -> list[Graph]:
    """rowsum_2's graphs (the import, then each step) at ``params``, each step checked against
    the reference executor."""
    r = Recipe.load(recipe_path("rowsum_2")).with_params(params)
    return [res.graph for res in apply_recipe(r, imported("rowsum"))]


def test_the_worked_example():
    *_, before, g = steps(W=2, M=4, N=3)
    s = g.node("sum")
    assert (s.kind, s.attrs["brm"], s.attrs["code"]) == (
        "accelerated",
        "lane_accumulate",
        "out = a",
    )
    assert s.attrs["params"] == {"W": 2, "op": "add"}
    assert s.inputs["a"].to_dict() == {"data": "A", "subset": ["2 * i_t:2 * i_t + 2", "j"]}
    assert s.outputs["out"].to_dict() == {"data": "y", "subset": ["2 * i_t:2 * i_t + 2"]}
    assert s.attrs["replaced"] == before.node("sum_map_s").to_dict()
    loops, named = rates(g)
    assert loops == [("i_t", 2), ("j", 3)] and named == {"T": 3}  # JOB_BEATS = N
    assert 2 * 3 // named["T"] == 2  # JOBS = n / T = M / W
    a = np.arange(1, 13, dtype=np.int64).reshape(4, 3)
    inputs = {"A": a, "y": np.zeros(4, dtype=np.int64)}
    assert execute(g, inputs)["y"].tolist() == [6, 15, 24, 33]
    for got, ref in against_reference(load("rowsum"), inputs, lambda x: execute(g, x)).values():
        assert np.array_equal(got, ref)


def test_the_recipe_at_its_defaults():
    """W 4, M 8, N 64: JOBS 2, JOB_BEATS 64; unbind gives back the interchanged nest."""
    from snax_forge.sandbox import unbind

    *_, before, g = steps()
    assert rates(g) == ([("i_t", 2), ("j", 64)], {"T": 64})
    assert g.node("sum").attrs["params"] == {"W": 4, "op": "add"}
    assert unbind(g, "sum").to_json() == before.to_json()


def interchanged(**params: int) -> Graph:
    """rowsum_2's graph before its bind."""
    return steps(**params)[-2]


def test_refused_the_tasklet_instead_of_its_map():
    with pytest.raises(SandboxError, match="binds the spatial map around a folding tasklet, not "
                                           "the tasklet: bind its map"):  # fmt: skip
        bind(interchanged(), "sum", **LANE)


def test_refused_a_front_op_in_the_tasklet():
    d = copy.deepcopy(interchanged().to_dict())
    tasklet = d["body"][0]["body"][0]["body"][0]["body"][0]
    assert tasklet["id"] == "sum"
    tasklet["attrs"]["code"] = "out = 2 * in1"
    with pytest.raises(SandboxError, match="sum computes '2 \\* in1' before the fold, a front op; "
                                           "a lane_accumulate folds its one input as it is: bind "
                                           "mac"):  # fmt: skip
        bind(Graph.from_dict(d), "sum_map_s", **LANE)


def test_refused_a_front_op_in_a_fused_nest():
    """dot 4 before its bind: the fused multiply and its fold are mac's."""
    r, graphs = recipe_graphs("dot_4")
    k = next(i for i, st in enumerate(r.steps) if st.transform == "bind")
    node = r.steps[k].params["node"]
    with pytest.raises(SandboxError, match="computes 'in1 \\* in2' before the fold, a front op; a "
                                           "lane_accumulate folds its input as it is: bind mac"):  # fmt: skip
        bind(graphs[k], node, **LANE)
    assert bind(graphs[k], node, **(LANE | {"brm": "mac", "implementation": "chisel_mac"}))


def tagged(kinds: dict[str, str], **sizes: int) -> Graph:
    r = Recipe.from_dict({"name": "t", "kernel": "rowsum", "params": {"M": 4, "N": 4} | sizes,
                          "symbols": {"M": "M", "N": "N"},
                          "steps": [{"transform": "tag_map", "params": {"map": m, "kind": k}}
                                    for m, k in kinds.items()]})  # fmt: skip
    return apply_recipe(r, imported("rowsum"))[-1].graph


def test_refused_the_reduced_loop_spatial():
    """The column map tagged spatial and bound: its lanes are the loop the fold runs over."""
    g = tagged({"sum_map": "temporal", "sum_map_1": "spatial"})
    with pytest.raises(SandboxError, match="sum folds over 'sum_map_1' \\(j\\), which is spatial; a "
                                           "lane_accumulate folds each lane over the temporal maps "
                                           "around it, a fold across the lanes binds accumulate"):  # fmt: skip
        bind(g, "sum_map_1", **(LANE | {"params": {"W": 4}}))


def test_refused_an_output_not_indexed_by_the_lane():
    """A spatial map whose variable neither the input nor the output names."""
    d = copy.deepcopy(tagged({"sum_map": "temporal", "sum_map_1": "temporal"}).to_dict())
    col = d["body"][0]["body"][0]
    col["body"] = [{"id": "lane_map", "kind": "map",
                    "attrs": {"var": "l", "range": "0:4", "loop.kind": "spatial"},
                    "body": col["body"]}]  # fmt: skip
    with pytest.raises(SandboxError, match="sum's output y\\['i'\\] is not indexed by the lane "
                                           "variable 'l'; each lane folds its own element"):  # fmt: skip
        bind(Graph.from_dict(d), "lane_map", **LANE)


def test_a_spatial_map_of_a_plain_tasklet_is_not_a_bind_site():
    """vecadd's add_map_s holds a tasklet that does not fold: bind takes the tasklet, not it."""
    g = graph("vecadd_split")
    with pytest.raises(SandboxError, match="a map binds only as a fused nest.*or as a spatial map "
                                           "of one tasklet whose output folds"):  # fmt: skip
        bind(g, "add_map_s", **LANE)
