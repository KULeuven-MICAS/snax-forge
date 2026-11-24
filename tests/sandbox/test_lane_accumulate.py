"""rowsum 2: each lane keeps its own running sum down a row, W rows at a time (lane_accumulate,
``lane_fold`` without a front op).

The recipe splits the row map by W, tags the column map temporal, puts the spatial row map
inside it (``interchange_map``) and binds that map: one job is one block of W rows, JOBS = M / W,
JOB_BEATS = N. The worked example, W 2, M 4, N 3: A's rows (1, 2, 3), (4, 5, 6), (7, 8, 9),
(10, 11, 12) give y = (6, 15, 24, 33), JOBS 2, JOB_BEATS 3.
"""

from __future__ import annotations

import pytest

from snax_forge.dfg import Graph
from snax_forge.sandbox import Recipe, SandboxError, apply_recipe, bind
from tests.helpers import imported, recipe_path

from .helpers import recipe_graphs

LANE = {"brm": "lane_accumulate", "implementation": "chisel_lane_accumulate", "instance": "acc"}


def steps(**params: int) -> list[Graph]:
    """rowsum_v2's graphs (the import, then each step) at ``params``, each step checked against
    the reference executor."""
    r = Recipe.load(recipe_path("rowsum_v2")).with_params(params)
    return [res.graph for res in apply_recipe(r, imported("rowsum"))]


def interchanged(**params: int) -> Graph:
    """rowsum_v2's graph before its bind."""
    return steps(**params)[-2]


def test_refused_a_front_op_in_a_fused_nest():
    """dot 4 before its bind: the fused multiply and its fold are mac's."""
    r, graphs = recipe_graphs("dot_v4")
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
