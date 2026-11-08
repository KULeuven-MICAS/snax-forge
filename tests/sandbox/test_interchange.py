"""interchange_map: a map and its only child map trade places, each keeping its id, variable,
range and tag; applied to the child it undoes itself.

rowsum imports as ``sum_map`` (i) around ``sum_map_1`` (j) around ``sum``. Its lane accumulator
(rowsum_2) splits the row map by W, tags the column map temporal and puts the spatial row map
inside it: ``sum_map`` (i_t) -> ``sum_map_1`` (j) -> ``sum_map_s`` (i_s) -> ``sum``. Every step
is checked against the reference executor (run.py's ``checked_step``).
"""

from __future__ import annotations

import copy

import pytest

from snax_forge.dfg import REDUCE_OPS, Graph
from snax_forge.dfg.reduce import ReduceOp
from snax_forge.sandbox import Recipe, SandboxError, apply_recipe, interchange_map
from snax_forge.sandbox.transforms import TRANSFORMS, site_of
from tests.helpers import imported

from .helpers import recipe_graphs

SPLIT = {"transform": "split_map", "params": {"map": "sum_map", "factor": "W"}}
TAG = {"transform": "tag_map", "params": {"map": "sum_map_1", "kind": "temporal"}}


def swap(map_id: str) -> dict:
    return {"transform": "interchange_map", "params": {"map": map_id}}


def run(steps: list[dict], **params: int) -> list[Graph]:
    """rowsum's import (M 8, N 64, W 4 unless given) and each step's graph, every step checked
    against the reference executor."""
    r = Recipe.from_dict({"name": "t", "kernel": "rowsum",
                          "params": {"W": 4, "M": 8, "N": 64} | params,
                          "symbols": {"M": "M", "N": "N"}, "steps": steps})  # fmt: skip
    return [res.graph for res in apply_recipe(r, imported("rowsum"))]


def nest(g: Graph) -> list[tuple]:
    return [(n.id, n.attrs.get("var"), n.attrs.get("range"), n.attrs.get("loop.kind"))
            for n, _ in g.walk()]  # fmt: skip


def test_rowsum_2s_three_steps():
    *_, g = run([SPLIT, TAG, swap("sum_map_s")])
    assert nest(g) == [
        ("sum_map", "i_t", "0:M // 4", "temporal"),
        ("sum_map_1", "j", "0:N", "temporal"),
        ("sum_map_s", "i_s", "0:4", "spatial"),
        ("sum", None, None, None),
    ]
    s = g.node("sum")
    assert s.inputs["in1"].subset == ["4 * i_t + i_s", "j"]
    assert s.outputs["out"].subset == ["4 * i_t + i_s"]
    assert g.node("sum_map").attrs["loop.split"] == {"var": "i", "range": "0:M"}


def test_twice_gives_the_graph_back():
    _, _, before, after, back = run([SPLIT, TAG, swap("sum_map_s"), swap("sum_map_1")])
    assert back.to_json() == before.to_json()
    assert after.to_json() != before.to_json()


def test_the_site_fills_the_map():
    _, _, g = run([SPLIT, TAG])
    assert TRANSFORMS["interchange_map"].site(g, "sum_map_s") == {"map": "sum_map_s"}
    assert site_of("interchange_map", g, ["sum_map"]) == {"map": "sum_map"}


def test_three_levels_the_middle_map():
    """rowsum with its column map split: sum_map (i) -> sum_map_1 (j_t) -> sum_map_1_s (j_s);
    the middle map and its child trade places, the outer map untouched."""
    split_cols = {"transform": "split_map", "params": {"map": "sum_map_1", "factor": "W"}}
    _, before, g = run([split_cols, swap("sum_map_1")])
    assert nest(before)[:3] == [
        ("sum_map", "i", "0:M", None),
        ("sum_map_1", "j_t", "0:N // 4", "temporal"),
        ("sum_map_1_s", "j_s", "0:4", "spatial"),
    ]
    assert nest(g) == [
        ("sum_map", "i", "0:M", None),
        ("sum_map_1_s", "j_s", "0:4", "spatial"),
        ("sum_map_1", "j_t", "0:N // 4", "temporal"),
        ("sum", None, None, None),
    ]
    assert g.node("sum_map_1").attrs == before.node("sum_map_1").attrs
    assert interchange_map(g, "sum_map_1_s").to_json() == before.to_json()


def refused(g: Graph, map_id: str, match: str) -> None:
    with pytest.raises(SandboxError, match=match):
        interchange_map(g, map_id)
    with pytest.raises(SandboxError, match=match):  # the site refuses as the transform does
        site_of("interchange_map", g, map_id)


def test_refused_a_body_other_than_one_map():
    g = imported("rowsum")
    refused(g, "sum_map_1", "its body must be one map and nothing else; it holds the tasklet 'sum'")
    refused(g, "sum", "a 'tasklet' is not a map")
    d = copy.deepcopy(g.to_dict())
    d["body"][0]["body"].append(copy.deepcopy(d["body"][0]["body"][0]) | {"id": "again"})
    d["body"][0]["body"][1]["body"][0]["id"] = "sum_2"
    refused(Graph.from_dict(d), "sum_map",
            "it holds the map 'sum_map_1', the map 'again'")  # fmt: skip


def test_refused_a_range_naming_the_other_variable():
    d = copy.deepcopy(imported("rowsum").to_dict())
    d["body"][0]["body"][0]["attrs"]["range"] = "0:i"  # a triangle: j runs to the row
    refused(Graph.from_dict(d), "sum_map",
            "sum_map_1's range 0:i names sum_map's variable 'i'; the two cannot trade places")  # fmt: skip


def test_refused_a_bound_unit_inside():
    *_, g = run([{"transform": "tag_map", "params": {"map": "sum_map", "kind": "temporal"}},
                 {"transform": "split_map", "params": {"map": "sum_map_1", "factor": "W"}},
                 {"transform": "bind", "params": {"node": "sum", "brm": "accumulate",
                                                  "implementation": "chisel_adder_tree",
                                                  "instance": "acc"}}])  # fmt: skip
    refused(g, "sum_map", "the bound unit 'sum' is inside it \\(unbind first\\)")


def test_refused_inside_an_accelerated_node():
    """An accelerated node with a nested block (C§11) holding a nest."""
    d = copy.deepcopy(imported("rowsum").to_dict())
    rowsum = d["body"][0]
    d["body"][0] = {"id": "blk", "kind": "accelerated",
                    "inputs": {"a": {"data": "A", "subset": ["0", "0:1"]}},
                    "outputs": {"out": {"data": "y", "subset": ["0:1"]}},
                    "attrs": {"instance": "blk", "brm": "accumulate",
                              "implementation": "chisel_adder_tree", "code": "out = a",
                              "params": {"W": 1, "op": "add"}, "replaced": None},
                    "body": [rowsum]}  # fmt: skip
    rowsum["body"][0]["body"] = [{"id": "inner", "kind": "map",
                                  "attrs": {"var": "k", "range": "0:2"},
                                  "body": rowsum["body"][0]["body"]}]  # fmt: skip
    refused(Graph.from_dict(d), "sum_map_1",
            "it is inside the accelerated node 'blk' \\(unbind first\\)")  # fmt: skip


def test_refused_a_fused_nests_outer_map():
    """dot 4's recipe fuses the split multiply and partial-sum nests into ``mult_map``
    (``loop.fused``) around ``mult_map_s``; a map put around it by hand has the fused nest as
    its only child."""
    _, graphs = recipe_graphs("dot_4")
    fused = next(g for g in graphs[1:] if any("loop.fused" in n.attrs for n, _ in g.walk()))
    (top,) = [n for n in fused.body if n.attrs.get("loop.fused")]
    refused(fused, top.id, f"'{top.id}' is a fused nest's outer map \\(unfuse_map first\\)")
    d = copy.deepcopy(fused.to_dict())
    i = next(k for k, n in enumerate(d["body"]) if n["id"] == top.id)
    d["body"][i] = {"id": "around", "kind": "map", "attrs": {"var": "r", "range": "0:1"},
                    "body": [d["body"][i]]}  # fmt: skip
    refused(Graph.from_dict(d), "around",
            f"'{top.id}' is a fused nest's outer map \\(unfuse_map first\\)")  # fmt: skip


def test_refused_an_order_dependent_fold(monkeypatch):
    """A fold over the outer map with an op that is not order-free; every registered op is
    order-free (D28), so add is marked as one that is not."""
    split_cols = {"transform": "split_map", "params": {"map": "sum_map_1", "factor": "W"}}
    _, g = run([split_cols])
    monkeypatch.setitem(REDUCE_OPS, "add", ReduceOp(REDUCE_OPS["add"].ufunc,
                                                    REDUCE_OPS["add"].identity, False))  # fmt: skip
    refused(g, "sum_map_1", "sum's out folds over 'j_t' with 'add', whose result depends on the "
                            "order")  # fmt: skip
    # the row map is no fold: y[i] moves with it, so each element's writes keep their order
    assert interchange_map(g, "sum_map").body[0].id == "sum_map_1"


@pytest.mark.slow
def test_the_editor_offers_it_and_undo_takes_it_back(tmp_path, monkeypatch):
    """The editor lists interchange_map on a map whose only child is a map (its site), not on
    one holding a tasklet; Undo moves the cursor back (the session replays, nothing to
    register), and interchange_map on the child is the step's own inverse."""
    from snax_forge.sandbox import session
    from snax_forge.sandbox.session import Roots, Session

    monkeypatch.setattr(session, "_import", imported)
    roots = Roots(tmp_path / "sessions", tmp_path / "recipes", tmp_path / "sandbox")
    s = Session.from_kernel("rowsum", {"M": 8, "N": 64}, roots=roots)
    assert s.apply("split_map", {"map": "sum_map", "factor": 4})["ok"]
    assert s.apply("tag_map", {"map": "sum_map_1", "kind": "temporal"})["ok"]
    sites = s.sites(2, "sum_map_s")["interchange_map"]
    assert sites["ok"] and sites["params"] == {"map": "sum_map_s"}
    assert [f["name"] for f in sites["form"]] == ["map"]
    assert not s.sites(2, "sum_map_1")["interchange_map"]["ok"]  # it holds the tasklet
    before = s.graphs[2].to_json()
    assert s.apply("interchange_map", {"map": "sum_map_s"})["ok"]
    assert s.move(2)["ok"] and s.graphs[s.cursor].to_json() == before
    assert s.move(3)["ok"] and s.apply("interchange_map", {"map": "sum_map_1"}, drop=True)["ok"]
    assert s.graphs[4].to_json() == before
