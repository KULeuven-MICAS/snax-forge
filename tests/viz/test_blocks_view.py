"""Blocks in the DFG view (REUSE1, api.py's ``fold``): a block not expanded drawn as one box
whose ports are the containers it reads and writes from outside, an expanded one as its nodes;
the blocks sharing units expand together. mvt at M = 8, N = 16, built from matvec_dot3 on each
half (matvec ⊃ dot), and from dot_3 on each half's rows with half 2 shared on half 1's units."""

from __future__ import annotations

from pathlib import Path

from snax_forge.dfg import Graph
from snax_forge.sandbox import Recipe, apply_recipe
from snax_forge.sandbox.blocks import lift
from snax_forge.sandbox.share import share
from snax_forge.sandbox.transforms import tag_map
from snax_forge.sandbox.transpose import mark_transpose
from snax_forge.viz.dfg.api import Entry, _layout, graph_view
from tests.helpers import imported
from tests.sandbox.helpers import mvt

from .test_columns import drawn, pairs

HALVES = (["tmp0_mult_map", "sum_map"], ["tmp2_mult_map", "sum_map_2"])


def _mvt_of_matvecs(g: Graph | None = None) -> Graph:
    r = Recipe.from_dict({"name": "mvt_reuse", "kernel": "mvt", "params": {"W": 4},
                          "symbols": {"M": 8, "N": 16}, "steps": [
        {"transform": "apply_recipe", "params": {"recipe": "matvec/matvec_dot3", "region": h,
                                                 "params": {"W": "W"}}} for h in HALVES]})  # fmt: skip
    return apply_recipe(r, imported("mvt") if g is None else g)[-1].graph


def view(g: Graph, *expanded: str) -> dict:
    return graph_view(Entry(g.name, Path(g.name), g, None), expanded=expanded)


def test_each_half_is_one_box_until_expanded():
    v = view(_mvt_of_matvecs())
    assert drawn(v["rows"]) == [{
        0: [["A@0", "y1@0"], "matvec", ["x1@0", "tmp0@1"], "augassign_18_4_map", ["x1@1"]],
        1: [["A@0~2", "y2@0"], "matvec_1", ["x2@0", "tmp2@1"], "augassign_19_4_map", ["x2@1"]],
    }]  # fmt: skip
    assert ("A@0", "matvec.in:A") in pairs(v) and ("matvec.out:tmp0", "tmp0@1") in pairs(v)
    node = next(r["node"] for r in v["rows"][0]["columns"][0]["rows"] if r["kind"] == "node")
    assert node["text"] == "matvec · matvec/matvec_dot3 · W = 4"
    assert [(b["name"], b["collapsed"], b["hidden"]) for b in v["blocks"]] == [
        ("matvec", True, False), ("dot", False, True), ("matvec_1", True, False),
        ("dot_1", False, True)]  # fmt: skip
    d = v["details"]["block:matvec"]
    assert (d["recipe"], d["params"], d["blocks"], d["units"]) == (
        "matvec/matvec_dot3",
        {"W": 4},
        ["dot"],
        ["mul", "sum"],
    )


def test_expanding_shows_the_inner_block_then_the_nodes():
    g = _mvt_of_matvecs()
    one = view(g, "matvec")
    # the rows' loop stays drawn around the dot (both nests' row maps as one), its ports the
    # slice one firing takes: row m_t of A in half 1, column n_t in half 2
    (row,) = [r["node"] for r in one["rows"][0]["columns"][0]["rows"] if r["kind"] == "node"][:1]
    assert row["text"] == "tmp0_mult_map + sum_map · m_t · 8 · temporal"
    assert [n["id"] for n in row["body"]] == ["dot"]
    assert next(b for b in one["blocks"] if b["name"] == "matvec")["tops"] == ["tmp0_mult_map"]
    slices = {(e["from"], e["to"]): e["slice"] for e in one["edges"] if "slice" in e}
    assert slices[("A@0", "dot.in:A")] == "A[m_t, 0:N]"
    assert slices[("dot.out:tmp0", "tmp0@1")] == "tmp0[m_t]"
    two = view(g, "matvec_1")
    slices = {(e["from"], e["to"]): e["slice"] for e in two["edges"] if "slice" in e}
    assert slices[("A@0~2", "dot_1.in:A")] == "A[0:M, n_t]"
    both = view(g, "matvec", "dot", "matvec_1", "dot_1")
    plain = _layout(g)  # the graph's own nodes, no block folded
    assert not [b for b in both["blocks"] if b["collapsed"]]
    assert pairs(both) == pairs(plain) and ("tmp0_mult.out", "tmp0_prod.in") in pairs(both)


def test_a_transposed_read_keeps_its_label_on_the_box():
    g = _mvt_of_matvecs(mark_transpose(imported("mvt"), "tmp2_mult"))
    v = view(g)
    (e,) = [e for e in v["edges"] if e["to"] == "matvec_1.in:A"]
    assert e.get("transpose") and drawn(v["rows"])[0] == ["A@0"]  # A drawn once


def test_halves_sharing_units_expand_together():
    g = lift(mvt(), "dot")
    for m in ("tmp0_mult_map", "sum_map", "tmp2_mult_map", "sum_map_2"):
        g = tag_map(g, m, "temporal")
    for b in ("dot", "dot_1"):
        g = apply_recipe(Recipe.from_dict({"name": "x", "kernel": "mvt", "steps": [
            {"transform": "apply_recipe", "params": {"recipe": "dot/dot_3", "block": b,
                                                     "params": {"W": 4}}}]}), g)[-1].graph  # fmt: skip
    g = share(g, "dot_1", **{"with": "dot"})
    v = view(g)
    assert {b["name"]: b["together"] for b in v["blocks"]} == {"dot": ["dot_1"], "dot_1": ["dot"]}


def test_the_step_route_takes_the_blocks_expanded(tmp_path):
    """View mode over matvec_dot3's steps: its last step one dot box, ``?expand=dot`` its
    nodes."""
    from snax_forge.sandbox import write_steps
    from snax_forge.viz.dfg.server import make_dfg_server
    from tests.helpers import recipe_path

    from .helpers import get_json, served

    r = Recipe.load(recipe_path("matvec_dot3")).with_params({"M": 4, "N": 64})
    write_steps(r, apply_recipe(r, imported(r.kernel)), tmp_path / "s")
    with served(make_dfg_server([tmp_path / "s"], 0)) as srv:
        _, shut = get_json(srv, "/api/step/3")
        _, open_ = get_json(srv, "/api/step/3?expand=dot")
    assert drawn(shut["rows"])[1] == "mult_map"  # the rows' loop, the dot box inside it
    assert shut["marks"]["blocks"] == []  # applied: neither unlift nor apply_recipe takes it
    assert [x for x in drawn(open_["rows"]) if isinstance(x, str)] == [
        "mult_map",
        "tmp0",
        "sum_map",
    ]
