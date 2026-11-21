"""Blocks in the DFG view (api.py's ``fold``): a block not expanded drawn as one box
whose ports are the containers it reads and writes from outside, an expanded one as its nodes;
the blocks sharing units expand together. mvt at M = 8, N = 16, built from matvec_v3_1 on each
half (matvec ⊃ dot), and from dot_v3 on each half's rows with half 2 shared on half 1's units."""

from __future__ import annotations

from pathlib import Path

from snax_forge.dfg import Graph
from snax_forge.sandbox import Recipe, apply_recipe
from snax_forge.sandbox.blocks import lift
from snax_forge.sandbox.share import share
from snax_forge.sandbox.transforms import tag_map
from snax_forge.sandbox.transpose import mark_transpose
from snax_forge.viz.dfg.api import Entry, graph_view
from tests.helpers import imported
from tests.sandbox.helpers import mvt

from .helpers import boxes, containers, ends

HALVES = (["tmp0_mult_map", "sum_map"], ["tmp2_mult_map", "sum_map_2"])


def _mvt_of_matvecs(g: Graph | None = None) -> Graph:
    r = Recipe.from_dict({"name": "mvt_reuse", "kernel": "mvt", "params": {"W": 4},
                          "symbols": {"M": 8, "N": 16}, "steps": [
        {"transform": "apply_recipe", "params": {"recipe": "matvec/matvec_v3_1", "region": h,
                                                 "params": {"W": "W"}}} for h in HALVES]})  # fmt: skip
    return apply_recipe(r, imported("mvt") if g is None else g)[-1].graph


def view(g: Graph, *expanded: str) -> dict:
    return graph_view(Entry(g.name, Path(g.name), g, None), expanded=expanded)


def test_each_half_is_one_box_until_expanded():
    v = view(_mvt_of_matvecs())
    assert containers(v, first=True) == ["A@0", "y1@0", "x1@0", "y2@0", "x2@0"]
    assert ("A@0", "matvec.in:A") in ends(v) and ("matvec.out:tmp0", "tmp0@1") in ends(v)
    assert ("A@0", "matvec_1.in:A") in ends(v)  # A one box, read by both halves
    b = boxes(v)["matvec"]
    assert (b["kind"], b["text"], b["block"]) == (
        "block",
        "matvec · matvec/matvec_v3_1 · W = 4",
        "matvec",
    )
    assert [(b["name"], b["collapsed"], b["hidden"]) for b in v["blocks"]] == [
        ("matvec", True, False), ("dot", False, True), ("matvec_1", True, False),
        ("dot_1", False, True)]  # fmt: skip
    d = v["details"]["block:matvec"]
    assert (d["recipe"], d["params"], d["blocks"], d["units"]) == (
        "matvec/matvec_v3_1",
        {"W": 4},
        ["dot"],
        ["mul", "sum"],
    )


def test_expanding_shows_the_inner_block_then_the_nodes():
    g = _mvt_of_matvecs()
    one = view(g, "matvec")
    # the rows' loop stays drawn round the dot (both nests' row maps as one, the other a chip)
    # in the block's outline, its ports the slice one firing takes: row m_t of A in half 1
    row = boxes(one)["tmp0_mult_map"]
    assert (row["text"], row["sub"], row["chips"], row["parent"]) == (
        "tmp0_mult_map · temporal", "m_t · 8", [["sum_map", None]], "block:matvec")  # fmt: skip
    assert [(o["id"], o["text"]) for o in one["drawing"]["outlines"]] == [
        ("block:matvec", "▾ matvec · matvec/matvec_v3_1 · W = 4")]  # fmt: skip
    subs = {e["id"]: e.get("sub") for e in one["drawing"]["edges"]}
    assert subs["tmp0_mult_map->dot.in:A"] == "A[m_t, 0:N]"
    assert subs["dot.out:tmp0->tmp0@1"] == "tmp0[m_t]"
    assert ("y1@0", "dot.in:y1") in ends(one)  # y1 whole in every m_t: past the row loop
    both = view(g, "matvec", "dot", "matvec_1", "dot_1")
    plain = graph_view(
        Entry(g.name, Path(g.name), g, None), expanded=[b["name"] for b in both["blocks"]]
    )
    assert not [b for b in both["blocks"] if b["collapsed"]]
    assert ends(both) == ends(plain) and not [
        b for b in boxes(both).values() if b["kind"] == "block"
    ]


def test_a_transposed_read_keeps_its_label_on_the_box():
    g = _mvt_of_matvecs(mark_transpose(imported("mvt"), "tmp2_mult"))
    v = view(g)
    (e,) = [e for e in v["drawing"]["edges"] if e["to"] == "matvec_1" and e["to_port"] == "in:A"]
    assert (e["label"], e["transpose"]) == ("Aᵀ", True)
    assert [c for c in containers(v) if c.startswith("A@")] == ["A@0"]  # A drawn once


def test_halves_sharing_units_expand_together():
    g = lift(mvt(), "dot")
    for m in ("tmp0_mult_map", "sum_map", "tmp2_mult_map", "sum_map_2"):
        g = tag_map(g, m, "temporal")
    for b in ("dot", "dot_1"):
        g = apply_recipe(Recipe.from_dict({"name": "x", "kernel": "mvt", "steps": [
            {"transform": "apply_recipe", "params": {"recipe": "dot/dot_v3", "block": b,
                                                     "params": {"W": 4}}}]}), g)[-1].graph  # fmt: skip
    g = share(g, "dot_1", **{"with": "dot"})
    v = view(g)
    assert {b["name"]: b["together"] for b in v["blocks"]} == {"dot": ["dot_1"], "dot_1": ["dot"]}


def test_the_step_route_takes_the_blocks_expanded(tmp_path):
    """View mode over matvec_v3_1's steps: its last step one dot box, ``?expand=dot`` its
    nodes."""
    from snax_forge.sandbox import write_steps
    from snax_forge.viz.dfg.server import make_dfg_server
    from tests.helpers import recipe_path

    from .helpers import get_json, served

    r = Recipe.load(recipe_path("matvec_v3_1")).with_params({"M": 4, "N": 64})
    write_steps(r, apply_recipe(r, imported(r.kernel)), tmp_path / "s")
    with served(make_dfg_server([tmp_path / "s"], 0)) as srv:
        _, shut = get_json(srv, "/api/step/3")
        _, open_ = get_json(srv, "/api/step/3?expand=dot")
    assert ("mult_map", "dot.in:A") in ends(shut)  # the rows' loop, then the dot box
    assert shut["marks"]["blocks"] == ["dot"]  # applied: unapply takes it
    assert "dot" not in boxes(open_) and {"mult_map", "unit:mul", "unit:sum"} <= set(boxes(open_))
