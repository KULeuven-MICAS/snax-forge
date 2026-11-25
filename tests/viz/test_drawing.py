"""The DFG page's drawing (``graph_view``'s ``drawing``): boxes, ports, edges and outlines.

Python decides what is drawn; the page lays it out. These tests check the structure exactly
(ids, ends, hardware drawn once, skip marks, task tags, chips, folds); where ELK puts the
boxes is checked by eye on the page.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pytest

from snax_forge.dfg import Graph
from snax_forge.sandbox import Recipe, SandboxError, apply_recipe
from snax_forge.sandbox.recipe import find_recipe
from snax_forge.viz.dfg.api import Entry, graph_view
from tests.helpers import REPO, imported, recipe_path


@cache
def graphs(name: str) -> tuple[Graph, ...]:
    """Every graph of recipe ``name`` at its defaults, its type preset included."""
    r = Recipe.load(recipe_path(name))
    return tuple(x.graph for x in apply_recipe(r, imported(r.kernel, r.dtypes, r.types)))


def drawn(name: str, k: int) -> dict:
    """Step ``k`` of recipe ``name`` as the page gets it."""
    return graph_view(Entry(str(k), Path(f"{k}.snaxdfg"), graphs(name)[k], None))["drawing"]


def problems(d: dict) -> list[str]:
    """What is wrong with drawing ``d``'s structure: repeated ids, an edge's end or port that
    does not exist, a parent that is not an outline."""
    out = []
    nodes = {n["id"]: n for n in d["nodes"]}
    outlines = {o["id"] for o in d["outlines"]}
    if len(nodes) != len(d["nodes"]):
        out.append("a node id twice")
    if len({e["id"] for e in d["edges"]}) != len(d["edges"]):
        out.append("an edge id twice")
    for e in d["edges"]:
        for end, port, side in (
            (e["from"], e["from_port"], "outs"),
            (e["to"], e["to_port"], "ins"),
        ):
            if end not in nodes:
                out.append(f"{e['id']}: no box {end}")
            elif port and port not in (nodes[end].get(side) or []):
                out.append(f"{e['id']}: {end} has no port {port}")
    for x in [*d["nodes"], *d["outlines"]]:
        if x.get("parent") and x["parent"] not in outlines:
            out.append(f"{x['id']}: no outline {x['parent']}")
    return out


def test_an_edge_skips_the_loops_its_subset_does_not_use():
    """y1 is read whole in every m_t: it goes straight to the inner loop, marked ``↻ m_t``."""
    (e,) = [e for e in drawn("mvt_v4_1_1", 18)["edges"] if e["from"] == "y1@0"]
    assert (e["to"], e["skip"], e["label"]) == ("tmp1_mult_map_1", ["m_t"], "↻ m_t")


@pytest.mark.slow
def test_every_recipes_every_step_is_well_formed():
    """Every checked-in recipe, every step: no structural problem (vecadd_v1_4 is refused on
    purpose)."""
    bad = {}
    for path in sorted((REPO / "recipes").glob("*/*.json")):
        try:
            gs = graphs(path.stem) if find_recipe(path.stem) == path else ()
        except SandboxError:
            assert path.stem == "vecadd_v1_4"
            continue
        for k, g in enumerate(gs):
            d = graph_view(Entry(str(k), Path(f"{k}.snaxdfg"), g, None))["drawing"]
            if p := problems(d):
                bad[f"{path.stem} {k}"] = p[:3]
    assert bad == {}


def test_folded_blocks_are_drawn_in_the_accelerator_share_ports_made():
    """gesummv_v2_1_2: each share_ports puts a dot block's group in accelerator gesummv; the
    block, folded, is drawn inside its outline beside the axpby, and the shared container
    carries the marker."""
    blocks = {}
    for k in (10, 11, 12):
        d = drawn("gesummv_v2_1_2", k)
        assert problems(d) == []
        blocks[k] = {n["id"]: n.get("parent") for n in d["nodes"] if n["kind"] == "block"}
    assert blocks[10] == {"matvec": None, "matvec_1": None}
    assert blocks[11] == {"matvec": "acc:gesummv", "matvec_1": None}
    assert blocks[12] == {"matvec": "acc:gesummv", "matvec_1": "acc:gesummv"}
    assert {n["share"] for n in d["nodes"] if n.get("share")} == {
        "share_ports · tmp1_sum", "share_ports · tmp3_sum"}  # fmt: skip
