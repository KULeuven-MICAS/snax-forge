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


def by_id(d: dict) -> dict[str, dict]:
    return {n["id"]: n for n in d["nodes"]}


def test_shared_hardware_is_drawn_once():
    """mvt_4_i8_dma_AT after ``share`` (step 18): one box per piece of hardware, keyed by it,
    the two tasks' nodes as chips, a temporal mux in front of the shared ports (one per port
    where a slicer sits) and a demux after the output; the slicer and packer ids no longer collide."""
    d = drawn("mvt_4_i8_dma_AT", 18)
    assert problems(d) == [] and d["tasks"] == ["mvt", "mvt_1"]
    n = by_id(d)
    hw = sorted(k for k in n if ":" in k)
    assert hw == ["demux:vsum_x2.out", "link:vmul_x2.out>vsum_x2.a", "mux:vmul_x2", "mux:vsum_x2.seed",
                  "packer:vsum_x2.out", "slicer:vsum_x2.seed", "unit:vmul_x2", "unit:vsum_x2"]  # fmt: skip
    assert n["unit:vsum_x2"]["chips"] == [["sum", "mvt"], ["sum_1", "mvt_1"]]
    assert n["link:vmul_x2.out>vsum_x2.a"]["chips"] == [
        ["tmp1_prod", "mvt"],
        ["tmp4_prod", "mvt_1"],
    ]
    assert n["mux:vsum_x2.seed"]["ins"] == ["seed·mvt", "seed·mvt_1"]
    assert n["demux:vsum_x2.out"]["outs"] == ["out·mvt", "out·mvt_1"]
    assert {o["id"]: o["text"] for o in d["outlines"]} == {
        "grp:mvt_1": "group mvt_1 · shared by mvt, mvt_1"}  # fmt: skip


def test_tasks_only_after_share():
    """Before ``share`` (step 8) no edge has a task and there is no mux; after it every edge of
    the shared hardware's two chains has one."""
    before, after = drawn("mvt_4_i8_dma_AT", 8), drawn("mvt_4_i8_dma_AT", 18)
    assert before["tasks"] == [] and all(e["task"] is None for e in before["edges"])
    assert not [n for n in before["nodes"] if n["kind"] in ("mux", "demux")]
    tasks = {e["id"]: e["task"] for e in after["edges"]}
    assert tasks["y1@0->tmp1_mult_map_1"] == "mvt" and tasks["y2@0->tmp4_mult_map_1"] == "mvt_1"


def test_an_edge_skips_the_loops_its_subset_does_not_use():
    """y1 is read whole in every m_t: it goes straight to the inner loop, marked ``↻ m_t``."""
    (e,) = [e for e in drawn("mvt_4_i8_dma_AT", 18)["edges"] if e["from"] == "y1@0"]
    assert (e["to"], e["skip"], e["label"]) == ("tmp1_mult_map_1", ["m_t"], "↻ m_t")


def test_a_link_draws_one_chain_and_the_readers_loops_as_chips():
    """After the link (step 8) vsum's loops match vmul's level by level: one chain, sum_map and
    sum_map_1 as chips on tmp1_mult_map and tmp1_mult_map_1; the fold says what it sums over."""
    n = by_id(drawn("mvt_4_i8_dma_AT", 8))
    assert "sum_map" not in n and "sum_map_1" not in n
    assert n["tmp1_mult_map"]["chips"] == [["sum_map", None]]
    assert n["tmp1_mult_map_1"]["chips"] == [["sum_map_1", None]]
    assert (n["unit:vsum_x1"]["fold"], n["unit:vsum_x1"]["sub"]) == (True, "over n_1_t")


def test_labels_only_where_the_dtype_changes():
    """int8 in, int32 out of the multiplier: only that edge carries a dtype."""
    d = drawn("mvt_4_i8_dma_AT", 18)
    assert [(e["id"], e["label"]) for e in d["edges"] if e["dt"]] == [
        ("unit:vmul_x2.out->link:vmul_x2.out>vsum_x2.a.in", "int32")]  # fmt: skip


@pytest.mark.slow
def test_every_recipes_every_step_is_well_formed():
    """Every checked-in recipe, every step: no structural problem (vecadd_undo is refused on
    purpose)."""
    bad = {}
    for path in sorted((REPO / "recipes").glob("*/*.json")):
        try:
            gs = graphs(path.stem) if find_recipe(path.stem) == path else ()
        except SandboxError:
            assert path.stem == "vecadd_undo"
            continue
        for k, g in enumerate(gs):
            d = graph_view(Entry(str(k), Path(f"{k}.snaxdfg"), g, None))["drawing"]
            if p := problems(d):
                bad[f"{path.stem} {k}"] = p[:3]
    assert bad == {}
