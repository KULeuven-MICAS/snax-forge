"""The DFG viewer: loading files, the view of a graph, the routes.

The drawing's boxes, ports, edges and outlines are what the page lays out, so they are checked
here exactly; where ELK puts them is checked by eye on the page (test_drawing.py has the
structural checks over every recipe).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from snax_forge.dfg import Graph
from snax_forge.viz.dfg.__main__ import main
from snax_forge.viz.dfg.api import (
    GraphSet,
    entry_names,
    expand,
    graph_view,  # fmt: skip
    load_entry,
)
from snax_forge.viz.dfg.server import make_dfg_server
from tests.helpers import FIXTURES, REPO, recipe_path

from .helpers import box_of, boxes, containers, ends, get, served


def view(path: Path) -> dict:
    return graph_view(load_entry(path.stem, path))


def writes(v: dict) -> list[tuple[str, str]]:
    """Every edge into a container's box: (from, its memlet in Details)."""
    b = boxes(v)
    return [(f, v["details"][f"edge:{e['id']}"]["memlet"]) for e, (f, _) in
            zip(v["drawing"]["edges"], ends(v), strict=True) if b[e["to"]]["kind"] == "container"]  # fmt: skip


def tasklet(tid, ins, outs, code="out = in1 + in2"):
    return {
        "id": tid,
        "kind": "tasklet",
        "inputs": {c: {"data": d, "subset": ["i"]} for c, d in ins.items()},
        "outputs": {c: {"data": d, "subset": ["i"]} for c, d in outs.items()},
        "attrs": {"code": code},
    }


def graph_file(tmp_path: Path, name: str, containers: dict, body: list) -> Path:
    cont = {k: {"shape": ["N"], "dtype": "int64", "transient": t} for k, t in containers.items()}
    g = Graph.from_dict({"name": name, "symbols": {"N": None}, "containers": cont, "body": body})
    path = tmp_path / f"{name}.snaxdfg"
    g.save(path)
    return path


def a_map(mid, *nodes):
    return {"id": mid, "kind": "map", "attrs": {"var": "i", "range": "0:N"}, "body": list(nodes)}


# =============================================================================
# Loading
# =============================================================================


def test_files_and_directories(tmp_path):
    for name in ("10_last", "2_bind", "0_input"):
        shutil.copy(FIXTURES / "vecadd.snaxdfg", tmp_path / f"{name}.snaxdfg")
    (tmp_path / "recipe.json").write_text("{}")
    files = expand([tmp_path, FIXTURES / "vecadd.snaxdfg"])
    assert [f.name for f in files] == [
        "0_input.snaxdfg", "2_bind.snaxdfg", "10_last.snaxdfg", "vecadd.snaxdfg"
    ]  # fmt: skip
    assert entry_names(files[:3]) == ["0_input", "2_bind", "10_last"]  # one folder: the stems
    with pytest.raises(FileNotFoundError, match="no such file"):
        expand([tmp_path / "nope"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no .snaxdfg file"):
        expand([tmp_path / "empty"])


def test_a_broken_file_keeps_its_error(tmp_path):
    bad = tmp_path / "bad.snaxdfg"
    bad.write_text((FIXTURES / "vecadd.snaxdfg").read_text().replace('"data": "B"', '"data": "X"'))
    (tmp_path / "notjson.snaxdfg").write_text("{")
    gs = GraphSet([tmp_path])
    assert [(e["name"], e["error"]) for e in gs.summaries()] == [
        ("bad", "node 'add'.inputs.in2: unknown container 'X'"),
        ("notjson", "invalid JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"),
    ]  # fmt: skip
    assert "drawing" not in graph_view(gs.get("bad"))
    bad.write_text((FIXTURES / "vecadd.snaxdfg").read_text())
    gs.reload()
    assert gs.get("bad").error is None and graph_view(gs.get("bad"))["drawing"]["nodes"]


# =============================================================================
# The view of a graph
# =============================================================================


def test_a_folding_write_names_its_op():
    """dot's sum map writes out[0] with reduce add: the edge's memlet says so, and the tasklet's
    text is the fold."""
    v = view(FIXTURES / "dot.snaxdfg")
    assert writes(v) == [("mult.out", "tmp0[n]"), ("sum.out", "out[0] (reduce add)")]
    assert [boxes(v)[x]["text"] for x in ("mult", "sum")] == ["mult · in1 * in2", "sum · fold +"]


@pytest.mark.slow
def test_the_bank_note_under_its_group():
    """vecmul at N = 2048: at W = 16 the group's summary has its bank note and Compare
    against W = 8 a ``bank bound`` row under the group, "–" on the side with none."""
    from snax_forge.design import Platform
    from snax_forge.viz.dfg.api import Entry, compare_view, step_view
    from tests.design.helpers import SNAX_FORGE
    from tests.helpers import steps

    platform = Platform.load(SNAX_FORGE)
    w8, w16 = (
        Entry(f"w{w}", Path(f"w{w}"), steps("vecmul_v1", N=2048, W=w)[-1], None) for w in (8, 16)
    )
    note = ("acc: the streams move 6144 words over 32 banks, at least 192 cycles of L1 time "
            "against the ideal 128")  # fmt: skip
    (g16,) = step_view(w16, "w16", platform)["summary"]["groups"]
    (g8,) = step_view(w8, "w8", platform)["summary"]["groups"]
    assert (g16["bank_note"], g8["bank_note"]) == (note, None)
    rows = compare_view(w8, w16, platform)["columns"]["estimate"]
    (row,) = [r for r in rows if r["label"] == "bank bound"]
    assert (row["under"], row["a"], row["b"], row["differs"]) == ("group acc", "–", note, True)


@pytest.mark.slow
def test_an_apply_recipes_params_are_bindings_not_unused():
    """mvt_v5_1: each apply_recipe step naming the session's params is an entry
    (``matvec_acc_v3 on matvec_acc · W_x1 = 8 · D_2 = 1``), so nothing is unused."""
    from snax_forge.sandbox import Recipe
    from snax_forge.viz.dfg.api import bindings_section
    from tests.helpers import steps

    r = Recipe.load(recipe_path("mvt_v5_1"))
    b = bindings_section(steps("mvt_v5_1")[-1], r, editable=True)
    recipes = [(e["label"], [(p["name"], p["value"]) for p in e["params"]])
               for e in b["entries"] if e["kind"] == "recipe"]  # fmt: skip
    assert recipes == [("matvec_acc_v3 on matvec_acc", [("D_2", 1), ("W_x1", 8)]),
                       ("matvec_acc_v4 on matvec_acc_1", [("W_x1", 8)])]  # fmt: skip
    assert b["unused"] == []


def test_inside_one_node_an_edge_goes_straight_from_the_writer(tmp_path):
    """t1 writes tmp0 and t2 reads it in the same map: the edge goes straight from t1 to t2,
    and the map's hop into t2, which no data takes, is an empty one."""
    path = graph_file(
        tmp_path, "chain", {"A": False, "B": False, "tmp0": True},
        [a_map("m", tasklet("t1", {"in1": "A"}, {"out": "tmp0"}, "out = in1"),
               tasklet("t2", {"in1": "tmp0"}, {"out": "B"}, "out = in1"))],
    )  # fmt: skip
    v = view(path)
    assert ends(v) == [("A@0", "m"), ("m", "t1.in1"), ("t1.out", "t2.in1"), ("t1.out", "tmp0@1"),
                       ("t2.out", "B@1"), ("m", "t2")]  # fmt: skip
    assert v["drawing"]["edges"][-1]["style"] == "empty"
    assert containers(v) == ["A@0", "tmp0@1", "B@1"]


# =============================================================================
# Server and command line
# =============================================================================


def test_routes(tmp_path):
    """View mode: the steps, a step, a reload; the old page's routes are gone."""
    path = tmp_path / "vecadd.snaxdfg"
    shutil.copy(FIXTURES / "vecadd.snaxdfg", path)
    with served(make_dfg_server([path], port=0)) as srv:
        status, ctype, body = get(srv, "/")
        assert status == 200 and ctype == "text/html" and b"dfg.js" in body
        assert get(srv, "/dfg.js")[1] == "text/javascript"
        assert json.loads(get(srv, "/api/steps")[2]) == [
            {"name": "vecadd", "folder": None, "label": "vecadd", "error": None}
        ]
        nodes = json.loads(get(srv, "/api/step/0")[2])["drawing"]["nodes"]
        assert [n["id"] for n in nodes if n.get("layer") == "first"] == ["A@0", "B@0"]
        for gone in ("/api/graphs", "/api/graph/vecadd", "/api/step/1"):
            with pytest.raises(HTTPError) as e:
                get(srv, gone)
            assert e.value.code == 404, gone
        path.write_text(path.read_text().replace('"N": null', '"N": 8'))
        req = Request(srv.url + "api/reload", data=b"{}", method="POST",
                      headers={"Content-Type": "application/json"})  # fmt: skip
        with urlopen(req, timeout=10) as r:
            assert json.loads(r.read()) == {"graphs": ["vecadd"]}
        assert json.loads(get(srv, "/api/step/0")[2])["symbols"] == {"N": 8}


def test_cli_rejects_a_missing_path(tmp_path, capsys):
    assert main([str(tmp_path / "nope.snaxdfg"), "--port", "0"]) == 2
    assert "no such file" in capsys.readouterr().err


# =============================================================================
# The cost in Compare and Details
# =============================================================================


def _dot5_steps():
    from snax_forge.design import Platform
    from snax_forge.viz.dfg.api import Entry
    from tests.helpers import steps

    gs = steps("dot_v5", W=8, N=2048)
    return [Entry(str(k), REPO, g, None) for k, g in enumerate(gs)], Platform.load(
        REPO / "platforms" / "snax_forge.json")  # fmt: skip


def _rows(rows, under):
    return {r["label"]: (r["a"], r["b"], r["differs"]) for r in rows if r["under"] == under}


@pytest.mark.slow
def test_compare_gives_two_columns_when_the_groups_differ():
    """dot 5's second bind (mac and sum, two groups) against its link (one group, dot): the
    Estimate in two columns, each side's texts, a group on one side only "–" on the other, the
    accelerator's cost on both; the Details of mult on both sides, its cost a row per text."""
    from snax_forge.viz.dfg.api import compare_view, step_view

    e, platform = _dot5_steps()
    five, six = (step_view(e[k], str(k), platform)["summary"] for k in (5, 6))
    assert [g["name"] for g in five["groups"]] == ["mac", "sum"]
    assert [g["name"] for g in six["groups"]] == ["dot"]
    c = compare_view(e[5], e[6], platform, labels=("5 bind", "6 link"))["columns"]
    assert c["labels"] == ["5 bind", "6 link"] and "about 30 %" in c["caveat"]
    rows = c["estimate"]
    assert not any("→" in r["a"] + r["b"] for r in rows)
    assert _rows(rows, "accelerator") == {"cycles": ("266", "265", True),
                                          "unmapped cycles": ("0", "0", False),
                                          "total cycles": ("266", "265", True),
                                          "streamers": (
                                              "2 streamer sets · 40 CSRs",
                                              "1 streamer set · 23 CSRs",
                                              True)}  # fmt: skip
    assert _rows(rows, "group mac")["energy"] == ("26.027 nJ", "–", True)
    assert _rows(rows, "group dot")["cycles"] == ("–", "265 = 256 + 9", True)
    # The link drops a streamer set (two streamers and 16 crossbar inputs), the banks the
    # same;
    # its 16 fewer accesses make it the cheaper one too, as an access has one price; the crossbar's
    # 24 + 9 and 16 + 1 read-only and write-only inputs priced as such
    bars = c["cost_bars"]
    assert [b["label"] for b in bars["area"]] == ["5 bind", "6 link"]
    assert [b["total_text"] for b in bars["area"]] == ["740 000 µm²", "626 000 µm²"]
    assert sum(x["width"] for x in bars["area"][0]["segments"]) == pytest.approx(1.0)
    assert sum(x["width"] for x in bars["area"][1]["segments"]) == pytest.approx(
        626283.5 / 739652.6, rel=1e-3
    )
    assert _rows(rows, "Cost estimate (first-order) on nangate45") == {
        "area": ("740 000 µm²", "626 000 µm²", True),
        "area without the banks": ("307 000 µm²", "194 000 µm²", True),
        "energy per run": ("59.755 nJ", "59.656 nJ", True),
        "datapath": ("61 000 µm² · 25.951 nJ", "61 000 µm² · 25.951 nJ", False),
        "control": ("1 110 µm² · 0.096 nJ", "1 090 µm² · 0.149 nJ", True),
        "streamers": ("132 000 µm² · 11.568 nJ", "68 000 µm² · 11.523 nJ", True),
        "crossbar": ("113 000 µm² · 7.562 nJ", "63 500 µm² · 7.535 nJ", True),
        "memory (banks)": ("433 000 µm² · 14.579 nJ", "433 000 µm² · 14.499 nJ", True),
        "L1 accesses": ("mac: 4104 words, 22.090 nJ; sum: 9 words, 0.051 nJ",
                        "dot: 4097 words, 22.034 nJ", True),
        "not characterized": ("none", "none", False)}  # fmt: skip
    mult = c["details"]["node:mult"]
    assert mult["where"] == "" and _rows(mult["rows"], None)["group"] == ("mac", "dot", True)
    assert _rows(mult["rows"], "cost")["per fire"] == ("101.302 pJ", "101.302 pJ", False)
    # the same fires: a held flop costs nothing, so the Mac's task is the same on both sides
    assert _rows(mult["rows"], "cost")["per task"] == (
        "25.933 nJ (256 fires, 512 bits)", "25.933 nJ (256 fires, 512 bits)", False)  # fmt: skip
    assert c["details"]["group:mac"]["where"] == "removed"
    assert c["details"]["group:dot"]["where"] == "added"
    assert {r["a"] for r in c["details"]["group:dot"]["rows"]} == {"–"}
    # from the unbound import: no cost on the left
    first = compare_view(e[0], e[6], platform)["columns"]["estimate"]
    assert _rows(first, "Cost estimate (first-order) on nangate45")["area"] == (
        "–",
        "626 000 µm²",
        True,
    )
    assert step_view(e[0], "0", platform)["summary"]["cost"] is None  # nothing bound
    assert not any(r["under"] and r["under"].startswith("Cost")
                   for r in compare_view(e[0], e[1], platform)["columns"]["estimate"])  # fmt: skip


# =============================================================================
# The Bindings tree, the params' names in the boxes and Details
# =============================================================================


def test_the_bindings_tree():
    """Dot 1 accelerator ▸ group ▸ unit (dot ▸ mul ▸ mul's entry); dot 2's linked group
    not encapsulated at the top; vecmul's unit alone at the top; a split not bound yet at the
    top beside a bound unit."""
    from dataclasses import replace

    from snax_forge.sandbox import Recipe
    from snax_forge.viz.dfg.api import bindings_section
    from tests.helpers import steps

    def tree(name, k=None):
        r = Recipe.load(recipe_path(name))
        k = len(r.steps) if k is None else k  # graph k is made by steps 1..k
        b = bindings_section(steps(name)[k], replace(r, steps=r.steps[:k]))
        return b["tree"], [e["label"] for e in b["entries"]]

    assert tree("dot_v1_1") == ([{"kind": "accelerator", "name": "dot", "children": [
        {"kind": "group", "name": "vmul", "children": [{"kind": "entry", "index": 0}]},
        {"kind": "group", "name": "vsum", "children": [{"kind": "entry", "index": 1}]}]}],
        ["vmul · simd", "vsum · accumulate"])  # fmt: skip
    # dot 1, two accelerators of one group of one unit each: both at the top
    assert tree("dot_v1") == ([{"kind": "entry", "index": 0}, {"kind": "entry", "index": 1}],
                             ["vmul · simd", "vsum · accumulate"])  # fmt: skip
    assert tree("dot_v2")[0] == [{"kind": "group", "name": "dot", "children": [
        {"kind": "entry", "index": 0}, {"kind": "entry", "index": 1}]}]  # fmt: skip
    assert tree("vecmul_v1")[0] == [{"kind": "entry", "index": 0}]
    # dot 1 at its step 3: vmul bound, sum_map split, nothing encapsulated: both at the top
    t, labels = tree("dot_v1", 3)
    assert t == [{"kind": "entry", "index": 0}, {"kind": "entry", "index": 1}]
    assert labels == ["vmul · simd", "sum_map split_map"]


# =============================================================================
# Scalars: axpy's alpha
# =============================================================================


def _axpy(name: str):
    from snax_forge.sandbox import Recipe, apply_recipe

    r = Recipe.load(recipe_path(name))
    return r, [x.graph for x in apply_recipe(r, Graph.load(FIXTURES / "axpy.snaxdfg"))]


def _views(r, graphs):
    from snax_forge.design import Platform
    from snax_forge.viz.dfg.api import Entry, step_view
    from tests.design.helpers import SNAX_FORGE

    p = Platform.load(SNAX_FORGE)
    return [step_view(Entry("s", Path("s"), g, None), "s", p, recipe=r) for g in graphs]


def test_a_scalar_read_as_csr_is_drawn_like_a_baked_one():
    """axpy's recipe: bake_scalar alpha csr first, so mult reads ``csr(alpha) * in2`` where
    axpy_v1_1's reads ``3 * in2``: alpha gets no box, edge or port, through the fuse and the
    split, and the bound unit reads ``k1 csr(alpha)``; Containers still lists alpha."""
    from snax_forge.viz.dfg.api import compact_text

    r, graphs = _axpy("axpy_v1")
    rb, baked = _axpy("axpy_v1_1")
    assert r.steps[0].params == {"container": "alpha", "value": "csr"}
    assert compact_text(graphs[1], graphs[1].node("mult")) == "mult · csr(alpha) * in2"
    for v, vb in zip(_views(r, graphs[1:]), _views(rb, baked[1:]), strict=True):
        assert "alpha" in {c["name"] for c in v["containers_section"]}
        assert not [b for b in boxes(v).values() if b.get("sel") == "alpha"]
        assert not [e for e in v["drawing"]["edges"] if e.get("data") == "alpha"]
        assert box_of(v, "mult")["ins"] == box_of(vb, "mult")["ins"]
        assert containers(v, first=True) == containers(vb, first=True)  # the same inputs


@pytest.mark.slow
def test_the_editor_offers_bake_unbake_and_fuse_map_on_axpy(tmp_path, monkeypatch):
    """The editor (a session on axpy): after the two splits fuse_map on mult_map (second
    augassign_19_4_map) and bake_scalar on alpha, its value an int field at alpha's seeded
    input (make_inputs' 3); after the bake,
    unbake_scalar on the baked tasklet, and alpha no longer a container to bake; fuse_map and
    bind then apply as the recipe does."""
    from snax_forge.sandbox import session
    from snax_forge.sandbox.session import Roots, Session
    from tests.helpers import imported

    monkeypatch.setattr(session, "_import", imported)
    s = Session.from_kernel("axpy", {"N": 64, "W": 4},
                            roots=Roots(tmp_path / "s", tmp_path / "r", tmp_path / "o"))  # fmt: skip
    assert s.sites(s.cursor, "mult_map")["fuse_map"]["ok"]  # the recipes' order: before any split
    for m in ("mult_map", "augassign_19_4_map"):
        assert s.apply("split_map", {"map": m, "factor": "W"})["ok"]
    at = s.sites(s.cursor, "mult_map")
    assert at["fuse_map"]["ok"] and at["fuse_map"]["params"] == {
        "first": "mult_map", "second": "augassign_19_4_map"}  # fmt: skip
    bake = s.sites(s.cursor, "alpha")["bake_scalar"]
    assert bake["ok"] and {f["name"]: f["type"] for f in bake["form"]} == {
        "container": "str", "value": "int"}  # fmt: skip
    assert {f["name"]: f["value"] for f in bake["form"]} == {"container": "alpha", "value": 3}
    assert "alpha" in s.marks(s.cursor)["containers"]
    assert not s.sites(s.cursor, "mult")["unbake_scalar"]["ok"]
    assert s.apply("bake_scalar", {"container": "alpha", "value": 3})["ok"]
    assert s.sites(s.cursor, "mult")["unbake_scalar"]["params"] == {"container": "alpha"}
    assert "mult" in s.marks(s.cursor)["nodes"]
    assert s.apply("fuse_map", {"first": "mult_map", "second": "augassign_19_4_map"})["ok"]
    got = s.apply("bind", {"node": "mult_map_s", "brm": "simd",
                           "implementation": "chisel_tiled_spatial", "instance": "acc"})  # fmt: skip
    assert got["ok"], got
    assert s.graphs[-1].node("mult").attrs["params"]["k1"] == 3


# =============================================================================
# The fold's seed: matvec_acc_v1
# =============================================================================


def _matvec_seed():
    from snax_forge.sandbox import Recipe, apply_recipe

    r = Recipe.load(recipe_path("matvec_acc_v1"))
    return r, [x.graph for x in apply_recipe(r, Graph.load(FIXTURES / "matvec_acc.snaxdfg"))]


# =============================================================================
# Partners: the maps a map fuses, seeds or interchanges with
# =============================================================================


def _bound_fixture(name: str, **symbols: int) -> Graph:
    d = json.loads((FIXTURES / f"{name}.snaxdfg").read_text())
    d["symbols"] = symbols
    return Graph.from_dict(d)


def _partners(g: Graph) -> dict[str, list[tuple[str, str, str | None]]]:
    from snax_forge.viz.dfg.api import Entry, graph_view

    v = graph_view(Entry("g", Path("g"), g, None))
    return {m: [(p["map"], p["transform"], (p.get("marker") or {}).get("text")) for p in ps
                if "near" not in p]
            for m, ps in v["partners"].items()}  # fmt: skip


def _agree(g: Graph, found: dict) -> None:
    """Every partner shown is one its transform's site takes, the pair in either role."""
    from snax_forge.sandbox import SandboxError
    from snax_forge.sandbox.transforms import _interchange_check, site_of

    for m, ps in found.items():
        for other, transform, _ in ps:
            if transform == "interchange_map":
                _interchange_check(g, m, other, "")
                continue
            try:
                site_of(transform, g, [m, other])
            except SandboxError:  # the partner is the first
                site_of(transform, g, [other, m])


MATVEC_FUSE = "tmp0[m, n] · m = m (0:8), n = n (0:512)"


def test_axpys_two_maps_fuse():
    """axpy at import: the multiply and the in-place add fuse over n (tmp0[n]); one-level
    nests, so nothing interchanges; no fold, so nothing seeds."""
    g = _bound_fixture("axpy", N=64)
    found = _partners(g)
    fuse = "tmp0[n] · n = n (0:64)"
    assert found == {"mult_map": [("augassign_19_4_map", "fuse_map", fuse)],
                     "augassign_19_4_map": [("mult_map", "fuse_map", fuse)]}  # fmt: skip
    _agree(g, found)


def test_a_seeded_or_bound_graph_has_no_seed_partner():
    """matvec_acc_v1's last steps: the seed taken, so no fuse_seed partner is left."""
    _, graphs = _matvec_seed()
    for g in graphs[6:]:
        assert not [p for ps in _partners(g).values() for p in ps if p[1] == "fuse_seed"]


# =============================================================================
# Folds: drawn as upside-down trapezoids
# =============================================================================
