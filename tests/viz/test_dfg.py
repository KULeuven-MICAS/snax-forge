"""The DFG viewer: loading files, the view of a graph, the routes.

The rows, boxes and edges are what the panel draws, so they are checked here
exactly; where the browser puts them, and the SVG curves, are checked by eye
(D55) on the three vecadd fixtures and a recipe's output directory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from snax_forge.dfg import Graph
from snax_forge.estimate.cost import L1_LABEL
from snax_forge.viz.dfg.__main__ import main
from snax_forge.viz.dfg.api import (
    GraphSet,
    entry_folders,
    entry_names,
    expand,
    find_step,
    graph_view,  # fmt: skip
    load_entry,
)
from snax_forge.viz.dfg.server import make_dfg_server
from tests.helpers import FIXTURES, REPO

from .helpers import get, served


def view(path: Path) -> dict:
    return graph_view(load_entry(path.stem, path))


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


def test_two_folders_with_the_same_stems_name_their_steps_by_folder(tmp_path):
    """Two designs' recipe folders with the same stems: each step is ``<folder>/<stem>`` and
    keeps its folder, in the order the folders were given; one folder keeps the bare stems."""
    for d in ("dot4_w8", "dot5_w8"):
        (tmp_path / d).mkdir()
        shutil.copy(FIXTURES / "vecadd.snaxdfg", tmp_path / d / "0_input.snaxdfg")
        shutil.copy(FIXTURES / "vecadd_accelerated.snaxdfg", tmp_path / d / "1_bind.snaxdfg")
    gs = GraphSet([tmp_path / "dot5_w8", tmp_path / "dot4_w8"])
    assert [(e["name"], e["folder"]) for e in gs.summaries()] == [
        ("dot5_w8/0_input", "dot5_w8"), ("dot5_w8/1_bind", "dot5_w8"),
        ("dot4_w8/0_input", "dot4_w8"), ("dot4_w8/1_bind", "dot4_w8")]  # fmt: skip
    assert gs.get("dot4_w8/1_bind").graph is not None
    one = GraphSet([tmp_path / "dot4_w8"])
    assert [(e["name"], e["folder"]) for e in one.summaries()] == [
        ("0_input", None), ("1_bind", None)]  # fmt: skip


def test_files_given_from_two_folders():
    """Files given one by one: from two folders, named by folder (two folders of one name
    still made unique); from one, the bare stems."""
    two = [Path("out/a/x.snaxdfg"), Path("out/b/y.snaxdfg")]
    assert (entry_names(two), entry_folders(two)) == (["a/x", "b/y"], ["a", "b"])
    same = [Path("p/a/x.snaxdfg"), Path("q/a/x.snaxdfg"), Path("q/a/z.snaxdfg")]
    assert entry_names(same) == ["a/x", "a/x-2", "a/z"]
    one = [Path("out/a/x.snaxdfg"), Path("out/a/./y.snaxdfg")]
    assert (entry_names(one), entry_folders(one)) == (["x", "y"], [None, None])


def test_an_old_hash_finds_its_step():
    """The URL hash's step: a full name; a bare stem when exactly one step has it (a hash from
    before the folders); a number (a hash from before the names); else none."""
    names = ["dot4_w8/0_input", "dot4_w8/6_link", "dot5_w8/0_input", "dot5_w8/7_bind"]
    assert find_step(names, "dot5_w8/0_input") == 2
    assert (find_step(names, "6_link"), find_step(names, "7_bind")) == (1, 3)
    assert find_step(names, "0_input") is None  # in both folders
    assert (find_step(names, "2"), find_step(names, "4"), find_step(names, "nope")) == (
        2,
        None,
        None,
    )
    assert find_step(["0_input", "6_link"], "6_link") == 1  # one folder: the name is the stem


def test_a_broken_file_keeps_its_error(tmp_path):
    bad = tmp_path / "bad.snaxdfg"
    bad.write_text((FIXTURES / "vecadd.snaxdfg").read_text().replace('"data": "B"', '"data": "X"'))
    (tmp_path / "notjson.snaxdfg").write_text("{")
    gs = GraphSet([tmp_path])
    assert [(e["name"], e["error"]) for e in gs.summaries()] == [
        ("bad", "node 'add'.inputs.in2: unknown container 'X'"),
        ("notjson", "invalid JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"),
    ]  # fmt: skip
    assert "rows" not in graph_view(gs.get("bad"))
    bad.write_text((FIXTURES / "vecadd.snaxdfg").read_text())
    gs.reload()
    assert gs.get("bad").error is None and graph_view(gs.get("bad"))["rows"]


# =============================================================================
# The view of a graph
# =============================================================================


def test_vecadd_accelerated():
    """The compact texts, and everything else in Details: the unit's BRM, params, timing and
    what it replaced; each edge's memlet and packing (int64: one per word, num_ch = W)."""
    v = view(FIXTURES / "vecadd_accelerated.snaxdfg")
    assert v["graph"] == "vecadd" and v["symbols"] == {"N": 64}
    assert [r["kind"] for r in v["rows"]] == ["containers", "node", "containers"]
    assert v["rows"][0]["boxes"] == ["A@0", "B@0"] and v["rows"][2]["boxes"] == ["C@1"]
    assert v["boxes"]["C@1"] == {
        "key": "C@1", "container": "C", "version": 1, "written_by": ["add"], "again": False
    }  # fmt: skip
    assert v["containers"]["A"] == {
        "name": "A",
        "text": "A",
        "badge": "L1 · 512 B",
        "transient": False,
    }
    top = v["rows"][1]["node"]
    assert (top["id"], top["loop_kind"], top["text"]) == (
        "add_map",
        "temporal",
        "add_map · i_t · 16 · temporal",
    )
    acc = top["body"][0]
    assert (acc["text"], acc["inputs"][0]) == ("acc · a + b · W 4", {"connector": "a", "data": "A"})
    d = v["details"]["node:add"]
    assert (d["brm"], d["implementation"], d["params"], d["replaced"], d["timing"]) == (
        "elementwise_add", "chisel_tiled_spatial", {"W": 4, "op": "add"}, "add_map_s › add",
        {"latency": 0, "ii": 1, "span": 1})  # fmt: skip
    assert v["details"]["node:add_map"] == {"kind": "map", "var": "i_t", "range": "0:N // 4",
                                           "loop.kind": "temporal",
                                           "loop.split": {"var": "i", "range": "0:N"}}  # fmt: skip
    assert v["details"]["container:A"]["sizes"] == [64]
    assert [(e["id"], e["dir"]) for e in v["edges"]] == [
        ("A@0->add.a", "read"), ("B@0->add.b", "read"), ("add.out->C@1", "write")]  # fmt: skip
    assert "text" not in v["edges"][0] and "packing" not in v["edges"][0]
    e = v["details"]["edge:A@0->add.a"]
    assert (e["memlet"], e["packing"], e["loops"], e["beats"], e["streamer"]) == (
        "A[4 * i_t:4 * i_t + 4]", "int64 · num_slc 1 · num_ch 4", [["i_t", 16]], 16, True)  # fmt: skip


def recipe_view(tmp_path: Path, recipe: str) -> dict:
    """The view of ``recipes/<recipe>.json``'s last (bound) graph."""
    from snax_forge.dfg.import_sdfg import import_kernel
    from snax_forge.sandbox import Recipe, apply_recipe

    r = Recipe.load(REPO / "recipes" / f"{recipe}.json")
    path = tmp_path / f"{recipe}.snaxdfg"
    apply_recipe(r, import_kernel(r.kernel, r.dtypes))[-1].graph.save(path)
    return view(path)


@pytest.mark.parametrize(
    ("recipe", "want"),
    [
        # int32, W = 4: two to a word, two channels a port
        ("vecadd_i32", {"A": "int32 · num_slc 2 · num_ch 2", "B": "int32 · num_slc 2 · num_ch 2", "C": "int32 · num_slc 2 · num_ch 2"}),
        # a fold's one-lane output shows its own type: one int64, one word
        ("vsum_tree_i32", {"A": "int32 · num_slc 2 · num_ch 2", "out": "int64 · num_slc 1 · num_ch 1"}),
        # int32 in, the int64 product out: 4 lanes take two words in and four out
        ("vecmul_i32_i64", {"A": "int32 · num_slc 2 · num_ch 2", "B": "int32 · num_slc 2 · num_ch 2", "C": "int64 · num_slc 1 · num_ch 4"}),
    ],
)  # fmt: skip
def test_a_port_edge_labels_its_packing(tmp_path, recipe, want):
    """Each accelerated port's edge: dtype, num_slc = 64 / b, num_ch = W / num_slc (D142)."""
    v = recipe_view(tmp_path, recipe)
    edges = [v["details"][f"edge:{e['id']}"] for e in v["edges"]]
    assert {e["data"]: e["packing"] for e in edges if e["packing"]} == want


def test_a_folding_write_names_its_op():
    """dot's sum map writes out[0] with reduce add: the edge's memlet says so, and the tasklet's
    text is the fold."""
    v = view(FIXTURES / "dot.snaxdfg")
    writes = [
        (e["from"], v["details"][f"edge:{e['id']}"]["memlet"])
        for e in v["edges"]
        if e["dir"] == "write"
    ]
    assert writes == [("mult.out", "tmp0[i]"), ("sum.out", "out[0] (reduce add)")]
    tops = [r["node"] for r in v["rows"] if r["kind"] == "node"]
    assert [t["body"][0]["text"] for t in tops] == ["mult · in1 * in2", "sum · fold +"]


def test_the_bound_dot_shows_both_accelerators():
    """dot after its recipe: mul and sum, each over N // 4 beats; tmp0 between them."""
    v = view(FIXTURES / "dot_accelerated.snaxdfg")
    tops = [r["node"] for r in v["rows"] if r["kind"] == "node"]
    assert [t["text"] for t in tops] == [
        "mult_map · i_t · 16 · temporal",
        "sum_map · i_t_1 · 16 · temporal",
    ]
    assert [t["body"][0]["text"] for t in tops] == ["mul · a * b · W 4", "sum · fold + · W 4"]
    writes = [
        (e["from"], v["details"][f"edge:{e['id']}"]["memlet"])
        for e in v["edges"]
        if e["dir"] == "write"
    ]
    assert writes == [("mult.out", "tmp0[4 * i_t:4 * i_t + 4]"), ("sum.out", "out[0:1]")]


def test_dot_1_draws_one_accelerator_of_two_groups(tmp_path):
    """dot 1 (D148): the accelerator dot, its groups mul and sum in the graph's order, each
    unit in its group and its group's box around its top-level map; each unit's panel names
    its group and accelerator."""
    from tests.gen.helpers import bound_dot1

    path = tmp_path / "dot_1.snaxdfg"
    path.write_text(bound_dot1().to_json())
    v = view(path)
    assert v["accelerators"] == [{"name": "dot", "text": "dot", "groups": [
        {"name": "mul", "text": "mul", "tops": ["mult_map"], "links": [],
         "units": [{"instance": "mul", "nodes": ["mult"]}]},
        {"name": "sum", "text": "sum", "tops": ["sum_map"], "links": [],
         "units": [{"instance": "sum", "nodes": ["sum"]}]},
    ]}]  # fmt: skip
    units = [v["details"][f"node:{n}"] for n in ("mult", "sum")]
    assert [(u["accelerator"], u["group"]) for u in units] == [("dot", "mul"), ("dot", "sum")]
    assert v["details"]["group:mul"]["accelerator"] == "dot"
    assert v["details"]["accelerator:dot"]["groups"] == ["mul", "sum"]


def test_without_encapsulate_each_group_is_its_own_accelerator():
    v = view(FIXTURES / "dot_accelerated.snaxdfg")
    assert [(a["name"], [g["name"] for g in a["groups"]]) for a in v["accelerators"]] == [
        ("mul", ["mul"]), ("sum", ["sum"])]  # fmt: skip
    units = [v["details"][f"node:{n}"] for n in ("mult", "sum")]
    assert [(u["accelerator"], u["group"]) for u in units] == [("mul", "mul"), ("sum", "sum")]


@pytest.mark.slow
def test_a_maps_bar_starts_with_its_id():
    """C§12: vecmul as imported at N = 2048, its map's bar ``mult_map · i · 2048 · untagged``."""
    from snax_forge.viz.dfg.api import Entry
    from tests.helpers import steps

    g = steps("vecmul", N=2048)[0]
    v = graph_view(Entry("0_input", Path("0_input.snaxdfg"), g, None))
    assert v["rows"][1]["node"]["text"] == "mult_map · i · 2048 · untagged"


@pytest.mark.slow
def test_the_containers_section_of_vecmul_and_matvec():
    """C§12's Containers, in the graph's order: vecmul's A, B, C each ``N``, 2048 int64, 16 KiB
    in L1 over N; matvec's matrix over M and N, its vectors over one each; a step view has it."""
    from snax_forge.design import Platform
    from snax_forge.estimate import estimate
    from snax_forge.viz.dfg.api import Entry, containers_section, step_view
    from tests.design.helpers import SNAX_FORGE
    from tests.helpers import steps

    platform = Platform.load(SNAX_FORGE)
    g = steps("vecmul", N=2048, W=8)[-1]
    got = containers_section(g, estimate(g, platform))
    keys = ("name", "shape_text", "sizes", "dtype", "bytes", "bytes_text", "placement", "symbols")
    assert [tuple(c[k] for k in keys) for c in got] == [
        (x, "N", [2048], "int64", 16384, "16 KiB", "l1", ["N"]) for x in "ABC"]  # fmt: skip
    assert got[0]["dims"] == [{"parts": [{"text": "N", "symbol": "N"}]}]
    m = steps("matvec_1")[-1]
    mv = {c["name"]: c for c in containers_section(m, estimate(m, platform))}
    assert (mv["A"]["shape_text"], mv["A"]["sizes"], mv["A"]["symbols"]) == (
        "M × N",
        [8, 512],
        ["M", "N"],
    )
    assert [(mv[x]["shape_text"], mv[x]["symbols"]) for x in ("p", "q")] == [
        ("N", ["N"]),
        ("M", ["M"]),
    ]
    assert [c["line"] for c in got] == ["N · L1 · 16 KiB"] * 3
    assert mv["A"]["line"] == "M × N · L1 · 32 KiB"
    view = step_view(Entry("2_bind", Path("2_bind.snaxdfg"), g, None), "2 bind", platform)
    assert view["containers_section"] == got
    assert containers_section(g)[0]["placement"] is None  # no estimate, no placement


def _cost_rows(t):
    return [(r["block"], r["datapath"], r["area_text"], r["energy_text"], r["share_text"])
            for r in [*t["rows"], t["total"]]]  # fmt: skip


@pytest.mark.slow
def test_the_cost_tables_of_vecmul_and_dot_1():
    """C§12's cost table, a row per unit, buffer link and control unit, the L1 accesses (D154),
    then the total, one table per accelerator: vecmul at W = 8 (N = 2048) its unit 55 300 µm²,
    23.137 nJ, 46.0 %, its control unit 702 µm² (flops and an equal comb), 0.128 nJ, its L1
    accesses 6144 words, 27.005 nJ, 53.7 %; dot 1 at W = 8 one table for its two groups, tmp0's
    round trip in its L1 accesses; Compare's cells before → after."""
    from snax_forge.design import Platform
    from snax_forge.estimate import estimate
    from snax_forge.viz.dfg.api import Entry, compare_view, cost_tables, step_view
    from tests.design.helpers import SNAX_FORGE
    from tests.helpers import steps

    platform = Platform.load(SNAX_FORGE)
    g = steps("vecmul", N=2048, W=8)[-1]
    (t,) = cost_tables(estimate(g, platform), g)
    assert t["accelerator"] == "acc" and _cost_rows(t) == [
        ("acc", "ElementwiseLanes", "55 300", "23.137", "46.0%"),
        ("acc ctrl", "control unit", "702", "0.128", "0.3%"),
        (L1_LABEL, "6144 words, 393 216 bits", "–", "27.005", "53.7%"),
        ("total", "", "56 000", "50.269", "100.0%")]  # fmt: skip
    d = steps("dot_1", N=2048, W=8)[-1]
    (t,) = cost_tables(estimate(d, platform), d)
    assert t["accelerator"] == "dot" and _cost_rows(t) == [
        ("mul", "ElementwiseLanes", "55 300", "23.137", "37.8%"),
        ("mul ctrl", "control unit", "702", "0.257", "0.4%"),
        ("sum", "TreeAccumulate", "2 710", "1.231", "2.0%"),
        ("sum ctrl", "control unit", "1 390", "0.509", "0.8%"),
        (L1_LABEL, "8193 words, 524 352 bits", "–", "36.011", "58.9%"),
        ("total", "", "60 100", "61.144", "100.0%")]  # fmt: skip
    v = step_view(Entry("2_bind", Path("2_bind.snaxdfg"), g, None), "2 bind", platform)
    assert v["summary"]["cost"]["tables"][0]["rows"][0]["area_text"] == "55 300"
    w16 = steps("vecmul", N=2048, W=16)[-1]
    c = compare_view(Entry("w8", Path("w8"), g, None), Entry("w16", Path("w16"), w16, None),
                     platform)["columns"]["cost_tables"]  # fmt: skip
    (unit, ctrl, l1, total) = [*c[0]["rows"], c[0]["total"]]
    assert unit["block"] == "acc" and unit["area_text"].startswith("55 300 → ") and unit["differs"]
    assert ctrl["datapath"] == "control unit" and total["block"] == "total"
    assert l1["block"] == L1_LABEL and "→" not in l1["datapath"] + l1["energy_text"]  # same words


@pytest.mark.slow
def test_the_bank_note_under_its_group():
    """vecmul at N = 2048 (D155): at W = 16 the group's summary has its bank note and Compare
    against W = 8 a ``bank bound`` row under the group, "–" on the side with none."""
    from snax_forge.design import Platform
    from snax_forge.viz.dfg.api import Entry, compare_view, step_view
    from tests.design.helpers import SNAX_FORGE
    from tests.helpers import steps

    platform = Platform.load(SNAX_FORGE)
    w8, w16 = (
        Entry(f"w{w}", Path(f"w{w}"), steps("vecmul", N=2048, W=w)[-1], None) for w in (8, 16)
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
def test_the_estimate_panel_gives_the_unmapped_and_total_cycles():
    """D155 in the page: vecmul at N = 2048 as imported, all 2048 cycles unmapped (the assumption
    in its title and the note), the bound step 256 on the accelerator; Compare's rows 2048 → 256."""
    from snax_forge.design import Platform
    from snax_forge.viz.dfg.api import Entry, compare_view, step_view
    from tests.design.helpers import SNAX_FORGE
    from tests.helpers import steps

    platform = Platform.load(SNAX_FORGE)
    g0, g1, g2 = steps("vecmul", N=2048, W=8)
    a, b = Entry("0_input", Path("0"), g0, None), Entry("2_bind", Path("2"), g2, None)
    tagged = step_view(Entry("1_split_map", Path("1"), g1, None), "1 split_map", platform)
    assert [c["value"] for c in tagged["summary"]["cycles"]] == [
        0,
        2048,
        2048,
    ]  # a tag runs nothing
    s0 = step_view(a, "0 input", platform)["summary"]
    assert [(c["label"], c["value"]) for c in s0["cycles"]] == [
        ("accelerator", 0), ("unmapped", 2048), ("total", 2048)]  # fmt: skip
    assert (
        s0["unmapped_note"]
        == s0["cycles"][1]["title"]
        == (
            "2048 iterations no unit runs, at 1 cycle each: an assumption (a core takes several), D155"
        )
    )
    s2 = step_view(b, "2 bind", platform)["summary"]
    assert [c["value"] for c in s2["cycles"]] == [256, 0, 256] and s2["unmapped_note"] is None
    rows = compare_view(a, b, platform)["columns"]["estimate"]
    got = {r["label"]: (r["a"], r["b"]) for r in rows if r["under"] == "accelerator"}
    assert got == {
        "cycles": ("0", "256"),
        "unmapped cycles": ("2048", "0"),
        "total cycles": ("2048", "256"),
    }  # fmt: skip (one unmapped count, D156)


@pytest.mark.slow
def test_the_symbols_section():
    """C§12's Symbols: each kernel symbol with its value and the containers using it; an input
    on the param of its name with a session's recipe in edit mode, read only otherwise."""
    from snax_forge.sandbox import Recipe
    from snax_forge.viz.dfg.api import symbols_section
    from tests.helpers import REPO, steps

    g = steps("vecmul", N=2048, W=8)[-1]
    r = Recipe.load(REPO / "recipes" / "vecmul.json").with_params({"N": 2048, "W": 8})
    assert symbols_section(g, r, editable=True) == [
        {"name": "N", "value": 2048, "param": "N", "editable": True, "containers": ["A", "B", "C"]}
    ]
    (ro,) = symbols_section(g)
    assert (ro["param"], ro["editable"]) == (None, False)
    assert not symbols_section(g, r)[0]["editable"]  # view mode: the folder's recipe, read only
    m = steps("matvec_1")[-1]
    got = {x["name"]: x for x in symbols_section(m)}
    assert (got["M"]["value"], got["M"]["containers"]) == (8, ["A", "q", "tmp0"])
    assert (got["N"]["value"], got["N"]["containers"]) == (512, ["A", "p", "tmp0"])


def _editor_vecmul():
    """vecmul at N = 2048 as the editor builds it: split by a param W = 8, two tags, the bind."""
    from snax_forge.sandbox import Recipe
    from snax_forge.sandbox.recipe import Step

    return Recipe("vecmul_e", "vecmul", {"N": 2048, "W": 8}, {"N": "N"}, [
        Step("split", {"map": "mult_map", "factor": "W"}),
        Step("tag_map", {"map": "mult_map", "kind": "temporal"}),
        Step("tag_map", {"map": "mult_map_inner", "kind": "spatial"}),
        Step("bind", {"node": "mult", "brm": "elementwise_mul",
                      "implementation": "chisel_tiled_spatial", "instance": "acc"})])  # fmt: skip


def _entries(b):
    return [(e["kind"], e["label"], [(p["name"], p["value"], p["editable"], p["shared_with"])
                                     for p in e["params"]], e["note"]) for e in b["entries"]]  # fmt: skip


@pytest.mark.slow
def test_the_bindings_of_vecmul_dot_1_and_dot_4():
    """C§12's Bindings: vecmul's split before its bind (``mult_map split · W = 8 · not bound``),
    then the unit (``acc · elementwise_mul · W = 8``); dot 1 with W_mul and W_sum, two entries,
    nothing shared; dot 4 at W = 4, one W shared by mac and sum; a param in none is unused."""
    from dataclasses import replace

    from snax_forge.sandbox import Recipe, apply_recipe
    from snax_forge.sandbox.recipe import Step
    from snax_forge.viz.dfg.api import bindings_section
    from tests.helpers import REPO, imported, steps

    r = _editor_vecmul()
    gs = [x.graph for x in apply_recipe(r, imported("vecmul"))]
    one = bindings_section(gs[1], replace(r, steps=r.steps[:1]), editable=True)
    assert _entries(one) == [("split", "mult_map split", [("W", 8, True, [])], "not bound")]
    four = bindings_section(gs[4], r, editable=True)
    assert _entries(four) == [("unit", "acc · elementwise_mul", [("W", 8, True, [])], None)]
    assert (
        four["unused"] == []
        and not bindings_section(gs[4], r)["entries"][0]["params"][0]["editable"]
    )
    d = Recipe.load(REPO / "recipes" / "dot_1.json")
    factor = {"mult_map": "W_mul", "sum_map": "W_sum"}
    d = Recipe(d.name, d.kernel, {"N": 2048, "W_mul": 8, "W_sum": 4}, d.symbols,
               [Step(x.transform, {**x.params, "factor": factor[x.params["map"]]})
                if "factor" in x.params else x for x in d.steps])  # fmt: skip
    dot1 = bindings_section(apply_recipe(d, imported("dot"))[-1].graph, d, editable=True)
    assert _entries(dot1) == [("unit", "mul · elementwise_mul", [("W_mul", 8, True, [])], None),
                              ("unit", "sum · accumulate", [("W_sum", 4, True, [])], None)]  # fmt: skip
    d4 = Recipe.load(REPO / "recipes" / "dot_4.json")
    dot4 = bindings_section(steps("dot_4")[-1], d4, editable=True)
    assert _entries(dot4) == [("unit", "mac · mac", [("W", 4, True, ["sum · accumulate"])], None),
                              ("unit", "sum · accumulate", [("W", 4, True, ["mac · mac"])], None)]  # fmt: skip
    spare = replace(d4, params={**d4.params, "K": 3})
    assert bindings_section(steps("dot_4")[-1], spare, editable=True)["unused"] == [
        {"name": "K", "value": 3, "editable": True}]  # fmt: skip
    bare = bindings_section(steps("dot_4")[-1])  # no recipe: the design params, read only
    assert [(e["label"], e["params"][0]["name"], e["params"][0]["editable"]) for e in bare["entries"]] == [
        ("mac · mac", "W", False), ("sum · accumulate", "W", False)]  # fmt: skip


@pytest.mark.slow
def test_view_mode_reads_the_folders_recipe(tmp_path):
    """A sandbox folder's recipe.json names the params in view mode, read only, up to the step
    shown; the step views carry the left column's three sections."""
    from snax_forge.design import Platform
    from snax_forge.sandbox import apply_recipe, write_steps
    from snax_forge.viz.dfg.api import folder_recipe, step_view
    from tests.design.helpers import SNAX_FORGE
    from tests.helpers import imported

    r = _editor_vecmul()
    write_steps(r, apply_recipe(r, imported("vecmul")), tmp_path)
    one, four = (
        load_entry(p.stem, p) for p in (tmp_path / "1_split.snaxdfg", tmp_path / "4_bind.snaxdfg")
    )
    assert len(folder_recipe(one).steps) == 1 and len(folder_recipe(four).steps) == 4
    v = step_view(four, "4 bind", Platform.load(SNAX_FORGE), recipe=folder_recipe(four))
    (unit,) = v["bindings_section"]["entries"]
    assert (unit["label"], unit["params"][0]["name"], unit["params"][0]["editable"]) == (
        "acc · elementwise_mul", "W", False)  # fmt: skip
    assert v["symbols_section"][0]["param"] == "N" and not v["symbols_section"][0]["editable"]
    assert v["containers_section"][0]["line"] == "N · L1 · 16 KiB"


def test_iterations_need_bound_symbols():
    plain = view(FIXTURES / "vecadd.snaxdfg")
    assert plain["rows"][1]["node"]["text"] == "add_map · i · ? · untagged"
    assert plain["details"]["container:A"]["sizes"] is None
    assert plain["containers"]["A"]["badge"] is None and plain["estimate_error"]
    split = view(FIXTURES / "vecadd_split.snaxdfg")
    inner = split["rows"][1]["node"]["body"][0]
    assert (inner["id"], inner["loop_kind"], inner["text"]) == (
        "add_map_s",
        "spatial",
        "add_map_s · i_s · 4 · spatial",
    )


def test_a_transient_between_two_maps_and_a_container_shown_again(tmp_path):
    """C = (A + B) * B: tmp0 between the maps, B shown again above the second map."""
    path = graph_file(
        tmp_path, "two", {"A": False, "B": False, "C": False, "tmp0": True},
        [a_map("add_map", tasklet("add", {"in1": "A", "in2": "B"}, {"out": "tmp0"})),
         a_map("mult_map", tasklet("mult", {"in1": "tmp0", "in2": "B"}, {"out": "C"}, "out = in1 * in2"))],
    )  # fmt: skip
    v = view(path)
    assert [r.get("boxes") for r in v["rows"]] == [
        ["A@0", "B@0"], None, ["tmp0@1", "B@0~1"], None, ["C@1"]
    ]  # fmt: skip
    assert v["boxes"]["B@0~1"]["again"] and v["boxes"]["tmp0@1"]["written_by"] == ["add"]
    assert [(e["from"], e["to"]) for e in v["edges"]] == [
        ("A@0", "add.in1"), ("B@0", "add.in2"), ("add.out", "tmp0@1"),
        ("tmp0@1", "mult.in1"), ("B@0~1", "mult.in2"), ("mult.out", "C@1"),
    ]  # fmt: skip


def test_in_place_gets_a_new_version(tmp_path):
    """A read by the first map and written by the second: A@0 at the top, A@1 at the bottom."""
    path = graph_file(
        tmp_path, "inplace", {"A": False, "B": False, "tmp0": True},
        [a_map("add_map", tasklet("add", {"in1": "A", "in2": "B"}, {"out": "tmp0"})),
         a_map("copy_map", tasklet("copy", {"in1": "tmp0"}, {"out": "A"}, "out = in1"))],
    )  # fmt: skip
    v = view(path)
    assert [r.get("boxes") for r in v["rows"]] == [["A@0", "B@0"], None, ["tmp0@1"], None, ["A@1"]]
    assert ("copy.out", "A@1") in [(e["from"], e["to"]) for e in v["edges"]]


def test_inside_one_node_an_edge_goes_straight_from_the_writer(tmp_path):
    path = graph_file(
        tmp_path, "chain", {"A": False, "B": False, "tmp0": True},
        [a_map("m", tasklet("t1", {"in1": "A"}, {"out": "tmp0"}, "out = in1"),
               tasklet("t2", {"in1": "tmp0"}, {"out": "B"}, "out = in1"))],
    )  # fmt: skip
    v = view(path)
    assert [(e["from"], e["to"], e["data"]) for e in v["edges"]] == [
        ("A@0", "t1.in1", "A"), ("t1.out", "t2.in1", "tmp0"),
        ("t1.out", "tmp0@1", "tmp0"), ("t2.out", "B@1", "B"),
    ]  # fmt: skip
    assert v["rows"][-1]["boxes"] == ["tmp0@1", "B@1"]


def test_unused_containers_are_in_the_first_row(tmp_path):
    path = graph_file(
        tmp_path, "unused", {"A": False, "B": False, "Z": False},
        [a_map("m", tasklet("t", {"in1": "A"}, {"out": "B"}, "out = in1"))],
    )  # fmt: skip
    assert view(path)["rows"][0]["boxes"] == ["A@0", "Z@0"]


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
        assert json.loads(get(srv, "/api/step/0")[2])["rows"][0]["boxes"] == ["A@0", "B@0"]
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


def test_the_run_viewer_still_serves_its_own_page(tmp_path):
    from .helpers import run_dir
    from .helpers import serving as run_serving

    with run_serving([run_dir(tmp_path / "vecadd", "vecadd", "off")]) as srv:
        assert b"app.js" in get(srv, "/")[2]


def test_cli_rejects_a_missing_path(tmp_path, capsys):
    assert main([str(tmp_path / "nope.snaxdfg"), "--port", "0"]) == 2
    assert "no such file" in capsys.readouterr().err


def _link_and_group(v: dict) -> tuple[dict, list]:
    """The view's one link node and its accelerators."""
    (link,) = [r["node"] for r in v["rows"] if r["kind"] == "node" and r["node"]["kind"] == "link"]
    return link, v["accelerators"]


def test_dot4_draws_its_wire_and_boxes_its_group(tmp_path):
    """dot 4: the link node in the accelerated node's style, ``out_partial = wire``,
    what it replaced; mac, the link and sum in one group box of the accelerator dot."""
    v = recipe_view(tmp_path, "dot_4")
    link, accs = _link_and_group(v)
    assert link["text"] == "out_partial · wire"
    assert v["details"][f"node:{link['id']}"] == {"kind": "link", "container": "out_partial",
                                                  "depth": 0, "from": "mac", "to": "sum"}  # fmt: skip
    assert v["containers"]["out_partial"]["badge"] == "wire"
    # out_partial gets no box: the edges run through the link, straight between the connectors
    boxes = [b for r in v["rows"] if r["kind"] == "containers" for b in r["boxes"]]
    assert boxes == ["A@0", "B@0", "tmp0@0", "out@1"]  # tmp0, the fused nest's local value, stays
    assert [(e["from"], e["to"]) for e in v["edges"] if e["data"] == "out_partial"] == [
        ("mult.out", f"{link['id']}.in"), (f"{link['id']}.out", "sum_final.a")]  # fmt: skip
    assert v["details"]["container:out_partial"]["placement"] == "link"
    (acc,) = accs
    (grp,) = acc["groups"]
    assert (acc["name"], grp["name"], grp["links"]) == ("dot", "dot", [link["id"]])
    assert [u["instance"] for u in grp["units"]] == ["mac", "sum"]
    assert grp["tops"] == ["mult_map", link["id"], "sum_final"]
    assert v["details"]["node:mult"]["group"] == "dot"


def test_dot6_draws_a_buffer_with_its_depth(tmp_path):
    v = recipe_view(tmp_path, "dot_6")
    link, accs = _link_and_group(v)
    assert (link["text"], v["details"][f"node:{link['id']}"]["depth"]) == (
        "out_partial · FIFO 1",
        1,
    )
    assert [g["name"] for g in accs[0]["groups"]] == ["dot"]


# =============================================================================
# The cost in Compare and Details (C§12, C§20)
# =============================================================================


def _dot5_steps():
    from snax_forge.design import Platform
    from snax_forge.viz.dfg.api import Entry
    from tests.helpers import steps

    gs = steps("dot_5", W=8, N=2048)
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
    assert c["labels"] == ["5 bind", "6 link"] and "about 20 %" in c["caveat"]
    rows = c["estimate"]
    assert not any("→" in r["a"] + r["b"] for r in rows)
    assert _rows(rows, "accelerator") == {"cycles": ("266", "265", True),
                                          "unmapped cycles": ("0", "0", False),
                                          "total cycles": ("266", "265", True)}  # fmt: skip
    assert _rows(rows, "group mac")["energy"] == ("26.136 nJ", "–", True)
    assert _rows(rows, "group dot")["cycles"] == ("–", "265 = 256 + 9", True)
    assert _rows(rows, "Cost estimate (first-order) on nangate45") == {
        "area": ("64 400 µm²", "64 100 µm²", True), "energy per run": ("44.491 nJ", "44.346 nJ", True),
        "L1 accesses": ("mac: 4104 words, 18.038 nJ; sum: 9 words, 0.040 nJ",
                        "dot: 4097 words, 18.008 nJ", True),
        "not characterized": ("none", "none", False)}  # fmt: skip
    mult = c["details"]["node:mult"]
    assert mult["where"] == "" and _rows(mult["rows"], None)["group"] == ("mac", "dot", True)
    assert _rows(mult["rows"], "cost")["per fire"] == ("97.259 pJ", "97.259 pJ", False)
    assert _rows(mult["rows"], "cost")["per task"] == (
        "25.872 nJ (256 fires, 512 bits)", "25.868 nJ (256 fires, 512 bits)", True)  # fmt: skip
    assert c["details"]["group:mac"]["where"] == "removed"
    assert c["details"]["group:dot"]["where"] == "added"
    assert {r["a"] for r in c["details"]["group:dot"]["rows"]} == {"–"}
    # from the unbound import: no cost on the left
    first = compare_view(e[0], e[6], platform)["columns"]["estimate"]
    assert _rows(first, "Cost estimate (first-order) on nangate45")["area"] == (
        "–",
        "64 100 µm²",
        True,
    )
    assert step_view(e[0], "0", platform)["summary"]["cost"] is None  # nothing bound
    assert not any(r["under"] and r["under"].startswith("Cost")
                   for r in compare_view(e[0], e[1], platform)["columns"]["estimate"])  # fmt: skip


@pytest.mark.slow
def test_compare_gives_two_columns_for_the_same_group():
    """dot 5 at W = 8 against W = 4: one group, dot, on both sides; its rows side by side, the
    equal ones unmarked."""
    from snax_forge.viz.dfg.api import Entry, compare_view
    from tests.helpers import steps

    e, platform = _dot5_steps()
    w4 = Entry("w4", REPO, steps("dot_5", W=4, N=2048)[-1], None)
    rows = compare_view(e[6], w4, platform)["columns"]["estimate"]
    assert _rows(rows, "group dot") == {
        "cycles": ("265 = 256 + 9", "517 = 512 + 5", True), "expected COUNTER": ("268", "520", True),
        "channels": ("17", "9", True), "area": ("64 100 µm²", "33 300 µm²", True),
        "energy": ("26.339 nJ", "26.615 nJ", True)}  # fmt: skip
    assert _rows(rows, "memory")["reads"] == ("4096 words", "4096 words", False)


def test_value_text_is_the_pages():
    from snax_forge.viz.dfg.api import value_text

    assert [value_text(v) for v in (None, True, 3, [], ["a", 1], {"W": 8, "op": "add"})] == [
        "–", "true", "3", "–", "a, 1", '{"W":8,"op":"add"}']  # fmt: skip


def test_a_block_with_no_flops_holds_dash():
    """vecmul's ElementwiseLanes has no register: its Details' holding reads "–"."""
    from snax_forge.design import Platform
    from snax_forge.viz.dfg.api import Entry, step_view
    from tests.helpers import steps

    v = step_view(Entry("v", REPO, steps("vecmul", W=8)[-1], None), "v",
                  Platform.load(REPO / "platforms" / "snax_forge.json"))  # fmt: skip
    cost = v["details"]["node:mult"]["cost"]
    assert cost["holding"] == "–" and cost["per task"].endswith("(8 fires, 0 bits)")


# =============================================================================
# DRY1-2 R3: the Bindings tree, the params' names in the boxes and Details
# =============================================================================


def test_the_bindings_tree():
    """D156: dot 1 accelerator ▸ group ▸ unit (dot ▸ mul ▸ mul's entry); dot 2's linked group
    not encapsulated at the top; vecmul's unit alone at the top; a split not bound yet at the
    top beside a bound unit."""
    from dataclasses import replace

    from snax_forge.sandbox import Recipe
    from snax_forge.viz.dfg.api import bindings_section
    from tests.helpers import steps

    def tree(name, k=None):
        r = Recipe.load(REPO / "recipes" / f"{name}.json")
        k = len(r.steps) if k is None else k  # graph k is made by steps 1..k
        b = bindings_section(steps(name)[k], replace(r, steps=r.steps[:k]))
        return b["tree"], [e["label"] for e in b["entries"]]

    assert tree("dot_1") == ([{"kind": "accelerator", "name": "dot", "children": [
        {"kind": "group", "name": "mul", "children": [{"kind": "entry", "index": 0}]},
        {"kind": "group", "name": "sum", "children": [{"kind": "entry", "index": 1}]}]}],
        ["mul · elementwise_mul", "sum · accumulate"])  # fmt: skip
    assert tree("dot_2")[0] == [{"kind": "group", "name": "dot", "children": [
        {"kind": "entry", "index": 0}, {"kind": "entry", "index": 1}]}]  # fmt: skip
    assert tree("vecmul")[0] == [{"kind": "entry", "index": 0}]
    # dot 1 at its step 3: mul bound, sum_map split, nothing encapsulated: both at the top
    t, labels = tree("dot_1", 3)
    assert t == [{"kind": "entry", "index": 0}, {"kind": "entry", "index": 1}]
    assert labels == ["mul · elementwise_mul", "sum_map split_map"]


def test_the_recipes_param_names_in_the_boxes_and_details():
    """D156: with the recipe, a unit's box gives its split's param (``W_mul 8``, not ``W 8``)
    and a split map's Details its range with the param and its iterations (``0:N // W_mul =
    256``); dot 2's shared W in both boxes; without a recipe the boxes keep ``W``."""
    from snax_forge.design import Platform
    from snax_forge.sandbox import Recipe, apply_recipe
    from snax_forge.viz.dfg.api import Entry, map_ranges, step_view, unit_params
    from tests.design.helpers import SNAX_FORGE
    from tests.helpers import dot1_split, imported

    r = dot1_split(8, 8)
    g = apply_recipe(r, imported("dot"))[-1].graph
    assert unit_params(g, r) == {"mul": ["W_mul"], "sum": ["W_sum"]}
    assert map_ranges(g, r) == {"mult_map": "0:N // W_mul = 256", "sum_map": "0:N // W_sum = 256"}
    v = step_view(Entry("5", Path("5"), g, None), "5", Platform.load(SNAX_FORGE), recipe=r)
    texts = json.dumps(v["rows"], ensure_ascii=False)
    assert "mul · a * b · W_mul 8" in texts and "sum · fold + · W_sum 8" in texts
    assert v["details"]["node:mult_map"]["range"] == "0:N // W_mul = 256"
    tail = dot1_split(10, 10)
    gt = apply_recipe(tail, imported("dot"))[-1].graph
    assert map_ranges(gt, tail)["mult_map"] == "0:(N + W_mul - 1) // W_mul = 205"
    plain = step_view(Entry("5", Path("5"), g, None), "5", Platform.load(SNAX_FORGE))
    assert "mul · a * b · W 8" in json.dumps(plain["rows"], ensure_ascii=False)
    two = Recipe.load(REPO / "recipes" / "dot_2.json").with_params({"N": 2048, "W": 8})
    g2 = apply_recipe(two, imported("dot"))[-1].graph
    assert unit_params(g2, two) == {"mul": ["W"], "sum": ["W"]}
