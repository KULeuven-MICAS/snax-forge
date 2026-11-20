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
    assert "drawing" not in graph_view(gs.get("bad"))
    bad.write_text((FIXTURES / "vecadd.snaxdfg").read_text())
    gs.reload()
    assert gs.get("bad").error is None and graph_view(gs.get("bad"))["drawing"]["nodes"]


# =============================================================================
# The view of a graph
# =============================================================================


def test_vecadd_accelerated():
    """The boxes' texts, and everything else in Details: the unit's BRM, params, timing and
    what it replaced; each edge's memlet and packing (int64: one per word, num_ch = W)."""
    v = view(FIXTURES / "vecadd_accelerated.snaxdfg")
    assert v["graph"] == "vecadd" and v["symbols"] == {"N": 64}
    assert (containers(v), containers(v, first=True)) == (["A@0", "B@0", "C@1"], ["A@0", "B@0"])
    b = boxes(v)
    assert b["A@0"] == {"id": "A@0", "kind": "container", "sel": "A", "name": "A", "text": "A[N]",
                        "dtype": "int64", "badge": "L1 · 512 B", "transient": False,
                        "scalar": False, "layer": "first"}  # fmt: skip
    loop, unit = b["add_map"], b["unit:acc"]
    assert (loop["text"], loop["sub"], loop["loop"]) == (
        "add_map · temporal",
        "n_1_t · 16",
        "temporal",
    )
    assert (unit["text"], unit["sel"], unit["ins"], unit["outs"]) == (
        "acc · a + b · W 4", "add", ["a", "b"], ["out"])  # fmt: skip
    d = v["details"]["node:add"]
    assert (d["brm"], d["implementation"], d["params"], d["replaced"], d["timing"]) == (
        "simd", "chisel_tiled_spatial", {"W": 4, "op": "add", "streams": 2}, "add_map_s › add",
        {"latency": 0, "ii": 1, "span": 1})  # fmt: skip
    assert v["details"]["node:add_map"] == {"kind": "map", "var": "n_1_t", "range": "0:N // 4",
                                           "loop.kind": "temporal",
                                           "loop.split": {"var": "n", "range": "0:N"}}  # fmt: skip
    assert v["details"]["container:A"]["sizes"] == [64]
    assert ends(v) == [("A@0", "add_map"), ("add_map", "unit:acc.a"), ("B@0", "add_map"),
                       ("add_map", "unit:acc.b"), ("unit:acc.out", "C@1")]  # fmt: skip
    e = v["details"]["edge:add_map->unit:acc.a"]
    assert (e["memlet"], e["packing"], e["loops"], e["beats"], e["streamer"]) == (
        "A[4 * n_1_t:4 * n_1_t + 4]", "int64 · num_slc 1 · num_ch 4", [["n_1_t", 16]], 16, True)  # fmt: skip


def recipe_view(tmp_path: Path, recipe: str, params: dict | None = None) -> dict:
    """The view of ``recipes/<folder>/<recipe>.json``'s last (bound) graph, with ``params`` set
    (``--set``)."""
    from snax_forge.dfg.import_sdfg import import_kernel
    from snax_forge.sandbox import Recipe, apply_recipe

    r = Recipe.load(recipe_path(recipe)).with_params(params or {})
    path = tmp_path / f"{recipe}.snaxdfg"
    apply_recipe(r, import_kernel(r.kernel, r.dtypes, r.types))[-1].graph.save(path)
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
    """Each accelerated port's edge: dtype, num_slc = 64 / b, num_ch = W / num_slc."""
    v = recipe_view(tmp_path, recipe)
    edges = [v["details"][f"edge:{e['id']}"] for e in v["drawing"]["edges"]]
    assert {e["data"]: e["packing"] for e in edges if e.get("packing")} == want


def test_a_packed_output_draws_its_packer(tmp_path):
    """vsum at i8, W = 8: a FifoPacker box between the sum's output and out (D174); at int64
    the output is wired, no packer."""
    v = recipe_view(tmp_path, "vsum_tree", {"types": "i8", "W": 8})
    pk = boxes(v)["packer:acc.out"]
    assert (pk["kind"], pk["text"]) == ("packer", "FifoPacker · d_out 2")
    assert ends(v, "out") == [
        ("unit:acc.out", "packer:acc.out.in"),
        ("packer:acc.out.out", "out@1"),
    ]
    d = v["details"]["node:packer:acc.out"]
    assert (d["kind"], d["unit"], d["port"], d["lanes"], d["d_out"], d["bits"]) == (
        "packer", "acc", "out", 1, 2, 33)  # fmt: skip
    plain = recipe_view(tmp_path, "vsum_tree")
    assert not [b for b in boxes(plain).values() if b["kind"] == "packer"]


def test_a_sliced_input_draws_its_slicer(tmp_path):
    """vecadd_i32 at W = 1: a FifoSlicer box between A's loop and the unit's a, and one for B
    (D180), the edge into each the port's packing; the packer below."""
    v = recipe_view(tmp_path, "vecadd_i32", {"W": 1})
    assert {k: b["kind"] for k, b in boxes(v).items() if b["kind"] in ("slicer", "packer")} == {
        "slicer:acc.a": "slicer", "slicer:acc.b": "slicer", "packer:acc.out": "packer"}  # fmt: skip
    assert boxes(v)["slicer:acc.a"]["text"] == "FifoSlicer · d_in 2"
    assert ends(v, "A") == [("A@0", "add_map"), ("add_map", "slicer:acc.a.in")]
    assert ("slicer:acc.a.out", "unit:acc.a") in ends(v)  # the wire into the port
    edge = v["details"]["edge:add_map->slicer:acc.a.in"]
    assert edge["packing"] == "int32 · num_slc 2 · d_in 2 · num_ch 1"
    d = v["details"]["node:slicer:acc.a"]
    assert (d["kind"], d["unit"], d["port"], d["lanes"], d["d_in"], d["bits"]) == (
        "slicer", "acc", "a", 1, 2, 66
    )  # fmt: skip
    assert "placed by the flow" in d["what"]
    plain = recipe_view(tmp_path, "vecadd_i32")
    assert not [b for b in boxes(plain).values() if b["kind"] in ("slicer", "packer")]


def test_a_packed_output_edge_shows_its_d_out(tmp_path):
    """vsum at i8, W = 8: the int32 sum packed 2 to a word on one channel (D174)."""
    v = recipe_view(tmp_path, "vsum_tree", {"types": "i8", "W": 8})
    edges = [v["details"][f"edge:{e['id']}"] for e in v["drawing"]["edges"]]
    assert {e["data"]: e["packing"] for e in edges if e.get("packing")} == {
        "A": "int8 · num_slc 8 · num_ch 1", "out": "int32 · num_slc 2 · d_out 2 · num_ch 1"
    }  # fmt: skip


def test_a_folding_write_names_its_op():
    """dot's sum map writes out[0] with reduce add: the edge's memlet says so, and the tasklet's
    text is the fold."""
    v = view(FIXTURES / "dot.snaxdfg")
    assert writes(v) == [("mult.out", "tmp0[n]"), ("sum.out", "out[0] (reduce add)")]
    assert [boxes(v)[x]["text"] for x in ("mult", "sum")] == ["mult · in1 * in2", "sum · fold +"]


def test_the_bound_dot_shows_both_accelerators():
    """dot after its recipe: mul and sum, each over N // 4 beats; tmp0 between them."""
    v = view(FIXTURES / "dot_accelerated.snaxdfg")
    b = boxes(v)
    assert [(b[x]["text"], b[x]["sub"]) for x in ("mult_map", "sum_map")] == [
        ("mult_map · temporal", "n_1_t · 16"), ("sum_map · temporal", "n_1_t · 16")]  # fmt: skip
    assert [b[x]["text"] for x in ("unit:mul", "unit:sum")] == [
        "mul · a * b · W 4",
        "sum · fold + · W 4",
    ]
    assert writes(v) == [
        ("unit:mul.out", "tmp0[4 * n_1_t:4 * n_1_t + 4]"),
        ("unit:sum.out", "out[0:1]"),
    ]


def test_dot_1_enc_draws_one_accelerator_of_two_groups(tmp_path):
    """dot_1_enc: the accelerator dot, its groups vmul and vsum in the graph's order, each unit
    inside the accelerator's outline (a group of one unit and no link has none); each unit's
    Details name its group and accelerator. dot 1 has two accelerators and no outline."""
    from tests.gen.helpers import bound_dot1, bound_dot1_enc

    path = tmp_path / "dot_1_enc.snaxdfg"
    path.write_text(bound_dot1_enc().to_json())
    v = view(path)
    assert [(o["id"], o["text"], o["sel"]) for o in v["drawing"]["outlines"]] == [
        ("acc:dot", "accelerator dot", "dot")]  # fmt: skip
    assert [boxes(v)[u]["parent"] for u in ("unit:vmul", "unit:vsum")] == ["acc:dot", "acc:dot"]
    units = [v["details"][f"node:{n}"] for n in ("mult", "sum")]
    assert [(u["accelerator"], u["group"]) for u in units] == [("dot", "vmul"), ("dot", "vsum")]
    assert v["details"]["group:vmul"]["accelerator"] == "dot"
    assert v["details"]["accelerator:dot"]["groups"] == ["vmul", "vsum"]
    plain = tmp_path / "dot_1.snaxdfg"
    plain.write_text(bound_dot1().to_json())
    p = view(plain)
    assert p["drawing"]["outlines"] == []
    assert [k for k in p["details"] if k.startswith("accelerator:")] == [
        "accelerator:vmul",
        "accelerator:vsum",
    ]


def test_without_encapsulate_each_group_is_its_own_accelerator():
    v = view(FIXTURES / "dot_accelerated.snaxdfg")
    assert [(k, d["groups"]) for k, d in v["details"].items() if k.startswith("accelerator:")] == [
        ("accelerator:mul", ["mul"]), ("accelerator:sum", ["sum"])]  # fmt: skip
    units = [v["details"][f"node:{n}"] for n in ("mult", "sum")]
    assert [(u["accelerator"], u["group"]) for u in units] == [("mul", "mul"), ("sum", "sum")]


@pytest.mark.slow
def test_a_loop_box_names_its_map_kind_variable_and_iterations():
    """Vecmul as imported at N = 2048: its map's box ``mult_map · untagged`` over ``n · 2048``."""
    from snax_forge.viz.dfg.api import Entry
    from tests.helpers import steps

    g = steps("vecmul", N=2048)[0]
    v = graph_view(Entry("0_input", Path("0_input.snaxdfg"), g, None))
    b = boxes(v)["mult_map"]
    assert (b["text"], b["sub"]) == ("mult_map · untagged", "n · 2048")


@pytest.mark.slow
def test_the_containers_section_of_vecmul_and_matvec():
    """the Containers, in the graph's order: vecmul's A, B, C each ``N``, 2048 int64, 16 KiB
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
    """the cost table, a row per unit, buffer link and control unit, a row per port's
    streamer and the accelerator's words through the crossbar and the banks, then the
    total, one table per accelerator: vecmul at W = 8 (N = 2048) its unit 55 300 µm², 23.137 nJ,
    its control unit 362 µm² (flops and an equal comb, 34 bits: no copy of JOBS), 0.047 nJ,
    three streamers of 8 channels,
    its 4096 reads and 2048 writes 38.251 nJ; dot_1_enc at W = 8 one table for its two groups,
    tmp0's round trip in its L1 row; Compare's cells before → after."""
    from snax_forge.design import Platform
    from snax_forge.estimate import estimate
    from snax_forge.viz.dfg.api import Entry, compare_view, cost_parts, cost_tables, step_view
    from tests.design.helpers import SNAX_FORGE
    from tests.helpers import steps

    platform = Platform.load(SNAX_FORGE)
    g = steps("vecmul", N=2048, W=8)[-1]
    (t,) = cost_tables(estimate(g, platform), g)
    strm = "streamer · FIFO 8, 1 loop, 8 ch"
    assert t["accelerator"] == "acc" and _cost_rows(t) == [
        ("acc", "SimdLanes", "55 300", "23.137", "29.4%"),
        ("acc ctrl", "control unit", "360", "0.047", "0.1%"),
        ("acc_a", strm, "31 700", "5.760", "7.3%"),
        ("acc_b", strm, "31 700", "5.760", "7.3%"),
        ("acc_out", strm, "31 700", "5.760", "7.3%"),
        (L1_LABEL, "4 096 reads, 2 048 writes", "–", "38.251", "48.6%"),
        ("total", "", "151 000", "78.714", "100.0%")]  # fmt: skip
    parts = [(p["name"], p["area_text"], p["area_share_text"], p["energy_text"],
              p["energy_share_text"]) for p in cost_parts(estimate(g, platform))]  # fmt: skip
    # the crossbar's 16 read-only and 8 write-only inputs priced as such
    assert parts == [("datapath", "55 300 µm²", "8.3%", "23.137 nJ", "29.4%"),
                     ("control", "360 µm²", "0.1%", "0.047 nJ", "0.1%"),
                     ("streamers", "95 200 µm²", "14.2%", "17.279 nJ", "22.0%"),
                     ("crossbar", "84 600 µm²", "12.7%", "10.513 nJ", "13.4%"),
                     ("memory", "433 000 µm²", "64.8%", "27.738 nJ", "35.2%")]  # fmt: skip
    d = steps("dot_1_enc", N=2048, W=8)[-1]
    (t,) = cost_tables(estimate(d, platform), d)
    assert t["accelerator"] == "dot" and _cost_rows(t) == [
        ("vmul", "SimdLanes", "55 300", "23.137", "23.9%"),
        ("vmul ctrl", "control unit", "360", "0.095", "0.1%"),
        ("vsum", "TreeAccumulate", "2 550", "1.033", "1.1%"),
        ("vsum ctrl", "control unit", "710", "0.187", "0.2%"),
        ("vmul_a", strm, "31 700", "5.760", "6.0%"),
        ("vmul_b", strm, "31 700", "5.760", "6.0%"),
        ("vmul_out", strm, "31 700", "5.760", "6.0%"),
        ("vsum_a", strm, "31 700", "5.760", "6.0%"),
        ("vsum_out", "streamer · FIFO 8, 1 loop, 1 ch", "4 560", "0.003", "0.0%"),
        (L1_LABEL, "6 144 reads, 2 049 writes", "–", "49.272", "50.9%"),
        ("total", "", "190 000", "96.766", "100.0%")]  # fmt: skip
    v = step_view(Entry("2_bind", Path("2_bind.snaxdfg"), g, None), "2 bind", platform)
    assert v["summary"]["cost"]["tables"][0]["rows"][0]["area_text"] == "55 300"
    w16 = steps("vecmul", N=2048, W=16)[-1]
    c = compare_view(Entry("w8", Path("w8"), g, None), Entry("w16", Path("w16"), w16, None),
                     platform)["columns"]["cost_tables"]  # fmt: skip
    (unit, ctrl, *strm, l1, total) = [*c[0]["rows"], c[0]["total"]]
    assert unit["block"] == "acc" and unit["area_text"].startswith("55 300 → ") and unit["differs"]
    assert ctrl["datapath"] == "control unit" and total["block"] == "total"
    assert [x["block"] for x in strm] == ["acc_a", "acc_b", "acc_out"]
    assert all(x["datapath"].endswith("8 ch → streamer · FIFO 8, 1 loop, 16 ch") for x in strm)
    assert l1["block"] == L1_LABEL and "→" not in l1["datapath"]  # the same words


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
    """The unmapped estimate in the page: vecmul at N = 2048 as imported, all 2048 cycles unmapped (the assumption
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
        == ("2048 iterations no unit runs, at 1 cycle each: an assumption (a core takes several)")
    )
    s2 = step_view(b, "2 bind", platform)["summary"]
    assert [c["value"] for c in s2["cycles"]] == [256, 0, 256] and s2["unmapped_note"] is None
    rows = compare_view(a, b, platform)["columns"]["estimate"]
    got = {r["label"]: (r["a"], r["b"]) for r in rows if r["under"] == "accelerator"}
    assert got == {
        "cycles": ("0", "256"),
        "unmapped cycles": ("2048", "0"),
        "total cycles": ("2048", "256"),
        "streamers": (
            "0 streamer sets",
            "1 streamer set · 22 CSRs",
        ),
    }  # fmt: skip (one unmapped count)


@pytest.mark.slow
def test_the_symbols_section():
    """the Symbols: each kernel symbol with its value and the containers using it; an input
    on the param of its name with a session's recipe in edit mode, read only otherwise."""
    from snax_forge.sandbox import Recipe
    from snax_forge.viz.dfg.api import symbols_section
    from tests.helpers import steps

    g = steps("vecmul", N=2048, W=8)[-1]
    r = Recipe.load(recipe_path("vecmul")).with_params({"N": 2048, "W": 8})
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
        Step("bind", {"node": "mult", "brm": "simd",
                      "implementation": "chisel_tiled_spatial", "instance": "acc"})])  # fmt: skip


def _entries(b):
    return [(e["kind"], e["label"], [(p["name"], p["value"], p["editable"], p["shared_with"])
                                     for p in e["params"]], e["note"]) for e in b["entries"]]  # fmt: skip


@pytest.mark.slow
def test_the_bindings_of_vecmul_dot_1_and_dot_4():
    """the Bindings: vecmul's split before its bind (``mult_map split · W = 8 · not bound``),
    then the unit (``acc · simd · W = 8``); dot 1 with W_mul and W_sum, two entries,
    nothing shared; dot 4 at W = 4, one W shared by mac and sum; a param in none is unused."""
    from dataclasses import replace

    from snax_forge.sandbox import Recipe, apply_recipe
    from snax_forge.sandbox.recipe import Step
    from snax_forge.viz.dfg.api import bindings_section
    from tests.helpers import imported, steps

    r = _editor_vecmul()
    gs = [x.graph for x in apply_recipe(r, imported("vecmul"))]
    one = bindings_section(gs[1], replace(r, steps=r.steps[:1]), editable=True)
    assert _entries(one) == [("split", "mult_map split", [("W", 8, True, [])], "not bound")]
    four = bindings_section(gs[4], r, editable=True)
    assert _entries(four) == [("unit", "acc · simd", [("W", 8, True, [])], None)]
    assert (
        four["unused"] == []
        and not bindings_section(gs[4], r)["entries"][0]["params"][0]["editable"]
    )
    d = Recipe.load(recipe_path("dot_1"))
    factor = {"mult_map": "W_mul", "sum_map": "W_sum"}
    d = Recipe(d.name, d.kernel, {"N": 2048, "W_mul": 8, "W_sum": 4}, d.symbols,
               [Step(x.transform, {**x.params, "factor": factor[x.params["map"]]})
                if "factor" in x.params else x for x in d.steps])  # fmt: skip
    dot1 = bindings_section(apply_recipe(d, imported("dot"))[-1].graph, d, editable=True)
    assert _entries(dot1) == [("unit", "vmul · simd", [("W_mul", 8, True, [])], None),
                              ("unit", "vsum · accumulate", [("W_sum", 4, True, [])], None)]  # fmt: skip
    d4 = Recipe.load(recipe_path("dot_4"))
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
        "acc · simd", "W", False)  # fmt: skip
    assert v["symbols_section"][0]["param"] == "N" and not v["symbols_section"][0]["editable"]
    assert v["containers_section"][0]["line"] == "N · L1 · 16 KiB"


def test_iterations_need_bound_symbols():
    plain = view(FIXTURES / "vecadd.snaxdfg")
    b = boxes(plain)
    assert (b["add_map"]["text"], b["add_map"]["sub"]) == ("add_map · untagged", "n · ?")
    assert plain["details"]["container:A"]["sizes"] is None
    assert b["A@0"]["badge"] is None and plain["estimate_error"]
    inner = boxes(view(FIXTURES / "vecadd_split.snaxdfg"))["add_map_s"]
    assert (inner["loop"], inner["text"], inner["sub"]) == (
        "spatial",
        "add_map_s · spatial",
        "n_2_s · 4",
    )


def test_a_transient_between_two_maps_and_an_input_read_twice(tmp_path):
    """C = (A + B) * B: tmp0 between the maps, B one box that both maps read."""
    path = graph_file(
        tmp_path, "two", {"A": False, "B": False, "C": False, "tmp0": True},
        [a_map("add_map", tasklet("add", {"in1": "A", "in2": "B"}, {"out": "tmp0"})),
         a_map("mult_map", tasklet("mult", {"in1": "tmp0", "in2": "B"}, {"out": "C"}, "out = in1 * in2"))],
    )  # fmt: skip
    v = view(path)
    assert containers(v) == ["A@0", "B@0", "tmp0@1", "C@1"]
    assert ends(v) == [
        ("A@0", "add_map"), ("add_map", "add.in1"), ("B@0", "add_map"), ("add_map", "add.in2"),
        ("add.out", "tmp0@1"), ("tmp0@1", "mult_map"), ("mult_map", "mult.in1"),
        ("B@0", "mult_map"), ("mult_map", "mult.in2"), ("mult.out", "C@1"),
    ]  # fmt: skip


def test_in_place_gets_a_new_version(tmp_path):
    """A read by the first map and written by the second: A@0 an input, A@1 a box of its own."""
    path = graph_file(
        tmp_path, "inplace", {"A": False, "B": False, "tmp0": True},
        [a_map("add_map", tasklet("add", {"in1": "A", "in2": "B"}, {"out": "tmp0"})),
         a_map("copy_map", tasklet("copy", {"in1": "tmp0"}, {"out": "A"}, "out = in1"))],
    )  # fmt: skip
    v = view(path)
    assert containers(v) == ["A@0", "B@0", "tmp0@1", "A@1"]
    assert ("copy.out", "A@1") in ends(v)


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


def test_unused_containers_are_in_the_first_row(tmp_path):
    path = graph_file(
        tmp_path, "unused", {"A": False, "B": False, "Z": False},
        [a_map("m", tasklet("t", {"in1": "A"}, {"out": "B"}, "out = in1"))],
    )  # fmt: skip
    assert containers(view(path), first=True) == ["A@0", "Z@0"]


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


def test_the_run_viewer_still_serves_its_own_page(tmp_path):
    from .helpers import run_dir
    from .helpers import serving as run_serving

    with run_serving([run_dir(tmp_path / "vecadd", "vecadd", "off")]) as srv:
        assert b"app.js" in get(srv, "/")[2]


def test_cli_rejects_a_missing_path(tmp_path, capsys):
    assert main([str(tmp_path / "nope.snaxdfg"), "--port", "0"]) == 2
    assert "no such file" in capsys.readouterr().err


def _link_and_group(v: dict) -> tuple[dict, list]:
    """The drawing's one link box and its outlines."""
    (link,) = [b for b in v["drawing"]["nodes"] if b["kind"] == "link"]
    return link, v["drawing"]["outlines"]


def test_dot4_draws_its_wire_and_boxes_its_group(tmp_path):
    """dot 4: the link box ``out_partial · wire`` keyed by its ports, mac, the link and sum in
    one group outline; out_partial and the fused nest's tmp0 get no box, the edges run through
    the link."""
    v = recipe_view(tmp_path, "dot_4")
    link, outlines = _link_and_group(v)
    assert (link["id"], link["text"], link["sel"]) == (
        "link:mac.out>sum.a",
        "out_partial · wire",
        "out_partial",
    )
    assert v["details"]["node:out_partial"] == {"kind": "link", "container": "out_partial",
                                                "depth": 0, "from": "mac", "to": "sum"}  # fmt: skip
    assert containers(v) == ["A@0", "B@0", "out@1"]
    assert ends(v, "out_partial") == [
        ("unit:mac.out", f"{link['id']}.in"),
        (f"{link['id']}.out", "unit:sum.a"),
    ]
    assert v["details"]["container:out_partial"]["placement"] == "link"
    assert [(o["id"], o["text"]) for o in outlines] == [("grp:dot", "group dot")]
    assert [boxes(v)[x]["parent"] for x in ("unit:mac", link["id"], "unit:sum")] == ["grp:dot"] * 3
    grp = v["details"]["group:dot"]
    assert (grp["accelerator"], grp["units"], grp["links"]) == (
        "dot",
        ["mac", "sum"],
        ["out_partial"],
    )
    assert v["details"]["node:mult"]["group"] == "dot"


def test_a_fused_nests_local_value_is_an_edge_inside_the_map_not_a_box():
    """dot 4 after fuse_map: tmp0 (``loop.local``) gets no box; mult's output joins sum's input
    straight, the edge local and labelled tmp0; the Containers line and Details give no bytes
    and the fused map; dot 1's tmp0 in L1 keeps its box."""
    from snax_forge.design import Platform
    from snax_forge.estimate import estimate
    from snax_forge.sandbox import Recipe
    from snax_forge.viz.dfg.api import Entry, step_view
    from tests.design.helpers import SNAX_FORGE
    from tests.helpers import steps

    platform = Platform.load(SNAX_FORGE)
    names = [s.transform for s in Recipe.load(recipe_path("dot_4")).steps]
    k = names.index("fuse_map") + 1
    g = steps("dot_4")[k]
    v = step_view(Entry("3_fuse_map", Path("3_fuse_map.snaxdfg"), g, None), "3 fuse_map", platform)
    assert not any(c.startswith("tmp0@") for c in containers(v))
    (edge,) = [e for e in v["drawing"]["edges"] if e.get("data") == "tmp0"]
    assert (ends(v, "tmp0"), edge["style"], edge["label"]) == (
        [("mult.out", "sum.in1")],
        "local",
        "tmp0",
    )
    line = {c["name"]: c for c in v["containers_section"]}["tmp0"]
    assert (line["line"], line["bytes"]) == ("N · local to mult_map", None)
    d = v["details"]["container:tmp0"]
    assert (d["placement"], d["bytes"], d["writers"], d["readers"]) == (
        "local to mult_map, no memory", None, ["mult"], ["sum"])  # fmt: skip
    assert estimate(g, platform).containers["tmp0"].placement == "local"  # the estimate's own
    one = steps("dot_1")[-1]
    w = step_view(Entry("8_bind", Path("8_bind.snaxdfg"), one, None), "8 bind", platform)
    assert "tmp0@1" in containers(w)
    assert not [e for e in w["drawing"]["edges"] if e["style"] == "local"]


def test_dot6_draws_a_buffer_with_its_depth(tmp_path):
    v = recipe_view(tmp_path, "dot_6")
    link, outlines = _link_and_group(v)
    assert (link["text"], v["details"][f"node:{link['sel']}"]["depth"]) == (
        "out_partial · FIFO 1",
        1,
    )
    assert [o["id"] for o in outlines] == ["grp:dot"]


# =============================================================================
# The cost in Compare and Details
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
    assert _rows(rows, "group mac")["energy"] == ("26.030 nJ", "–", True)
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
        "energy per run": ("59.811 nJ", "59.656 nJ", True),
        "datapath": ("61 000 µm² · 25.951 nJ", "61 000 µm² · 25.951 nJ", False),
        "control": ("1 110 µm² · 0.152 nJ", "1 090 µm² · 0.149 nJ", True),
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
        "cycles": ("265 = 256 + 9", "517 = 512 + 5", True), "expected BUSY_COUNTER": ("268", "520", True),
        "channels": ("17", "9", True), "area": ("62 100 µm²", "31 900 µm²", True),
        "energy": ("26.099 nJ", "26.104 nJ", True)}  # fmt: skip
    assert _rows(rows, "memory")["reads"] == ("4096 words", "4096 words", False)


def test_value_text_is_the_pages():
    from snax_forge.viz.dfg.api import value_text

    assert [value_text(v) for v in (None, True, 3, [], ["a", 1], {"W": 8, "op": "add"})] == [
        "–", "true", "3", "–", "a, 1", '{"W":8,"op":"add"}']  # fmt: skip


def test_a_block_with_no_flops_has_no_flop_area():
    """vecmul's SimdLanes has no register: its Details' area has 0 µm² of flops."""
    from snax_forge.design import Platform
    from snax_forge.viz.dfg.api import Entry, step_view
    from tests.helpers import steps

    v = step_view(Entry("v", REPO, steps("vecmul", W=8)[-1], None), "v",
                  Platform.load(REPO / "platforms" / "snax_forge.json"))  # fmt: skip
    cost = v["details"]["node:mult"]["cost"]
    assert "flops 0)" in cost["area"] and cost["per task"].endswith("(8 fires, 0 bits)")
    assert "holding" not in cost


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

    assert tree("dot_1_enc") == ([{"kind": "accelerator", "name": "dot", "children": [
        {"kind": "group", "name": "vmul", "children": [{"kind": "entry", "index": 0}]},
        {"kind": "group", "name": "vsum", "children": [{"kind": "entry", "index": 1}]}]}],
        ["vmul · simd", "vsum · accumulate"])  # fmt: skip
    # dot 1, two accelerators of one group of one unit each: both at the top
    assert tree("dot_1") == ([{"kind": "entry", "index": 0}, {"kind": "entry", "index": 1}],
                             ["vmul · simd", "vsum · accumulate"])  # fmt: skip
    assert tree("dot_2")[0] == [{"kind": "group", "name": "dot", "children": [
        {"kind": "entry", "index": 0}, {"kind": "entry", "index": 1}]}]  # fmt: skip
    assert tree("vecmul")[0] == [{"kind": "entry", "index": 0}]
    # dot 1 at its step 3: vmul bound, sum_map split, nothing encapsulated: both at the top
    t, labels = tree("dot_1", 3)
    assert t == [{"kind": "entry", "index": 0}, {"kind": "entry", "index": 1}]
    assert labels == ["vmul · simd", "sum_map split_map"]


def test_the_recipes_param_names_in_the_boxes_and_details():
    """With the recipe, a unit's box gives its split's param (``W_mul 8``, not ``W 8``)
    and a split map's Details its range with the param and its iterations (``0:N // W_mul =
    256``); dot 2's shared W in both boxes; without a recipe the boxes keep ``W``."""
    from snax_forge.design import Platform
    from snax_forge.sandbox import Recipe, apply_recipe
    from snax_forge.viz.dfg.api import Entry, map_ranges, step_view, unit_params
    from tests.design.helpers import SNAX_FORGE
    from tests.helpers import dot1_split, imported

    r = dot1_split(8, 8)
    g = apply_recipe(r, imported("dot"))[-1].graph
    assert unit_params(g, r) == {"vmul": ["W_mul"], "vsum": ["W_sum"]}
    assert map_ranges(g, r) == {"mult_map": "0:N // W_mul = 256", "sum_map": "0:N // W_sum = 256"}
    v = step_view(Entry("5", Path("5"), g, None), "5", Platform.load(SNAX_FORGE), recipe=r)
    texts = json.dumps(v["drawing"], ensure_ascii=False)
    assert "vmul · a * b · W_mul 8" in texts and "vsum · fold + · W_sum 8" in texts
    assert v["details"]["node:mult_map"]["range"] == "0:N // W_mul = 256"
    tail = dot1_split(10, 10)
    gt = apply_recipe(tail, imported("dot"))[-1].graph
    assert map_ranges(gt, tail)["mult_map"] == "0:(N + W_mul - 1) // W_mul = 205"
    plain = step_view(Entry("5", Path("5"), g, None), "5", Platform.load(SNAX_FORGE))
    assert "vmul · a * b · W 8" in json.dumps(plain["drawing"], ensure_ascii=False)
    two = Recipe.load(recipe_path("dot_2")).with_params({"N": 2048, "W": 8})
    g2 = apply_recipe(two, imported("dot"))[-1].graph
    assert unit_params(g2, two) == {"mul": ["W"], "sum": ["W"]}


def test_a_unit_with_a_slot_shows_it():
    """relu's unit: its slot k beside its lanes, in the drawing and in the Bindings, read only;
    streams is no lanes param, so not shown."""
    from snax_forge.dfg import Graph
    from snax_forge.sandbox import Recipe, apply_recipe
    from snax_forge.viz.dfg.api import bindings_section, compact_text
    from tests.helpers import FIXTURES, recipe_path

    r = Recipe.load(recipe_path("relu"))
    g = apply_recipe(r, Graph.load(FIXTURES / "relu.snaxdfg"))[-1].graph
    assert compact_text(g, g.node("numpy_maximum")) == "acc · max(a, k) · W 4 · k 0"
    for recipe in (r, None):
        (e,) = bindings_section(g, recipe)["entries"]
        assert e["label"] == "acc · simd"
        assert [(p["name"], p["value"], p["editable"]) for p in e["params"]] == [
            ("W", 4, False), ("k", 0, False)]  # fmt: skip


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


def _node_in(view: dict, node_id: str) -> dict:
    return box_of(view, node_id)


def test_a_scalar_is_a_parameter_hexagon_its_edge_broadcast():
    """alpha before any bake: listed ``scalar · int32``, its box marked scalar (the hexagon), its
    edge straight into mult's port (no loop: alpha[0] uses none) labelled broadcast."""
    r, graphs = _axpy("axpy")
    (v,) = _views(r, graphs[:1])
    line = {c["name"]: c["line"] for c in v["containers_section"]}
    assert line["alpha"] == "scalar · int32" and line["x"] == "N · L1 · 256 B"
    b = boxes(v)
    assert b["alpha@0"]["scalar"] and not b["x@0"]["scalar"]
    (e,) = [e for e in v["drawing"]["edges"] if e.get("data") == "alpha"]
    assert ((e["to"], e["to_port"]), e["label"]) == (("mult", "in1"), "broadcast · ↻ n")
    assert b["mult"]["ins"] == ["in1", "in2"]


def test_a_broadcast_input_is_labelled_on_its_wire_into_the_unit():
    """mvt's column-form mac (D186): y2's slicer feeds mac_x2's b, labelled ``b · broadcast``;
    A and the seed come in unlabelled."""
    from snax_forge.viz.dfg.api import fold
    from snax_forge.viz.dfg.drawing import drawing
    from tests.helpers import bound_mvt_column

    g = bound_mvt_column()
    tops, blocks = fold(g)
    d = drawing(g, tops, None, blocks)
    got = {e["to_port"]: (e["from"], e["label"]) for e in d["edges"] if e["to"] == "unit:mac_x2"}
    assert got == {"a": ("tmp4_mult_map_1", None), "b": ("slicer:mac_x2.b", "b · broadcast"),
                   "seed": ("tmp4_mult_map", None)}  # fmt: skip


def test_a_scalar_read_as_csr_is_drawn_like_a_baked_one():
    """axpy's recipe: bake_scalar alpha csr first, so mult reads ``csr(alpha) * in2`` where
    axpy_baked's reads ``3 * in2``: alpha gets no box, edge or port, through the fuse and the
    split, and the bound unit reads ``k1 csr(alpha)``; Containers still lists alpha."""
    from snax_forge.viz.dfg.api import compact_text

    r, graphs = _axpy("axpy")
    rb, baked = _axpy("axpy_baked")
    assert r.steps[0].params == {"container": "alpha", "value": "csr"}
    assert compact_text(graphs[1], graphs[1].node("mult")) == "mult · csr(alpha) * in2"
    for v, vb in zip(_views(r, graphs[1:]), _views(rb, baked[1:]), strict=True):
        assert "alpha" in {c["name"] for c in v["containers_section"]}
        assert not [b for b in boxes(v).values() if b.get("sel") == "alpha"]
        assert not [e for e in v["drawing"]["edges"] if e.get("data") == "alpha"]
        assert box_of(v, "mult")["ins"] == box_of(vb, "mult")["ins"]
        assert containers(v, first=True) == containers(vb, first=True)  # the same inputs


def test_a_units_slots_show_csr_or_baked():
    """After bind the unit's label gives each slot as it gives W 4, no "=": k1 csr(alpha),
    or k1 3 baked; the Bindings the same, read only."""
    from snax_forge.viz.dfg.api import bindings_section, compact_text

    for name, slot in (("axpy", "csr(alpha)"), ("axpy_baked", 3)):
        r, graphs = _axpy(name)
        g = graphs[-1]
        assert compact_text(g, g.node("mult")) == f"acc · k1 * a + b · W 4 · k1 {slot}"
        (e,) = bindings_section(g, r)["entries"]
        assert [(p["name"], p["value"], p["editable"]) for p in e["params"]][-1] == (
            "k1", slot, False)  # fmt: skip


def test_a_container_box_names_its_shape():
    """A box reads the container's name and shape over the kernel's symbols: rowsum's A[M, N]
    (a matrix) and y[M] (a vector); a scalar argument its name alone (axpy's alpha)."""
    from snax_forge.viz.dfg.api import container_text
    from tests.helpers import imported

    g = imported("rowsum")
    assert (container_text(g, "A"), container_text(g, "y")) == ("A[M, N]", "y[M]")
    r, graphs = _axpy("axpy")
    a = graphs[0]
    assert [container_text(a, c) for c in ("alpha", "x", "y")] == ["alpha", "x[N]", "y[N]"]
    (v,) = _views(r, graphs[:1])
    assert {c: boxes(v)[f"{c}@0"]["text"] for c in ("alpha", "x", "y")} == {
        "alpha": "alpha", "x": "x[N]", "y": "y[N]"}  # fmt: skip


def test_a_baked_tasklet_reads_the_value():
    """After bake_scalar 3 the tasklet reads ``mult · 3 * in2``, one input port, no alpha."""
    r, graphs = _axpy("axpy_baked")
    assert r.steps[0].params == {"container": "alpha", "value": 3}
    g = graphs[1]
    from snax_forge.viz.dfg.api import compact_text

    assert compact_text(g, g.node("mult")) == "mult · 3 * in2"
    (v,) = _views(r, [g])
    assert not [b for b in boxes(v).values() if b.get("sel") == "alpha"]
    assert boxes(v)["mult"]["ins"] == ["in2"]
    assert compact_text(g, g.node("augassign_19_4")) == "augassign_19_4 · in1 + in2"


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


def test_the_lane_accumulator_reads_as_a_fold():
    """rowsum_2 bound: its unit reads ``acc · fold + · W 4`` over the column loop, under the row
    block loop and the column loop; its Details give the W lanes down a column."""
    from snax_forge.design import Platform
    from snax_forge.sandbox import Recipe, apply_recipe
    from snax_forge.viz.dfg.api import Entry, step_view
    from tests.design.helpers import SNAX_FORGE
    from tests.helpers import imported

    r = Recipe.load(recipe_path("rowsum_2"))
    g = apply_recipe(r, imported("rowsum"))[-1].graph
    v = step_view(Entry("4", Path("4"), g, None), "4", Platform.load(SNAX_FORGE), recipe=r)
    b = boxes(v)
    assert [(b[x]["text"], b[x].get("sub")) for x in ("sum_map", "sum_map_1", "unit:acc")] == [
        ("sum_map · temporal", "m_1_t · 2"), ("sum_map_1 · temporal", "n_t · 64"),
        ("acc · fold + · W 4", "over n_t")]  # fmt: skip
    assert ends(v, "A") == [
        ("A@0", "sum_map"),
        ("sum_map", "sum_map_1"),
        ("sum_map_1", "unit:acc.a"),
    ]
    d = v["details"]["node:sum"]
    assert (d["brm"], d["replaced"]) == ("lane_accumulate", "sum_map_s › sum")
    assert d["inputs"] == {"a": "A[4 * m_1_t:4 * m_1_t + 4, n_t]"}
    assert d["outputs"] == {"out": "y[4 * m_1_t:4 * m_1_t + 4]"}


# =============================================================================
# The fold's seed: matvec_seed
# =============================================================================


def _matvec_seed():
    from snax_forge.sandbox import Recipe, apply_recipe

    r = Recipe.load(recipe_path("matvec_seed"))
    return r, [x.graph for x in apply_recipe(r, Graph.load(FIXTURES / "matvec_acc.snaxdfg"))]


def test_the_seed_is_two_edges_of_x_on_the_fold():
    """Before fuse_seed the sum writes tmp1, which the elementwise nest adds to q; after it q
    is read by the fold's seed port (a seed edge, its memlet in Details) and written by its
    output, and tmp1 has no box; bound, the unit's seed port takes the seed's edge."""
    r, graphs = _matvec_seed()
    before, seeded, bound = _views(r, graphs[5:8])
    assert "tmp1@1" in containers(before)
    assert not [e for e in before["drawing"]["edges"] if e["style"] == "seed"]
    for v, box, subset in ((seeded, "sum", "q[m_t]"), (bound, "unit:sum", "q[m_t:m_t + 1]")):
        assert "tmp1@1" not in containers(v)
        assert ends(v, "q") == [
            ("q@0", "sum_map"),
            ("sum_map", f"{box}.seed"),
            (f"{box}.out", "q@1"),
        ]
        (seed,) = [e for e in v["drawing"]["edges"] if e["style"] == "seed"]
        assert v["details"][f"edge:{seed['id']}"]["memlet"] == subset
        assert v["details"]["node:sum"]["seed"] == subset
        assert "seed" in boxes(v)[box]["ins"]
    assert seeded["details"]["node:sum"]["outputs"] == {
        "out": "q[m_t] (reduce add, seed q[m_t])"}  # fmt: skip
    d = bound["details"]["node:sum"]
    assert d["inputs"] == {"a": "tmp0[m_t, 4 * n_1_t:4 * n_1_t + 4]", "seed": "q[m_t:m_t + 1]"}
    assert d["params"] == {"W": 4, "op": "add", "seed": 1}
    assert bound["details"]["container:q"]["readers"] == ["sum"]


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


def _near(g: Graph) -> dict[str, list[tuple[str, str]]]:
    from snax_forge.viz.dfg.api import Entry, graph_view

    v = graph_view(Entry("g", Path("g"), g, None))
    return {m: [(p["map"], p["near"]) for p in ps if "near" in p]
            for m, ps in v["partners"].items() if any("near" in p for p in ps)}  # fmt: skip


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


def test_matvec_acc_partners_at_import():
    """matvec_acc at import: mult and sum fuse (tmp0, both loops matched), the sum seeds from
    the elementwise nest (tmp1, its kept loop m against the nest's m), each nest's two maps
    interchange; no partner for a pair the transforms refuse (the sum's nest and the
    elementwise nest are of different depths for fuse_map)."""
    g = _bound_fixture("matvec_acc", M=8, N=512)
    found = _partners(g)
    seed = "tmp1[m] · m = m (0:8)"
    assert found == {
        "mult_map": [("sum_map", "fuse_map", MATVEC_FUSE), ("mult_map_1", "interchange_map", None)],
        "mult_map_1": [("mult_map", "interchange_map", None)],
        "sum_map": [("mult_map", "fuse_map", MATVEC_FUSE),
                    ("augassign_19_4_map", "fuse_seed", seed),
                    ("sum_map_1", "interchange_map", None)],
        "sum_map_1": [("augassign_19_4_map", "fuse_seed", seed),
                      ("sum_map", "interchange_map", None)],
        "augassign_19_4_map": [("sum_map", "fuse_seed", seed), ("sum_map_1", "fuse_seed", seed)],
    }  # fmt: skip
    _agree(g, found)
    from snax_forge.viz.dfg.api import Entry, graph_view

    v = graph_view(Entry("g", Path("g"), g, None))
    marks = {p["transform"]: p["marker"] for p in v["partners"]["sum_map"] if "marker" in p}
    assert {t: (m["box"], m["container"]) for t, m in marks.items()} == {
        "fuse_map": ("tmp0@1", "tmp0"), "fuse_seed": ("tmp1@1", "tmp1")}  # fmt: skip
    assert {boxes(v)[m["box"]]["sel"] for m in marks.values()} == {"tmp0", "tmp1"}


def test_matvec_acc_partners_after_the_tag_and_split():
    """After tag_map sum_map temporal, mult (untagged) still fuses with sum (it takes the
    tag). After split_map sum_map_1 4 too, the two no longer fuse: mult_map is a near partner
    of sum_map, its tag the refusal naming split_map mult_map_1 4; every map of the sum's nest
    seeds with the elementwise nest, its kept loop m_t, and the three maps of the sum's nest
    interchange pairwise (interchange_map takes any two maps of one perfect nest). After
    split_map mult_map_1 4, mult and sum fuse again, three loops matched."""
    from snax_forge.sandbox.transforms import split_map, tag_map

    t = tag_map(_bound_fixture("matvec_acc", M=8, N=512), "sum_map", "temporal")
    fuse = "tmp0[m_t, n] · m = m_t (0:8), n = n (0:512)"
    assert _partners(t)["mult_map"][0] == ("sum_map", "fuse_map", fuse)
    assert _near(t) == {}
    g = split_map(t, "sum_map_1", 4)
    found = _partners(g)
    seed = "tmp1[m] · m_t = m (0:8)"
    ew = "augassign_19_4_map"
    assert found == {
        "mult_map": [("mult_map_1", "interchange_map", None)],
        "mult_map_1": [("mult_map", "interchange_map", None)],
        "sum_map": [(ew, "fuse_seed", seed), ("sum_map_1", "interchange_map", None),
                    ("sum_map_1_s", "interchange_map", None)],
        "sum_map_1": [(ew, "fuse_seed", seed), ("sum_map", "interchange_map", None),
                      ("sum_map_1_s", "interchange_map", None)],
        "sum_map_1_s": [(ew, "fuse_seed", seed), ("sum_map", "interchange_map", None),
                        ("sum_map_1", "interchange_map", None)],
        ew: [("sum_map", "fuse_seed", seed), ("sum_map_1", "fuse_seed", seed),
             ("sum_map_1_s", "fuse_seed", seed)],
    }  # fmt: skip
    _agree(g, found)
    why = ("fuse_map 'mult_map' 'sum_map': mult_map_1 runs 0:512, sum_map_1 runs 0:128 and its "
           "child 0:4: split_map mult_map_1 4 first")  # fmt: skip
    assert _near(g) == {"mult_map": [("sum_map", why)], "sum_map": [("mult_map", why)]}
    both = split_map(g, "mult_map_1", 4)
    assert _near(both) == {}
    fused = ("tmp0[m_t, 4 * n_1_t + n_2_s] · m = m_t (0:8), n_1_t = n_1_t (0:128), "
             "n_2_s = n_2_s (0:4)")  # fmt: skip
    assert _partners(both)["mult_map"][0] == ("sum_map", "fuse_map", fused)


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
    """matvec_seed's last steps: the seed taken, so no fuse_seed partner is left."""
    _, graphs = _matvec_seed()
    for g in graphs[6:]:
        assert not [p for ps in _partners(g).values() for p in ps if p[1] == "fuse_seed"]


# =============================================================================
# Folds: drawn as upside-down trapezoids
# =============================================================================


def _fold_nodes(v: dict) -> list[str]:
    return [b["sel"] for b in v["drawing"]["nodes"] if b.get("fold")]


def test_a_folding_node_is_marked_fold():
    """A tasklet whose output memlet has a reduce (an identity or a seed) and a bound unit whose
    replaced nest folds: matvec_acc's sum, seeded or not, its bound accumulate; dot 4's mac and
    final adder tree; rowsum 2's lane accumulator. The multiplies, the elementwise add, a simd
    unit and a link do not."""
    from snax_forge.viz.dfg.api import Entry, graph_view
    from tests.sandbox.helpers import recipe_graphs

    def view_of(g: Graph) -> dict:
        return graph_view(Entry("g", Path("g"), g, None))

    assert _fold_nodes(view_of(_bound_fixture("matvec_acc", M=8, N=512))) == ["sum"]
    _, graphs = _matvec_seed()
    assert [_fold_nodes(view_of(g)) for g in graphs[5:]] == [["sum"]] * 4
    assert "fold" not in _node_in(view_of(graphs[-1]), "mult")
    for name, want in (("dot_4", ["mult", "sum_final"]), ("rowsum_2", ["sum"]),
                       ("axpy", [])):  # fmt: skip
        v = view_of(recipe_graphs(name)[1][-1])
        assert sorted(_fold_nodes(v)) == want, name
    links = [b for b in view_of(recipe_graphs("dot_4")[1][-1])["drawing"]["nodes"]
             if b["kind"] == "link"]  # fmt: skip
    assert links and not any(b.get("fold") for b in links)
