"""The memory layout of a run: what the memory tab is given (memory.py, api.py).

1. the folded lines of the flow runs, written out
2. the lines do not grow with the depth of a memory
3. expanding every fold and empty run puts every element exactly once, at
   the address its region gives
4. layouts that are not one row per step: 2D, transposed, overlapping
5. a memory without regions, and the limits of a rows request
6. the marks: the ones a level allows, conflict counts and times on the rows
7. the cycle overlay: a cycle's accesses and the reads whose data comes back in it

Runs are flow folders with SNAX-GVSoC stubbed (helpers.run_dir), regions from
their design points and ELFs (run.regions.regions_of).
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from snax_forge.design.platform import L1Config, L2Config
from snax_forge.run import Region
from snax_forge.viz import api, memory
from tests.flow.helpers import stubbed_flow

from .helpers import L1, Events, cluster

B_PIN = [("B.l1.base", L1 + 576)]  # B from bank 8 of row 2
FLOWS = {
    "default": ("vecadd", "stall", {}),
    "W8": ("vecadd", "off", {"recipe_sets": {"W": 8}}),
    "B_pinned": (
        "vecadd",
        "off",
        {"memory_sets": B_PIN},
    ),  # the synthetic stalls need A, B on one bank
    "B_pinned_1024": (
        "vecadd",
        "off",
        {"memory_sets": B_PIN, "platform_sets": [("l1.rows", 1024)]},
    ),
    "task": ("vecadd", "task", {}),
    "beat": ("vecadd", "beat", {}),
    "vsum": ("vsum", "off", {}),
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The flow runs, loaded as the viewer loads them."""
    base = tmp_path_factory.mktemp("flows")
    return {name: api.load_run(stubbed_flow(base, design, level, out=base / name, **kw), name)
            for name, (design, level, kw) in FLOWS.items()}  # fmt: skip


def mem(view, m):
    return next(x for x in view["memories"] if x["mem"] == m)


def text(line, regions):
    """A line as a short string: ``row 4: _x8 B[0..7]``, ``fold 1-2: A+16``, ``empty 13-63``;
    a region's beat padding as ``out[pad]x7`` (its words)."""
    names = [r["name"] for r in regions]
    if line["kind"] == "empty":
        return f"empty {line['from']}-{line['to']}"
    if line["kind"] == "fold":
        steps = " ".join(f"{names[int(k)]}{v:+d}" for k, v in line["step"].items())
        return f"fold {line['from']}-{line['to']}: {steps}"
    parts: list[list] = []  # [name, first, last, count] runs of neighbouring cells
    for cell in line["cells"]:
        if not cell:
            if parts and parts[-1][0] == "_":
                parts[-1][3] += 1
            else:
                parts.append(["_", 0, 0, 1])
            continue
        ((ri, flat),) = cell
        if flat == memory.BEAT_PAD:
            pad = f"{names[ri]}[pad]"
            if parts and parts[-1][0] == pad:
                parts[-1][3] += 1
            else:
                parts.append([pad, 0, 0, 1])
            continue
        if parts and parts[-1][0] == names[ri] and parts[-1][2] + 1 == flat:
            parts[-1][2] = flat
        else:
            parts.append([names[ri], flat, flat, 1])
    words = [f"{p[0]}x{p[3]}" if p[0].endswith("[pad]") else f"_x{p[3]}" if p[0] == "_"
             else f"{p[0]}[{p[1]}..{p[2]}]" for p in parts]  # fmt: skip
    return f"row {line['row']}: " + " ".join(words)


def texts(view, m):
    x = mem(view, m)
    return [text(line, x["regions"]) for line in x["lines"]]


# =============================================================================
# 1. The folded lines of the flow runs
# =============================================================================

L1_CONTIGUOUS = [
    "row 0: A[0..31]", "row 1: A[32..63]", "row 2: B[0..31]", "row 3: B[32..63]",
    "row 4: C[0..31]", "row 5: C[32..63]", "empty 6-511",
]  # fmt: skip

L2_PACKED = [  # the stubbed ELF's symbols: at the plan's L2 bases
    "row 0: A[0..7]", "fold 1-6: A+8", "row 7: A[56..63]",
    "row 8: B[0..7]", "fold 9-14: B+8", "row 15: B[56..63]",
    "row 16: C[0..7]", "fold 17-22: C+8", "row 23: C[56..63]",
    f"empty 24-{(1 << 31) // 64 - 1}",
]  # fmt: skip


@pytest.mark.parametrize("name", ["default", "W8"])
def test_the_contiguous_runs(runs, name):
    view = api.memory_view(runs[name])
    assert view["has_regions"] and [m["mem"] for m in view["memories"]] == ["l1", "l2"]
    assert texts(view, "l1") == L1_CONTIGUOUS and texts(view, "l2") == L2_PACKED
    geo = mem(view, "l1")["geometry"]
    assert (geo["columns"], geo["rows"], geo["column_label"], geo["group"]) == (32, 512, "bank", 8)
    assert (geo["base_addr"], geo["word_bytes"], geo["size_bytes"]) == (L1, 8, 128 * 1024)
    geo2 = mem(view, "l2")["geometry"]
    assert (geo2["columns"], geo2["row_label"], geo2["base_addr"]) == (8, "beat", 0x80000000)


def test_l2_is_where_the_elf_put_the_arrays(runs):
    rv = runs["default"]
    l2 = mem(api.memory_view(rv), "l2")
    syms = rv.outputs.elf_symbols
    assert [(r["name"], r["start"]) for r in l2["regions"]] == [
        (n, syms[n]["addr"]) for n in ("A", "B", "C")]  # fmt: skip
    rows = [(syms[n]["addr"] - 0x80000000) // 64 for n in ("A", "B", "C")]
    assert [r["rows"][0] for r in l2["regions"]] == rows
    got = texts(api.memory_view(rv), "l2")
    assert [x for x in got if x.endswith(": A[0..7]")] == [f"row {rows[0]}: A[0..7]"]


def test_the_b_pinned_run(runs):
    """B at L1 + 576 starts in bank 8 of row 2; the placement puts C after it at L1 + 1088."""
    view = api.memory_view(runs["B_pinned"])
    assert texts(view, "l1") == [
        "row 0: A[0..31]", "row 1: A[32..63]",
        "row 2: _x8 B[0..23]", "row 3: B[24..55]", "row 4: B[56..63] C[0..23]",
        "row 5: C[24..55]", "row 6: C[56..63] _x24",
        "empty 7-511",
    ]  # fmt: skip
    l1 = mem(view, "l1")
    b = next(r for r in l1["regions"] if r["name"] == "B")
    assert (b["start"], b["end"], b["bytes"], b["rows"]) == (L1 + 576, L1 + 1088, 512, [2, 4])
    assert l1["geometry"]["size_bytes"] == 128 * 1024
    assert l1["used"] == {"words": 192, "share": 192 / 16384}
    assert b["banks"] == list(range(32))


def test_a_folded_line(runs):
    """vsum's A fills L2 rows; its output is one word."""
    view = api.memory_view(runs["vsum"])
    l2 = mem(view, "l2")
    fold = next(x for x in l2["lines"] if x["kind"] == "fold")
    a = fold["from"] - 1
    assert fold == {"kind": "fold", "from": a + 1, "to": a + 6, "hidden": 6,
                    "step": {"0": 8}, "span": {"0": [8, 55]}}  # fmt: skip
    # out is one word the DMA moves: seven more to a whole 64-byte beat, kept free (D144)
    assert [x for x in texts(view, "l1") if "out" in x] == ["row 2: out[0..0] out[pad]x7 _x24"]


# =============================================================================
# 2. The lines do not grow with the depth
# =============================================================================


def test_a_deeper_l1_gives_as_many_lines(runs):
    short = texts(api.memory_view(runs["B_pinned"]), "l1")
    deep = texts(api.memory_view(runs["B_pinned_1024"]), "l1")
    assert len(deep) == len(short) and deep[:-1] == short[:-1] and deep[-1] == "empty 7-1023"


# =============================================================================
# 3. Expanding every fold puts every element once, where its region says
# =============================================================================


def expand(rv, m):
    """Every row of memory ``m`` that is not in a long empty run, as the viewer gets them."""
    view = mem(api.memory_view(rv), m)
    rows = {}
    for line in view["lines"]:
        if line["kind"] == "row":
            rows[line["row"]] = line
            continue
        a, b = line["from"], line["to"] + 1
        if line["kind"] == "empty" and b - a > 4 * memory.MAX_ROWS:
            got = (
                api.memory_rows(rv, m, a, a + 1)["rows"] + api.memory_rows(rv, m, b - 1, b)["rows"]
            )
            assert not any(c for r in got for c in r["cells"]), "an empty run is empty"
            continue
        for s in range(a, b, memory.MAX_ROWS):
            for r in api.memory_rows(rv, m, s, min(b, s + memory.MAX_ROWS))["rows"]:
                assert line["kind"] == "fold" or not any(r["cells"]), "an empty run is empty"
                rows[r["row"]] = r
    return view, rows


@pytest.mark.parametrize("name", ["default", "B_pinned", "B_pinned_1024", "vsum"])
@pytest.mark.parametrize("m", ["l1", "l2"])
def test_every_element_once_at_its_address(runs, name, m):
    rv = runs[name]
    view, rows = expand(rv, m)
    assert all(0 <= r < view["geometry"]["rows"] for r in rows)
    regions = [r for r in rv.regions if r.memory == m]
    seen = {}
    for line in rows.values():
        for col, cell in enumerate(line["cells"]):
            for ri, flat in cell:
                if flat == memory.BEAT_PAD:  # a word of the region's beat padding
                    lo, hi = view["regions"][ri]["beat_pad"]
                    assert lo <= line["addr"][col] + view["geometry"]["word_bytes"] - 1
                    assert line["addr"][col] < hi
                    continue
                assert (ri, flat) not in seen
                seen[ri, flat] = line["addr"][col]
    for ri, reg in enumerate(regions):
        for flat in range(int(np.prod(reg.shape))):
            idx = np.array([np.unravel_index(flat, reg.shape)])
            assert seen.pop((ri, flat)) == int(reg.layout.address(idx)[0])
    assert not seen
    if m == "l1":  # a cell's column and row are its address's bank and row
        for line in rows.values():
            for col, a in enumerate(line["addr"]):
                assert rv.cluster.bank_row(a) == (col, line["row"])


# =============================================================================
# 4. Layouts that are not one row per step
# =============================================================================


def layout(*regs):
    point = SimpleNamespace(platform=SimpleNamespace(l1=L1Config(), l2=L2Config()))
    return memory.summary(memory.layout_of("l1", cluster(), point, list(regs)))


def test_a_2d_region_folds_by_its_rows():
    m = Region("M", "l1", L1, (8, 32), (256, 8), "int64")  # one matrix row per bank row
    got = layout(m)
    assert [text(x, got["regions"]) for x in got["lines"]] == [
        "row 0: M[0..31]", "fold 1-6: M+32", "row 7: M[224..255]", "empty 8-511"
    ]  # fmt: skip


def test_a_transposed_region_folds_with_step_one():
    """(i, j) at 8 i + 256 j: bank i, row j, so row j holds flats j, 8 + j, ... (step 1)."""
    t = Region("T", "l1", L1, (32, 8), (8, 256), "int64")
    lines = layout(t)["lines"]
    assert [x["kind"] for x in lines] == ["row", "fold", "row", "empty"]
    assert [c[0][1] for c in lines[0]["cells"]] == [8 * i for i in range(32)]
    assert lines[1]["step"] == {"0": 1} and (lines[1]["from"], lines[1]["to"]) == (1, 6)


def test_overlapping_regions_share_cells():
    """Two names for one buffer (a reused buffer): each cell holds both, in region order."""
    a = Region("A", "l1", L1, (32,), (8,), "int64")
    b = Region("B", "l1", L1, (32,), (8,), "int64")
    got = layout(a, b)
    assert got["lines"][0]["cells"][3] == [[0, 3], [1, 3]]
    assert got["used"]["words"] == 32 and [r["bytes"] for r in got["regions"]] == [256, 256]


def test_a_region_change_ends_the_run():
    a = Region("A", "l1", L1, (3, 32), (256, 8), "int64")
    b = Region("B", "l1", L1 + 3 * 256, (1, 32), (256, 8), "int64")
    got = layout(a, b)
    assert [text(x, got["regions"]) for x in got["lines"]] == [
        "row 0: A[0..31]", "fold 1-1: A+32", "row 2: A[64..95]", "row 3: B[0..31]", "empty 4-511"
    ]  # fmt: skip


def test_a_region_outside_the_memory_is_refused():
    with pytest.raises(memory.MemoryViewError, match="outside"):
        layout(Region("A", "l1", L1 - 8, (4,), (8,), "int64"))


# =============================================================================
# 5. A memory without regions, and the limits of a rows request
# =============================================================================


def test_a_memory_without_regions():
    got = layout()
    assert got["regions"] == [] and got["used"]["words"] == 0
    assert got["lines"] == [{"kind": "empty", "from": 0, "to": 511}]


def test_the_limits_of_a_rows_request(runs):
    rv = runs["B_pinned_1024"]
    assert len(api.memory_rows(rv, "l1", 0, memory.MAX_ROWS)["rows"]) == memory.MAX_ROWS
    for a, b in [(0, memory.MAX_ROWS + 1), (5, 5), (-1, 3), (1020, 1025)]:
        with pytest.raises(memory.MemoryViewError):
            api.memory_rows(rv, "l1", a, b)
    with pytest.raises(memory.MemoryViewError):
        api.memory_rows(rv, "l3", 0, 1)


# =============================================================================
# 6. Marks
# =============================================================================


@pytest.mark.parametrize("name, have", [
    ("W8", []), ("task", []), ("beat", ["arrival", "use", "wait"]),
    ("default", ["conflicts", "arrival", "use", "wait"]),
])  # fmt: skip
def test_the_marks_a_level_allows(runs, name, have):
    view = api.memory_view(runs[name])
    assert list(view["marks_available"]) == have and view["marks"] is None
    assert sorted(view["marks_missing"]) == sorted(
        {"conflicts", "arrival", "use", "wait"} - set(have)
    )
    for m, why in view["marks_missing"].items():
        assert why.startswith("needs --trace ")
        with pytest.raises(memory.MemoryViewError, match="needs --trace"):
            api.memory_view(runs[name], m)
    assert "marks" not in view["memories"][0]["lines"][0]  # none unless asked
    with pytest.raises(memory.MemoryViewError, match="must be one of"):
        api.memory_view(runs[name], "later")


def test_conflict_marks_fold_the_memory_tab(runs):
    """The synthetic run's acc_b beat 0 stalls behind acc_a's on banks 0-3: the words B[0..3]
    waited (row 2) and A[0..3] were served (row 0)."""
    rv = runs["default"]
    l1 = api.memory_view(rv, "conflicts")["memories"][0]
    rows = {x["row"]: x for x in l1["lines"] if x["kind"] == "row"}
    assert rows[0]["marks"] == [1] * 4 + [0] * 28 and rows[2]["marks"] == [1] * 4 + [0] * 28
    assert rows[1]["marks"] == [0] * 32
    stalls = [e for e in rv.events if e["k"] == "stall"]
    assert len(stalls) == 4 and l1["mark_range"] == [0, 1]
    total = sum(sum(x["marks"]) for x in l1["lines"] if x["kind"] == "row")
    total += sum(f["marked"] for f in l1["lines"] if f["kind"] == "fold")
    assert total == 2 * len(stalls)  # each stall marks the word that waited and the word served
    assert mem(api.memory_view(rv, "conflicts"), "l2")["mark_range"] is None


def test_time_marks(runs):
    """The synthetic run: A's and B's DMA beats land one a cycle, the streamers read beat b of
    every lane at t + 1 + b (acc_b one later), acc_out writes beat b at t + 5 + b."""
    rv = runs["default"]
    arrival = {m["mem"]: m for m in api.memory_view(rv, "arrival")["memories"]}
    rows = {x["row"]: x["marks"] for x in arrival["l1"]["lines"] if x["kind"] == "row"}
    dma = {(e["id"], e["i"]): e["t"] for e in rv.events if e["k"] == "dma" and e["op"] == "w"}
    assert rows[0] == [dma[(2, 0)]] * 8 + [dma[(2, 1)]] * 8 + [dma[(2, 2)]] * 8 + [dma[(2, 3)]] * 8
    out = [e for e in rv.events if e["k"] == "grant" and e["op"] == "w"]
    assert rows[4][:4] == [min(e["t"] for e in out)] * 4
    wait = {x["row"]: x["marks"] for x in api.memory_view(rv, "wait")["memories"][0]["lines"]
            if x["kind"] == "row"}  # fmt: skip
    use = {x["row"]: x["marks"] for x in api.memory_view(rv, "use")["memories"][0]["lines"]
           if x["kind"] == "row"}  # fmt: skip
    assert wait[0] == [u - a for u, a in zip(use[0], rows[0], strict=True)]
    assert wait[4] == [None] * 32  # C is written, then read by the DMA only
    # in L2: A was there from the start, C arrives with the store's bursts
    l2 = arrival["l2"]
    assert l2["mark_range"] is not None
    assert all(m is None for x in l2["lines"] if x["kind"] == "row" and x["cells"][0]
               and x["cells"][0][0][0] == 0 for m in x["marks"])  # fmt: skip
    got = api.memory_rows(rv, "l1", 0, 2, "arrival")["rows"]
    assert [r["marks"] for r in got] == [rows[0], rows[1]]


def test_the_time_step_rule():
    ts = memory._time_step
    assert ts([1, 2, None], [3, 4, None]) == (True, 2)
    assert ts([1, 2, None], [3, 5, None]) == (False, None)
    assert ts([1, None], [3, 4]) == (False, None)  # a word with a time next to one without
    assert ts([None, None], [None, None]) == (True, None)
    point = SimpleNamespace(platform=SimpleNamespace(l1=L1Config(), l2=L2Config()))
    lay = memory.layout_of(
        "l1", cluster(), point, [Region("A", "l1", L1, (4, 32), (256, 8), "int64")]
    )
    marks = memory.row_marks(lay, {("l1", L1 + 256 * r + 8 * c): t for r, t in enumerate([1, 3, 5, 8])
                                   for c in range(32)}, "time")  # fmt: skip
    kinds = [(x["kind"], x.get("row", x.get("from"))) for x in memory.fold(lay, marks)]
    assert kinds == [
        ("row", 0),
        ("fold", 1),
        ("row", 2),
        ("row", 3),
        ("empty", 4),
    ]  # step 2, then 3


# =============================================================================
# 7. The cycle overlay
# =============================================================================


def test_every_read_is_in_the_overlay_of_the_cycle_its_data_comes_back(runs):
    rv = runs["default"]
    reads = [e for e in rv.events if e["k"] in ("grant", "dma", "burst") and e.get("op") == "r"]
    assert reads and {e["k"] for e in reads} == {"grant", "dma", "burst"}
    for e in reads:
        assert e in api.memory_cycle(rv, e["back"])["returns"]
    for t in range(rv.total_cycles):
        at = api.memory_cycle(rv, t)
        assert all(e["back"] == t for e in at["returns"])
        assert at["events"] == [e for e in rv.events if e["t"] == t and e["k"] in api.OVERLAY_KINDS]


def test_a_read_back_long_after_its_grant_is_in_that_cycle_only(runs):
    """A held read: acc_a's grant at 10 gets its data at 210, 200 cycles later."""
    ev = Events()
    late = ev.grant(10, "acc_a.lane0", L1, b=0, task="add_acc_a", back=210)
    rv = replace(runs["default"], index=ev.index())
    assert api.memory_cycle(rv, 210) == {"t": 210, "level": "stall", "events": [],
                                         "returns": [late]}  # fmt: skip
    assert api.memory_cycle(rv, 10)["events"] == [late]
    assert not any(api.memory_cycle(rv, t)["returns"] for t in range(300) if t != 210)


# =============================================================================
# 5. Packing and padding (D142), from design points
# =============================================================================


def designed_l1(tmp_path, recipe):
    """L1's summary for ``recipes/<recipe>.json`` designed on snax_forge (no run: no L2)."""
    from snax_forge.design import DesignPoint
    from snax_forge.gen import generate
    from snax_forge.run import load_cluster, regions_of
    from tests.gen.helpers import designed

    rel = designed(tmp_path, recipe, recipe)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        generate(rel, tmp_path / "gen", root=tmp_path)
    point = DesignPoint.load(tmp_path / rel)
    cl = load_cluster(tmp_path / "gen" / "snax_forge.hjson", tmp_path / rel)
    return memory.summary(memory.layout_of("l1", cl, point, regions_of(point, None)))


def test_a_tail_and_a_beat_padded(tmp_path):
    """rowsum_3x7: A[3, 7] laid out over its padded [3, 8], each row's pad at A[r, 7]; y, three
    int64 the DMA moves, padded to a whole 64-byte beat. The extents are the checks'."""
    s = designed_l1(tmp_path, "rowsum_3x7")
    a, y = s["regions"]
    assert (a["shape"], a["padded_shape"], a["pad_elements"]) == ([3, 7], [3, 8], 3)
    assert (a["data_end"] - a["start"], a["end"], a["beat_pad"]) == (192, a["data_end"], None)
    assert (y["data_end"] - y["start"], y["end"] - y["start"]) == (24, 64)
    assert y["beat_pad"] == [y["data_end"], y["end"]]
    pads = sorted((ri, f, x["addr"][c]) for x in s["lines"] if x["kind"] == "row"
                  for c, cell in enumerate(x["cells"]) for ri, f in cell if f < 0)  # fmt: skip
    a_pads = [(f, addr) for ri, f, addr in pads if ri == 0]
    assert sorted(a_pads, key=lambda x: x[1]) == [(memory.tail_pad(p), a["start"] + 8 * p)
                                                  for p in (7, 15, 23)]  # fmt: skip
    assert [addr for ri, f, addr in pads if ri == 1] == list(range(y["data_end"], y["end"], 8))
    assert {f for ri, f, _ in pads if ri == 1} == {memory.BEAT_PAD}


def test_two_int32_to_a_word(tmp_path):
    """vecadd_i32: each L1 word holds two int32, A[2i] and A[2i + 1]; no padding."""
    s = designed_l1(tmp_path, "vecadd_i32")
    assert [r["pad_elements"] for r in s["regions"]] == [0, 0, 0]
    assert [r["beat_pad"] for r in s["regions"]] == [None] * 3
    first = next(x for x in s["lines"] if x["kind"] == "row")
    assert first["cells"][:2] == [[[0, 0], [0, 1]], [[0, 2], [0, 3]]]
