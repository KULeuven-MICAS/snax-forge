"""The memory layout of a run (VIS4a, D95, D96): what the memory tab is given.

1. the folded lines of the flow runs, written out
2. the lines do not grow with the depth of a memory
3. expanding every fold and empty run puts every element exactly once, at
   the address its region gives
4. layouts that are not one row per step: 2D, transposed, overlapping
5. a run without regions, and the limits of a rows request
"""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.flow import run_flow
from snax_forge.snax_model.scenario import ClusterConfig, Region
from snax_forge.viz import api, memory
from tests.design.helpers import FIXTURES, RECIPE, SMALL16

from .helpers import SCEN, run_dir

PLAIN = FIXTURES / "vecadd.snaxdfg"
B_PIN = [("B.l1.base", 576)]
FLOWS = {
    "default": {},
    "W8": {"recipe_sets": {"W": 8}},
    "B_pinned": {"memory_sets": B_PIN},
    "B_pinned_1024": {"memory_sets": B_PIN, "platform_sets": [("l1.rows", 1024)]},
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The flow runs of the acceptance, loaded as the viewer loads them."""
    base = tmp_path_factory.mktemp("flows")
    out = {}
    for name, kw in FLOWS.items():
        f = run_flow(RECIPE, SMALL16, graph_path=PLAIN, out=base / name, trace_level="off", **kw)
        out[name] = api.load_run(f.out / "run", name)
    return out


def mem(view, m):
    return next(x for x in view["memories"] if x["mem"] == m)


def text(line, regions):
    """A line as a short string: ``row 4: _x8 B[0..7]``, ``fold 1-2: A+16``, ``empty 13-63``."""
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
        if parts and parts[-1][0] == names[ri] and parts[-1][2] + 1 == flat:
            parts[-1][2] = flat
        else:
            parts.append([names[ri], flat, flat, 1])
    words = [f"_x{p[3]}" if p[0] == "_" else f"{p[0]}[{p[1]}..{p[2]}]" for p in parts]
    return f"row {line['row']}: " + " ".join(words)


def texts(view, m):
    x = mem(view, m)
    return [text(line, x["regions"]) for line in x["lines"]]


# =============================================================================
# 1. The folded lines of the flow runs
# =============================================================================

L1_CONTIGUOUS = [
    "row 0: A[0..15]", "fold 1-2: A+16", "row 3: A[48..63]",
    "row 4: B[0..15]", "fold 5-6: B+16", "row 7: B[48..63]",
    "row 8: C[0..15]", "fold 9-10: C+16", "row 11: C[48..63]",
    "empty 12-63",
]  # fmt: skip

L2_PACKED = [
    "row 0: A[0..7]", "fold 1-6: A+8", "row 7: A[56..63]",
    "row 8: B[0..7]", "fold 9-14: B+8", "row 15: B[56..63]",
    "row 16: C[0..7]", "fold 17-22: C+8", "row 23: C[56..63]",
    "empty 24-511",
]  # fmt: skip


def test_the_b_pinned_run(runs):
    """B at 576 starts in bank 8 of row 4; the placement puts C after it at 1088."""
    view = api.memory_view(runs["B_pinned"])
    assert texts(view, "l1") == [
        "row 0: A[0..15]", "fold 1-2: A+16", "row 3: A[48..63]",
        "row 4: _x8 B[0..7]", "row 5: B[8..23]", "fold 6-6: B+16", "row 7: B[40..55]",
        "row 8: B[56..63] C[0..7]", "row 9: C[8..23]", "fold 10-10: C+16", "row 11: C[40..55]",
        "row 12: C[56..63] _x8",
        "empty 13-63",
    ]  # fmt: skip
    assert texts(view, "l2") == L2_PACKED
    l1 = mem(view, "l1")
    b = next(r for r in l1["regions"] if r["name"] == "B")
    assert (b["start"], b["end"], b["bytes"], b["rows"]) == (576, 1088, 512, [4, 8])
    assert b["share"] == 512 / (16 * 64 * 8) and l1["used"] == {"words": 192, "share": 192 / 1024}
    fold = next(x for x in l1["lines"] if x["kind"] == "fold" and x["from"] == 1)
    assert fold == {"kind": "fold", "from": 1, "to": 2, "hidden": 2,
                    "step": {"0": 16}, "span": {"0": [16, 47]}}  # fmt: skip


@pytest.mark.parametrize("name", ["default", "W8"])
def test_the_contiguous_runs(runs, name):
    view = api.memory_view(runs[name])
    assert view["has_regions"] and [m["mem"] for m in view["memories"]] == ["l1", "l2"]
    assert texts(view, "l1") == L1_CONTIGUOUS and texts(view, "l2") == L2_PACKED
    geo = mem(view, "l1")["geometry"]
    assert (geo["columns"], geo["rows"], geo["column_label"], geo["group"]) == (16, 64, "bank", 8)
    geo2 = mem(view, "l2")["geometry"]
    assert (geo2["columns"], geo2["rows"], geo2["row_label"]) == (8, 512, "beat")


# =============================================================================
# 2. The lines do not grow with the depth
# =============================================================================


def test_a_deeper_l1_gives_as_many_lines(runs):
    short = texts(api.memory_view(runs["B_pinned"]), "l1")
    deep = texts(api.memory_view(runs["B_pinned_1024"]), "l1")
    assert len(deep) == len(short) and deep[:-1] == short[:-1] and deep[-1] == "empty 13-1023"


# =============================================================================
# 3. Expanding every fold puts every element once, where its region says
# =============================================================================


def expand(rv, m):
    """Every row of memory ``m``, as the viewer gets them: row lines plus requests."""
    view = mem(api.memory_view(rv), m)
    rows = {}
    for line in view["lines"]:
        if line["kind"] == "row":
            rows[line["row"]] = line
            continue
        a, b = line["from"], line["to"] + 1
        for s in range(a, b, memory.MAX_ROWS):
            got = api.memory_rows(rv, m, s, min(b, s + memory.MAX_ROWS))["rows"]
            for r in got:
                assert line["kind"] == "fold" or not any(r["cells"]), "an empty run is empty"
                rows[r["row"]] = r
    return view, rows


@pytest.mark.parametrize("name", list(FLOWS))
@pytest.mark.parametrize("m", ["l1", "l2"])
def test_every_element_once_at_its_address(runs, name, m):
    rv = runs[name]
    view, rows = expand(rv, m)
    assert sorted(rows) == list(range(view["geometry"]["rows"]))  # every row, once
    regions = [r for r in rv.outputs.regions if r.mem == m]
    seen = {}
    for line in rows.values():
        for col, cell in enumerate(line["cells"]):
            for ri, flat in cell:
                assert (ri, flat) not in seen
                seen[ri, flat] = line["addr"][col]
    for ri, reg in enumerate(regions):
        for flat in range(reg.size):
            idx = np.unravel_index(flat, reg.shape)
            assert seen.pop((ri, flat)) == reg.address(idx)
    assert not seen


# =============================================================================
# 4. Layouts that are not one row per step
# =============================================================================


def alu4():
    return ClusterConfig.load(SCEN / "clusters" / "alu4.json")


def test_a_2d_region_folds_by_its_rows():
    m = Region("M", "l1", 0, (8, 16), (128, 8))  # row-major 8 x 16: one matrix row per bank row
    got = memory.summary(memory.layout_of("l1", alu4(), [m]))
    assert [text(x, got["regions"]) for x in got["lines"]] == [
        "row 0: M[0..15]", "fold 1-6: M+16", "row 7: M[112..127]", "empty 8-63"
    ]  # fmt: skip


def test_a_transposed_region_folds_with_step_one():
    """(i, j) at 8 i + 128 j: bank i, row j, so row j holds flats j, 8 + j, ... (step 1)."""
    t = Region("T", "l1", 0, (16, 8), (8, 128))
    got = memory.summary(memory.layout_of("l1", alu4(), [t]))
    lines = got["lines"]
    assert [x["kind"] for x in lines] == ["row", "fold", "row", "empty"]
    assert [c[0][1] for c in lines[0]["cells"]] == [8 * i for i in range(16)]
    assert lines[1]["step"] == {"0": 1} and (lines[1]["from"], lines[1]["to"]) == (1, 6)


def test_overlapping_regions_share_cells():
    """Two names for one buffer (a reused buffer): each cell holds both, in region order."""
    a = Region("A", "l1", 0, (32,), (8,))
    b = Region("B", "l1", 0, (32,), (8,))
    got = memory.summary(memory.layout_of("l1", alu4(), [a, b]))
    row0 = got["lines"][0]
    assert row0["cells"][3] == [[0, 3], [1, 3]]
    assert got["used"]["words"] == 32 and [r["bytes"] for r in got["regions"]] == [256, 256]


def test_a_row_that_changes_step_ends_the_run():
    """A[0..15], A[16..31], then A skips a row's worth: two runs, no fold across the jump."""
    a = Region("A", "l1", 0, (3, 16), (128, 8))
    b = Region("B", "l1", 3 * 128, (1, 16), (128, 8))  # row 3 holds another region
    got = memory.summary(memory.layout_of("l1", alu4(), [a, b]))
    assert [text(x, got["regions"]) for x in got["lines"]] == [
        "row 0: A[0..15]", "fold 1-1: A+16", "row 2: A[32..47]", "row 3: B[0..15]", "empty 4-63"
    ]  # fmt: skip


# =============================================================================
# 5. A run without regions, and the limits of a rows request
# =============================================================================


def test_a_run_without_regions(tmp_path):
    rv = api.load_run(run_dir(tmp_path / "dma", "dma", "off"))
    view = api.memory_view(rv)
    assert view["has_regions"] is False
    for x in view["memories"]:
        assert x["regions"] == [] and x["used"]["words"] == 0
        assert x["lines"] == [{"kind": "empty", "from": 0, "to": x["geometry"]["rows"] - 1}]


def test_the_limits_of_a_rows_request(runs):
    rv = runs["B_pinned_1024"]
    assert len(api.memory_rows(rv, "l1", 0, memory.MAX_ROWS)["rows"]) == memory.MAX_ROWS
    for a, b in [(0, memory.MAX_ROWS + 1), (5, 5), (-1, 3), (1020, 1025)]:
        with pytest.raises(memory.MemoryViewError):
            api.memory_rows(rv, "l1", a, b)
    with pytest.raises(memory.MemoryViewError):
        api.memory_rows(rv, "l3", 0, 1)
