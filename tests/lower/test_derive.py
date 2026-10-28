"""Design point -> task list (D64)."""

from __future__ import annotations

import pytest

from snax_forge.design import Design, DesignPoint, MemorySpec, Platform, run_checks, streamers
from snax_forge.design.streamers import Loop
from snax_forge.dfg import Memlet
from snax_forge.lower import Layout, LowerError, TaskList, memlet_values, placed_waits, task_list
from snax_forge.lower import __main__ as cli
from snax_forge.lower.derive import addresses
from tests.brm.helpers import affine_elementwise_add
from tests.design.helpers import (
    SNAX_FORGE,
    bound_matadd,
    bound_rowsum,
    bound_vecadd,
    bound_w8,
    chained,
    design,
    two_loops,
)

from .helpers import VECADD_TASKS

PINS = [("B.l1.base", 576), ("C.l1.base", 1152)]


def point(graph="vecadd_accelerated", memory=(), sets=None, edit_graph=None) -> DesignPoint:
    d = design(graph, sets=sets, edit_graph=edit_graph)
    d.memory_spec = MemorySpec().with_changes(list(memory))
    assert run_checks(d) == []
    return DesignPoint.of(d, "vecadd")


# =============================================================================
# 1. vecadd
# =============================================================================


def test_vecadd_on_snax_forge_gives_its_checked_in_task_list():
    """Equal to tests/run/fixtures/vecadd/tasks.json, the run fixture's (C§9)."""
    d = Design(Platform.load(SNAX_FORGE), bound_vecadd(4))
    assert run_checks(d) == []
    got = task_list(DesignPoint.of(d, "vecadd"))
    want = TaskList.load(VECADD_TASKS)
    assert got == want and got.to_dict() == want.to_dict()


def test_the_default_plan_differs_from_the_pinned_one_only_in_the_l1_bases():
    got = task_list(point()).to_dict()["steps"]
    want = task_list(point(memory=PINS)).to_dict()["steps"]
    diff = [(a["task_name"], a["values"]) for a, b in zip(got, want, strict=True) if a != b]
    assert [name for name, _ in diff] == ["load_B", "add_acc_b", "add_acc_out", "store_C"]
    assert diff[0][1]["dst"]["base"] == 512 and diff[1][1]["base"] == 512
    assert diff[2][1]["base"] == 1024 and diff[3][1]["src"]["base"] == 1024


def test_w8_gives_eight_lanes():
    d = design(bound_w8())
    assert run_checks(d) == []
    p = DesignPoint.of(d, "vecadd_w8")
    tl = task_list(p)
    acc = tl.configured()["add_acc"]
    assert acc.values == {"n": 8}
    assert tl.configured()["add_acc_a"].values == {
        "base": 0, "temporal_bounds": [8], "temporal_strides": [64], "spatial_strides": [8]
    }  # fmt: skip


# =============================================================================
# 2. More than one loop, more than one node
# =============================================================================


def test_two_temporal_maps_give_two_streamer_loops():
    p = point(edit_graph=two_loops)  # two streamer loops, from the graph (D84)
    tl = task_list(p)
    assert tl.configured()["add_acc_a"].values == {
        "base": 0, "temporal_bounds": [8, 2], "temporal_strides": [32, 256], "spatial_strides": [8]
    }  # fmt: skip
    assert tl.configured()["add_acc"].values == {"n": 16}


def test_matadd_lowers_its_two_firing_loops():
    """matadd (int32, D142) at W = 4, M = 8, N = 16 on snax_forge (D73, D84): the column beats
    innermost (4, 16 bytes apart), the rows outer (8, 64 bytes), 4 lanes as 2 channels a word
    (8 bytes) apart, each base the plan's; n = 32 firings; the streamers' two temporal loops
    are the graph's."""
    d = Design(Platform.load(SNAX_FORGE), bound_matadd())
    assert run_checks(d) == []
    p = DesignPoint.of(d, "matadd")
    assert {s.name: s.temporal_dims for s in p.streamers.values()} == {
        "acc_a": 2, "acc_b": 2, "acc_out": 2
    }  # fmt: skip
    c = task_list(p).configured()
    for port, cont in (("a", "A"), ("b", "B"), ("out", "C")):
        assert c[f"add_acc_{port}"].values == {
            "base": p.memory.layouts[cont]["l1"].base, "temporal_bounds": [4, 8],
            "temporal_strides": [16, 64], "spatial_strides": [8],
        }  # fmt: skip
    assert c["add_acc"].values == {"n": 32}
    lay = p.memory.layouts["C"]["l1"]
    visited = addresses(c["add_acc_out"].values, [2])  # beat i * 4 + j_t, channel (a word)
    assert visited[1 * 4 + 1, 1] == lay.base + 88 == lay.address([1, 6])  # elements (1, 6:8)


@pytest.mark.parametrize(
    "name, reader, writer, n",
    [
        ("rowsum", ([16, 8], [32, 512]), ([8], [8]), 128),
        ("rowsum_serial", ([16, 4], [8, 128]), ([4], [8]), 64),
    ],
)
def test_rowsum_restarts_every_row(name, reader, writer, n):
    """rowsum on snax_forge (open item 81): A's reader runs the columns innermost (T = N / W
    beats) and the rows outer, two temporal loops; y's writer has one firing loop, the rows,
    one lane, one word per row; n = M * N / W beats and len = T = 16 (M sums)."""
    d = Design(Platform.load(SNAX_FORGE), bound_rowsum(name))
    assert run_checks(d) == []
    p = DesignPoint.of(d, name)
    assert {s.name: s.temporal_dims for s in p.streamers.values()} == {"acc_a": 2, "acc_out": 1}
    c = task_list(p).configured()
    for port, cont, (bounds, strides) in (("a", "A", reader), ("out", "y", writer)):
        assert c[f"sum_acc_{port}"].values == {
            "base": p.memory.layouts[cont]["l1"].base, "temporal_bounds": bounds,
            "temporal_strides": strides, "spatial_strides": [8],
        }  # fmt: skip
    assert c["sum_acc"].values == {"n": n, "T": 16}


def test_a_chain_of_two_nodes_orders_its_tasks_by_the_data():
    p = point(edit_graph=chained)
    tl = task_list(p)
    names = [s.task_name for s in tl.steps if s.op == "configure"]
    assert names == [
        "load_A", "load_B", "add_acc_a", "add_acc_b", "add_acc_out", "add_acc", "store_C",
        "add2_acc2_a", "add2_acc2_b", "add2_acc2_out", "add2_acc2", "store_D",
    ]  # fmt: skip
    c = tl.configured()
    assert c["add2_acc2_a"].after == ["add_acc_out"]  # C comes from the first node, in L1
    assert c["add2_acc2_b"].after == ["load_B"] and c["store_D"].after == ["add2_acc2_out"]
    assert [s.task for s in tl.steps if s.op == "sync"] == ["store_C", "store_D"]


# =============================================================================
# dot: a multiplier chained into an accumulator through L1 (D105)
# =============================================================================


def test_dot_gives_its_task_list():
    """mul, then sum after mul's writer; sum's out one beat, T = N / W; out stored as one beat."""
    tl = task_list(point("dot_accelerated"))
    names = [s.task_name for s in tl.steps if s.op == "configure"]
    assert names == [
        "load_A", "load_B", "mult_mul_a", "mult_mul_b", "mult_mul_out", "mult_mul",
        "sum_sum_a", "sum_sum_out", "sum_sum", "store_out",
    ]  # fmt: skip
    c = tl.configured()
    assert c["sum_sum_a"].after == ["mult_mul_out"]  # tmp0 comes from mul, through L1
    assert c["mult_mul"].values == {"n": 16}
    assert c["sum_sum"].values == {"n": 16, "T": 16}
    assert c["sum_sum_out"].values == {
        "base": 1024, "temporal_bounds": [1], "temporal_strides": [0], "spatial_strides": [8]
    }  # fmt: skip
    assert c["store_out"].values["src"] == {"base": 1024, "bounds": [1], "strides": [64]}
    assert [s.task for s in tl.steps if s.op == "sync"] == ["store_out"]


def test_dot_waits_once_between_its_accelerators():
    """Of the placed waits, one is on an accelerator's writer: mul_out, before sum starts."""
    p = point("dot_accelerated")
    waits = [w.component for w in placed_waits(task_list(p), p)]
    assert waits == ["dma", "dma", "mul_out", "sum_out", "dma"]


# =============================================================================
# dot 1: the two groups in one accelerator (D148), a task each
# =============================================================================


def dot1() -> DesignPoint:
    from tests.gen.helpers import bound_dot1

    d = Design(Platform.load(SNAX_FORGE), bound_dot1())
    assert run_checks(d) == []
    return DesignPoint.of(d, "dot_1")


# every temporal stride a word: a reader at stride 0 on its first loop would start (Reader.scala)
IDLE = {"base": 0, "temporal_bounds": [0], "temporal_strides": [8], "spatial_strides": [0]}


def test_dot_1_configures_every_mover_per_task_the_other_groups_idle():
    """Each group's task configures all five movers in the shell's order and starts them with
    its unit; the other group's are idle, every bound 0 (C§12)."""
    tl = task_list(dot1())
    c = tl.configured()
    starts = [s.tasks for s in tl.steps if s.op == "start" and len(s.tasks) > 1]
    assert starts == [
        ["mult_mul_a", "mult_mul_b", "mult_mul_out", "mult_sum_a", "mult_sum_out", "mult_mul"],
        ["sum_mul_a", "sum_mul_b", "sum_mul_out", "sum_sum_a", "sum_sum_out", "sum_sum"],
    ]  # fmt: skip
    for t in ("mult_sum_a", "mult_sum_out", "sum_mul_a", "sum_mul_b", "sum_mul_out"):
        assert (c[t].values, c[t].after) == (IDLE, []), t
    assert c["sum_sum_a"].after == ["mult_mul_out"]  # tmp0 from mul, through L1
    assert c["sum_sum_a"].values["base"] == c["mult_mul_out"].values["base"]
    assert (c["mult_mul"].values, c["sum_sum"].values) == ({"n": 16}, {"n": 16, "T": 16})


def test_dot_1_waits_for_mul_s_writer_before_sum_and_never_on_an_idle_mover():
    p = dot1()
    waits = placed_waits(task_list(p), p)
    assert [(w.component, sorted(w.covers)) for w in waits] == [
        ("dma", ["dma"]), ("dma", ["dma"]),
        ("mul_out", ["mul", "mul_a", "mul_b", "mul_out"]),
        ("sum_out", ["sum", "sum_a", "sum_out"]),
        ("dma", ["dma"]),
    ]  # fmt: skip


def test_dot_without_encapsulate_is_two_accelerators_with_no_idle_movers():
    """Each group its own accelerator: its tasks configure its own movers only."""
    names = [s.task_name for s in task_list(point("dot_accelerated")).steps if s.op == "configure"]
    assert not any(n.startswith(("mult_sum", "sum_mul")) for n in names)


def _backwards(g):
    """C written back to front: beat t at elements 60 - 4t .. 63 - 4t."""
    g["body"][0]["body"][0]["outputs"]["out"]["subset"] = ["60 - 4 * i_t:64 - 4 * i_t"]


def test_a_memlet_the_nest_disagrees_with_is_refused(monkeypatch):
    """An affine BRM: the addresses are right, the order is not the BRM's nest's."""
    monkeypatch.setattr(streamers, "load_brm", lambda name: affine_elementwise_add())
    with pytest.raises(LowerError, match="different orders"):
        task_list(point(edit_graph=_backwards))


def test_a_stream_brm_takes_the_memlets_order():
    """The library's elementwise_add (stream, D70): the memlet is the order, so C back to front
    lowers, its streamer starting at element 60 and stepping back a beat at a time."""
    out = task_list(point(edit_graph=_backwards)).configured()["add_acc_out"].values
    assert out["temporal_bounds"] == [16] and out["temporal_strides"] == [-32]
    assert out["spatial_strides"] == [8]


# =============================================================================
# 3. memlet_values
# =============================================================================


def test_memlet_values_through_row_and_column_major_layouts():
    m = Memlet("X", ["i_t", "0:8"])
    loops = [Loop("m", "i_t", 0, 4, 1)]
    row, sb = memlet_values(m, loops, Layout.contiguous(0, (4, 8), 8), {}, "x")
    assert row == {"base": 0, "temporal_bounds": [4], "temporal_strides": [64],
                   "spatial_strides": [8]} and sb == [8]  # fmt: skip
    col, _ = memlet_values(m, loops, Layout(1024, (4, 8), (8, 32)), {}, "x")
    assert col == {"base": 1024, "temporal_bounds": [4], "temporal_strides": [8],
                   "spatial_strides": [32]}  # fmt: skip
    assert addresses(col, sb)[1].tolist() == [1032 + 32 * k for k in range(8)]


def test_memlet_values_with_offsets_steps_and_no_loop():
    m = Memlet("X", ["2 * i + 1:2 * i + 9:2"])
    v, sb = memlet_values(m, [Loop("m", "i", 3, 2, 4)], Layout(0, (64,), (8,)), {}, "x")
    # i = 3, 7: first element 7, then 15; lanes step 2
    assert v == {"base": 56, "temporal_bounds": [2], "temporal_strides": [64],
                 "spatial_strides": [16]} and sb == [4]  # fmt: skip
    one, _ = memlet_values(Memlet("X", ["0:4"]), [], Layout(0, (64,), (8,)), {}, "x")
    assert one["temporal_bounds"] == [1] and one["temporal_strides"] == [0]


def test_memlet_values_refuse_what_is_not_affine():
    with pytest.raises(LowerError, match="not affine"):
        memlet_values(Memlet("X", ["i * i"]), [Loop("m", "i", 0, 4, 1)],
                      Layout(0, (64,), (8,)), {}, "x")  # fmt: skip
    with pytest.raises(LowerError, match="changes length"):
        memlet_values(Memlet("X", ["i:2 * i"]), [Loop("m", "i", 0, 4, 1)],
                      Layout(0, (64,), (8,)), {}, "x")  # fmt: skip


# =============================================================================
# 4. Command line
# =============================================================================


def test_cli_writes_the_task_list_beside_the_point(tmp_path, capsys):
    point(memory=PINS).save(tmp_path / "design_point.json")
    assert cli.main(["tasks", str(tmp_path / "design_point.json")]) == 0
    assert TaskList.load(tmp_path / "tasks.json") == task_list(point(memory=PINS))
    out = capsys.readouterr().out
    assert "task list of vecadd (12 steps):" in out
    assert "sync      store_C (poll)" in out


def test_cli_rejects_a_bad_point(tmp_path, capsys):
    (tmp_path / "dp.json").write_text("{}")
    assert cli.main(["tasks", str(tmp_path / "dp.json")]) == 1
    assert "[point.keys]" in capsys.readouterr().err
    assert cli.main(["other"]) == 2


@pytest.mark.parametrize(
    "first",
    ["snax_forge.design", "snax_forge.lower", "snax_forge.design.memory",
     "snax_forge.lower.layout", "snax_forge.design.point", "snax_forge.lower.derive"],
)  # fmt: skip
def test_design_and_lower_import_in_any_order(first):
    """design.memory reads lower.layout and lower reads design: no import cycle either way."""
    import subprocess
    import sys

    code = f"import {first}; import snax_forge.design, snax_forge.lower"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


def test_int32_vecadd_lowers_in_words():
    """int32 vecadd at W = 4 (D142): each beat two words (spatial stride 8, the streamer's
    num_ch W / num_slc = 2), a beat on every 16 bytes; each 64 x 4 = 256-byte container moved as
    four whole DMA beats."""

    def int32(g):
        for c in g["containers"].values():
            c["dtype"] = "int32"

    steps = {
        s.task_name: s for s in task_list(point(edit_graph=int32)).steps if hasattr(s, "task_name")
    }
    assert steps["add_acc_a"].values == {
        "base": 0, "temporal_bounds": [16], "temporal_strides": [16], "spatial_strides": [8]
    }  # fmt: skip
    assert steps["load_A"].values["src"]["bounds"] == [4]


def test_matadd_int32_4x6_in_2x2_tiles_in_words():
    """R1's accepted 2-D case (D142): the fastest spatial loop one word (8 bytes, a channel of
    2 int32), the row loop 6 x 4 bytes; the tiles step 2 columns (8 bytes) and 2 rows
    (48 bytes)."""
    m = Memlet("A", ["2 * i:2 * i + 2", "2 * j:2 * j + 2"])
    loops = [Loop("mi", "i", 0, 2, 1), Loop("mj", "j", 0, 3, 1)]
    v, sb = memlet_values(m, loops, Layout.contiguous(0, (4, 6), 4), {}, "x", num_slc=2)
    assert sb == [2, 2]
    assert v == {"base": 0, "temporal_bounds": [3, 2], "temporal_strides": [8, 48],
                 "spatial_strides": [8, 24]}  # fmt: skip
    words = (addresses(v, [1, 2]) // 8).tolist()  # 1 channel of 2 int32 by 2 rows
    assert words[:3] == [[0, 3], [1, 4], [2, 5]]  # the first row of tiles
