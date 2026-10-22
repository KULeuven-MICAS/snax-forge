"""The schedule's queries (viz/schedule.py) on hand-written events with obvious answers.

1. segments: a quiet stretch of 16 cycles or more folds, a shorter one stays open; a DMA copy
   keeps its cycles open; ``open`` and ``cycle`` open a fold; the segments cover the window;
   a fold's per-core counts add up to its length; the boot
2. lanes: one per core, the DMA, each mover, the accelerator; bars with their totals and,
   from ``beat``, their class runs; the recorded run's function band from 0 in main
3. detail: the beat and stall lines of a range per lane
4. below the level: the reason
"""

from itertools import pairwise

import pytest

from snax_forge.run import RunOutputs
from snax_forge.viz import api
from snax_forge.viz import schedule as sc
from snax_forge.viz.index import RunIndex

from .helpers import L1, Events, cluster, recorded_run, regions, tasks

TWO = tasks("add", "add2")


def two_tasks(gap_from=20, gap_to=60, total=81, level="task"):
    """The accelerator's task add_acc from 0 to ``gap_from``, add2_acc from ``gap_to`` to 80."""
    ev = Events()
    ev.task(0, "acc", "add_acc", gap_from)
    ev.task(gap_to, "acc", "add2_acc", 80)
    return ev


def plain(segs):
    return [(s["from"], s["to"], s["folded"]) for s in segs["segments"]]


# =============================================================================
# 1. Segments
# =============================================================================


def test_two_tasks_40_cycles_apart_give_one_fold():
    """Active 0..20 and 60..80 (start to done, both in): quiet 21..59, 39 cycles, folds."""
    segs = sc.segments(two_tasks().index("task", task_list=TWO, total=81))
    assert plain(segs) == [(0, 21, False), (21, 60, True), (60, 81, False)]
    assert [s["foldable"] for s in segs["segments"]] == [False, True, False]


def test_a_10_cycle_gap_stays_open():
    segs = sc.segments(two_tasks(20, 31).index("task", task_list=TWO, total=81))
    assert plain(segs) == [(0, 81, False)]


def test_a_dma_copy_alone_keeps_its_cycles_open():
    """Without the copy 21..59 folds; with it issued at 30 and done at 45, 30..45 stay open
    and the stretches around it, 9 and 14 cycles, are too short to fold."""
    ev = two_tasks()
    ev.copy(30, 2, "load_A", 45)
    segs = sc.segments(ev.index("task", task_list=TWO, total=81))
    assert plain(segs) == [(0, 81, False)]
    ev = two_tasks(10, 70)
    ev.copy(30, 2, "load_A", 45)
    segs = sc.segments(ev.index("task", task_list=TWO, total=81))
    assert plain(segs) == [(0, 11, False), (11, 30, True), (30, 46, False), (46, 70, True),
                           (70, 81, False)]  # fmt: skip


def test_open_and_cycle_open_a_fold():
    idx = two_tasks().index("task", task_list=TWO, total=81)
    assert plain(sc.segments(idx, open=[21])) == [(0, 21, False), (21, 60, False), (60, 81, False)]
    assert plain(sc.segments(idx, open=[22]))[1] == (21, 60, True)  # only its first cycle
    assert plain(sc.segments(idx, cycle=40))[1] == (21, 60, False)
    assert plain(sc.segments(idx, cycle=60))[1] == (21, 60, True)
    opened = sc.segments(idx, open=[21])["segments"][1]
    assert opened["foldable"] and "cores" not in opened


def test_the_segments_cover_the_window_with_no_overlap():
    ev = two_tasks(10, 70)
    ev.copy(30, 2, "load_A", 45)
    for open_ in ([], [11], [11, 46]):
        segs = sc.segments(ev.index("task", task_list=TWO, total=81), open=open_)["segments"]
        assert segs[0]["from"] == 0 and segs[-1]["to"] == 81
        assert all(a["to"] == b["from"] for a, b in pairwise(segs))
        assert all(s["from"] < s["to"] for s in segs)
        assert all(s["to"] - s["from"] >= sc.FOLD_MIN for s in segs if s["folded"])


def test_a_folds_core_counts_add_up_to_its_length():
    """Fold 21..59 (39 cycles): core0 runs 21-24, sleeps 25-29, polls 30-33 (4 reads), sleeps
    34-44, runs 45-59; core1 runs throughout, entering forge_core_1 at 50."""
    ev = two_tasks()
    ev.core(25, "state", mode="sleep")
    ev.core(30, "poll", pc=0x80000378, csr=0x3D4, block="acc", value=1, count=4, last=33)
    ev.core(45, "state", mode="run")
    ev.core(50, "func", "core1", name="forge_core_1", pc=0x800003A4)
    ev.csr(52, 0x3C0, 7)
    (fold,) = [s for s in sc.segments(ev.index("task", task_list=TWO, total=81))["segments"]
               if s["folded"]]  # fmt: skip
    c0, c1 = fold["cores"]["core0"], fold["cores"]["core1"]
    assert c0["cycles"] == {"run": 19, "poll": 4, "barrier": 0, "sleep": 16}
    assert c1["cycles"] == {"run": 39, "poll": 0, "barrier": 0, "sleep": 0}
    for c in (c0, c1):
        assert sum(c["cycles"].values()) == fold["to"] - fold["from"]
    assert (c0["csr_reads"], c0["csr_writes"]) == (4, 1)
    assert c1["funcs"] == [None, "forge_core_1"]


def test_the_boot_is_a_fold_that_never_opens():
    idx = two_tasks().index("task", task_list=TWO, total=81)
    idx.boot = 2812
    boot = sc.segments(idx, open=[-2812], cycle=-5)["boot"]
    assert boot == {"from": -2812, "to": 0, "cycles": 2812, "folded": True}


# =============================================================================
# 2. Lanes
# =============================================================================


def test_one_lane_per_core_the_dma_each_mover_and_the_accelerator():
    ev = two_tasks()
    ev.task(0, "acc_a", "add_acc_a", 19)
    ev.copy(1, 2, "load_A", 9)
    lanes = sc.lanes(ev.index("task", task_list=TWO, total=81))["lanes"]
    assert [(x["name"], x["kind"]) for x in lanes] == [
        ("core0", "core"), ("core1", "core"), ("dma", "dma"), ("acc_a", "mover"),
        ("acc_b", "mover"), ("acc_out", "mover"), ("acc", "accelerator")]  # fmt: skip
    dma = lanes[2]["bars"]
    assert dma == [{"task": "load_A", "id": 2, "from": 1, "to": 9, "dir": "l2_to_l1",
                    "bytes": 512}]  # fmt: skip
    (a,) = lanes[3]["bars"]
    assert (a["task"], a["from"], a["to"], a["cycles"]["busy"]) == ("add_acc_a", 0, 19, 18)
    assert "classes" not in a  # below beat
    assert [b["task"] for b in lanes[-1]["bars"]] == ["add_acc", "add2_acc"]
    assert lanes[0]["spans"] == [{"from": 0, "to": 81, "class": "run"}]


def test_a_bar_has_its_class_runs_from_beat():
    """add_acc from 0 to 20 counts 19 cycles: stall_in 1-3, busy 4-17, flush 18-19."""
    ev = two_tasks()
    ev.events = [e for e in ev.events if e.get("task") != "add_acc"]
    ev.task(0, "acc", "add_acc", 20, {"busy": 14, "stall_in": 3, "stall_out": 0, "flush": 2})
    for t, mode in ((1, "stall_in"), (4, "busy"), (18, "flush")):
        ev.add({"t": t, "k": "class", "src": "acc", "task": "add_acc", "mode": mode})
    lanes = sc.lanes(ev.index("beat", task_list=TWO, total=81))["lanes"]
    bar = lanes[-1]["bars"][0]
    assert bar["classes"] == [{"from": 1, "to": 4, "mode": "stall_in"},
                              {"from": 4, "to": 18, "mode": "busy"},
                              {"from": 18, "to": 20, "mode": "flush"}]  # fmt: skip
    assert bar["cycles"] == {"busy": 14, "stall_in": 3, "stall_out": 0, "flush": 2}


def test_the_recorded_runs_function_band_starts_at_0_in_main(tmp_path):
    rv = api.load_run(recorded_run(tmp_path / "rec", "task"), "rec")
    for lane in sc.lanes(rv.index)["lanes"][:2]:
        first, second = lane["funcs"][:2]
        assert (first["from"], first["name"]) == (0, "main")
        assert first["to"] == second["from"] and second["name"].startswith("forge_core_")


# =============================================================================
# 3. Detail
# =============================================================================


def test_detail_groups_a_ranges_lines_per_lane():
    ev = two_tasks()
    ev.grant(5, "acc_a.lane0", L1, b=0, task="add_acc_a")
    ev.stall(5, "acc_b.lane0", L1 + 512, "acc_a.lane0", L1, b=0, task="add_acc_b")
    ev.dma(6, 2, 0, "w", L1, 0x80000000, "load_A")
    ev.fire(7, 0, {"acc_a": 0, "acc_b": 0}, {"acc_a": [1], "acc_b": [100]})
    ev.add({"t": 8, "k": "fifo", "src": "acc_a", "task": "add_acc_a", "lane": 0, "count": 1})
    ev.grant(30, "acc_a.lane0", L1 + 8, b=1, task="add_acc_a")
    d = sc.detail(ev.index("stall", task_list=TWO, total=81), 0, 10)
    assert {p: [e["k"] for e in es] for p, es in d["ports"].items()} == {
        "acc_a.lane0": ["grant"], "acc_b.lane0": ["stall"]}  # fmt: skip
    assert [e["k"] for e in d["dma"]] == ["dma"] and [e["k"] for e in d["acc"]] == ["fire"]
    assert [e["count"] for e in d["fifo"]["acc_a.lane0"]] == [1]
    assert d["ports"]["acc_a.lane0"][0]["vals"] == [1]  # as the converter wrote it


# =============================================================================
# 4. Below the level
# =============================================================================


def test_at_off_there_is_no_schedule_and_the_reason():
    idx = RunIndex.from_run(RunOutputs({}, None, None), cluster(), regions(), TWO)
    with pytest.raises(sc.ScheduleUnavailable, match="needs --trace task; this run was traced "
                       "at off"):  # fmt: skip
        sc.schedule(idx)


def test_below_beat_there_is_no_detail():
    with pytest.raises(sc.ScheduleUnavailable, match="needs --trace beat"):
        sc.detail(two_tasks().index("task", task_list=TWO, total=81), 0, 10)
