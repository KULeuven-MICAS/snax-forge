"""The schedule's queries (viz/schedule.py) on hand-written events with obvious answers.

1. segments: a quiet stretch of 16 cycles or more folds, a shorter one stays open; a DMA copy
   keeps its cycles open; ``open`` and ``cycle`` open a fold; the segments cover the window;
   a fold's per-core counts add up to its length; the boot
2. lanes: one per core, the DMA, each mover, the accelerator; bars with their totals and,
   from ``beat``, their class runs; the recorded run's function band from 0 in main
3. detail: the beat and stall lines of a range per lane
4. below the level: the reason
"""

from snax_forge.viz import schedule as sc

from .helpers import Events, tasks

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


# =============================================================================
# 2. Lanes
# =============================================================================


# =============================================================================
# 3. Detail
# =============================================================================


# =============================================================================
# 4. Below the level
# =============================================================================
