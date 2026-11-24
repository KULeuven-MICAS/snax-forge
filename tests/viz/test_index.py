"""The run index (viz/index.py) on hand-written events with obvious keys, and its marks.

1. each key holds the events it names, in time order; null tasks are kept; the reads by the
   cycle their data comes back
2. ids keyed by task: two runs of one component, a full key named twice, the pairing of an
   accelerator task with its streamer tasks from the task list and its refusals
3. the marks: conflicts, arrival, use, wait; which a run has, and why not
"""

import pytest

from snax_forge.lower.tasks import Configure, Start, TaskList, TaskListError
from snax_forge.run import RunError
from snax_forge.viz import index as ix

from .helpers import L1, L2, Events, tasks

# A[0..7] in L1 row 0 banks 0-7, B[0..7] at L1 + 512: row 2 banks 0-7, C at L1 + 1024


def vecadd_beat0():
    """Beat 0 of a vecadd: A and B loaded by one DMA beat each, read on lane 0 by acc_a and
    acc_b (acc_b stalled one cycle behind acc_a on bank 0), fired, pushed, written, stored."""
    ev = Events()
    ev.task(0, "acc_a", "add_acc_a", 20)
    ev.task(1, "acc", "add_acc", 21)
    ev.burst(2, 2, "r", L2, 512, "load_A")
    ev.dma(3, 2, 0, "w", L1, L2, "load_A")
    ev.burst(4, 3, "r", L2 + 512, 512, "load_B")
    ev.dma(5, 3, 0, "w", L1 + 512, L2 + 512, "load_B")
    ev.grant(10, "acc_a.lane0", L1, b=0, task="add_acc_a")
    ev.stall(10, "acc_b.lane0", L1 + 512, "acc_a.lane0", L1, b=0, task="add_acc_b")
    ev.grant(11, "acc_b.lane0", L1 + 512, b=0, task="add_acc_b")
    ev.fire(13, 0, {"acc_a": 0, "acc_b": 0}, {"acc_a": [1], "acc_b": [100]})
    ev.push(13, 0, 0, 0, [101])
    ev.grant(14, "acc_out.lane0", L1 + 1024, op="w", b=0, task="add_acc_out")
    ev.dma(30, 4, 0, "r", L1 + 1024, L2 + 1024, "store_C")
    ev.burst(30, 4, "w", L2 + 1024, 64, "store_C")
    ev.add({"t": 40, "k": "fifo", "src": "acc_a", "task": None, "lane": 0, "count": 0})
    return ev


# =============================================================================
# 1. The keys
# =============================================================================


def test_a_word_has_every_access_that_covers_it():
    idx = vecadd_beat0().index()
    # the DMA beat covers words 0..7 of A in L1, the burst all of A in L2
    for j in range(8):
        assert [e["k"] for e in idx.word[("l1", L1 + 8 * j)]][:1] == ["dma"]
    assert [e["k"] for e in idx.word[("l1", L1)]] == ["dma", "grant"]
    assert [e["k"] for e in idx.word[("l1", L1 + 512)]] == ["dma", "stall", "grant"]
    assert len([k for k in idx.word if k[0] == "l2" and L2 <= k[1] < L2 + 512]) == 64
    assert ("l1", L1 + 64) not in idx.word  # no access touched A[8]'s word in L1


# =============================================================================
# 2. Ids keyed by task
# =============================================================================


def test_a_full_key_named_twice_is_refused_naming_both_lines():
    ev = Events()
    ev.fire(5, 0, {"acc_a": 0}, {"acc_a": [1]}, task="add_acc")
    ev.fire(50, 0, {"acc_a": 0}, {"acc_a": [2]}, task="add_acc")
    with pytest.raises(RunError, match=r"firing \('add_acc', 0\).*t 5 fire.*t 50 fire"):
        ev.index()
    ev = Events()
    ev.grant(5, "acc_a.lane0", L1, b=0, task="add_acc_a")
    ev.grant(9, "acc_a.lane0", L1 + 8, b=0, task="add_acc_a")
    with pytest.raises(RunError, match="add_acc_a acc_a beat 0 lane 0.*t 5 grant.*t 9 grant"):
        ev.index()


def test_a_start_with_two_streamer_tasks_for_one_mover_is_refused():
    """The task list's own check refuses it when the list is made; the index refuses a list
    whose steps changed after that check."""
    with pytest.raises(TaskListError):
        TaskList("bad", [*tasks().steps[:4], Configure("x_acc_a", "streamer", "acc_a", {}),
                         *tasks().steps[4:]])  # fmt: skip
    tl = tasks()
    start = next(s for s in tl.steps if isinstance(s, Start) and "add_acc" in s.tasks)
    tl.steps.insert(tl.steps.index(start), Configure("x_acc_a", "streamer", "acc_a", {}))
    start.tasks.append("x_acc_a")
    with pytest.raises(RunError, match=r"'add_acc' has 2 streamer tasks for acc_a "
                                       r"\(\['add_acc_a', 'x_acc_a'\]\)"):  # fmt: skip
        Events().index(task_list=tl)


# =============================================================================
# 3. Marks
# =============================================================================


def test_marks_are_registered_once_and_named():
    with pytest.raises(ValueError, match="registered already"):
        ix.register_mark("use", "beat", lambda idx: {})
    with pytest.raises(ValueError, match="kind"):
        ix.register_mark("late", "beat", lambda idx: {}, "colour")
    with pytest.raises(ValueError, match="must be one of"):
        ix.mark(Events().index(), "later")


def test_a_registered_mark_is_offered(monkeypatch):
    monkeypatch.setattr(ix, "MARKS", dict(ix.MARKS))
    ix.register_mark("bank", "beat", lambda idx: {k: idx.cluster.bank_row(k[1])[0]
                                                  for k in idx.word if k[0] == "l1"}, "count")  # fmt: skip
    idx = vecadd_beat0().index("beat")
    assert "bank" in ix.marks_of(idx)[0] and ix.mark(idx, "bank")[("l1", L1 + 520)] == 1
