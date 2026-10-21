"""The run index (viz/index.py) on hand-written events with obvious keys, and its marks.

1. each key holds the events it names, in time order; null tasks are kept
2. a key named twice (two runs of one component) unlinks the ids
3. the marks: conflicts, arrival, use, wait; which a run has, and why not
"""

import pytest

from snax_forge.viz import index as ix

from .helpers import L1, L2, Events

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


def test_an_element_has_its_events_in_both_memories_in_time_order():
    idx = vecadd_beat0().index()
    got = [(e["t"], e["k"], e.get("mem", "l1"), e.get("op")) for e in idx.elem[("A", 0)]]
    assert got == [(2, "burst", "l2", "r"), (3, "dma", "l1", "w"), (10, "grant", "l1", "r")]
    assert [e["k"] for e in idx.elem[("C", 0)]] == ["grant", "dma", "burst"]
    assert [e["k"] for e in idx.elem[("B", 0)]] == ["burst", "dma", "stall", "grant"]
    assert [e["k"] for e in idx.elem[("A", 8)]] == ["burst"]  # its DMA beat was not traced


def test_a_word_has_every_access_that_covers_it():
    idx = vecadd_beat0().index()
    # the DMA beat covers words 0..7 of A in L1, the burst all of A in L2
    for j in range(8):
        assert [e["k"] for e in idx.word[("l1", L1 + 8 * j)]][:1] == ["dma"]
    assert [e["k"] for e in idx.word[("l1", L1)]] == ["dma", "grant"]
    assert [e["k"] for e in idx.word[("l1", L1 + 512)]] == ["dma", "stall", "grant"]
    assert len([k for k in idx.word if k[0] == "l2" and L2 <= k[1] < L2 + 512]) == 64
    assert ("l1", L1 + 64) not in idx.word  # no access touched A[8]'s word in L1


def test_the_ids():
    idx = vecadd_beat0().index()
    assert [e["src"] for e in idx.beat[("acc_a", 0)].values()] == ["acc_a.lane0"]
    assert list(idx.beat[("acc_out", 0)]) == [0]  # by lane
    assert idx.fire[0]["b"] == {"acc_a": 0, "acc_b": 0} and idx.push[0]["vals"] == [101]
    assert idx.folded[0] is idx.push[0]
    assert idx.dma[(2, 0)]["elems"] == list(range(8)) and idx.dma[(4, 0)]["op"] == "r"
    assert [e["op"] for e in idx.burst[4]] == ["w"] and idx.burst[2][0]["size"] == 512
    assert [(e["src"], e["task"]) for e in idx.stalls_bank[0]] == [("acc_b.lane0", "add_acc_b")]
    assert list(idx.stalls_port) == ["acc_b.lane0"]
    assert [e["t"] for e in idx.grants_port["acc_a.lane0"]] == [10]
    assert idx.start["add_acc"]["t"] == 1 and idx.done["add_acc_a"]["t"] == 20
    assert idx.linked is None and idx.filtered is None


def test_a_null_task_event_is_kept():
    idx = vecadd_beat0().index()
    assert {"t": 40, "k": "fifo", "src": "acc_a", "task": None, "lane": 0, "count": 0} in idx.events
    ev = Events()
    ev.grant(5, "acc_a.lane0", L1, b=0, task=None)
    idx = ev.index()
    assert idx.beat[("acc_a", 0)][0]["task"] is None and idx.elem[("A", 0)][0]["t"] == 5


def test_a_push_folds_its_firings():
    ev = Events()
    for n in range(4):
        ev.fire(n, n, {"acc_a": n}, {"acc_a": [n]})
    ev.push(4, 0, 0, 3, [6])
    idx = ev.index()
    assert all(idx.folded[n] is idx.push[0] for n in range(4)) and 4 not in idx.folded


def test_no_trace_gives_an_empty_index():
    from snax_forge.run import RunOutputs

    ev = Events()
    idx = ix.RunIndex.from_run(RunOutputs({}, None, None), ev.c, ev.regions)
    assert idx.level == "off" and idx.events == [] and dict(idx.word) == {}


def test_a_filter_without_the_dma_is_said():
    idx = vecadd_beat0().index(filter=["acc", "acc_a"])
    assert idx.filtered == {"sources": ["acc", "acc_a"]}


# =============================================================================
# 2. Two runs of one component
# =============================================================================


def test_a_key_named_twice_unlinks_the_ids():
    ev = Events()
    ev.fire(5, 0, {"acc_a": 0}, {"acc_a": [1]}, task="add_acc")
    ev.fire(50, 0, {"acc_a": 0}, {"acc_a": [2]}, task="add2_acc")
    idx = ev.index()
    assert idx.linked is not None and "firing 0" in idx.linked


# =============================================================================
# 3. Marks
# =============================================================================


def test_conflict_marks_count_the_wanted_and_the_served_word():
    ev = vecadd_beat0()
    ev.stall(11, "acc_b.lane1", L1 + 520, "dma", L1 + 8, b=0, task="add_acc_b")
    got = ix.mark(ev.index(), "conflicts")
    assert got == {("l1", L1 + 512): 1, ("l1", L1): 1, ("l1", L1 + 520): 1, ("l1", L1 + 8): 1}


def test_arrival_use_and_wait():
    idx = vecadd_beat0().index("beat")
    arrival, use, wait = (ix.mark(idx, m) for m in ("arrival", "use", "wait"))
    assert arrival[("l1", L1)] == 3 and use[("l1", L1)] == 10 and wait[("l1", L1)] == 7
    assert arrival[("l1", L1 + 512)] == 5 and wait[("l1", L1 + 512)] == 6
    # C: written by acc_out at 14, then read by the DMA only: no use
    assert arrival[("l1", L1 + 1024)] == 14 and ("l1", L1 + 1024) not in use
    # in L2, A was there from the start; C arrives with the store's burst
    assert ("l2", L2) not in arrival and arrival[("l2", L2 + 1024)] == 30
    assert ("l2", L2) not in use and ("l2", L2) not in wait


def test_a_read_before_the_arrival_is_not_its_use():
    ev = Events()
    ev.grant(2, "acc_a.lane0", L1, b=0)
    ev.dma(5, 2, 0, "w", L1, L2)
    ev.grant(9, "acc_a.lane0", L1, b=1)
    idx = ev.index("beat")
    assert ix.first_use(idx, ("l1", L1)) == 9 and ix.mark(idx, "wait")[("l1", L1)] == 4


@pytest.mark.parametrize("level, have", [
    ("task", []), ("beat", ["arrival", "use", "wait"]),
    ("stall", ["conflicts", "arrival", "use", "wait"]),
    ("finegrained", ["conflicts", "arrival", "use", "wait"]),
])  # fmt: skip
def test_the_marks_a_run_has(level, have):
    idx = Events().index(level)
    got, lack = ix.marks_of(idx)
    assert got == have and sorted(lack) == sorted(set(ix.MARKS) - set(have))
    for name, why in lack.items():
        assert f"needs --trace {ix.MARKS[name].level}" in why and f"traced at {level}" in why
        with pytest.raises(ValueError, match="needs --trace"):
            ix.mark(idx, name)


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
