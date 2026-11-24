"""The cluster view's queries (viz/cluster.py) on hand-written events with obvious answers.

1. at a cycle: a grant's element and value, on its lane and its bank; a read's data at its
   ``back``; a FIFO count held between two ``fifo`` lines; a mover's class inside a class
   run; the DMA's open copy, its beat and its read data back from L1 and L2; only that
   cycle's lines; what a level lacks
2. stepping: it skips empty cycles, goes both ways, stops at the window's ends
3. recorded runs: an accelerator of two groups (dot 1), a shared mover's port mux (dot 1
   shared), a linked group at task (dot 4), a group of two outputs (bicg_v3)
"""

import pytest

from snax_forge.viz import cluster as cv

from .helpers import L1, L2, Events


def run():
    """add_acc_a from 0 to 20: A[1] (value 2) granted on acc_a.lane0 at 10, its data back at
    12; the lane's queue 2 from 5, 0 from 9; acc_a busy 1-4, stall_xbar 5-9, idle 10-19; a
    copy of A issued at 2, its first beat at 3, done at 6; acc fires at 13."""
    ev = Events()
    ev.task(0, "acc_a", "add_acc_a", 20, {"busy": 4, "stall_xbar": 5, "stall_fifo": 0, "idle": 10})
    ev.task(0, "acc", "add_acc", 20)
    for t, mode in ((1, "busy"), (5, "stall_xbar"), (10, "idle")):
        ev.add({"t": t, "k": "class", "src": "acc_a", "task": "add_acc_a", "mode": mode})
    ev.add({"t": 1, "k": "class", "src": "acc", "task": "add_acc", "mode": "busy"})
    ev.grant(10, "acc_a.lane0", L1 + 8, b=1, task="add_acc_a", back=12)
    ev.stall(10, "acc_b.lane1", L1 + 8, "acc_a.lane0", L1 + 8, b=1, task="add_acc_b")
    for t, n in ((5, 2), (9, 0)):
        ev.add({"t": t, "k": "fifo", "src": "acc_a", "task": "add_acc_a", "lane": 0, "count": n})
    ev.copy(2, 2, "load_A", 6)
    ev.dma(3, 2, 0, "w", L1, L2, "load_A")
    ev.dma(4, 2, 1, "w", L1 + 64, L2 + 64, "load_A")
    ev.fire(13, 0, {"acc_a": 0, "acc_b": 0}, {"acc_a": [1], "acc_b": [100]})
    return ev


# =============================================================================
# 1. At a cycle
# =============================================================================


def test_a_fifo_count_holds_between_its_lines():
    idx = run().index("stall", total=30)
    got = [cv.cluster_at(idx, t)["movers"]["acc_a.lane0"]["fifo"] for t in range(3, 11)]
    assert got == [0, 0, 2, 2, 2, 2, 0, 0]
    assert cv.cluster_at(idx, 7)["movers"]["acc_a.lane1"]["fifo"] == 0


# =============================================================================
# 2. Stepping
# =============================================================================


def test_a_bad_step_is_refused():
    with pytest.raises(ValueError):
        cv.step(run().index("stall", total=30), 0, 2, "event")


# =============================================================================
# The recorded runs: two groups on shared movers, two sets, a group of two outputs
# =============================================================================


def _recorded(tmp_path_factory, fixture, level):
    from snax_forge.viz import api
    from tests.run.helpers import RECORDED

    from .helpers import recorded_run

    name = f"{fixture}_{level}"
    out = recorded_run(tmp_path_factory.mktemp(fixture) / name, level, RECORDED / fixture)
    return api.load_run(out, name)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    """mvt_v1_2_2 at stall: one accelerator of two groups, mvt and vadd, on shared movers."""
    return _recorded(tmp_path_factory, "mvt_v1_2_2", "stall")


@pytest.fixture(scope="module")
def two_sets(tmp_path_factory):
    """mvt_v5_sequential at task: two accelerators, mvt and mac_x2, each its own set."""
    return _recorded(tmp_path_factory, "mvt_v5_sequential", "task")


def test_the_cluster_view_names_the_unit_that_fires(shared):
    """At an mvt firing ``acc`` is mvt, its task and fire its own, vadd's entry quiet; at a vadd
    firing the other way round."""
    idx = shared.index
    for unit, other, task in (
        ("mvt", "vadd", "tmp0_mult_mvt"),
        ("vadd", "mvt", "augassign_18_4_vadd"),
    ):
        t = next(e["t"] for e in idx.events if e["k"] == "fire" and e["src"] == unit)
        a = cv.cluster_at(idx, t)["acc"]
        assert (a["name"], a["accelerator"], a["fire"]["src"]) == (unit, "mvt_vadd", unit)
        units = {u["name"]: u for u in a["units"]}
        assert units[other]["task"] is None and units[other]["fire"] is None
        assert units[unit]["task"] == task


def test_the_schedule_splits_the_detail_by_unit(shared):
    from snax_forge.viz import schedule

    idx = shared.index
    bars = [b for lane in schedule.lanes(idx)["lanes"] for b in lane.get("bars", []) if "idle" in b]
    assert bars and not any(b["idle"] for b in bars)  # every mover is used by both groups
    d = schedule.detail(idx, 0, idx.total_cycles)
    assert {u: {e["src"] for e in es} for u, es in d["units"].items()} == {
        "mvt": {"mvt"}, "vadd": {"vadd"}}  # fmt: skip
