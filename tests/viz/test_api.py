"""Tests for the visualiser's API functions (VIS1, D55, D56).

The report itself is HTML and is checked by eye (D55); what is tested here
is what the viewer is given:

  1. a run loads as read_outputs gives it
  2. the events window is exactly [A, B), of the asked sources and kinds
  3. the FIFO busy window equals a direct count from beat-level fifo events
  4. each DMA task's direction matches the memory its source beats read (D58)
"""

import pytest

from snax_forge.snax_model.scenario import ClusterConfig, read_outputs
from snax_forge.viz import api

from .helpers import run_dir


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    """vecadd and vecadd_conflict at beat level, vecadd at task and off."""
    base = tmp_path_factory.mktemp("runs")
    return {
        (name, level): run_dir(base / f"{name}-{level}", name, level)
        for name, level in [
            ("vecadd", "beat"),
            ("vecadd_conflict", "beat"),
            ("vecadd", "task"),
            ("vecadd", "off"),
        ]
    }


# =============================================================================
# 1. Loading
# =============================================================================


@pytest.mark.parametrize("level", ["beat", "off"])
def test_run_loads_as_read_outputs(dirs, level):
    d = dirs[("vecadd", level)]
    rv, ref = api.load_run(d), read_outputs(d)
    assert rv.name == d.name
    assert rv.outputs.run == ref.run and rv.outputs.profile == ref.profile
    assert (rv.outputs.l1 == ref.l1).all() and (rv.outputs.l2 == ref.l2).all()
    assert rv.cluster == ClusterConfig.from_dict(ref.run["cluster"])
    if level == "off":
        assert rv.outputs.trace is None and rv.events == []
    else:
        assert rv.outputs.trace.to_dict() == ref.trace.to_dict()
        assert rv.events == [e.to_dict() for e in ref.trace.events]


def test_same_directory_names_get_suffixes(tmp_path):
    assert api.run_names([tmp_path / "a" / "x", tmp_path / "b" / "x", tmp_path / "y"]) == [
        "x",
        "x-2",
        "y",
    ]


# =============================================================================
# 2. Events window
# =============================================================================


@pytest.mark.parametrize(
    "a, b, srcs, kinds",
    [
        (0, None, None, None),
        (20, 60, None, None),
        (20, 60, ["ra", "ctl"], None),
        (60, 20, None, None),
        (70, 71, ["xbar"], None),
        (0, None, None, ["cmd", "start", "done"]),  # the schedule's task events (D57)
        (20, 80, ["xbar"], ["stall"]),
    ],
)
def test_events_window(dirs, a, b, srcs, kinds):
    rv = api.load_run(dirs[("vecadd", "beat")])
    got = api.events_window(rv, a, b, srcs, kinds)
    stop = 10**9 if b is None else b
    want = [
        e
        for e in rv.events
        if a <= e["t"] < stop
        and (srcs is None or e["src"] in srcs)
        and (kinds is None or e["k"] in kinds)
    ]
    assert got == want
    if (a, b) == (20, 60):
        assert got  # the window is not trivially empty


# =============================================================================
# 3. FIFO busy window (D56)
# =============================================================================


def direct_count(rv, fifo, lanes, window):
    """Per lane, cycles per count over ``window``, replayed from the fifo events."""
    changes = [e for e in rv.events if e["k"] == "fifo" and e["src"] == fifo]
    hists = []
    for lane in range(lanes):
        at = {e["t"]: e["count"] for e in changes if e["lane"] == lane}
        count, hist = 0, {}
        for t in range(rv.total_cycles):
            count = at.get(t, count)
            if any(a <= t < b for a, b in window):
                hist[count] = hist.get(count, 0) + 1
        hists.append(hist)
    return hists


@pytest.mark.parametrize("name", ["vecadd", "vecadd_conflict"])
def test_fifo_window_equals_direct_count(dirs, name):
    rv = api.load_run(dirs[(name, "beat")])
    res = api.fifo_windows(rv)
    assert res["available"] and set(res["streamers"]) == {"ra", "rb", "wr"}
    for s, win in res["streamers"].items():
        assert win["reason"] is None and win["owners"] == [s, "acc"]
        f = rv.outputs.profile.streamers[s].fifo
        want = direct_count(rv, f.name, len(f.hist), win["window"])
        for lane, got in enumerate(win["lanes"]):
            assert {c: n for c, n in enumerate(got["hist"]) if n} == want[lane]
            assert sum(got["hist"]) == win["window_cycles"]
            n = win["window_cycles"]
            assert got["mean"] == pytest.approx(sum(c * k for c, k in want[lane].items()) / n)


def test_fifo_window_needs_task_events(dirs):
    task = api.fifo_windows(api.load_run(dirs[("vecadd", "task")]))
    beat = api.fifo_windows(api.load_run(dirs[("vecadd", "beat")]))
    assert task == beat  # start and done are task events: beat events add nothing
    off = api.fifo_windows(api.load_run(dirs[("vecadd", "off")]))
    assert not off["available"] and off["streamers"] == {} and "off" in off["reason"]


# =============================================================================
# 4. DMA task directions (D58)
# =============================================================================


@pytest.mark.parametrize("name", ["vecadd", "vecadd_conflict"])
def test_dma_direction_matches_source_beats(dirs, name):
    """The direction from the register writes equals where the task's reads went."""
    rv = api.load_run(dirs[(name, "beat")])
    tasks = api.dma_tasks(rv)
    assert set(tasks) == {c.name for c in rv.cluster.components if c.kind == "dma"}
    seen = set()
    for dma, ts in tasks.items():
        assert [(t["start"], t["done"]) for t in ts] == api.task_spans(rv, dma)
        for t in ts:
            src = {
                e["mem"]
                for e in api.events_window(rv, t["start"], t["done"] + 1, [dma], ["dma_beat"])
                if e["side"] == "src"
            }
            assert src == {t["direction"].split("_to_")[0]}
            seen.add(t["direction"])
    if name == "vecadd_conflict":
        assert seen == {"l2_to_l1", "l1_to_l2"}  # both directions are exercised


def test_dma_direction_needs_task_events(dirs):
    task = api.dma_tasks(api.load_run(dirs[("vecadd", "task")]))
    assert task == api.dma_tasks(api.load_run(dirs[("vecadd", "beat")]))
    off = api.dma_tasks(api.load_run(dirs[("vecadd", "off")]))
    assert off and all(ts == [] for ts in off.values())
