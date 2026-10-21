"""The class part of the converter (C§7, FLOW3 round 3): class lines against their run's totals
on hand-written snippets with obvious answers, the totals on ``done`` from the summary lines,
and a core's spans (run/cores.py).

1. class runs: a run of busy, stall_in, busy covers its span; a gap, a sum off by one, a
   repeated class, a line outside every run are each named
2. the totals on done: an accelerator's and a mover's run from its summary line; a run
   without one is a named error; the class lines become events with their task
3. a core's spans: barrier > poll > sleep > run, a state before the window, the cover
"""

import json
from pathlib import Path

import pytest

from snax_forge.lower.tasks import TaskList
from snax_forge.run import RunError, cluster_from, core_counts, core_spans, parse_line
from snax_forge.run.beats import class_invariants, class_runs
from snax_forge.run.convert import convert

from .helpers import files, synthetic

TASKS = Path(__file__).parent / "fixtures" / "vecadd" / "tasks.json"
UNIT = {"busy": 6, "stall_in": 3, "stall_out": 0, "flush": 0}


def run(*lines, cycles=None, start=10):
    """The accelerator's run ``add_acc`` started at ``start``, its ``done`` with ``cycles``,
    and its class lines ``(t, mode)``."""
    cycles = dict(UNIT if cycles is None else cycles)
    end = start + 1 + sum(cycles.values())
    events = [{"t": start, "k": "start", "src": "acc", "task": "add_acc"},
              {"t": end, "k": "done", "src": "acc", "task": "add_acc", "cycles": cycles}]  # fmt: skip
    classes = [{"t": t, "k": "class", "src": "acc", "task": "add_acc", "mode": m}
               for t, m in lines]  # fmt: skip
    return events + classes, classes


# =============================================================================
# 1. Class runs
# =============================================================================


def test_busy_stall_in_busy_covers_its_span():
    """Start at 10: cycles 11 to 19 counted, busy 11-12, stall_in 13-15, busy 16-19."""
    events, classes = run((11, "busy"), (13, "stall_in"), (16, "busy"))
    assert class_runs(classes, 20) == [(11, 13, "busy"), (13, 16, "stall_in"), (16, 20, "busy")]
    assert class_invariants(events, classes, ["acc"]) == []


@pytest.mark.parametrize(
    "lines, cycles, says",
    [
        # a gap at the start: the first counted cycle is 11
        ([(12, "busy"), (13, "stall_in"), (16, "busy")], None, "no class line at its first cycle 11"),
        # a sum off by one: busy 7 cycles where the summary says 6
        ([(11, "busy"), (13, "stall_in"), (15, "busy")], None, "class runs {'busy': 7, 'stall_in': 2"),
        ([(11, "busy"), (13, "stall_in"), (16, "busy")], {**UNIT, "busy": 5, "stall_in": 4},
         "class runs {'busy': 6, 'stall_in': 3"),
        # two lines of one class in a row, and two lines in one cycle
        ([(11, "busy"), (13, "busy"), (16, "stall_in")], None, "overlap or repeat a class"),
        ([(11, "busy"), (13, "stall_in"), (13, "busy")], None, "overlap or repeat a class"),
        # a class the summary has no count for
        ([(11, "busy"), (13, "idle")], None, "class idle at 13, not one of"),
    ],
)  # fmt: skip
def test_a_broken_class_run_is_named(lines, cycles, says):
    events, classes = run(*lines, cycles=cycles)
    bad = class_invariants(events, classes, ["acc"])
    assert len(bad) == 1 and says in bad[0], bad


def test_a_class_line_outside_every_run_is_named():
    events, classes = run((11, "busy"), (13, "stall_in"), (16, "busy"))
    stray = {"t": 40, "k": "class", "src": "acc", "task": None, "mode": "busy"}
    bad = class_invariants(events, [*classes, stray], ["acc"])
    assert bad == ["acc: class busy at 40 outside its runs"]


# =============================================================================
# 2. The totals on done, and the class events
# =============================================================================


@pytest.fixture
def cluster():
    return cluster_from(*files())


@pytest.fixture
def tasks():
    return TaskList.from_dict(json.loads(TASKS.read_text()))


def test_done_carries_its_runs_summary_classes(cluster, tasks):
    """The synthetic run's summary: the unit busy 16, stall_in 2; each mover busy 16,
    stall_xbar 1, idle 2."""
    lines = [parse_line(x) for x in synthetic(cluster, tasks)]
    _, trace = convert(lines, cluster, tasks, 226, "task")
    (acc,) = trace.of_kind("done", "acc")
    assert acc["cycles"] == {"busy": 16, "stall_in": 2, "stall_out": 0, "flush": 0}
    for m in ("acc_a", "acc_b", "acc_out"):
        (done,) = trace.of_kind("done", m)
        assert done["cycles"] == {"busy": 16, "stall_xbar": 1, "stall_fifo": 0, "idle": 2}
    assert not trace.of_kind("class")  # class lines are beat-level


def test_a_run_without_its_summary_line_is_named(cluster, tasks):
    lines = [x for x in synthetic(cluster, tasks) if "mover mover=reader1" not in x]
    with pytest.raises(RunError, match=r"acc_b \(add_acc_b\): no summary line for its run 0") as e:
        convert([parse_line(x) for x in lines], cluster, tasks, 226, "task")
    assert e.value.code == "convert.summary"


def test_class_lines_become_events_with_their_task(cluster, tasks):
    lines = [parse_line(x) for x in synthetic(cluster, tasks, "beat")]
    _, trace = convert(lines, cluster, tasks, 226, "beat", None, None,
                       ["acc", "acc_a", "acc_b", "acc_out", "dma"])  # fmt: skip
    (start,) = trace.of_kind("start", "acc")
    got = [(e["t"] - start["t"], e["mode"], e["task"]) for e in trace.of_kind("class", "acc")]
    assert got == [(1, "stall_in", "add_acc"), (3, "busy", "add_acc")]
    assert [e["mode"] for e in trace.of_kind("class", "acc_out")] == ["busy", "stall_xbar", "idle"]


def test_a_source_not_traced_has_no_class_events(cluster, tasks):
    lines = [parse_line(x) for x in synthetic(cluster, tasks, "beat")]
    _, trace = convert(lines, cluster, tasks, 226, "beat", None, None, ["acc", "dma"])
    assert {e["src"] for e in trace.of_kind("class")} == {"acc"}


# =============================================================================
# 3. A core's spans
# =============================================================================


def ev(k, t, **kw):
    return {"t": t, "k": k, **kw}


def test_a_barrier_over_a_poll_over_a_sleep():
    """Sleep 10-29; a poll 20-39; a barrier 25-34: barrier wins 25-34, poll 20-24 and
    35-39, sleep 10-19."""
    spans = core_spans([ev("state", 10, mode="sleep"), ev("state", 30, mode="run"),
                        ev("poll", 20, last=39), ev("barrier", 25, last=34)], 50)  # fmt: skip
    assert spans == [(0, 10, "run"), (10, 20, "sleep"), (20, 25, "poll"), (25, 35, "barrier"),
                     (35, 40, "poll"), (40, 50, "run")]  # fmt: skip
    assert core_counts(spans) == {"run": 20, "poll": 10, "barrier": 10, "sleep": 10}


def test_a_state_and_a_wait_before_the_window_count_inside_it():
    spans = core_spans([ev("state", -5, mode="sleep"), ev("barrier", -3, last=4),
                        ev("state", 8, mode="run")], 12)  # fmt: skip
    assert spans == [(0, 5, "barrier"), (5, 8, "sleep"), (8, 12, "run")]


def test_the_spans_cover_the_window_merged():
    spans = core_spans([ev("poll", 2, last=3), ev("poll", 4, last=6), ev("barrier", 30, last=90)],
                       40)  # fmt: skip
    assert spans == [(0, 2, "run"), (2, 7, "poll"), (7, 30, "run"), (30, 40, "barrier")]
    assert [a for a, _, _ in spans[1:]] == [b for _, b, _ in spans[:-1]]
    assert core_spans([], 7) == [(0, 7, "run")]
