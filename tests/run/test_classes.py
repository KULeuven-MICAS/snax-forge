"""The class part of the converter (C§7): class lines against their run's totals
on hand-written snippets with obvious answers, the totals on ``done`` from the summary lines,
and a core's spans (run/cores.py).

1. class runs: a run of busy, stall_in, busy covers its span; a gap, a sum off by one, a
   repeated class, a line outside every run are each named
2. the totals on done: an accelerator's and a mover's run from its summary line; a run
   without one is a named error; the class lines become events with their task
3. a core's spans: barrier > poll > sleep > run, a state before the window, the cover
4. what each core is in at 0: its function and state from before the window, null without a
   ``func`` line, the index's first function span from 0
"""

import pytest

from snax_forge.run import (
    RunError,
    RunOutputs,
    core_counts,
    core_spans,
    parse_line,
    start_mode,
)
from snax_forge.run.beats import class_invariants, class_runs
from snax_forge.run.convert import convert
from snax_forge.viz.index import RunIndex

from .helpers import BOOT, MAIN, line, synthetic

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


# =============================================================================
# 4. What each core is in at 0
# =============================================================================


def with_states(cluster, tasks, *states):
    """The synthetic run with core1's extra ``state`` lines ``(cycle, mode)``."""
    lines = synthetic(cluster, tasks)
    lines += [line("pe1", "events", f"{c} state mode={m}", c) for c, m in states]
    return convert([parse_line(x) for x in lines], cluster, tasks, 226, "task")


def test_start_from_a_func_and_a_state_before_the_window(cluster, tasks):
    """The synthetic run, core1 asleep 6 to 2 cycles before the window: each core starts in
    main (its func line at MAIN) and running; the index's function spans start at 0 in main."""
    profile, trace = with_states(cluster, tasks, (BOOT - 6, "sleep"), (BOOT - 2, "run"))
    assert trace.start == {"core0": {"func": "main", "mode": "run"},
                           "core1": {"func": "main", "mode": "run"}}  # fmt: skip
    assert list(trace.start) == cluster.cores
    assert not any(e["k"] == "func" and e["t"] == 0 for e in trace.events)
    idx = RunIndex.from_run(RunOutputs({}, profile, trace), cluster, [], tasks)
    for core in cluster.cores:
        assert idx.funcs[core][0] == (0, MAIN + 10 - BOOT, "main")
        assert idx.funcs[core][1][2] == f"forge_core_{core[-1]}"


def test_start_mode_is_the_state_the_classes_start_from(cluster, tasks):
    """core1 put to sleep 2 cycles before the window and woken 3 cycles in: asleep at 0, and
    its first 3 cycles count sleep, in the profile and in the index's spans."""
    profile, trace = with_states(cluster, tasks, (BOOT - 2, "sleep"), (BOOT + 3, "run"))
    assert trace.start["core1"] == {"func": "main", "mode": "sleep"}
    assert profile.cores["core1"].cycles["sleep"] == 3
    idx = RunIndex.from_run(RunOutputs({}, profile, trace), cluster, [], tasks)
    assert idx.core_spans["core1"][:2] == [
        (0, 3, "sleep"),
        (3, idx.core_spans["core1"][1][1], "run"),
    ]
    for core in cluster.cores:
        assert core_counts(idx.core_spans[core]) == profile.cores[core].cycles


def test_a_core_without_a_func_line_before_the_window_starts_in_null(cluster, tasks):
    lines = [x for x in synthetic(cluster, tasks)
             if not ("/pe1/" in x and " func " in x and int(x.split(": ")[1]) <= BOOT)]  # fmt: skip
    _, trace = convert([parse_line(x) for x in lines], cluster, tasks, 226, "task")
    assert trace.start["core1"] == {"func": None, "mode": "run"}
    idx = RunIndex.from_run(RunOutputs({}, None, trace), cluster, [], tasks)
    assert idx.funcs["core1"][0] == (0, MAIN + 10 - BOOT, None)


def test_the_start_mode_is_the_one_the_spans_start_from():
    for evs in ([], [ev("state", -9, mode="sleep")], [ev("state", -9, mode="sleep"),
                ev("state", -2, mode="run")], [ev("state", 0, mode="sleep")]):  # fmt: skip
        assert start_mode(evs) == core_spans(evs, 10)[0][2]
