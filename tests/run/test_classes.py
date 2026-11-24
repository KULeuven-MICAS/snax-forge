"""The class part of the converter: class lines against their run's totals
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
    parse_line,
)
from snax_forge.run.beats import class_invariants
from snax_forge.run.convert import convert
from snax_forge.viz.index import RunIndex

from .helpers import BOOT, line, synthetic

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


def test_a_run_without_its_summary_line_is_named(cluster, tasks):
    lines = [x for x in synthetic(cluster, tasks) if "mover mover=reader1" not in x]
    with pytest.raises(RunError, match=r"acc_b \(add_acc_b\): no summary line for its run 0") as e:
        convert([parse_line(x) for x in lines], cluster, tasks, 226, "task")
    assert e.value.code == "convert.summary"


def test_a_source_not_traced_has_no_class_events(cluster, tasks):
    lines = [parse_line(x) for x in synthetic(cluster, tasks, "beat")]
    _, trace = convert(lines, cluster, tasks, 226, "beat", None, None, ["acc", "dma"])
    assert {e["src"] for e in trace.of_kind("class")} == {"acc"}


# =============================================================================
# 3. A core's spans
# =============================================================================


def ev(k, t, **kw):
    return {"t": t, "k": k, **kw}


# =============================================================================
# 4. What each core is in at 0
# =============================================================================


def with_states(cluster, tasks, *states):
    """The synthetic run with core1's extra ``state`` lines ``(cycle, mode)``."""
    lines = synthetic(cluster, tasks)
    lines += [line("pe1", "events", f"{c} state mode={m}", c) for c, m in states]
    return convert([parse_line(x) for x in lines], cluster, tasks, 226, "task")


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
