"""Tests for the task-list format (D64) and its values against a design point.

Sections:
  1. round trips (the checked-in fixture, derived task lists)
  2. structural errors
  3. format errors
  4. values against the design point (check_values)
"""

import json

import pytest

from snax_forge.common import to_json
from snax_forge.lower import TaskList, TaskListError, check_values, components
from snax_forge.lower import task_list as derive

from .helpers import (
    VECADD_TASKS,
    add,
    dma,
    start,
    stream,
    sync,
    task_list,
    vecadd_point,
    vsum_point,
)

# =============================================================================
# 1. Round trips
# =============================================================================


def test_the_checked_in_task_list_round_trips():
    """vecadd's on snax_forge, as tests/run keeps it: the same data back, byte for byte."""
    d = json.loads(VECADD_TASKS.read_text())
    tasks = TaskList.from_dict(d)
    assert tasks.to_dict() == d
    assert to_json(tasks.to_dict()) == VECADD_TASKS.read_text()


@pytest.mark.parametrize("point", [vecadd_point, vsum_point], ids=["vecadd", "vsum"])
def test_a_derived_task_list_round_trips_and_its_values_fit(point):
    tasks = derive(point())
    assert TaskList.from_dict(json.loads(to_json(tasks.to_dict()))) == tasks
    check_values(tasks, point())


def test_missing_optional_keys_take_defaults():
    t = task_list({"op": "configure", "task_name": "a", "type": "dma", "component": "dma",
                   "values": dma("a")["values"]}, {"op": "start", "tasks": ["a"]},
                  {"op": "sync", "task": "a"})  # fmt: skip
    conf, _, s = t.steps
    assert (conf.after, conf.wait_mode, s.mode) == ([], "poll", "poll")
    assert t.to_dict()["steps"][0]["after"] == []  # every field is written back


# =============================================================================
# 2. Structural errors (no design point needed)
# =============================================================================


@pytest.mark.parametrize(
    "steps, match",
    [
        ([dma("a"), start("a"), dma("a"), start("a")], "configured twice"),
        ([stream("r", "acc_a", after=["a"]), start("r")], "'a' is not an earlier task"),
        ([stream("r", "acc_a", after=["r"]), start("r")], "'r' is not an earlier task"),
        ([dma("a"), dma("b"), start("a"), start("b")], "configured again before its task 'a'"),
        ([start("a")], "'a' is not configured before"),
        ([dma("a"), start("a"), start("a")], "'a' is started twice"),
        ([stream("r", "acc_a"), stream("s", "acc_a"), start("r", "s")], "configured again"),
        (
            [dma("a"), stream("r", "acc_a", after=["a"]), start("a", "r")],
            "needs 'a', which has not",
        ),
        ([dma("a"), stream("r", "acc_a", after=["a"]), start("r"), start("a")], "needs 'a'"),
        ([dma("a"), sync("a"), start("a")], "'a' is not started before"),
        ([dma("a"), start("a"), sync("a", "spin")], "mode must be one of"),
        ([dma("a", wait_mode="spin"), start("a")], "wait_mode must be one of"),
        ([dma("a")], "configured but never started"),
        ([dma("a"), start("a"), {"op": "read", "reg": "busy"}], "not 'block.register'"),
        ([{"op": "start", "tasks": []}], "no tasks"),
    ],
)
def test_structural_errors(steps, match):
    with pytest.raises(TaskListError, match=match):
        task_list(*steps)


def test_a_task_listed_twice_in_one_start():
    """Two tasks on one component cannot meet in one start: the second
    configure is refused first (above)."""
    with pytest.raises(TaskListError, match="listed twice"):
        task_list(stream("r", "acc_a"), start("r", "r"))


# =============================================================================
# 3. Format errors
# =============================================================================


@pytest.mark.parametrize(
    "d, match",
    [
        ({"steps": []}, "needs a 'name'"),
        ({"name": "t", "steps": [], "cluster": "x"}, "unknown keys"),
        ({"name": "t", "steps": [{"op": "wait", "task": "a"}]}, "op must be one of"),
        ({"name": "t", "steps": [{**add("x"), "group": "g"}]}, "unknown keys"),
        ({"name": "t", "steps": [{"op": "configure", "task_name": "x", "type": "dma"}]}, "missing"),
        ({"name": "t", "steps": [{**add("x"), "values": [16]}]}, "values must be an object"),
        ({"name": "t", "steps": [{"op": "sync"}]}, "missing"),
    ],
)
def test_format_errors(d, match):
    with pytest.raises(TaskListError, match=match):
        TaskList.from_dict(d)


# =============================================================================
# 4. Values against the design point
# =============================================================================


def test_the_components_are_the_design_points():
    assert components(vecadd_point()) == {
        "dma": "dma", "acc_a": "streamer", "acc_b": "streamer", "acc_out": "streamer",
        "acc": "accel",
    }  # fmt: skip


def bad_values(values: dict) -> dict:
    return {"op": "configure", "task_name": "x", "type": "dma", "component": "dma",
            "values": values}  # fmt: skip


GOOD_DMA = dma("x")["values"]
GOOD_STREAM = stream("x", "acc_a")["values"]


@pytest.mark.parametrize(
    "steps, match",
    [
        ([{**dma("x"), "component": "dma2"}, start("x")], "no component 'dma2'"),
        ([{**dma("x"), "type": "streamer"}, start("x")], "dma is a dma, not a streamer"),
        ([{**dma("x"), "type": "gpu", "component": "dma"}, start("x")], "is a dma, not a gpu"),
        ([bad_values({k: v for k, v in GOOD_DMA.items() if k != "dst"}), start("x")], "missing"),
        ([bad_values({**GOOD_DMA, "burst": 4}), start("x")], "unknown"),
        ([bad_values({**GOOD_DMA, "direction": "l1_to_l1"}), start("x")], "direction must be one of"),
        (
            [bad_values({**GOOD_DMA, "src": {"symbol": "A", "bounds": [0], "strides": [64]}}), start("x")],
            "bounds must be >= 1",
        ),
        (
            [{**stream("x", "acc_a"), "values": {**GOOD_STREAM, "temporal_bounds": [-1]}},
             start("x")],
            r"temporal bounds must be >= 1, or all 0 \(an idle mover\)",
        ),
        ([bad_values({**GOOD_DMA, "src": {"symbol": "A", "bounds": [8]}}), start("x")], "missing"),
        (
            [bad_values({**GOOD_DMA, "src": {"symbol": "A", "bounds": 8, "strides": [64]}}), start("x")],
            "must be a list",
        ),
        ([{**add("x"), "values": {"n": 16, "T": 4}}, start("x")], "parameters"),
        ([{**stream("x", "acc_a"), "values": {"base": 0}}, start("x")], "missing"),
        (
            [{**stream("x", "acc_a"), "values": {**GOOD_STREAM, "temporal_bounds": [4, 4],
                                                 "temporal_strides": [32, 128]}}, start("x")],
            "acc_a: 2 loops, registers hold 1",
        ),
        (
            [{**stream("x", "acc_a"), "values": {**GOOD_STREAM, "temporal_strides": [32, 0]}},
             start("x")],
            "1 temporal bounds but 2 temporal strides",
        ),
        (
            [{**stream("x", "acc_a"), "values": {**GOOD_STREAM, "spatial_strides": [8, 32]}},
             start("x")],
            "1 spatial bounds but 2 spatial strides",
        ),
    ],
)  # fmt: skip
def test_values_that_do_not_fit_the_design_point(steps, match):
    with pytest.raises(TaskListError, match=match):
        check_values(task_list(*steps), vecadd_point())


def test_an_idle_mover_has_every_bound_0():
    """Another group's mover in an accelerator of several (D148): every temporal bound 0."""
    idle = {**GOOD_STREAM, "temporal_bounds": [0], "temporal_strides": [0]}
    check_values(task_list({**stream("x", "acc_a"), "values": idle}, start("x")), vecadd_point())
