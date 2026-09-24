"""Tests for the task-list format (LOW1b, D64).

Sections:
  1. round trips (checked-in files, builder, values)
  2. structural errors
  3. format errors
"""

import json

import pytest

from snax_forge.lower import TaskList, TaskListError, Tasks, arg_of, values_of
from snax_forge.snax_model import DmaDescriptor, DmaPattern, StreamerRegs
from snax_forge.snax_model.config import to_json
from snax_forge.snax_model.scenario import register_map_of

from .helpers import SCEN, TASK_SCENARIOS, add, cluster, dma, start, stream, sync, task_list

# =============================================================================
# 1. Round trips
# =============================================================================


def test_every_scenario_is_a_task_list():
    """Every scenario folder has a hand-written task list (D66)."""
    scenarios = tuple(sorted(p.parent.name for p in SCEN.glob("*/scenario.py")))
    assert TASK_SCENARIOS == scenarios
    assert "fmul" in scenarios and "vecadd" in scenarios


@pytest.mark.parametrize("name", TASK_SCENARIOS)
def test_checked_in_task_list_round_trips(name):
    """Hand-written files: the same data back, whatever the layout of the file."""
    d = json.loads((SCEN / name / "tasks.json").read_text())
    tasks = TaskList.from_dict(d)
    assert tasks.to_dict() == d
    assert TaskList.from_dict(json.loads(to_json(tasks.to_dict()))) == tasks


def test_missing_optional_keys_take_defaults():
    t = task_list({"op": "configure", "task_name": "a", "type": "dma", "component": "dma",
                   "values": dma("a")["values"]}, {"op": "start", "tasks": ["a"]},
                  {"op": "sync", "task": "a"})  # fmt: skip
    conf, _, s = t.steps
    assert (conf.after, conf.wait_mode, s.mode) == ([], "poll", "poll")
    assert t.to_dict()["steps"][0]["after"] == []  # every field is written back


def test_builder_writes_the_values_of_its_arguments():
    t = Tasks("b", cluster("alu4"))
    desc = DmaDescriptor("l1_to_l2", DmaPattern(64, (2, 3), (64, 128)), DmaPattern(0, (6,), (64,)))
    t.configure("st", "dma", desc, wait_mode="signal")
    t.configure("rd", "ra", StreamerRegs(8, (16,), (32,), (4,), (8,)), after=["st"])
    t.configure("x", "acc", {"n": 16})
    t.start("st")
    t.start("rd", "x")
    d = t.task_list().to_dict()
    assert d["steps"][0] == {
        "op": "configure", "task_name": "st", "type": "dma", "component": "dma", "after": [],
        "wait_mode": "signal",
        "values": {"direction": "l1_to_l2", "src": {"base": 64, "bounds": [2, 3], "strides": [64, 128]},
                   "dst": {"base": 0, "bounds": [6], "strides": [64]}},
    }  # fmt: skip
    assert d["steps"][1]["values"] == {
        "base": 8, "temporal_bounds": [16], "temporal_strides": [32], "spatial_strides": [8]
    }  # fmt: skip
    assert d["steps"][1]["after"] == ["st"]
    assert d["steps"][2]["values"] == {"n": 16}


def test_builder_checks_arguments_where_they_are_made():
    t = Tasks("b", cluster("alu4"))
    with pytest.raises(ValueError, match="design-time"):
        t.configure("r", "ra", StreamerRegs(0, (16,), (32,), (2, 2), (8, 64)))
    with pytest.raises(TaskListError, match="no component"):
        t.configure("r", "rc", StreamerRegs(0, (16,), (32,), (4,), (8,)))


@pytest.mark.parametrize(
    "block, arg",
    [
        (
            "dma",
            DmaDescriptor(
                "l2_to_l1", DmaPattern(0, (8, 2), (1024, 64)), DmaPattern(0, (16,), (64,))
            ),
        ),
        ("ra", StreamerRegs(24, (4,), (-32,), (4,), (8,))),
        ("acc", {"n": 7}),
    ],
)
def test_values_and_start_arguments_are_inverses(block, arg):
    b = register_map_of(cluster("alu4"))[block]
    assert arg_of(b.kind, values_of(b.kind, arg), b) == arg


# =============================================================================
# 2. Structural errors (no cluster needed)
# =============================================================================


@pytest.mark.parametrize(
    "steps, match",
    [
        ([dma("a"), start("a"), dma("a"), start("a")], "configured twice"),
        ([stream("r", "ra", after=["a"]), start("r")], "'a' is not an earlier task"),
        ([stream("r", "ra", after=["r"]), start("r")], "'r' is not an earlier task"),
        ([dma("a"), dma("b"), start("a"), start("b")], "configured again before its task 'a'"),
        ([start("a")], "'a' is not configured before"),
        ([dma("a"), start("a"), start("a")], "'a' is started twice"),
        ([stream("r", "ra"), stream("s", "ra"), start("r", "s")], "configured again"),
        ([dma("a"), stream("r", "ra", after=["a"]), start("a", "r")], "needs 'a', which has not"),
        ([dma("a"), stream("r", "ra", after=["a"]), start("r"), start("a")], "needs 'a'"),
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
        task_list(stream("r", "ra"), start("r", "r"))


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
