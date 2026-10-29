"""Tests for the visualiser's API functions on flow runs (C§7, run/).

The report itself is HTML and is checked by eye (D55); what is tested here
is what the viewer is given:

  1. a run loads as read_run and load_cluster give it, from its folder or its run/
  2. the memory view answers on every run (its cases are test_memory.py's);
     the movement views say why not below beat (test_movement.py's)

A task's bars and a range of events are the schedule's (test_schedule.py).
"""

import pytest

from snax_forge.run import RunError, load_cluster, read_run
from snax_forge.viz import api

from .helpers import run_dir


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    """vecadd at task and off, vsum at task."""
    base = tmp_path_factory.mktemp("runs")
    return {(n, lv): run_dir(base / f"{n}-{lv}", n, lv)
            for n, lv in [("vecadd", "task"), ("vecadd", "off"), ("vsum", "task")]}  # fmt: skip


# =============================================================================
# 1. Loading
# =============================================================================


@pytest.mark.parametrize("level", ["task", "off"])
def test_a_run_loads_as_run_gives_it(dirs, level):
    d = dirs[("vecadd", level)]
    rv = api.load_run(d)
    out = read_run(d / "run")
    assert (rv.name, rv.path, rv.folder) == (f"vecadd-{level}", d / "run", d)
    assert rv.outputs == out and rv.total_cycles == out.profile.total_cycles == 226
    assert rv.cluster == load_cluster(d / "gen" / "snax_forge.hjson",
                                      d / "design" / "design_point.json")  # fmt: skip
    assert rv.trace_level == level
    assert (rv.events == []) == (level == "off")
    assert [e["t"] for e in rv.events] == sorted(e["t"] for e in rv.events)
    assert api.load_run(d / "run").outputs == out  # its run/ too


def test_not_a_flow_run(tmp_path):
    with pytest.raises(RunError):
        api.load_run(tmp_path)


def test_names_are_the_folders_and_unique(dirs, tmp_path):
    d = dirs[("vecadd", "task")]
    assert api.run_names([d, d / "run", d]) == ["vecadd-task", "vecadd-task-2", "vecadd-task-3"]


def test_detail_and_summary(dirs):
    rv = api.load_run(dirs[("vecadd", "task")])
    s = api.run_summary(rv)
    assert (s["design"], s["trace_level"], s["total_cycles"]) == ("vecadd", "task", 226)
    d = api.run_detail(rv)
    assert d["profile"] == rv.outputs.profile.to_dict() and d["cluster"] == rv.cluster.to_dict()
    assert d["trace"] == {**rv.outputs.trace.meta(), "n_events": len(rv.events)}
    assert d["total_cycles"] == 226 and "tasks" not in d
    assert api.run_detail(api.load_run(dirs[("vecadd", "off")]))["trace"] is None


# =============================================================================
# 2. Memory, and not yet
# =============================================================================


def test_the_memory_view_answers_and_movement_says_why_not(dirs):
    rv = api.load_run(dirs[("vecadd", "task")])
    assert api.memories(rv) == ["l1", "l2"]
    view = api.memory_view(rv)
    assert [m["mem"] for m in view["memories"]] == ["l1", "l2"] and view["marks_available"] == {}
    for view in (api.conflicts_view(rv), api.journey_view(rv, "A", 0)):
        assert view["available"] is False and view["reason"].endswith("traced at task")


def test_a_run_has_its_regions_and_index(dirs):
    rv = api.load_run(dirs[("vecadd", "task")])
    assert [(r.container, r.memory) for r in rv.regions] == [
        (c, m) for c in "ABC" for m in ("l2", "l1")]  # fmt: skip
    assert rv.index.level == "task" and rv.index.events == rv.outputs.trace.events
    assert api.load_run(dirs[("vecadd", "off")]).index.level == "off"
