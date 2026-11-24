"""Tests for the visualiser's API functions on flow runs (run/).

The report itself is HTML and is checked by eye; what is tested here
is what the viewer is given:

  1. a run loads as read_run and load_cluster give it, from its folder or its run/
  2. the memory view answers on every run (its cases are test_memory.py's);
     the movement views say why not below beat (test_movement.py's)

A task's bars and a range of events are the schedule's (test_schedule.py).
"""

import pytest

from snax_forge.viz import api

from .helpers import run_dir


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    """vecadd at task and off, vsum at task."""
    base = tmp_path_factory.mktemp("runs")
    return {(n, lv): run_dir(base / f"{n}-{lv}", f"{n}_v1", lv)
            for n, lv in [("vecadd", "task"), ("vecadd", "off"), ("vsum", "task")]}  # fmt: skip


# =============================================================================
# 1. Loading
# =============================================================================


def test_names_are_the_folders_and_unique(dirs, tmp_path):
    d = dirs[("vecadd", "task")]
    assert api.run_names([d, d / "run", d]) == ["vecadd-task", "vecadd-task-2", "vecadd-task-3"]


# =============================================================================
# 2. Memory, and not yet
# =============================================================================
