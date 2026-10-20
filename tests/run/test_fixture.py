"""run/ on a recorded run: vecadd at trace level task (fixtures/vecadd/README.md), converted
without GVSoC. No expected numbers: the conversion goes through, its invariants hold
(convert raises otherwise) and it reads back."""

import json
from pathlib import Path

import pytest

from snax_forge.lower.tasks import TaskList
from snax_forge.run import load_cluster, read_events, read_run, write_run
from snax_forge.run.convert import convert, invariants

HERE = Path(__file__).parent / "fixtures" / "vecadd"


@pytest.fixture(scope="module")
def recorded():
    cluster = load_cluster(HERE / "snax_alu.hjson", HERE / "design_point.json")
    tasks = TaskList.from_dict(json.loads((HERE / "tasks.json").read_text()))
    check = json.loads((HERE / "check.json").read_text())
    return cluster, tasks, check, read_events(HERE / "events.log")


def test_task_level(recorded, tmp_path):
    cluster, tasks, check, lines = recorded
    profile, trace = convert(lines, cluster, tasks, check["total_cycles"], "task",
                             check["functional_check"])  # fmt: skip
    assert invariants(profile, trace, cluster) == []
    assert trace.boot > 0 and 0 < trace.window and trace.sources == cluster.sources
    assert {e["src"] for e in trace.events} <= set(cluster.sources)
    # every task of the list starts (or is issued) in the window
    named = {e["task"] for e in trace.events if e["k"] in ("start", "issue")}
    assert named == set(tasks.configured())
    assert all(c.cycles is not None for c in profile.cores.values())
    (tmp_path / "run.json").write_text("{}")
    write_run(tmp_path, profile, trace)
    back = read_run(tmp_path)
    assert (back.profile, back.trace) == (profile, trace)


def test_off_from_the_summary_lines(recorded):
    cluster, tasks, check, lines = recorded
    summary = [x for x in lines if x.trace == "summary"]
    profile, trace = convert(summary, cluster, tasks, check["total_cycles"], "off")
    full, _ = convert(lines, cluster, tasks, check["total_cycles"], "task")
    assert trace is None
    for c in full.cores.values():
        c.cycles, c.waits = None, []
    assert full == profile
