"""run/ on a recorded run: vecadd at trace levels task and stall (fixtures/vecadd/README.md),
converted without GVSoC. No expected numbers: the conversion goes through, its invariants hold
(convert raises otherwise) and it reads back."""

import json
from pathlib import Path

import pytest

from snax_forge.run import read_events, read_run, write_run
from snax_forge.run.cluster import core_name
from snax_forge.run.convert import convert, invariants

from .helpers import load_recorded

HERE = Path(__file__).parent / "fixtures" / "vecadd"


@pytest.fixture(scope="module")
def recorded():
    return load_recorded("vecadd")


def test_task_level(recorded, tmp_path):
    cluster, tasks, check, lines, regions, _ = recorded
    profile, trace = convert(lines, cluster, tasks, check["total_cycles"], "task",
                             check["functional_check"], regions)  # fmt: skip
    assert invariants(profile, trace, cluster, None, regions) == []
    assert trace.boot > 0 and trace.total_cycles == check["total_cycles"]
    assert trace.sources == cluster.sources
    assert all(sum(c.cycles.values()) == trace.total_cycles for c in profile.cores.values())
    assert {e["src"] for e in trace.events} <= set(cluster.sources)
    # every task of the list starts (or is issued) in the window
    named = {e["task"] for e in trace.events if e["k"] in ("start", "issue")}
    assert named == set(tasks.configured())
    assert all(c.cycles is not None for c in profile.cores.values())
    # what each core is in at 0: main, running, on the accelerator's core
    assert trace.start[core_name(cluster.accelerator.core)] == {"func": "main", "mode": "run"}
    assert list(trace.start) == cluster.cores
    (tmp_path / "run.json").write_text("{}")
    write_run(tmp_path, profile, trace)
    back = read_run(tmp_path)
    assert (back.profile, back.trace) == (profile, trace)


def test_off_from_the_summary_lines(recorded):
    cluster, tasks, check, lines, *_ = recorded
    summary = [x for x in lines if x.trace == "summary"]
    profile, trace = convert(summary, cluster, tasks, check["total_cycles"], "off")
    full, _ = convert(lines, cluster, tasks, check["total_cycles"], "task")
    assert trace is None
    for c in full.cores.values():
        c.cycles, c.waits = None, []
    assert full == profile


def test_stall_level(recorded, tmp_path):
    """The stall-level recording: every beat and stall invariant holds on real events, and the
    profile is the task level's."""
    cluster, tasks, check, lines, regions, _ = recorded
    sources = ["acc", *(m.name for m in cluster.accelerator.movers), "dma"]
    profile, trace = convert(read_events(HERE / "events_stall.log"), cluster, tasks,
                             check["total_cycles"], "stall", check["functional_check"],
                             regions, sources)  # fmt: skip
    kinds = {e["k"] for e in trace.events}
    assert {"grant", "dma", "burst", "fire", "push", "stall", "fifo"} <= kinds
    assert trace.filter == sources and trace.level == "stall"
    task, _ = convert(lines, cluster, tasks, check["total_cycles"], "task",
                      check["functional_check"])  # fmt: skip
    assert profile.to_dict() == task.to_dict()
    assert {e["region"] for e in trace.events if e["k"] in ("grant", "dma", "burst")} == {
        "A", "B", "C"}  # fmt: skip
    # a burst's region is the container whose symbol holds its address
    syms = json.loads((HERE / "elf_symbols.json").read_text())
    for e in trace.of_kind("burst"):
        (name,) = [c for c, s in syms.items() if s["addr"] <= e["addr"] < s["addr"] + s["size"]]
        assert e["region"] == name and e["elems"]
    (tmp_path / "run.json").write_text("{}")
    write_run(tmp_path, profile, trace)
    assert read_run(tmp_path).trace.events == trace.events
