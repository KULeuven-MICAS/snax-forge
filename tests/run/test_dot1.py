"""run/ on a recorded run of an accelerator of two groups (D148): dot 1 at trace levels task
and stall (fixtures/dot_1/README.md), converted without GVSoC.

The cluster has the accelerator dot with its units mul and sum, each with its group, its
SNAX-GVSoC path (``unit_mul``, ``unit_sum``), its group's CSRs and its movers; the converter
names each unit's lines after it, gives each a profile, matches each group's task and the
movers' tasks (the other group's idle) in order. Above task each beat line goes to its unit
and each streamer line to the task launched with its unit's (DRY1-2 R2, open item 96).
"""

import json

import pytest

from snax_forge.run import RunError, Unit
from snax_forge.run.convert import convert, invariants

from .helpers import load_recorded


@pytest.fixture(scope="module")
def recorded():
    return load_recorded("dot_1")


def test_the_cluster_has_a_unit_per_group(recorded):
    cluster, *_ = recorded
    a = cluster.accelerator
    assert a.name == "dot" and a.streamer == "dot_streamer"
    first = a.csrs.streamer_last + 1
    assert a.units == [Unit("mul", "mul", "unit_mul", first, first + 2, first + 3),
                       Unit("sum", "sum", "unit_sum", first + 4, first + 7, first + 8)]  # fmt: skip
    assert [(m.name, m.gvsoc) for m in a.movers] == [
        ("mul_a", "reader0"), ("mul_b", "reader1"), ("sum_a", "reader2"),
        ("mul_out", "writer0"), ("sum_out", "writer1"),
    ]  # fmt: skip
    assert cluster.sources[:7] == ["mul", "sum", "mul_a", "mul_b", "sum_a", "mul_out", "sum_out"]
    assert [a.block_of(first + k) for k in (0, 3, 4, 6, 8, 9)] == [
        "mul", "mul", "sum", "sum", "sum", None]  # fmt: skip


def test_each_unit_converts_with_its_own_tasks(recorded):
    cluster, tasks, check, lines, regions, _ = recorded
    profile, trace = convert(lines, cluster, tasks, check["total_cycles"], "task",
                             check["functional_check"], regions)  # fmt: skip
    assert invariants(profile, trace, cluster, None, regions) == []
    assert {k: a.firings for k, a in profile.accelerators.items()} == {"mul": 16, "sum": 16}
    starts = {e["src"]: [] for e in trace.of_kind("start")}
    for e in trace.of_kind("start"):
        starts[e["src"]].append(e["task"])
    assert starts["mul"] == ["mult_mul"] and starts["sum"] == ["sum_sum"]
    assert starts["sum_a"] == ["mult_sum_a", "sum_sum_a"]  # idle with mul, then its own
    idle = [e for e in trace.of_kind("done") if e["task"] in ("mult_sum_a", "sum_mul_a")]
    assert [e["cycles"]["busy"] for e in idle] == [0, 0]
    # each CSR access names its group's unit (mul's STATUS read once, mul done by then)
    blocks = [e["block"] for e in trace.events if e["k"] in ("csr", "poll") and e["src"] == "core0"]
    assert {"mul", "sum", "dot_streamer"} <= set(blocks) <= {"mul", "sum", "dot_streamer", "-"}
    assert [w.block for w in profile.cores["core0"].waits if w.block != "barrier"] == ["sum"]
    assert {t: x["group"] for t, x in check["functional_check"]["tasks"].items()} == {
        "mult_mul": "mul", "sum_sum": "sum"}  # fmt: skip


def test_each_unit_has_its_own_movers(recorded):
    cluster, *_ = recorded
    a = cluster.accelerator
    assert [m.name for m in a.movers_of("mul")] == ["mul_a", "mul_b", "mul_out"]
    assert [m.name for m in a.movers_of("sum")] == ["sum_a", "sum_out"]
    assert (a.unit_of_mover("sum_a"), a.unit_of_mover("mul_out")) == ("sum", "mul")


@pytest.fixture(scope="module")
def stalled():
    from snax_forge.run.sources import all_sources

    cluster, tasks, check, lines, regions, _ = load_recorded("dot_1", "events_stall.log")
    return cluster, convert(lines, cluster, tasks, check["total_cycles"], "stall",
                            check["functional_check"], regions, all_sources(cluster))  # fmt: skip


def test_both_groups_convert_at_the_beat_levels(stalled, recorded):
    """At stall (so beat too), every beat and stall invariant holding on both units: each unit's
    16 firings (N = 64, W = 4) under its own task, the mul's 16 pushes into tmp0 by mul_out and the
    sum's one into out, sum_a reading tmp0's words; the profile is the task level's."""
    _, (profile, trace) = stalled
    fires = {u: [e for e in trace.of_kind("fire") if e["src"] == u] for u in ("mul", "sum")}
    assert {u: len(x) for u, x in fires.items()} == {"mul": 16, "sum": 16}
    assert {e["task"] for e in fires["mul"]} == {"mult_mul"}
    assert {e["task"] for e in fires["sum"]} == {"sum_sum"}
    assert set(fires["mul"][0]["b"]) == {"mul_a", "mul_b"} and set(fires["sum"][0]["b"]) == {
        "sum_a"
    }
    pushes = [(e["src"], e["task"]) for e in trace.of_kind("push")]
    assert pushes.count(("mul", "mult_mul")) == 16 and pushes.count(("sum", "sum_sum")) == 1
    grants = [e for e in trace.of_kind("grant") if e["region"] == "tmp0"]
    writes = {e["src"].split(".")[0] for e in grants if e["op"] == "w"}
    reads = {e["src"].split(".")[0] for e in grants if e["op"] == "r"}
    assert (writes, reads) == ({"mul_out"}, {"sum_a"})
    assert {e["task"] for e in grants if e["op"] == "w"} == {"mult_mul_out"}
    assert {e["task"] for e in grants if e["op"] == "r"} == {"sum_sum_a"}
    assert all(v is not None for e in fires["sum"] for v in e["vals"].values())
    rc, rtasks, rcheck, rlines, rregions, _ = recorded
    task_profile, _ = convert(rlines, rc, rtasks, rcheck["total_cycles"], "task",
                              rcheck["functional_check"], rregions)  # fmt: skip
    assert {k: (a.firings, a.cycles) for k, a in profile.accelerators.items()} == {
        k: (a.firings, a.cycles) for k, a in task_profile.accelerators.items()}  # fmt: skip


def test_beat_levels_are_refused_for_a_linked_group():
    """A group of several linked units (dot 4) stays refused above task (open item 96)."""
    cluster, tasks, check, lines, regions, _ = load_recorded("dot_4")
    with pytest.raises(RunError) as e:
        convert(lines, cluster, tasks, check["total_cycles"], "beat", None, regions, [])
    assert e.value.code == "convert.level" and "open item 96" in str(e.value)
    assert "several linked units" in str(e.value)


def test_the_cluster_round_trips(recorded):
    from snax_forge.run import Cluster

    cluster, *_ = recorded
    assert Cluster.from_dict(json.loads(json.dumps(cluster.to_dict()))) == cluster
