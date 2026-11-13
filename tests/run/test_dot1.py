"""run/ on a recorded run of an accelerator of two groups: dot 1 at trace levels task
and stall (fixtures/dot_1/README.md), converted without GVSoC.

The cluster has the accelerator dot with its units mul and sum, each with its group, its
SNAX-GVSoC path (``unit_vmul``, ``unit_vsum``), its group's CSRs and its movers; the converter
names each unit's lines after it, gives each a profile, matches each group's task and the
movers' tasks (the other group's idle) in order. Above task each beat line goes to its unit
and each streamer line to the task launched with its unit's.
"""

import json

import pytest

from snax_forge.run import RunError, Unit
from snax_forge.run.convert import convert, invariants

from .helpers import load_recorded


@pytest.fixture(scope="module")
def recorded():
    return load_recorded("dot_1_enc")


def test_the_cluster_has_a_unit_per_group(recorded):
    cluster, *_ = recorded
    a = cluster.accelerator
    assert a.name == "dot" and a.streamer == "dot_streamer"
    first = a.csrs.sets[0].last + 1
    assert a.units == [Unit("vmul", "vmul", "unit_vmul", first, first + 2, first + 3),
                       Unit("vsum", "vsum", "unit_vsum", first + 4, first + 7, first + 8)]  # fmt: skip
    assert [(m.name, m.gvsoc) for m in a.movers] == [
        ("vmul_a", "reader0"), ("vmul_b", "reader1"), ("vsum_a", "reader2"),
        ("vmul_out", "writer0"), ("vsum_out", "writer1"),
    ]  # fmt: skip
    assert cluster.sources[:7] == [
        "vmul",
        "vsum",
        "vmul_a",
        "vmul_b",
        "vsum_a",
        "vmul_out",
        "vsum_out",
    ]
    assert [a.block_of(first + k) for k in (0, 3, 4, 6, 8, 9)] == [
        "vmul", "vmul", "vsum", "vsum", "vsum", None]  # fmt: skip


def test_each_unit_converts_with_its_own_tasks(recorded):
    cluster, tasks, check, lines, regions, _ = recorded
    profile, trace = convert(lines, cluster, tasks, check["total_cycles"], "task",
                             check["functional_check"], regions)  # fmt: skip
    assert invariants(profile, trace, cluster, None, regions) == []
    assert {k: a.firings for k, a in profile.accelerators.items()} == {"vmul": 16, "vsum": 16}
    starts = {e["src"]: [] for e in trace.of_kind("start")}
    for e in trace.of_kind("start"):
        starts[e["src"]].append(e["task"])
    assert starts["vmul"] == ["mult_vmul"] and starts["vsum"] == ["sum_vsum"]
    assert starts["vsum_a"] == ["mult_vsum_a", "sum_vsum_a"]  # idle with mul, then its own
    idle = [e for e in trace.of_kind("done") if e["task"] in ("mult_vsum_a", "sum_vmul_a")]
    assert [e["cycles"]["busy"] for e in idle] == [0, 0]
    # each CSR access names its group's unit (mul's STATUS read once, mul done by then)
    blocks = [e["block"] for e in trace.events if e["k"] in ("csr", "poll") and e["src"] == "core0"]
    assert {"vmul", "vsum", "dot_streamer"} <= set(blocks) <= {"vmul", "vsum", "dot_streamer", "-"}
    assert [w.block for w in profile.cores["core0"].waits if w.block != "barrier"] == ["vsum"]
    assert {t: x["group"] for t, x in check["functional_check"]["tasks"].items()} == {
        "mult_vmul": "vmul", "sum_vsum": "vsum"}  # fmt: skip


def test_each_unit_has_its_own_movers(recorded):
    cluster, *_ = recorded
    a = cluster.accelerator
    assert [m.name for m in a.movers_of("vmul")] == ["vmul_a", "vmul_b", "vmul_out"]
    assert [m.name for m in a.movers_of("vsum")] == ["vsum_a", "vsum_out"]
    assert (a.unit_of_mover("vsum_a"), a.unit_of_mover("vmul_out")) == ("vsum", "vmul")


@pytest.fixture(scope="module")
def stalled():
    from snax_forge.run.sources import all_sources

    cluster, tasks, check, lines, regions, _ = load_recorded("dot_1_enc", "events_stall.log")
    return cluster, convert(lines, cluster, tasks, check["total_cycles"], "stall",
                            check["functional_check"], regions, all_sources(cluster))  # fmt: skip


def test_both_groups_convert_at_the_beat_levels(stalled, recorded):
    """At stall (so beat too), every beat and stall invariant holding on both units: each unit's
    16 firings (N = 64, W = 4) under its own task, the mul's 16 pushes into tmp0 by mul_out and the
    sum's one into out, sum_a reading tmp0's words; the profile is the task level's."""
    _, (profile, trace) = stalled
    fires = {u: [e for e in trace.of_kind("fire") if e["src"] == u] for u in ("vmul", "vsum")}
    assert {u: len(x) for u, x in fires.items()} == {"vmul": 16, "vsum": 16}
    assert {e["task"] for e in fires["vmul"]} == {"mult_vmul"}
    assert {e["task"] for e in fires["vsum"]} == {"sum_vsum"}
    assert set(fires["vmul"][0]["b"]) == {"vmul_a", "vmul_b"} and set(fires["vsum"][0]["b"]) == {
        "vsum_a"
    }
    pushes = [(e["src"], e["task"]) for e in trace.of_kind("push")]
    assert pushes.count(("vmul", "mult_vmul")) == 16 and pushes.count(("vsum", "sum_vsum")) == 1
    grants = [e for e in trace.of_kind("grant") if e["region"] == "tmp0"]
    writes = {e["src"].split(".")[0] for e in grants if e["op"] == "w"}
    reads = {e["src"].split(".")[0] for e in grants if e["op"] == "r"}
    assert (writes, reads) == ({"vmul_out"}, {"vsum_a"})
    assert {e["task"] for e in grants if e["op"] == "w"} == {"mult_vmul_out"}
    assert {e["task"] for e in grants if e["op"] == "r"} == {"sum_vsum_a"}
    assert all(v is not None for e in fires["vsum"] for v in e["vals"].values())
    rc, rtasks, rcheck, rlines, rregions, _ = recorded
    task_profile, _ = convert(rlines, rc, rtasks, rcheck["total_cycles"], "task",
                              rcheck["functional_check"], rregions)  # fmt: skip
    assert {k: (a.firings, a.cycles) for k, a in profile.accelerators.items()} == {
        k: (a.firings, a.cycles) for k, a in task_profile.accelerators.items()}  # fmt: skip


def test_beat_levels_are_refused_for_a_linked_group():
    """A group of several linked units (dot 4) stays refused above task."""
    cluster, tasks, check, lines, regions, _ = load_recorded("dot_4")
    with pytest.raises(RunError) as e:
        convert(lines, cluster, tasks, check["total_cycles"], "beat", None, regions, [])
    assert e.value.code == "convert.level" and "not read yet" in str(e.value)
    assert "several linked units" in str(e.value)


def test_the_cluster_round_trips(recorded):
    from snax_forge.run import Cluster

    cluster, *_ = recorded
    assert Cluster.from_dict(json.loads(json.dumps(cluster.to_dict()))) == cluster


# =============================================================================
# dot_1: two accelerators, two streamer sets
# =============================================================================


@pytest.fixture(scope="module")
def two_sets():
    return load_recorded("dot_1")


def test_dot_1_has_a_streamer_per_set(two_sets):
    """vmul: streamer 960-977, its CSRs 978-981; vsum: 982-994, 995-999."""
    cluster, *_ = two_sets
    a = cluster.accelerator
    assert (a.name, a.accelerators()) == (
        "snax_forge_dot_1",
        [("vmul", ["vmul"]), ("vsum", ["vsum"])],
    )
    assert [(x.name, x.first, x.last, x.trace) for x in a.csrs.sets] == [
        ("vmul", 960, 977, "snax_acc0/streamer_vmul"), ("vsum", 982, 994, "snax_acc0/streamer_vsum")]  # fmt: skip
    assert a.units == [Unit("vmul", "vmul", "unit_vmul", 978, 980, 981),
                       Unit("vsum", "vsum", "unit_vsum", 995, 998, 999)]  # fmt: skip
    assert [(m.name, m.gvsoc, m.set) for m in a.movers] == [
        ("vmul_a", "reader0", "vmul"), ("vmul_b", "reader1", "vmul"), ("vmul_out", "writer0", "vmul"),
        ("vsum_a", "reader0", "vsum"), ("vsum_out", "writer0", "vsum"),
    ]  # fmt: skip
    assert (a.window("vmul"), a.window("vsum")) == ((960, 981), (982, 999))
    assert [a.block_of(x) for x in (960, 978, 982, 995, 1000)] == [
        "vmul_streamer", "vmul", "vsum_streamer", "vsum", None]  # fmt: skip


def test_dot_1_converts_set_by_set(two_sets):
    """Each unit's task starts its own set's movers only (no idle task), each streamer's lines
    named at its own path; one wait between the accelerators."""
    cluster, tasks, check, lines, regions, _ = two_sets
    profile, trace = convert(lines, cluster, tasks, check["total_cycles"], "task",
                             check["functional_check"], regions)  # fmt: skip
    assert invariants(profile, trace, cluster, None, regions) == []
    assert {k: a.firings for k, a in profile.accelerators.items()} == {"vmul": 16, "vsum": 16}
    starts: dict[str, list[str]] = {}
    for e in trace.of_kind("start"):
        starts.setdefault(e["src"], []).append(e["task"])
    assert (starts["vsum_a"], starts["vmul_a"]) == (["sum_vsum_a"], ["mult_vmul_a"])
    assert not any(e.get("idle") for e in trace.events)
    blocks = {e["block"] for e in trace.events if e["k"] in ("csr", "poll") and e["src"] == "core0"}
    assert {"vmul", "vsum", "vmul_streamer", "vsum_streamer"} <= blocks
    assert check["functional_check"]["tasks"]["mult_vmul"]["counter"] == 19
    assert set(profile.streamers) == {"vmul_a", "vmul_b", "vmul_out", "vsum_a", "vsum_out"}


def test_dot_1_beats_on_both_accelerators():
    """At stall, beats traced on both accelerators: vmul writes tmp0 through vmul_out, vsum reads
    it through vsum_a, each crossbar port named in its set (snax_acc0.streamer_<set>...)."""
    from snax_forge.run.sources import all_sources

    cluster, tasks, check, lines, regions, _ = load_recorded("dot_1", "events_stall.log")
    profile, trace = convert(lines, cluster, tasks, check["total_cycles"], "stall",
                             check["functional_check"], regions, all_sources(cluster))  # fmt: skip
    fires = {u: [e for e in trace.of_kind("fire") if e["src"] == u] for u in ("vmul", "vsum")}
    assert {u: len(x) for u, x in fires.items()} == {"vmul": 16, "vsum": 16}
    grants = [e for e in trace.of_kind("grant") if e["region"] == "tmp0"]
    assert {e["src"].split(".")[0] for e in grants if e["op"] == "w"} == {"vmul_out"}
    assert {e["src"].split(".")[0] for e in grants if e["op"] == "r"} == {"vsum_a"}
    assert {"vmul_a.lane0", "vsum_a.lane0", "vsum_out.lane0"} <= set(profile.ports)
