"""run/ on a recorded run of two groups sharing one streamer's movers (D169): dot_1_shared at
trace levels task and stall (fixtures/dot_1_shared/README.md), converted without GVSoC.

vmul and vsum share the set dot: mover k of each role carries the k-th widest port of each group,
so dot_reader0 is vmul_a and vsum_a, dot_reader1 vmul_b only, dot_writer0 vmul_out (4 channels)
and vsum_out (its first channel, ENABLED_CHANNEL 1). Each group's task starts every mover of the
set, vsum's with vmul_b's idle task (every bound 0) on dot_reader1.
"""

import pytest

from snax_forge.run.beats import pairing
from snax_forge.run.convert import convert, invariants

from .helpers import load_recorded


@pytest.fixture(scope="module")
def recorded():
    return load_recorded("dot_1_shared")


@pytest.fixture(scope="module")
def stalled():
    from snax_forge.run.sources import all_sources

    cluster, tasks, check, lines, regions, _ = load_recorded("dot_1_shared", "events_stall.log")
    return cluster, regions, convert(lines, cluster, tasks, check["total_cycles"], "stall",
                                     check["functional_check"], regions, all_sources(cluster))  # fmt: skip


def test_the_cluster_has_the_shared_movers_with_their_users(recorded):
    cluster, *_ = recorded
    a = cluster.accelerator
    assert [(m.name, m.gvsoc, m.set) for m in a.movers] == [
        ("dot_reader0", "reader0", "dot"), ("dot_reader1", "reader1", "dot"),
        ("dot_writer0", "writer0", "dot"),
    ]  # fmt: skip
    assert [(m.users, m.user_units, m.user_channels) for m in a.movers] == [
        (("vmul_a", "vsum_a"), ("vmul", "vsum"), (4, 4)),
        (("vmul_b",), ("vmul",), (4,)),
        (("vmul_out", "vsum_out"), ("vmul", "vsum"), (4, 1)),
    ]
    assert [m.name for m in a.movers_of("vmul")] == ["dot_reader0", "dot_reader1", "dot_writer0"]
    assert [m.name for m in a.movers_of("vsum")] == ["dot_reader0", "dot_writer0"]
    assert cluster.sources[:5] == ["vmul", "vsum", "dot_reader0", "dot_reader1", "dot_writer0"]


def test_the_task_level_converts(recorded):
    cluster, tasks, check, lines, regions, _ = recorded
    profile, trace = convert(lines, cluster, tasks, check["total_cycles"], "task",
                             check["functional_check"], regions)  # fmt: skip
    assert invariants(profile, trace, cluster, None, regions) == []
    assert trace.total_cycles == check["total_cycles"] == 287
    assert {k: a.firings for k, a in profile.accelerators.items()} == {"vmul": 16, "vsum": 16}
    assert set(profile.streamers) == {"dot_reader0", "dot_reader1", "dot_writer0"}


def test_the_stall_level_converts(stalled, recorded):
    """convert checks the beat invariants itself at stall (it raises otherwise); the profile is
    the task level's."""
    cluster, regions, (profile, trace) = stalled
    assert invariants(profile, trace, cluster, None, regions) == []
    rc, rtasks, rcheck, rlines, rregions, _ = recorded
    task_profile, _ = convert(rlines, rc, rtasks, rcheck["total_cycles"], "task",
                              rcheck["functional_check"], rregions)  # fmt: skip
    assert {k: (a.firings, a.cycles) for k, a in profile.accelerators.items()} == {
        k: (a.firings, a.cycles) for k, a in task_profile.accelerators.items()}  # fmt: skip


def test_vsum_pushes_one_word_granted_once_on_writer0(stalled):
    """vsum's out is 1 word on writer0's first channel: each push holds 1 word, and writer0 has
    one grant per push in vsum's task, on lane 0."""
    _, _, (_, trace) = stalled
    pushes = [e for e in trace.of_kind("push") if e["src"] == "vsum"]
    assert len(pushes) == 1 and all(len(e["value"]) == 1 for e in pushes)
    grants = [e for e in trace.of_kind("grant")
              if e["src"].startswith("dot_writer0.") and e["task"] == "sum_vsum_out"]  # fmt: skip
    assert len(grants) == len(pushes)
    assert [(e["src"], e["vals"]) for e in grants] == [("dot_writer0.lane0", pushes[0]["vals"])]
    vmul = [e for e in trace.of_kind("grant")
            if e["src"].startswith("dot_writer0.") and e["task"] == "mult_vmul_out"]  # fmt: skip
    assert len(vmul) == 16 * 4


def test_each_task_pairs_its_groups_movers(recorded, stalled):
    """Each unit task's movers are its own group's; on dot_reader1, which vsum does not use, vsum's
    task pairs vmul_b's idle task (bound 0), busy 0 cycles."""
    cluster, tasks, *_ = recorded
    assert pairing(tasks, cluster) == {
        "mult_vmul": {"dot_reader0": "mult_vmul_a", "dot_reader1": "mult_vmul_b",
                      "dot_writer0": "mult_vmul_out"},
        "sum_vsum": {"dot_reader0": "sum_vsum_a", "dot_reader1": "sum_vmul_b",
                     "dot_writer0": "sum_vsum_out"},
    }  # fmt: skip
    _, _, (_, trace) = stalled
    starts: dict[str, list[str]] = {}
    for e in trace.of_kind("start"):
        starts.setdefault(e["src"], []).append(e["task"])
    assert starts["dot_reader0"] == ["mult_vmul_a", "sum_vsum_a"]
    assert starts["dot_reader1"] == ["mult_vmul_b", "sum_vmul_b"]
    assert starts["dot_writer0"] == ["mult_vmul_out", "sum_vsum_out"]
    idle = [e for e in trace.of_kind("done") if e["task"] == "sum_vmul_b"]
    assert [e["cycles"]["busy"] for e in idle] == [0]
    fires = {u: {b for e in trace.of_kind("fire") if e["src"] == u for b in e["b"]}
             for u in ("vmul", "vsum")}  # fmt: skip
    assert fires == {"vmul": {"dot_reader0", "dot_reader1"}, "vsum": {"dot_reader0"}}
