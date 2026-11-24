"""run/ on a recorded run of one accelerator of two groups sharing one streamer's movers:
mvt_v1_2_2 at W = 8, M = N = 8, trace levels task and stall (fixtures/mvt_v1_2_2/README.md),
converted without GVSoC.

The groups ``mvt`` (``vmul_x1`` linked into ``vsum_x1``, one SNAX-GVSoC component,
``group_mvt``) and ``vadd`` share the set ``mvt_vadd``: mover k of each role carries the k-th
widest port of each group, so reader0 is ``vmul_x1_a`` and ``vadd_a``, reader1 ``vmul_x1_b`` and
``vadd_b``, writer0 ``vsum_x1_out`` (its first channel) and ``vadd_out`` (8 channels). Each of the
four tasks starts every mover of the set; the converter names each group's lines after it, gives
each a profile and matches each group's tasks in order.
"""

import json

import pytest

from snax_forge.run import Cluster, Unit
from snax_forge.run.beats import pairing
from snax_forge.run.convert import convert, invariants
from snax_forge.run.sources import all_sources

from .helpers import load_recorded

SET = "mvt_vadd"
TASKS = ["tmp0_mult_mvt", "augassign_18_4_vadd", "tmp2_mult_mvt", "augassign_19_4_vadd"]


@pytest.fixture(scope="module")
def recorded():
    return load_recorded("mvt_v1_2_2")


@pytest.fixture(scope="module")
def converted(recorded):
    cluster, tasks, check, lines, regions, _ = recorded
    return convert(lines, cluster, tasks, check["total_cycles"], "task", check["functional_check"],
                   regions)  # fmt: skip


@pytest.fixture(scope="module")
def stalled():
    cluster, tasks, check, lines, regions, _ = load_recorded("mvt_v1_2_2", "events_stall.log")
    return cluster, regions, convert(lines, cluster, tasks, check["total_cycles"], "stall",
                                     check["functional_check"], regions, all_sources(cluster))  # fmt: skip


def test_the_cluster_has_a_unit_per_group_and_the_shared_movers(recorded):
    cluster, *_ = recorded
    a = cluster.accelerator
    assert (a.name, a.streamer, a.accelerators()) == (
        SET,
        f"{SET}_streamer",
        [(SET, ["mvt", "vadd"])],
    )
    first = a.csrs.sets[0].last + 1
    assert a.units == [Unit("mvt", "mvt", "group_mvt", first, first + 3, first + 4),
                       Unit("vadd", "vadd", "unit_vadd", first + 5, first + 7, first + 8)]  # fmt: skip
    assert [(m.name, m.gvsoc, m.set) for m in a.movers] == [
        (f"{SET}_reader0", "reader0", SET), (f"{SET}_reader1", "reader1", SET),
        (f"{SET}_writer0", "writer0", SET),
    ]  # fmt: skip
    assert [(m.users, m.user_units, m.user_channels) for m in a.movers] == [
        (("vmul_x1_a", "vadd_a"), ("vmul_x1", "vadd"), (8, 8)),
        (("vmul_x1_b", "vadd_b"), ("vmul_x1", "vadd"), (8, 8)),
        (("vsum_x1_out", "vadd_out"), ("vsum_x1", "vadd"), (1, 8)),
    ]
    movers = [m.name for m in a.movers]
    assert [m.name for m in a.movers_of("mvt")] == [m.name for m in a.movers_of("vadd")] == movers
    assert cluster.sources[:5] == ["mvt", "vadd", *movers]
    assert [a.block_of(first + k) for k in (0, 4, 5, 8, 9)] == ["mvt", "mvt", "vadd", "vadd", None]
    assert Cluster.from_dict(json.loads(json.dumps(cluster.to_dict()))) == cluster


def test_each_group_converts_with_its_own_tasks(recorded, converted):
    cluster, _, check, _, regions, _ = recorded
    profile, trace = converted
    assert invariants(profile, trace, cluster, None, regions) == []
    assert trace.total_cycles == check["total_cycles"] == 509
    assert {k: a.firings for k, a in profile.accelerators.items()} == {"mvt": 16, "vadd": 2}
    starts: dict[str, list[str]] = {}
    for e in trace.of_kind("start"):
        starts.setdefault(e["src"], []).append(e["task"])
    assert starts["mvt"] == ["tmp0_mult_mvt", "tmp2_mult_mvt"]
    assert starts["vadd"] == ["augassign_18_4_vadd", "augassign_19_4_vadd"]
    assert starts[f"{SET}_writer0"] == ["sum_vsum_x1_out", "augassign_18_4_vadd_out",
                                        "sum_1_vsum_x1_out", "augassign_19_4_vadd_out"]  # fmt: skip
    blocks = {e["block"] for e in trace.events if e["k"] in ("csr", "poll") and e["src"] == "core0"}
    assert {"mvt", "vadd", f"{SET}_streamer"} <= blocks <= {"mvt", "vadd", f"{SET}_streamer", "-"}
    assert {t: x["group"] for t, x in check["functional_check"]["tasks"].items()} == dict(
        zip(TASKS, ["mvt", "vadd", "mvt", "vadd"], strict=True))  # fmt: skip
    assert set(profile.streamers) == {m.name for m in cluster.accelerator.movers}


def test_each_task_pairs_its_groups_movers(recorded):
    cluster, tasks, *_ = recorded
    paired = pairing(tasks, cluster)
    assert list(paired) == TASKS
    assert paired["tmp0_mult_mvt"] == {f"{SET}_reader0": "tmp0_mult_vmul_x1_a",
                                       f"{SET}_reader1": "tmp0_mult_vmul_x1_b",
                                       f"{SET}_writer0": "sum_vsum_x1_out"}  # fmt: skip
    assert paired["augassign_18_4_vadd"] == {f"{SET}_reader0": "augassign_18_4_vadd_a",
                                             f"{SET}_reader1": "augassign_18_4_vadd_b",
                                             f"{SET}_writer0": "augassign_18_4_vadd_out"}  # fmt: skip


def test_the_stall_level_converts(stalled, converted):
    """convert checks the beat invariants itself at stall (it raises otherwise); the profile is
    the task level's."""
    cluster, regions, (profile, trace) = stalled
    assert invariants(profile, trace, cluster, None, regions) == []
    task_profile, _ = converted
    assert {k: (a.firings, a.cycles) for k, a in profile.accelerators.items()} == {
        k: (a.firings, a.cycles) for k, a in task_profile.accelerators.items()}  # fmt: skip


def test_each_group_reads_and_writes_through_the_shared_movers(stalled):
    """mvt's 8 firings a task fire on readers 0 and 1, its 8 pushes (a row's sum each) one word
    on writer0's lane 0; vadd's one firing a task pushes 8 words, one grant a lane; vadd reads
    the sum mvt wrote (tmp0, tmp2) on reader1."""
    _, _, (_, trace) = stalled
    fires = {u: [e for e in trace.of_kind("fire") if e["src"] == u] for u in ("mvt", "vadd")}
    assert {u: len(x) for u, x in fires.items()} == {"mvt": 16, "vadd": 2}
    assert {u: {b for e in x for b in e["b"]} for u, x in fires.items()} == {
        u: {f"{SET}_reader0", f"{SET}_reader1"} for u in ("mvt", "vadd")}  # fmt: skip
    pushes = [(e["src"], e["task"]) for e in trace.of_kind("push")]
    assert [pushes.count((u, t)) for u, t in zip(["mvt", "vadd"] * 2, TASKS, strict=True)] == [
        8,
        1,
        8,
        1,
    ]
    sums = [e for e in trace.of_kind("grant") if e["task"] == "sum_vsum_x1_out"]
    assert len(sums) == 8 and {e["src"] for e in sums} == {f"{SET}_writer0.lane0"}
    vadd = [e for e in trace.of_kind("grant") if e["task"] == "augassign_18_4_vadd_out"]
    assert len(vadd) == 8
    reads = {(e["task"], e["region"]) for e in trace.of_kind("grant") if e["op"] == "r"}
    assert {("augassign_18_4_vadd_b", "tmp0"), ("augassign_19_4_vadd_b", "tmp2")} <= reads
