"""run/ on a recorded run of two accelerators in two streamer sets: mvt_v5_sequential at
M = N = 16, trace level task (fixtures/mvt_v5_sequential/README.md), converted without GVSoC.

The accelerators ``mvt`` (a linked group, ``vmul_x1`` into a seeded ``vsum_x1``, one SNAX-GVSoC
component, ``group_mvt``, under one control model) and ``mac_x2`` (one unit, ``unit_mac_x2``)
each have their own set, its streamer at its own path and its CSRs after it. The converter names
each streamer's lines at its own path, each task starting its own set's movers only (no idle
task), the linked group's lines after the group, its done with its run's classes.
"""

import pytest

from snax_forge.run import Unit
from snax_forge.run.convert import convert, invariants

from .helpers import load_recorded


@pytest.fixture(scope="module")
def recorded():
    return load_recorded("mvt_v5_sequential")


@pytest.fixture(scope="module")
def converted(recorded):
    cluster, tasks, check, lines, regions, _ = recorded
    return convert(lines, cluster, tasks, check["total_cycles"], "task", check["functional_check"],
                   regions)  # fmt: skip


def test_a_streamer_per_set(recorded):
    """mvt: streamer 960-986, its CSRs 987-991; mac_x2: 992-1018, 1019-1023."""
    cluster, *_ = recorded
    a = cluster.accelerator
    assert (a.name, a.accelerators()) == (
        "snax_forge_mvt_v5",
        [("mvt", ["mvt"]), ("mac_x2", ["mac_x2"])],
    )
    assert [(x.name, x.first, x.last, x.trace) for x in a.csrs.sets] == [
        ("mvt", 960, 986, "snax_acc0/streamer_mvt"),
        ("mac_x2", 992, 1018, "snax_acc0/streamer_mac_x2"),
    ]
    assert a.units == [Unit("mvt", "mvt", "group_mvt", 987, 990, 991),
                       Unit("mac_x2", "mac_x2", "unit_mac_x2", 1019, 1022, 1023)]  # fmt: skip
    assert [(m.name, m.gvsoc, m.set) for m in a.movers] == [
        ("vmul_x1_a", "reader0", "mvt"), ("vmul_x1_b", "reader1", "mvt"),
        ("vsum_x1_seed", "reader2", "mvt"), ("vsum_x1_out", "writer0", "mvt"),
        ("mac_x2_a", "reader0", "mac_x2"), ("mac_x2_b", "reader1", "mac_x2"),
        ("mac_x2_seed", "reader2", "mac_x2"), ("mac_x2_out", "writer0", "mac_x2"),
    ]  # fmt: skip
    assert (a.window("mvt"), a.window("mac_x2")) == ((960, 991), (992, 1023))
    assert [a.block_of(x) for x in (960, 987, 992, 1019, 1024)] == [
        "mvt_streamer", "mvt", "mac_x2_streamer", "mac_x2", None]  # fmt: skip
    assert [m.name for m in a.movers_of("mac_x2")] == ["mac_x2_a", "mac_x2_b", "mac_x2_seed",
                                                       "mac_x2_out"]  # fmt: skip


def test_it_converts_set_by_set(recorded, converted):
    """Each unit's task starts its own set's movers only, no idle task; each streamer and each
    accelerator's CSRs named."""
    cluster, _, check, _, regions, _ = recorded
    profile, trace = converted
    assert invariants(profile, trace, cluster, None, regions) == []
    assert trace.total_cycles == check["total_cycles"] == 521
    starts: dict[str, list[str]] = {}
    for e in trace.of_kind("start"):
        starts.setdefault(e["src"], []).append(e["task"])
    assert (starts["mvt"], starts["mac_x2"]) == (["tmp1_mult_mvt"], ["tmp4_mult_mac_x2"])
    assert (starts["vsum_x1_seed"], starts["mac_x2_a"]) == (
        ["sum_vsum_x1_seed"],
        ["tmp4_mult_mac_x2_a"],
    )
    assert not any(e.get("idle") for e in trace.events)
    blocks = {e["block"] for e in trace.events if e["k"] in ("csr", "poll") and e["src"] == "core0"}
    assert blocks == {"mvt", "mac_x2", "mvt_streamer", "mac_x2_streamer"}
    assert set(profile.streamers) == {m.name for m in cluster.accelerator.movers}


def test_the_linked_group_converts_as_one_unit(recorded, converted):
    """mvt's group is one accelerator with its run's classes: 32 firings a cycle apart (no gap),
    BUSY_COUNTER 34, every busy cycle in one class; mac_x2's 32 firings after a fill of 3,
    BUSY_COUNTER 36."""
    check = recorded.check
    profile, trace = converted
    cycles = {k: {c: n for c, n in a.cycles.items() if c != "idle"}
              for k, a in profile.accelerators.items()}  # fmt: skip
    assert {k: a.firings for k, a in profile.accelerators.items()} == {"mvt": 32, "mac_x2": 32}
    assert cycles == {"mvt": {"busy": 33, "stall_in": 0, "stall_out": 0, "flush": 1},
                      "mac_x2": {"busy": 32, "stall_in": 3, "stall_out": 0, "flush": 1}}  # fmt: skip
    tasks = check["functional_check"]["tasks"]
    assert [tasks[t]["counter"] for t in ("tmp1_mult_mvt", "tmp4_mult_mac_x2")] == [34, 36]
    assert [sum(c.values()) for c in cycles.values()] == [34, 36]
    dones = [(e["src"], e["task"], e["last"] - e["first"] + 1) for e in trace.of_kind("done")
             if e["src"] in ("mvt", "mac_x2")]  # fmt: skip
    assert dones == [("mvt", "tmp1_mult_mvt", 32), ("mac_x2", "tmp4_mult_mac_x2", 32)]
