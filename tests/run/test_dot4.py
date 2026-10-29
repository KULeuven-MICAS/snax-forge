"""run/ on an accelerator of one linked group (D148): dot 4, whose units mac and sum are one
SNAX-GVSoC component, ``snax_acc0/group_dot``, under one control model (snax-gvsoc's
SnaxGroup). The recorded run (fixtures/dot_4/README.md: dot 4 at N = 512, W = 4) converts
without GVSoC: the converter names the group's lines after the group, matches the start to the
group's task and gives the done its run's classes; beat levels are refused (open item 96).
"""

import pytest

from snax_forge.run import RunError, Unit
from snax_forge.run.convert import convert, invariants

from .helpers import load_recorded

# =============================================================================
# The recorded run (fixtures/dot_4)
# =============================================================================


@pytest.fixture(scope="module")
def recorded():
    return load_recorded("dot_4")


def test_the_recorded_run_converts_the_group_as_one_unit(recorded):
    """The group is one accelerator, dot, with its run's classes: 128 firings a cycle apart
    from the first (no gap), COUNTER 133 = C§17's C4 (JT + 2 = 130) + the streamers' fill 3,
    every busy cycle in one class; its task and its movers' each start once."""
    cluster, tasks, check, lines, regions, _ = recorded
    assert cluster.accelerator.units == [Unit("dot", "dot", "group_dot", *_csrs(cluster))]
    profile, trace = convert(lines, cluster, tasks, check["total_cycles"], "task",
                             check["functional_check"], regions)  # fmt: skip
    assert invariants(profile, trace, cluster, None, regions) == []
    (dot,) = profile.accelerators.values()
    assert list(profile.accelerators) == ["dot"] and dot.firings == 128
    cycles = {k: v for k, v in dot.cycles.items() if k != "idle"}
    assert cycles == {"busy": 129, "stall_in": 3, "stall_out": 0, "flush": 1}
    counter = check["functional_check"]["tasks"]["mult_dot"]["counter"]
    assert counter == sum(cycles.values()) == 128 + 2 + 3
    (done,) = [e for e in trace.of_kind("done") if e["src"] == "dot"]
    assert (done["task"], done["last"] - done["first"] + 1) == ("mult_dot", 128)
    starts = sorted((e["src"], e["task"]) for e in trace.of_kind("start") if e["src"] != "dma")
    assert starts == [("dot", "mult_dot"), ("mac_a", "mult_mac_a"), ("mac_b", "mult_mac_b"),
                      ("sum_out", "sum_final_sum_out")]  # fmt: skip


def test_the_recorded_run_refuses_beat_levels(recorded):
    cluster, tasks, check, lines, regions, _ = recorded
    with pytest.raises(RunError) as e:
        convert(lines, cluster, tasks, check["total_cycles"], "beat", None, regions, [])
    assert e.value.code == "convert.level"


def _csrs(cluster):
    first = cluster.accelerator.csrs.streamer_last + 1
    return first, first + 3, first + 4
