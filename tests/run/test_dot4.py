"""run/ on an accelerator of one linked group (D148): dot 4, whose units mac and sum are one
SNAX-GVSoC component, ``snax_acc0/group_dot``, under one control model (snax-gvsoc's
SnaxGroup).

The first tests' group lines are hand-written from snax-gvsoc's shared group case
``tests/snax/group/cases/dot4_n8_j1.txt``
(dot 4 at N = 8, W = 4: T = 2, JOBS 1, busy 4 cycles from the start): the start, the done and
the summary run line the component prints at its boundary (docs/EVENTS.md there), its
``count`` and ``firings`` the head unit's (the MAC's two), ``rate`` the head's firings per
result of the last unit (2). The converter names them after the group, matches the start to
the group's task and gives the done its run's classes; beat levels are refused (open item 96).
The last convert the recorded run of B1x's F3 (fixtures/dot_4/README.md: dot 4 at N = 512,
W = 4), without GVSoC.
"""

import json
from pathlib import Path

import pytest

from snax_forge.design import DesignPoint
from snax_forge.gen import generate
from snax_forge.lower.derive import task_list
from snax_forge.lower.tasks import TaskList
from snax_forge.run import RunError, Unit, load_cluster, read_events, regions_of
from snax_forge.run.convert import convert, invariants, source_of, trace_events
from snax_forge.run.events import parse_line
from tests.gen.helpers import designed

S = 100  # the start's cycle
GROUP = "/chip/soc/cluster_0/snax_acc0/group_dot"
# GVSoC's trace line: "<ps>: <cycle>: [<path>/<trace>] <cycle> <kind> <fields>"
LINES = [
    f"{S}00000: {S}: [{GROUP}/events] {S} start count=2 rate=2",
    f"{S + 4}00000: {S + 4}: [{GROUP}/events] {S + 4} done firings=2 first={S + 1} last={S + 2}",
    (
        f"{S + 4}00000: {S + 4}: [{GROUP}/summary] {S + 4} run index=0 firings=2 busy=3 stall_in=0 "
        "stall_out=0 flush=1"
    ),
]


@pytest.fixture(scope="module")
def dot4(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dot4")
    path = designed(tmp, "dot_4", "dot_4", ["N=8"])
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        manifest = generate(path, tmp / "gvsoc", tmp)
    cluster = load_cluster(tmp / "gvsoc" / "snax_forge.hjson", tmp / path)
    return manifest, cluster, task_list(DesignPoint.load(tmp / path))


def test_the_manifest_names_the_groups_component(dot4):
    manifest, *_ = dot4
    assert manifest.trace == "snax_acc0/group_dot"
    assert [(t.name, t.trace) for t in manifest.tasks] == [("mult_dot", None)]


def test_the_cluster_has_one_unit_for_the_group(dot4):
    _, cluster, _ = dot4
    a = cluster.accelerator
    first = a.csrs.streamer_last + 1
    assert a.units == [Unit("dot", "dot", "group_dot", first, first + 3, first + 4)]
    assert [m.name for m in a.movers] == ["mac_a", "mac_b", "sum_out"]


def test_the_groups_lines_convert_to_its_task(dot4):
    _, cluster, tasks = dot4
    named = [(source_of(x, cluster), x) for x in (parse_line(t) for t in LINES)]
    assert {s for s, _ in named} == {"dot"}
    events = [e for _, _, e in trace_events(named, cluster, tasks, 0, 10**6)]
    start, done = events
    assert (start["k"], start["src"], start["task"], start["t"]) == ("start", "dot", "mult_dot", S)
    assert (done["k"], done["task"], done["t"], done["firings"], done["first"], done["last"]) == (
        "done", "mult_dot", S + 5, 2, S + 1, S + 2)  # fmt: skip
    assert done["cycles"] == {"busy": 3, "stall_in": 0, "stall_out": 0, "flush": 1}


def test_beat_levels_are_refused(dot4):
    _, cluster, tasks = dot4
    with pytest.raises(RunError) as e:
        convert([parse_line(t) for t in LINES], cluster, tasks, 1000, "beat", None, [], [])
    assert e.value.code == "convert.level"


# =============================================================================
# The recorded run (fixtures/dot_4)
# =============================================================================

HERE = Path(__file__).parent / "fixtures" / "dot_4"


@pytest.fixture(scope="module")
def recorded():
    cluster = load_cluster(HERE / "snax_forge.hjson", HERE / "design_point.json")
    tasks = TaskList.from_dict(json.loads((HERE / "tasks.json").read_text()))
    check = json.loads((HERE / "check.json").read_text())
    point = DesignPoint.load(HERE / "design_point.json")
    regions = regions_of(point, json.loads((HERE / "elf_symbols.json").read_text()))
    return cluster, tasks, check, read_events(HERE / "events.log"), regions


def test_the_recorded_run_converts_the_group_as_one_unit(recorded):
    """The group is one accelerator, dot, with its run's classes: 128 firings a cycle apart
    from the first (no gap), COUNTER 133 = C§17's C4 (JT + 2 = 130) + the streamers' fill 3,
    every busy cycle in one class; its task and its movers' each start once."""
    cluster, tasks, check, lines, regions = recorded
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
    cluster, tasks, check, lines, regions = recorded
    with pytest.raises(RunError) as e:
        convert(lines, cluster, tasks, check["total_cycles"], "beat", None, regions, [])
    assert e.value.code == "convert.level"


def _csrs(cluster):
    first = cluster.accelerator.csrs.streamer_last + 1
    return first, first + 3, first + 4
