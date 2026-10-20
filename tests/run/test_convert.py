"""run/convert.py on snippets and on a synthetic run: names, the window, the trace, the
profile, the invariants (D137, C§7)."""

import json
from pathlib import Path

import pytest

from snax_forge.lower.tasks import TaskList
from snax_forge.run import RunError, cluster_from, parse_line
from snax_forge.run.convert import convert, direction, invariants, port_of, source_of

from .helpers import BOOT, MAIN, files, line, synthetic

TASKS = Path(__file__).parent / "fixtures" / "vecadd" / "tasks.json"


@pytest.fixture
def cluster():
    return cluster_from(*files())


@pytest.fixture
def tasks():
    return TaskList.from_dict(json.loads(TASKS.read_text()))


def parse(lines):
    return [parse_line(x) for x in lines]


@pytest.mark.parametrize(
    "path, trace, text, name",
    [
        ("snax_acc0/unit", "events", "5 start count=16 rate=1", "acc"),
        ("snax_acc0/streamer", "events", "5 start mover=reader1", "acc_b"),
        ("snax_acc0/streamer", "events", "5 done mover=writer0", "acc_out"),
        ("pe1", "events", "5 state mode=run", "core1"),
        ("idma/fe", "events", "5 done id=2", "dma"),
        ("tcdm/snax_xbar", "summary", "5 bank index=0 reads=0 writes=0 conflicts=0", "xbar"),
    ],
)
def test_paths_are_renamed(cluster, path, trace, text, name):
    assert source_of(parse_line(line(path, trace, text)), cluster) == name


@pytest.mark.parametrize(
    "path, text",
    [("snax_acc1/unit", "5 start count=1 rate=1"), ("pe2", "5 state mode=run"),
     ("snax_acc0/streamer", "5 start mover=reader7")],
)  # fmt: skip
def test_a_path_not_of_the_cluster_is_named(cluster, path, text):
    with pytest.raises(RunError) as e:
        source_of(parse_line(line(path, "events", text)), cluster)
    assert e.value.code in ("convert.path", "cluster.mover")


def test_crossbar_inputs_become_ports(cluster):
    assert port_of("snax_acc0.streamer.reader1.lane3", cluster) == "acc_b.lane3"
    assert port_of("snax_acc0.streamer.writer0.lane0", cluster) == "acc_out.lane0"
    assert [port_of(x, cluster) for x in ("core0", "core1", "axi")] == ["core0", "core1", "axi"]
    with pytest.raises(RunError):
        port_of("core5", cluster)


def test_dma_direction_by_the_l1(cluster):
    l1, l2 = 0x10000200, 0x80002900
    assert [direction(a, b, cluster) for a, b in ((l2, l1), (l1, l2), (l1, l1), (l2, l2))] == [
        "l2_to_l1",
        "l1_to_l2",
        "l1_to_l1",
        "l2_to_l2",
    ]


def test_the_synthetic_run(cluster, tasks):
    profile, trace = convert(parse(synthetic(cluster, tasks)), cluster, tasks, 226, "task",
                             {"passed": True})  # fmt: skip
    assert (trace.boot, trace.total_cycles) == (BOOT, 226)
    assert invariants(profile, trace, cluster) == []
    assert profile.functional_check == {"passed": True}
    # every event in the window, in time order, boot dropped
    ts = [e["t"] for e in trace.events]
    assert ts == sorted(ts) and 0 <= ts[0] and ts[-1] < trace.total_cycles
    # func lines stay as function spans; main's entry is before the window, its read at 0
    assert [e["name"] for e in trace.of_kind("func", "core0")] == [
        "forge_core_0",
        "main",
        "forge_check",
    ]
    assert not [e for e in trace.events if e["k"].startswith("mcycle")]
    # tasks named from the task list
    assert [e["task"] for e in trace.of_kind("issue")] == ["load_A", "load_B", "store_C"]
    assert [e["dir"] for e in trace.of_kind("issue")] == ["l2_to_l1", "l2_to_l1", "l1_to_l2"]
    assert [e["task"] for e in trace.of_kind("start")] == [
        "add_acc_a",
        "add_acc_b",
        "add_acc_out",
        "add_acc",
    ]
    (done,) = trace.of_kind("done", "acc")
    (start,) = trace.of_kind("start", "acc")
    assert done["firings"] == 16 and done["t"] == done["last"] + 1 > start["t"]


def test_poll_and_barrier_lines(cluster, tasks):
    profile, trace = convert(parse(synthetic(cluster, tasks)), cluster, tasks, 226, "task")
    (poll,) = trace.of_kind("poll", "core0")
    assert (poll["block"], poll["csr"], poll["value"]) == ("acc", 0x3D4, 1)
    reads = [e for e in trace.of_kind("csr", "core0") if e["op"] == "r"]
    assert [(e["block"], e["value"], e["t"]) for e in reads] == [("acc", 0, poll["last"] + 1)]
    writes = [e["block"] for e in trace.of_kind("csr", "core0") if e["op"] == "w"]
    assert writes == ["acc_streamer", "acc"]
    core0 = profile.cores["core0"]
    # the boot's barrier (before main) is dropped, the one after the run kept
    assert [(w.block, w.last - w.first) for w in core0.waits] == [
        ("acc", poll["last"] - poll["t"]),
        ("barrier", 3),
    ]
    assert core0.cycles["poll"] == poll["last"] - poll["t"] + 1
    assert core0.cycles["barrier"] == 4
    assert sum(core0.cycles.values()) == trace.total_cycles
    assert profile.cores["core1"].cycles == {"run": 225, "poll": 0, "barrier": 1, "sleep": 0}


def test_a_barrier_over_a_poll_and_sleep_under_a_poll(cluster, tasks):
    lines = synthetic(cluster, tasks)
    extra = [
        line("pe0", "events", f"{MAIN + 200} state mode=sleep", MAIN + 200),
        line("pe0", "events", f"{MAIN + 202} poll pc=0x0 csr=0x3d0 value=1 first={MAIN + 202} "
             f"last={MAIN + 210} count=5", MAIN + 202),
        line("pe0", "events", f"{MAIN + 205} barrier first={MAIN + 205} last={MAIN + 207}",
             MAIN + 205),
        line("pe0", "events", f"{MAIN + 215} state mode=run", MAIN + 215),
    ]  # fmt: skip
    profile, trace = convert(parse(lines + extra), cluster, tasks, 226, "task")
    c = profile.cores["core0"]
    assert sum(c.cycles.values()) == trace.total_cycles
    assert c.cycles["sleep"] == 2 + 4  # 200, 201, then 211 to 214
    assert [w.block for w in c.waits][-2:] == ["acc_streamer", "barrier"]


def test_main_and_forge_check_are_needed(cluster, tasks):
    lines = [x for x in synthetic(cluster, tasks) if "forge_check" not in x]
    with pytest.raises(RunError, match="forge_check") as e:
        convert(parse(lines), cluster, tasks, 226, "task")
    assert e.value.code == "convert.window"


@pytest.mark.parametrize("cycles", [225, 227])
def test_forge_cycles_must_be_the_reads_apart(cluster, tasks, cycles):
    """The two mcycle reads are 226 cycles apart: any other forge cycles is a named error."""
    with pytest.raises(RunError, match="226 cycles apart") as e:
        convert(parse(synthetic(cluster, tasks)), cluster, tasks, cycles, "task")
    assert e.value.code == "convert.window"


def test_the_reads_are_picked_by_their_function_spans(cluster, tasks):
    """A read of mcycle outside main's span (here after forge_core_0 is entered) is not main's."""
    lines = synthetic(cluster, tasks)
    moved = [x.replace(f"{BOOT} mcycle", f"{MAIN + 12} mcycle") if "/pe0/" in x else x
             for x in lines]  # fmt: skip
    with pytest.raises(RunError, match="no mcycle read in main") as e:
        convert(parse(moved), cluster, tasks, 226, "task")
    assert e.value.code == "convert.window"
    late = [x.replace(f"{BOOT + 226} mcycle", f"{MAIN + 226 + 40} mcycle") for x in lines]
    with pytest.raises(RunError, match="no mcycle read in forge_check"):
        convert(parse(late), cluster, tasks, 266, "task")


def test_a_copy_against_its_tasks_direction(cluster, tasks):
    lines = [x.replace("src=0x80000000", "src=0x10010000") for x in synthetic(cluster, tasks)]
    with pytest.raises(RunError, match="load_A") as e:
        convert(parse(lines), cluster, tasks, 226, "task")
    assert e.value.code == "convert.tasks"


def test_a_start_without_its_done_breaks_an_invariant(cluster, tasks):
    lines = [x for x in synthetic(cluster, tasks) if "done mover=writer0" not in x]
    with pytest.raises(RunError, match="acc_out: 1 starts and 0 dones") as e:
        convert(parse(lines), cluster, tasks, 226, "task")
    assert e.value.code == "convert.invariant"


def test_firings_against_the_summary_break_an_invariant(cluster, tasks):
    lines = [x.replace("done firings=16", "done firings=15") for x in synthetic(cluster, tasks)]
    with pytest.raises(RunError, match="15 firings in its done events, 16 in its summary"):
        convert(parse(lines), cluster, tasks, 226, "task")


def test_off_is_the_summary_lines_alone(cluster, tasks):
    lines = synthetic(cluster, tasks, level="off")
    assert all("/summary" in x for x in lines)
    profile, trace = convert(parse(lines), cluster, tasks, 226, "off")
    assert trace is None
    a = profile.accelerators["acc"]
    assert a.cycles == {"busy": 16, "stall_in": 2, "stall_out": 0, "flush": 0, "idle": 208}
    assert (a.firings, a.utilisation) == (16, 16 / 226)
    s = profile.streamers["acc_out"]
    assert (s.write, s.cycles["idle"], s.fifo.name, s.fifo.depth, s.fifo.max) == (
        True,
        226 - 17,
        "acc_out.fifo",
        8,
        [2, 2, 2, 2],
    )
    assert s.ports == [f"acc_out.lane{i}" for i in range(4)]
    assert (profile.cores["core0"].cycles, profile.cores["core0"].csr_writes) == (None, 18)
    assert profile.dmas["dma"].transfers == 3
    assert profile.ports["acc_b.lane2"].grants == 16 and profile.ports["core1"].whole_run
    assert len(profile.banks.reads) == 32
    # the same summary with the events: the same profile but the cores'
    full, _ = convert(parse(synthetic(cluster, tasks)), cluster, tasks, 226, "task")
    for c in full.cores.values():
        c.cycles, c.waits = None, []
    assert full == profile
