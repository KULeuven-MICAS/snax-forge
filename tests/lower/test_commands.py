"""Tests for the lowering of task lists into programs (LOW1b, D45, D64).

LOW1b acceptance (docs/STATUS.md): the program lowered from
scenarios/vecadd/tasks.json equals vecadd's hand-scheduled program, the one
scenarios/vecadd held before its program was generated (D65, D67), written
out here without the lowering. Plus: every scenario's generated program is
its task list lowered and keeps its cycle count, the wait rule on small
cases, and the errors that need the cluster.

Sections:
  1. the scenarios' task lists
  2. configure and start
  3. which waits a start gets
  4. sync and read
  5. errors that need the cluster
"""

import pytest

from snax_forge.lower import Program, TaskList, TaskListError, lower_program, upstream
from snax_forge.snax_model import DmaDescriptor, DmaPattern, StreamerRegs
from snax_forge.snax_model.scenario import Scenario, run

from .helpers import (
    SCEN,
    TASK_SCENARIOS,
    add,
    cluster,
    dma,
    lowered,
    shape,
    start,
    stream,
    sync,
)

# =============================================================================
# 1. The scenarios' task lists
# =============================================================================


def vecadd_by_hand() -> list:
    """vecadd's program as scheduled by hand (test_profile.run_vecadd, and the
    program scenarios/make.py wrote before D64): the LOW1b reference."""
    p = Program(cluster("alu4"))

    def beats(base: int) -> DmaPattern:
        return DmaPattern(base, (8,), (64,))

    def unit(word: int) -> StreamerRegs:
        return StreamerRegs(word * 8, (16,), (32,), (4,), (8,))

    p.config("dma", DmaDescriptor("l2_to_l1", beats(0), beats(0)))
    p.start("dma")
    p.config("dma", DmaDescriptor("l2_to_l1", beats(1024), beats(576)))
    p.wait("dma", "poll")
    p.start("dma")
    for block, word in (("ra", 0), ("rb", 72), ("wr", 144)):
        p.config(block, unit(word))
    p.config("acc", {"n": 16})
    p.wait("dma", "poll")
    for block in ("ra", "rb", "wr", "acc"):
        p.start(block)
    p.config("dma", DmaDescriptor("l1_to_l2", beats(1152), beats(2048)))
    p.wait("wr", "poll")
    p.start("dma")
    p.wait("dma", "poll")
    return p.cmds


def test_vecadd_task_list_gives_the_vecadd_program():
    """LOW1b acceptance, against a program written without the lowering."""
    program = lower_program(TaskList.load(SCEN / "vecadd" / "tasks.json"), cluster("alu4"))
    assert [c.to_dict() for c in program] == [c.to_dict() for c in vecadd_by_hand()]
    assert len(program) == 57


@pytest.mark.parametrize("name", TASK_SCENARIOS)
def test_the_scenario_program_is_its_task_list_lowered(name):
    sc = Scenario.load(SCEN / name / "scenario.json")
    tasks = TaskList.load(SCEN / name / "tasks.json")
    assert tasks.name == name
    program = lower_program(tasks, sc.cluster)
    assert [c.to_dict() for c in program] == [c.to_dict() for c in sc.program]


# The cycle counts of the scenarios as they were checked in before their programs
# were generated (D67): a change to a task list or to the lowering that moves one
# of them shows up here. fmul's program is the one it was scheduled by hand with.
CYCLES = {"vecadd": 77, "vecadd_conflict": 85, "vecadd_tiled": 471, "fmul": 525, "reduce": 35,
          "dma": 65}  # fmt: skip


@pytest.mark.parametrize("name", TASK_SCENARIOS)
def test_the_scenario_keeps_its_cycles(name):
    assert run(Scenario.load(SCEN / name / "scenario.json")).total_cycles == CYCLES[name]


def test_vecadd_step_by_step():
    """The walk-through of D64: configs before the waits, one wait per dependency."""
    tasks = TaskList.load(SCEN / "vecadd" / "tasks.json")
    program = [c.to_dict() for c in lower_program(tasks, cluster("alu4"))]
    assert shape(program) == [
        "Cdma", "Sdma",
        "Cdma", "Wdma:poll", "Sdma",
        "Cra", "Crb", "Cwr", "Cacc", "Wdma:poll", "Sra", "Srb", "Swr", "Sacc",
        "Cdma", "Wwr:poll", "Sdma",
        "Wdma:poll",
    ]  # fmt: skip


# =============================================================================
# 2. Configure and start
# =============================================================================


def test_configure_writes_every_register_through_the_adapter():
    """Values in the type's own terms become every register, unused loops padded."""
    prog = lowered(dma("x", src=1024, dst=576), start("x"))
    writes = {c["reg"]: c["value"] for c in prog if c["op"] == "csr_write"}
    assert writes["dma.direction"] == 0
    assert (writes["dma.src_base"], writes["dma.dst_base"]) == (1024, 576)
    assert (writes["dma.src_bound[0]"], writes["dma.src_bound[1]"]) == (8, 1)
    assert (writes["dma.src_stride[0]"], writes["dma.src_stride[1]"]) == (64, 0)
    assert prog[-1] == {"op": "csr_write", "reg": "dma.start", "value": 1}


def test_configure_can_run_ahead_of_its_start():
    """A configure goes where it is written: here while the DMA's first task runs."""
    prog = lowered(
        dma("a"), start("a"),
        stream("r", "ra", after=["a"]),
        dma("b", src=512, dst=512), start("b"),
        start("r"),
    )  # fmt: skip
    assert shape(prog) == ["Cdma", "Sdma", "Cra", "Cdma", "Wdma:poll", "Sdma", "Sra"]


def test_start_launches_its_tasks_in_list_order():
    prog = lowered(
        stream("w", "wr"), stream("r", "ra"), stream("s", "rb"), add("x"),
        start("x", "s", "r", "w"),
    )  # fmt: skip
    assert shape(prog)[-4:] == ["Sacc", "Srb", "Sra", "Swr"]


# =============================================================================
# 3. Which waits a start gets
# =============================================================================


def test_a_busy_component_is_waited_for():
    """Same component, no after: the DMA runs one task at a time."""
    prog = lowered(dma("a"), start("a"), dma("b", src=512), start("b"))
    assert shape(prog) == ["Cdma", "Sdma", "Cdma", "Wdma:poll", "Sdma"]


def test_no_after_no_wait_across_components():
    """Nothing ties the streamer to the load unless ``after`` says so."""
    prog = lowered(dma("a"), start("a"), stream("r", "ra"), start("r"))
    assert shape(prog) == ["Cdma", "Sdma", "Cra", "Sra"]


def test_one_wait_per_component_covers_its_earlier_tasks():
    """after on both loads: one wait on the DMA, which ends on its latest task."""
    prog = lowered(
        dma("a"), start("a"), dma("b", src=512, dst=512), start("b"),
        stream("r", "ra", after=["a", "b"]), start("r"),
    )  # fmt: skip
    assert shape(prog)[-3:] == ["Cra", "Wdma:poll", "Sra"]
    assert shape(prog).count("Wdma:poll") == 2  # the first one is b's busy wait


def test_a_covered_dependency_gets_no_wait():
    prog = lowered(dma("a"), start("a"), sync("a"), stream("r", "ra", after=["a"]), start("r"))
    assert shape(prog) == ["Cdma", "Sdma", "Wdma:poll", "Cra", "Sra"]


def test_waits_are_ordered_by_when_their_task_started():
    """ra's earlier task started before the DMA's load, so ra is waited for first."""
    first_ra = [stream("r0", "ra"), start("r0"), dma("a"), start("a")]
    prog = lowered(*first_ra, stream("r1", "ra", after=["a"]), start("r1"))
    assert shape(prog)[-3:] == ["Wra:poll", "Wdma:poll", "Sra"]
    first_dma = [dma("a"), start("a"), stream("r0", "ra"), start("r0")]
    prog = lowered(*first_dma, stream("r1", "ra", after=["a"]), start("r1"))
    assert shape(prog)[-3:] == ["Wdma:poll", "Wra:poll", "Sra"]


def test_the_wait_mode_is_the_waited_tasks():
    prog = lowered(
        dma("a", wait_mode="signal"), start("a"), stream("r", "ra", after=["a"]), start("r")
    )
    assert "Wdma:signal" in shape(prog)


def group(k: int, after: list[str] = ()) -> list[dict]:
    """ra, rb, wr and acc configured and started together as task group k."""
    return [
        stream(f"ra{k}", "ra", after=list(after)), stream(f"rb{k}", "rb", word=72),
        stream(f"wr{k}", "wr", word=144), add(f"acc{k}"),
        start(f"ra{k}", f"rb{k}", f"wr{k}", f"acc{k}"),
    ]  # fmt: skip


def test_a_wait_on_the_writer_covers_its_group():
    """wr finishes last in its group: after waiting for it, ra, rb and acc are free."""
    prog = lowered(
        *group(0), dma("c", "l1_to_l2", src=1152, dst=2048, after=["wr0"]), start("c"),
        *group(1),
    )  # fmt: skip
    tail = shape(prog)[shape(prog).index("Sdma") + 1 :]
    assert tail == ["Cra", "Crb", "Cwr", "Cacc", "Sra", "Srb", "Swr", "Sacc"]


def test_a_needed_wait_the_writer_covers_is_left_out():
    """ra and rb are busy too, but waiting for wr covers them (fmul's tiles, D66)."""
    prog = lowered(*group(0), *group(1))
    assert shape(prog)[-5:] == ["Wwr:poll", "Sra", "Srb", "Swr", "Sacc"]


def test_the_writer_covers_only_tasks_started_with_it():
    """ra's task started on its own: waiting for wr says nothing about it."""
    prog = lowered(
        stream("r", "ra"), start("r"),
        stream("s", "rb"), stream("w", "wr"), add("x"), start("s", "w", "x"),
        sync("w"),
        stream("r1", "ra"), stream("s1", "rb"), start("r1", "s1"),
    )  # fmt: skip
    assert shape(prog)[-3:] == ["Wra:poll", "Sra", "Srb"]


def test_upstream_follows_attach_and_the_write_flag():
    assert upstream(cluster("alu4")) == {"acc": ["ra", "rb"], "wr": ["acc"]}
    assert upstream(cluster("red4")) == {"acc": ["ra"], "wr": ["acc"]}


# =============================================================================
# 4. Sync and read
# =============================================================================


def test_sync_is_always_emitted():
    """A sync is a point in the program, even when a wait already covers it."""
    prog = lowered(dma("a"), start("a"), sync("a", "signal"), sync("a"))
    assert shape(prog) == ["Cdma", "Sdma", "Wdma:signal", "Wdma:poll"]


def test_sync_covers_later_dependencies():
    prog = lowered(dma("a"), start("a"), sync("a"), dma("b", after=["a"]), start("b"))
    assert shape(prog) == ["Cdma", "Sdma", "Wdma:poll", "Cdma", "Sdma"]


def test_read_is_a_csr_read_by_name():
    prog = lowered(dma("a"), start("a"), sync("a"), {"op": "read", "reg": "dma.busy_cycles"})
    assert prog[-1] == {"op": "csr_read", "reg": "dma.busy_cycles"}


# =============================================================================
# 5. Errors that need the cluster
# =============================================================================


def bad_values(values: dict) -> dict:
    return {"op": "configure", "task_name": "x", "type": "dma", "component": "dma",
            "values": values}  # fmt: skip


GOOD_DMA = dma("x")["values"]


@pytest.mark.parametrize(
    "steps, match",
    [
        ([{**dma("x"), "component": "dma2"}, start("x")], "no component 'dma2'"),
        ([{**dma("x"), "type": "streamer"}, start("x")], "dma is a dma, not a streamer"),
        ([{**dma("x"), "type": "gpu", "component": "dma"}, start("x")], "is a dma, not a gpu"),
        ([bad_values({k: v for k, v in GOOD_DMA.items() if k != "dst"}), start("x")], "missing"),
        ([bad_values({**GOOD_DMA, "burst": 4}), start("x")], "unknown"),
        ([bad_values({**GOOD_DMA, "direction": "l1_to_l1"}), start("x")], "direction"),
        ([bad_values({**GOOD_DMA, "src": {"base": 0, "bounds": [8]}}), start("x")], "missing"),
        (
            [
                bad_values(
                    {**GOOD_DMA, "src": {"base": 0, "bounds": [8, 1, 1], "strides": [64, 0, 0]}}
                ),
                start("x"),
            ],
            "loops",
        ),
        ([{**add("x"), "values": {"n": 16, "T": 4}}, start("x")], "parameters"),
        ([{**stream("x", "ra"), "values": {"base": 0}}, start("x")], "missing"),
        (
            [
                bad_values({**GOOD_DMA, "src": {"base": 0, "bounds": 8, "strides": [64]}}),
                start("x"),
            ],
            "must be a list",
        ),
        ([dma("x"), start("x"), sync("x"), {"op": "read", "reg": "dma.nothing"}], "no register"),
        ([dma("x"), start("x"), sync("x"), {"op": "read", "reg": "ctl.busy"}], "no register"),
    ],
)
def test_errors_that_need_the_cluster(steps, match):
    with pytest.raises(TaskListError, match=match):
        lowered(*steps)
