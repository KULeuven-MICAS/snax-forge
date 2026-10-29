"""Where a task list waits (D64): ``placed_waits`` on vecadd's and vsum's design points.

Each test writes a task list and reads its placed waits back in the shape of
helpers.shape: configures, waits and starts in program order.

Sections:
  1. configure and start
  2. which waits a start gets
  3. sync
  4. vecadd's and vsum's derived task lists
"""

from snax_forge.lower import placed_waits, task_list, upstream

from .helpers import (
    add,
    dma,
    shape,
    start,
    stream,
    sync,
    vecadd_point,
    vsum_point,
)

# =============================================================================
# 1. Configure and start
# =============================================================================


def test_configure_can_run_ahead_of_its_start():
    """A configure goes where it is written: here while the DMA's first task runs."""
    prog = shape(
        dma("a"), start("a"),
        stream("r", "acc_a", after=["a"]),
        dma("b", src=512, dst=512), start("b"),
        start("r"),
    )  # fmt: skip
    assert prog == ["Cdma", "Sdma", "Cacc_a", "Cdma", "Wdma:poll", "Sdma", "Sacc_a"]


def test_start_launches_its_tasks_in_list_order():
    prog = shape(
        stream("w", "acc_out"), stream("r", "acc_a"), stream("s", "acc_b"), add("x"),
        start("x", "s", "r", "w"),
    )  # fmt: skip
    assert prog[-4:] == ["Sacc", "Sacc_b", "Sacc_a", "Sacc_out"]


# =============================================================================
# 2. Which waits a start gets
# =============================================================================


def test_a_busy_component_is_waited_for():
    """Same component, no after: the DMA runs one task at a time."""
    prog = shape(dma("a"), start("a"), dma("b", src=512), start("b"))
    assert prog == ["Cdma", "Sdma", "Cdma", "Wdma:poll", "Sdma"]


def test_no_after_no_wait_across_components():
    """Nothing ties the streamer to the load unless ``after`` says so."""
    prog = shape(dma("a"), start("a"), stream("r", "acc_a"), start("r"))
    assert prog == ["Cdma", "Sdma", "Cacc_a", "Sacc_a"]


def test_one_wait_per_component_covers_its_earlier_tasks():
    """after on both loads: one wait on the DMA, which ends on its latest task."""
    prog = shape(
        dma("a"), start("a"), dma("b", src=512, dst=512), start("b"),
        stream("r", "acc_a", after=["a", "b"]), start("r"),
    )  # fmt: skip
    assert prog[-3:] == ["Cacc_a", "Wdma:poll", "Sacc_a"]
    assert prog.count("Wdma:poll") == 2  # the first one is b's busy wait


def test_a_covered_dependency_gets_no_wait():
    prog = shape(dma("a"), start("a"), sync("a"), stream("r", "acc_a", after=["a"]), start("r"))
    assert prog == ["Cdma", "Sdma", "Wdma:poll", "Cacc_a", "Sacc_a"]


def test_waits_are_ordered_by_when_their_task_started():
    """acc_a's earlier task started before the DMA's load, so acc_a is waited for first."""
    first_ra = [stream("r0", "acc_a"), start("r0"), dma("a"), start("a")]
    prog = shape(*first_ra, stream("r1", "acc_a", after=["a"]), start("r1"))
    assert prog[-3:] == ["Wacc_a:poll", "Wdma:poll", "Sacc_a"]
    first_dma = [dma("a"), start("a"), stream("r0", "acc_a"), start("r0")]
    prog = shape(*first_dma, stream("r1", "acc_a", after=["a"]), start("r1"))
    assert prog[-3:] == ["Wdma:poll", "Wacc_a:poll", "Sacc_a"]


def test_the_wait_mode_is_the_waited_tasks():
    prog = shape(
        dma("a", wait_mode="signal"), start("a"), stream("r", "acc_a", after=["a"]), start("r")
    )
    assert "Wdma:signal" in prog


def group(k: int, after: list[str] = ()) -> list[dict]:
    """acc_a, acc_b, acc_out and acc configured and started together as task group k."""
    return [
        stream(f"ra{k}", "acc_a", after=list(after)), stream(f"rb{k}", "acc_b", word=72),
        stream(f"wr{k}", "acc_out", word=144), add(f"acc{k}"),
        start(f"ra{k}", f"rb{k}", f"wr{k}", f"acc{k}"),
    ]  # fmt: skip


def test_a_wait_on_the_writer_covers_its_group():
    """acc_out finishes last in its group: after waiting for it, acc_a, acc_b and acc are free."""
    prog = shape(
        *group(0), dma("c", "l1_to_l2", src=1152, dst=2048, after=["wr0"]), start("c"),
        *group(1),
    )  # fmt: skip
    tail = prog[prog.index("Sdma") + 1 :]
    assert tail == ["Cacc_a", "Cacc_b", "Cacc_out", "Cacc", "Sacc_a", "Sacc_b", "Sacc_out", "Sacc"]


def test_a_needed_wait_the_writer_covers_is_left_out():
    """acc_a and acc_b are busy too, but waiting for acc_out covers them (fmul's tiles)."""
    prog = shape(*group(0), *group(1))
    assert prog[-5:] == ["Wacc_out:poll", "Sacc_a", "Sacc_b", "Sacc_out", "Sacc"]


def test_the_writer_covers_only_tasks_started_with_it():
    """acc_a's task started on its own: waiting for acc_out says nothing about it."""
    prog = shape(
        stream("r", "acc_a"), start("r"),
        stream("s", "acc_b"), stream("w", "acc_out"), add("x"), start("s", "w", "x"),
        sync("w"),
        stream("r1", "acc_a"), stream("s1", "acc_b"), start("r1", "s1"),
    )  # fmt: skip
    assert prog[-3:] == ["Wacc_a:poll", "Sacc_a", "Sacc_b"]


def test_upstream_follows_the_streamers_instance_and_write_flag():
    assert upstream(vecadd_point()) == {"acc": ["acc_a", "acc_b"], "acc_out": ["acc"]}
    assert upstream(vsum_point()) == {"acc": ["acc_a"], "acc_out": ["acc"]}


# =============================================================================
# 3. Sync
# =============================================================================


def test_sync_is_always_emitted():
    """A sync is a point in the program, even when a wait already covers it."""
    prog = shape(dma("a"), start("a"), sync("a", "signal"), sync("a"))
    assert prog == ["Cdma", "Sdma", "Wdma:signal", "Wdma:poll"]


def test_sync_covers_later_dependencies():
    prog = shape(dma("a"), start("a"), sync("a"), dma("b", after=["a"]), start("b"))
    assert prog == ["Cdma", "Sdma", "Wdma:poll", "Cdma", "Sdma"]


# =============================================================================
# 4. vecadd's and vsum's derived task lists
# =============================================================================


def test_vecadd_waits_for_each_load_then_its_writer_then_its_store():
    """acc_a needs load_A, acc_b load_B: two DMA waits; the writer's covers the adder's group;
    the sync on the store."""
    point = vecadd_point()
    waits = [(w.step, w.component, w.mode, w.covers) for w in placed_waits(task_list(point), point)]
    assert waits == [
        (3, "dma", "poll", {"dma": "load_A"}),
        (8, "dma", "poll", {"dma": "load_B"}),
        (10, "acc_out", "poll", {"acc_out": "add_acc_out", "acc": "add_acc", "acc_b": "add_acc_b",
                                 "acc_a": "add_acc_a"}),
        (11, "dma", "poll", {"dma": "store_C"}),
    ]  # fmt: skip


def test_vsum_waits_for_its_load_then_its_writer_then_its_store():
    point = vsum_point()
    waits = [(w.step, w.component, w.mode, w.covers) for w in placed_waits(task_list(point), point)]
    assert waits == [
        (5, "dma", "poll", {"dma": "load_A"}),
        (7, "acc_out", "poll", {"acc_out": "sum_acc_out", "acc": "sum_acc", "acc_a": "sum_acc_a"}),
        (8, "dma", "poll", {"dma": "store_out"}),
    ]
