"""The viz package's refusals, one table: each case builds its input and makes the call that
is refused, with the error and the message it must give."""

from __future__ import annotations

import pytest

from snax_forge.lower.tasks import Configure, Start, TaskList
from snax_forge.run import Region, RunError
from snax_forge.viz import memory
from tests.viz.helpers import L1, Events
from tests.viz.test_memory import layout


def _an_accelerator_task_without_a_streamer_task_for_a_mover_is_refused():
    tl = TaskList(
        "vecadd",
        [
            Configure("add_acc_a", "streamer", "acc_a", {}),
            Configure("add_acc_b", "streamer", "acc_b", {}),
            Configure("add_acc", "accel", "acc", {}),
            Start(["add_acc_a", "add_acc_b", "add_acc"]),
        ],
    )
    Events().index(task_list=tl)


def _a_region_outside_the_memory_is_refused():
    layout(Region("A", "l1", L1 - 8, (4,), (8,), "int64"))


CASES = [
    pytest.param(_an_accelerator_task_without_a_streamer_task_for_a_mover_is_refused, RunError, "'add_acc' has 0 streamer tasks for acc_out \\(\\[\\]\\)", id='an_accelerator_task_without_a_streamer_task_for_a_mover_is_refused'),
    pytest.param(_a_region_outside_the_memory_is_refused, memory.MemoryViewError, 'outside', id='a_region_outside_the_memory_is_refused'),
]  # fmt: skip


@pytest.mark.parametrize(("case", "error", "message"), CASES)
def test_refusal(case, error, message):
    with pytest.raises(error, match=message):
        case()
