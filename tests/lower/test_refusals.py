"""The lower package's refusals, one table: each case builds its input and makes the call that
is refused, with the error and the message it must give."""

from __future__ import annotations

import pytest

from snax_forge.lower import (
    LowerError,
    TaskList,
    TaskListError,
    check_values,
    task_list,
)
from tests.lower.helpers import add, start, stream, vsum_point
from tests.lower.helpers import task_list as listed
from tests.lower.test_derive import _seeded, in_place, point


def _a_write_into_its_read_container_elsewhere_is_refused():
    def edit(gd):
        in_place(gd, "4 * n_1_t + N:4 * n_1_t + N + 4", "4 * n_1_t + n_2_s + N")

    task_list(point(edit_graph=edit))


def _a_seed_on_other_elements_than_its_write_is_refused():
    def reversed_seed(d):
        for top in d["body"]:
            for n in top["body"]:
                if n["id"] == "sum_map_1":
                    n["body"][0]["inputs"]["seed"]["subset"] = ["7 - m_t:8 - m_t"]

    task_list(_seeded("matvec_acc_v1", reversed_seed))


def _a_task_listed_twice_in_one_start():
    listed(stream("r", "acc_a"), start("r", "r"))


def _an_accelerator_tasks_head_firings_stay_below_2_to_the_32():
    def vsum(n: int, t: int) -> TaskList:
        return listed({**add("x"), "values": {"n": n, "T": t}}, start("x"))

    check_values(vsum((1 << 32) - 4, 4), vsum_point())
    check_values(vsum(1 << 32, 4), vsum_point())


CASES = [
    pytest.param(_a_write_into_its_read_container_elsewhere_is_refused, LowerError, "node add\\.out: writes B in place of add_acc_b's read, but not the same elements", id='a_write_into_its_read_container_elsewhere_is_refused'),
    pytest.param(_a_seed_on_other_elements_than_its_write_is_refused, LowerError, "node sum\\.out: writes q in place of sum_sum_seed's read, but not the same elements in the same order", id='a_seed_on_other_elements_than_its_write_is_refused'),
    pytest.param(_a_task_listed_twice_in_one_start, TaskListError, 'listed twice', id='a_task_listed_twice_in_one_start'),
    pytest.param(_an_accelerator_tasks_head_firings_stay_below_2_to_the_32, TaskListError, 'the head fires JOBS · ∏T = 4294967296 times', id='an_accelerator_tasks_head_firings_stay_below_2_to_the_32'),
]  # fmt: skip


@pytest.mark.parametrize(("case", "error", "message"), CASES)
def test_refusal(case, error, message):
    with pytest.raises(error, match=message):
        case()
