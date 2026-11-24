"""The lower package's refusals, one table: each case builds its input and makes the call that
is refused, with the error and the message it must give."""

from __future__ import annotations

import pytest

from snax_forge.design import Design, DesignPoint, Platform, run_checks
from snax_forge.lower import (
    LowerError,
    TaskList,
    TaskListError,
    check_values,
    task_list,
)
from tests.design.helpers import SNAX_FORGE
from tests.helpers import steps
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


def _a_forks_buffer_shorter_than_the_beats_before_its_reader_fires():
    """atax_v4 at T = N / W = 2: a buffer of 1 fills before the Mac's first t, for ever."""
    g = steps("atax_v4", M=4, N=8, W=4, R=2, D=1)[-1]
    d = Design(Platform.load(SNAX_FORGE), g)
    assert run_checks(d) == []
    point = DesignPoint.of(d, "atax_v4")
    check_values(task_list(point), point)


CASES = [
    pytest.param(_a_write_into_its_read_container_elsewhere_is_refused, LowerError, "node add\\.out: writes B in place of add_acc_b's read, but not the same elements", id='a_write_into_its_read_container_elsewhere_is_refused'),
    pytest.param(_a_seed_on_other_elements_than_its_write_is_refused, LowerError, "node sum\\.out: writes q in place of sum_sum_seed's read, but not the same elements in the same order", id='a_seed_on_other_elements_than_its_write_is_refused'),
    pytest.param(_a_task_listed_twice_in_one_start, TaskListError, 'listed twice', id='a_task_listed_twice_in_one_start'),
    pytest.param(_an_accelerator_tasks_head_firings_stay_below_2_to_the_32, TaskListError, 'a unit fires 4294967296 times', id='an_accelerator_tasks_head_firings_stay_below_2_to_the_32'),
    pytest.param(_a_forks_buffer_shorter_than_the_beats_before_its_reader_fires, TaskListError, 'the fork of A buffers 1 beats for mac.a, which fires only after 2 beats on the other branch; link A with a depth of 2 or more', marks=pytest.mark.slow, id='a_forks_buffer_shorter_than_the_beats_before_its_reader_fires'),
]  # fmt: skip


@pytest.mark.parametrize(("case", "error", "message"), CASES)
def test_refusal(case, error, message):
    with pytest.raises(error, match=message):
        case()
