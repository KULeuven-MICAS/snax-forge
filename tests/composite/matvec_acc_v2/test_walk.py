"""matvec_acc_v2 walked once (two groups in one accelerator, a seeded fold): gen-gvsoc writes its
forge/ fixture, and each feature it carries for snax-gvsoc and the RTL, one a test."""

from tests.composite.helpers import FORGE, RTL_FIXTURES, same_files
from tests.helpers import requires_submodule
from tests.numbers import check_composite

pytestmark = requires_submodule


def test_gen_gvsoc_writes_the_forge_fixture(walked):
    same_files(walked.gen, FORGE / walked.design)


def test_gen_rtl_writes_the_rtl_fixture(walked):
    same_files(walked.rtl, RTL_FIXTURES / walked.design)


def test_the_estimate_has_a_group_per_group(walked):
    assert len(walked.estimate.groups) == len(walked.groups)
    assert walked.estimate.total_cycles > 0


def test_two_groups_in_one_accelerator(walked):
    (acc,) = walked.accelerator["accelerators"]
    assert [g["name"] for g in acc["groups"]] == ["mul", "sum"]
    assert acc["inputs"] == ["mul.a", "mul.b", "sum.a", "sum.seed"]


def test_the_fold_is_seeded(walked):
    s = walked.units["sum"]
    assert (s["datapath"], list(s["ports"])) == ("SerialAccumulate", ["a", "seed", "out"])


def test_a_task_per_group(walked):
    assert [t.component for t in walked.tasks.accel_steps()] == ["mul", "sum"]


def test_the_numbers_match_their_snapshot(walked):
    check_composite(walked)
