"""bicg_v4_1 walked once (the one pass in two strips: a fork, a bank, a hold, a carried seed): gen-
gvsoc writes its forge/ fixture and gen-rtl its RTL fixture, and each feature it carries for
snax-gvsoc and the RTL, one a test."""

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


def test_q_carries_its_seed_across_the_strips(walked):
    s = walked.units["sum"]
    assert s["ports"]["seed"]["carry"] and [c["name"] for c in s["csrs"]] == ["CARRY"]
    (task,) = walked.tasks.accel_steps()
    assert task.values["sum"] == {"n": 320, "T": 4, "carry": 40}  # the seed read back 40 jobs on


def test_s_is_a_bank_of_4_rows_with_r_held(walked):
    mac = walked.units["mac"]
    assert (
        mac["params"]["rows"] == 4 and mac["ports"]["b"]["hold"] and mac["ports"]["b"]["broadcast"]
    )


def test_a_reader_of_a_forks_to_mul_and_mac(walked):
    (acc,) = walked.accelerator["accelerators"]
    assert acc["inputs"][0] == ["mul.a", "mac.a"]


def test_the_numbers_match_their_snapshot(walked):
    check_composite(walked)
