"""vecadd_v1_1_w1 walked once (one lane at int32): gen-gvsoc writes its forge/ fixture and gen-rtl
its RTL fixture, and each feature it carries for snax-gvsoc and the RTL, one a test."""

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


def test_the_inputs_are_sliced_and_the_output_packed(walked):
    ports = walked.units["acc"]["ports"]
    assert [(p, ports[p]["lanes"], ports[p].get("d_in"), ports[p].get("d_out")) for p in ports] == [
        ("a", 1, 2, None), ("b", 1, 2, None), ("out", 1, None, 2)]  # fmt: skip


def test_a_beat_of_words_moves_two_jobs(walked):
    (s,) = [x for x in walked.cluster.accelerator.movers if x.name == "acc_out"]
    assert s.lanes == 1 and walked.tasks.accel_steps()[0].values == {"n": 8}


def test_the_numbers_match_their_snapshot(walked):
    check_composite(walked)
