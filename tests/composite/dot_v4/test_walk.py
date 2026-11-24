"""dot_v4 walked once (a linked group, Mac into TreeAccumulate over a wire): gen-gvsoc writes its
forge/ fixture and gen-rtl its RTL fixture, and each feature it carries for snax-gvsoc and the
RTL, one a test."""

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


def test_one_group_linked_by_a_wire(walked):
    (g,) = walked.groups
    assert [(u["name"], u["datapath"]) for u in g["units"]] == [
        ("mac", "Mac"),
        ("sum", "TreeAccumulate"),
    ]
    assert [(k["from"], k["to"], k["depth"]) for k in g["links"]] == [("mac.out", "sum.a", 0)]


def test_the_head_counts_job_beats(walked):
    assert [c["name"] for c in walked.units["mac"]["csrs"]] == ["JOB_BEATS"]
    (task,) = walked.tasks.accel_steps()
    assert task.values == {"mac": {"n": 2, "T": 2}, "sum": {"n": 1, "T": 1}}


def test_the_numbers_match_their_snapshot(walked):
    check_composite(walked)
