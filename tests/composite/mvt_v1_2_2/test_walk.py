"""mvt_v1_2_2 walked once (two groups on one streamer set's shared movers): gen-gvsoc writes its
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


def test_the_set_shares_its_movers_through_a_port_mux(walked):
    (acc,) = walked.accelerator["accelerators"]
    assert [g["name"] for g in acc["groups"]] == ["mvt", "vadd"]
    assert [m["role"] for m in acc["movers"]] == ["read", "read", "write"]
    assert acc["port_mux"]["groups"] == ["mvt", "vadd"]


def test_mvt_is_linked_by_a_buffer(walked):
    mvt = walked.groups[0]
    assert [(k["from"], k["to"], k["depth"]) for k in mvt["links"]] == [
        ("vmul_x1.out", "vsum_x1.a", 1)
    ]


def test_four_tasks_two_a_group(walked):
    assert [t.component for t in walked.tasks.accel_steps()] == ["mvt", "vadd"] * 2


def test_the_numbers_match_their_snapshot(walked):
    check_composite(walked)
