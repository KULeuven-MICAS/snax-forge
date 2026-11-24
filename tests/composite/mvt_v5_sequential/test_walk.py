"""mvt_v5_sequential walked once (two accelerators, a seeded Mac with y2 broadcast): gen-gvsoc
writes its forge/ fixture, and each feature it carries for snax-gvsoc and the RTL, one a test."""

from tests.composite.helpers import FORGE, RTL_FIXTURES, same_files
from tests.helpers import requires_submodule

pytestmark = requires_submodule


def test_gen_gvsoc_writes_the_forge_fixture(walked):
    same_files(walked.gen, FORGE / walked.design)


def test_gen_rtl_writes_the_rtl_fixture(walked):
    same_files(walked.rtl, RTL_FIXTURES / walked.design)


def test_the_estimate_has_a_group_per_group(walked):
    assert len(walked.estimate.groups) == len(walked.groups)
    assert walked.estimate.total_cycles > 0


def test_two_accelerators_a_set_each(walked):
    assert [a["name"] for a in walked.accelerator["accelerators"]] == ["mvt", "mac_x2"]
    assert [s.name for s in walked.cluster.accelerator.csrs.sets] == ["mvt", "mac_x2"]


def test_y2_is_broadcast_to_every_lane_of_the_seeded_mac(walked):
    mac = walked.units["mac_x2"]
    assert mac["ports"]["b"]["broadcast"] and mac["ports"]["b"]["d_in"] == 8
    assert list(mac["ports"]) == ["a", "b", "seed", "out"]
