"""mvt_v6 walked once (the one pass over A: a fork, a seeded bank, a hold): gen-gvsoc writes its
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


def test_a_reader_of_a_forks_to_both_chains(walked):
    (acc,) = walked.accelerator["accelerators"]
    assert acc["inputs"][0] == ["vmul_x1.a", "mac_x2.a"]


def test_the_mac_is_a_seeded_bank_of_8_rows_with_y2_held(walked):
    mac = walked.units["mac_x2"]
    assert mac["params"]["rows"] == 8 and [c["name"] for c in mac["csrs"]] == ["JOB_BEATS", "ROWS"]
    assert mac["ports"]["b"]["hold"] and "seed" in mac["ports"]


def test_a_jobs_csr_per_chain(walked):
    (g,) = walked.groups
    assert [c["name"] for c in g["csrs"]][:3] == ["START", "JOBS_VSUM_X1", "JOBS_MAC_X2"]


def test_the_numbers_match_their_snapshot(walked):
    check_composite(walked)
