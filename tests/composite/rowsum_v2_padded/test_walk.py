"""rowsum_v2_padded walked once (LaneAccumulate on the padded_rows layout): gen-gvsoc writes its
forge/ fixture, and each feature it carries for snax-gvsoc and the RTL, one a test."""

from tests.composite.helpers import FORGE, same_files
from tests.helpers import requires_submodule
from tests.numbers import check_composite

pytestmark = requires_submodule


def test_gen_gvsoc_writes_the_forge_fixture(walked):
    same_files(walked.gen, FORGE / walked.design)


def test_the_estimate_has_a_group_per_group(walked):
    assert len(walked.estimate.groups) == len(walked.groups)
    assert walked.estimate.total_cycles > 0


def test_each_row_is_a_word_longer(walked):
    assert walked.point.memory.spec.passes["layout"] == "padded_rows"
    assert "// A: input, 32 elements, 40 padded to beats" in (walked.gen / "data.h").read_text()


def test_one_lane_accumulate_with_job_beats(walked):
    acc = walked.units["acc"]
    assert (acc["datapath"], [c["name"] for c in acc["csrs"]]) == ("LaneAccumulate", ["JOB_BEATS"])


def test_the_numbers_match_their_snapshot(walked):
    check_composite(walked)
