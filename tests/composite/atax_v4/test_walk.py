"""atax_v4 walked once (atax's one group: t held over a link into the bank, A read once and
forked through a buffer): gen-gvsoc writes its forge/ fixture and gen-rtl its RTL fixture, and
the estimate's numbers stay at their snapshot."""

from tests.composite.helpers import FORGE, RTL_FIXTURES, same_files
from tests.helpers import requires_submodule
from tests.numbers import check_composite

pytestmark = requires_submodule


def test_gen_gvsoc_writes_the_forge_fixture(walked):
    same_files(walked.gen, FORGE / walked.design)


def test_gen_rtl_writes_the_rtl_fixture(walked):
    same_files(walked.rtl, RTL_FIXTURES / walked.design)


def test_one_group_of_three_units(walked):
    (g,) = walked.groups
    assert [u["name"] for u in g["units"]] == ["mul", "sum", "mac"]
    assert len(walked.estimate.groups) == 1 and walked.estimate.total_cycles > 0


def test_the_numbers_match_their_snapshot(walked):
    check_composite(walked)
