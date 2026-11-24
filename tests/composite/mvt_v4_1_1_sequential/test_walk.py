"""mvt_v4_1_1_sequential walked once (A copied transposed by the DMA, one accelerator run once a
half): gen-gvsoc writes its forge/ fixture, and each feature it carries for snax-gvsoc and the
RTL, one a test."""

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


def test_the_dma_copies_a_transposed(walked):
    main = (walked.gen / "main.c").read_text()
    assert "// dma transpose_A_T" in main
    assert "A_T" in walked.point.memory.layouts


def test_every_start_is_waited_on_at_once(walked):
    assert walked.point.schedule == "sequential"
    assert len(walked.tasks.accel_steps()) == 2


def test_the_seed_is_sliced_and_the_output_packed(walked):
    ports = walked.units["vsum_x2"]["ports"]
    assert (ports["seed"]["d_in"], ports["out"]["d_out"]) == (2, 2)
