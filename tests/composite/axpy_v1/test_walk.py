"""axpy_v1 walked once (axpby with k1 read from a CSR): gen-gvsoc writes its forge/ fixture, and
each feature it carries for snax-gvsoc and the RTL, one a test."""

from tests.composite.helpers import FORGE, same_files
from tests.helpers import requires_submodule
from tests.numbers import check_composite

pytestmark = requires_submodule


def test_gen_gvsoc_writes_the_forge_fixture(walked):
    same_files(walked.gen, FORGE / walked.design)


def test_the_estimate_has_a_group_per_group(walked):
    assert len(walked.estimate.groups) == len(walked.groups)
    assert walked.estimate.total_cycles > 0


def test_k1_is_a_unit_csr_pinned_to_its_slot(walked):
    acc = walked.units["acc"]
    assert (acc["params"], acc["scalars"]) == ({"op": "axpby"}, {"k1": "int32"})
    assert [(c["name"], c.get("pin")) for c in acc["csrs"]] == [("K1", "k1")]


def test_main_c_writes_k1_from_alpha_before_the_start(walked):
    main = (walked.gen / "main.c").read_text()
    k1, start = main.index("K1"), main.index("_RW_START")
    assert k1 < start and "alpha" in main


def test_an_input_is_padded_to_whole_beats(walked):
    data = (walked.gen / "data.h").read_text()
    assert "// x: input, 8 elements, 16 padded to beats" in data or "padded to beats" in data


def test_the_numbers_match_their_snapshot(walked):
    check_composite(walked)
