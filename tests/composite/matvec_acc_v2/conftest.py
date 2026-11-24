"""matvec_acc_v2's walk, once for the package (tests/composite/helpers.py)."""

import pytest

from tests.composite.helpers import walk

DESIGN = "matvec_acc_v2"


@pytest.fixture(scope="package")
def walked(tmp_path_factory):
    return walk(tmp_path_factory.mktemp(DESIGN), DESIGN)
