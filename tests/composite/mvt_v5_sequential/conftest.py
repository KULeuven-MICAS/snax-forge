"""mvt_v5_sequential's walk, once for the package (tests/composite/helpers.py)."""

import pytest

from tests.composite.helpers import walk

DESIGN = "mvt_v5_sequential"


@pytest.fixture(scope="package")
def walked(tmp_path_factory):
    return walk(tmp_path_factory.mktemp(DESIGN), DESIGN)
