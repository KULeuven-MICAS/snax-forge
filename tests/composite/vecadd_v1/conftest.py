"""vecadd_v1's walk, once for the package (tests/composite/helpers.py)."""

import pytest

from tests.composite.helpers import walk

DESIGN = "vecadd_v1"


@pytest.fixture(scope="package")
def walked(tmp_path_factory):
    return walk(tmp_path_factory.mktemp(DESIGN), DESIGN)
