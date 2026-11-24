"""The estimate's cycles: the closed cycles of dot's and matvec's variants, and
every recorded ``forge counter`` as a group's cycles plus the streamers' fill. Every
recipe is imported (DaCe), hence slow."""

from __future__ import annotations

import pytest

from .helpers import estimated

pytestmark = pytest.mark.slow

# the table: per shape and W, the cycles of variants 2 to 6
TABLE = {
    ("dot", 4): [130, 129, 130, 133, 134],
    ("dot", 8): [66, 65, 66, 73, 74],
    ("matvec", 4): [2050, 2049, 2050, 2098, 2054],
    ("matvec", 8): [1026, 1025, 1026, 1138, 1034],
}


# Open item 95: per design at its recipe's defaults, its forge counters in order (dot 1's were
# measured encapsulated, now dot_v1_1's; dot 1 as two accelerators gives the same groups)
COUNTERS = {
    "vecadd_v1_1": [19], "vsum_v1_1_2": [11], "dot_v1": [19, 19], "dot_v1_1": [19, 19],
    "dot_v4": [133], "vecadd_v1": [19],
    "vsum_v1": [67], "vecmul_v1_1_1": [19], "dot_v2": [133], "dot_v3": [132], "dot_v5": [136],
    "dot_v6": [137], "matvec_v5": [2101], "matvec_v6": [2057], "relu_v1": [18], "axpy_v1": [19],
    "axpy_v1_1": [19], "dot_v7": [19, 67],
}  # fmt: skip


@pytest.mark.parametrize("name", list(COUNTERS))
def test_cycles_and_the_fill_are_every_forge_counter(name):
    assert [g.cycles + g.fill for g in estimated(name).groups] == COUNTERS[name]
