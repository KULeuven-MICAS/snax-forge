"""The estimate's cycles (D151, C§19): C§17's closed cycles of dot's and matvec's variants, and
every ``forge counter`` of open item 95 as a group's cycles plus the streamers' fill. Every
recipe is imported (DaCe), hence slow."""

from __future__ import annotations

import pytest

from .helpers import estimated

pytestmark = pytest.mark.slow

# C§17's table: per shape and W, the cycles of variants 2 to 6
TABLE = {
    ("dot", 4): [130, 129, 130, 133, 134],
    ("dot", 8): [66, 65, 66, 73, 74],
    ("matvec", 4): [2050, 2049, 2050, 2098, 2054],
    ("matvec", 8): [1026, 1025, 1026, 1138, 1034],
}


@pytest.mark.parametrize(("kernel", "w"), list(TABLE))
def test_the_closed_cycles_of_c17(kernel, w):
    sizes = {"N": 512} | ({"M": 16} if kernel == "matvec" else {})
    got = []
    for k in range(2, 7):
        (g,) = estimated(f"{kernel}_{k}", W=w, **sizes).groups
        got.append(g.cycles)
    assert got == TABLE[kernel, w]


def test_the_row_dot_5_splits_into_bottleneck_ramp_and_stall():
    """matvec 5 at W = 4: the MAC's 2048 firings, its L 1 and the serial sum's L 1 + span 4 − 1,
    and the 45 cycles the MAC waits for the serial sum."""
    (g,) = estimated("matvec_5", W=4, N=512, M=16).groups
    assert (g.cycles, g.bottleneck, g.ramp, g.stall) == (2098, 2048, 5, 45)


# Open item 95: per design at its recipe's defaults, its forge counters in order
COUNTERS = {
    "vecadd_i32": [19], "vsum_tree_i8": [11], "dot_1": [19, 19], "dot_4": [133], "vecadd": [19],
    "vsum": [67], "vecmul_i32_i64": [19], "dot_2": [133], "dot_3": [132], "dot_5": [136],
    "dot_6": [137], "matvec_5": [2101], "matvec_6": [2057],
}  # fmt: skip


@pytest.mark.parametrize("name", list(COUNTERS))
def test_cycles_and_the_fill_are_every_forge_counter(name):
    assert [g.cycles + g.fill for g in estimated(name).groups] == COUNTERS[name]


def test_vecadd_and_dot_1():
    """vecadd at N = 64, W = 4: 16 firings at L 0. dot 1 at N = 512: mul 128, sum 128 plus its
    L 1, one after the other through L1."""
    assert [g.cycles for g in estimated("vecadd", N=64, W=4).groups] == [16]
    e = estimated("dot_1", N=512, W=4)
    assert [(g.name, g.cycles) for g in e.groups] == [("mul", 128), ("sum", 129)]
    assert e.accelerator_cycles == 257
