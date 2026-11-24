"""atax's one group on the estimate: t linked into the bank's held input (atax_v3), then A read
once through a buffer of D beats on the `Mac`'s branch (atax_v4). At 64 × 64, W = 8, T = N/W = 8
beats a row: the group runs the second read's cycles from D = T + 2, stalls a cycle a row at
T + 1 and two at T. Every recipe is imported (DaCe), hence slow."""

from __future__ import annotations

import pytest

from snax_forge import char
from snax_forge.design import Platform
from snax_forge.estimate import Estimate, estimate
from tests.helpers import steps

from .helpers import SNAX_FORGE

pytestmark = pytest.mark.slow
M, N, W = 64, 64, 8
T = N // W


def at(name: str, **params: int) -> Estimate:
    return estimate(steps(name, **params)[-1], Platform.load(SNAX_FORGE), "nangate45")


def test_t_held_over_a_link_runs_one_row_of_a_a_t_cycles():
    (g,) = at("atax_v3").groups
    assert g.units == ["mul", "sum", "mac"] and g.links == ["tmp2_prod", "tmp1"]
    # t[0] after the first row's T beats and the vsum's and link's latency, then a row a T
    assert g.cycles == M * T + T + 3 == 523


@pytest.mark.parametrize(("depth", "stall"), [(T + 2, 0), (T + 3, 0), (T + 1, 1), (T, 2)])
def test_a_buffer_short_of_t_plus_2_stalls_a_cycle_a_row_for_each_beat(depth, stall):
    (g,) = at("atax_v4", D=depth).groups
    assert g.cycles == M * T + T + 3 + stall * (M - 1)


def test_the_buffer_loads_one_entry_a_push():
    """D entries of W int8 lanes, a push loading one: the same energy at every D (comb aside)."""
    table, e = char.load("nangate45"), at("atax_v4")
    fifo = e.groups[0].cost.links["A"]
    assert fifo.datapath == "LinkBuffer" and fifo.fires == M * T  # one push a beat of A
    assert e.groups[0].bits["A"] == (T + 2) * W * 8
    got = char.lookup(table, "LinkBuffer", {"depth": T + 2, "lanes": W, "dtype": "int8"}, W, T + 2)
    assert fifo.e_fire_pj == pytest.approx(got.e_comb_pj + W * 8 * table.e_load_pj)
