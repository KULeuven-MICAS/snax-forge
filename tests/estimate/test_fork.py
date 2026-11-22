"""The estimate of bicg_v3, one pass over A (D189), against bicg_v2_1 at its size."""

from __future__ import annotations

from .helpers import estimated

SIZE = {"M": 64, "N": 32, "W": 32}


def test_one_group_of_two_chains_takes_the_longer_chains_cycles():
    e = estimated("bicg_v3")
    (g,) = e.groups
    assert (g.name, g.units, g.cycles) == ("bicg_types", ["mul", "sum", "mac"], 67)


def test_a_is_read_once_and_the_pass_costs_less_than_two():
    """Against bicg_v2_1's two passes at the same size: one reader and one control unit
    fewer, and A's 256 words read once; the time of the two passes run at once, plus a cycle."""
    one, two = estimated("bicg_v3"), estimated("bicg_v2_1", **SIZE)
    assert one.accelerator_cycles == 67 and [g.cycles for g in two.groups] == [66, 66]
    assert one.energy < two.energy and one.area_design < two.area_design
