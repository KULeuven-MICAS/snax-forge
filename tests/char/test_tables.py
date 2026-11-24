"""HWCOST's tables (snax_forge/char/tables.py): the files against SOURCE, the rule
against snax-forge-experiments' worked examples (its ``docs/cost_tables.md``), extrapolation
outside the grid, the bank lookup, and the crossbar's read-only and write-only inputs."""

from __future__ import annotations

import pytest

from snax_forge.char.tables import (
    Grid,
    load_costs,
    streamer_cost,
    xbar_cost,
)


@pytest.fixture(scope="module")
def costs():
    c = load_costs("nangate45")
    assert c is not None
    return c


def test_no_tables_for_a_technology_without_them():
    assert load_costs("nangate15") is None


def test_channels_extrapolate_along_the_last_two_grid_values(costs):
    """32 channels at depth 8, 1 loop: the line through 8 and 16 channels."""
    lo, hi = streamer_cost(costs, 8, 1, 8), streamer_cost(costs, 8, 1, 16)
    c = streamer_cost(costs, 8, 1, 32)
    assert c.energy_pj == pytest.approx(lo.energy_pj + 3 * (hi.energy_pj - lo.energy_pj))
    assert c.area_um2 == pytest.approx(lo.area_um2 + 3 * (hi.area_um2 - lo.area_um2))


def test_extrapolation_is_clamped_at_zero():
    """A line falling below zero gives zero, not a negative cost."""
    g = Grid({"x": (2, 4)}, {"v": {(2,): 1.0, (4,): 3.0}})
    assert g.at("v", x=1) == (0.0, ("x",))
    assert g.at("v", x=0.5) == (0.0, ("x",))
    assert g.at("v", x=3) == (2.0, ())


def test_xbar_kinds_are_the_imported_constants(costs):
    """xbar_kinds.json: the factor form's two shares, the crosspoint form's coefficients
    and the grid they were fitted on."""
    k = costs.kinds
    assert (k.f_ro, k.f_wo) == (0.395299, 0.409746)
    assert (k.c_rw, k.c_ro, k.c_wo, k.d, k.e, k.g) == (
        159.6181,
        95.4836,
        93.9346,
        22.6763,
        -17.1604,
        -103.7622,
    )
    assert (k.inputs, k.banks) == ((16, 32, 64), (16, 32, 64))


def test_xbar_example_6_all_read_only(costs):
    """Example 6: 16 inputs, 16 banks, all read-only: factor 42,048.5 × (1 − 0.395299) = 25,427;
    crosspoint 95.4836 · 16 · 16 + 22.6763 · 16 − 17.1604 · 16 − 103.7622 = 24,428."""
    assert round(xbar_cost(costs, 16, 16, 16, 0).area_um2) == 25427
    assert round(xbar_cost(costs, 16, 16, 16, 0, "crosspoint").area_um2) == 24428


@pytest.mark.parametrize("args, match", [
    ((36, 32, 30, 7), "30 read-only and 7 write-only of 36"),
    ((36, 32, -1, 0), "-1 read-only"),
    ((36, 32, 0, 0, "linear"), "crossbar form 'linear'"),
])  # fmt: skip
def test_xbar_kinds_refused(costs, args, match):
    with pytest.raises(ValueError, match=match):
        xbar_cost(costs, *args)
