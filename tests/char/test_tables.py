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


def test_leakage_beside_the_area(costs):
    """Every table carries leakage (µW), read as its area is: a bank's by lookup (512 × 64 at
    itrs-lstp, subthreshold plus gate), a streamer's at a grid point, the crossbar's as its area
    scaled by the table's leakage per µm² at its inputs and banks."""
    from snax_forge.char.tables import bank

    assert bank(costs, 512, 64).leak_uw == pytest.approx(0.140432)
    s = streamer_cost(costs, 8, 2, 4)
    assert 0.0179 < s.leak_uw / s.area_um2 < 0.0191
    full = xbar_cost(costs, 16, 32)
    ro = xbar_cost(costs, 16, 32, 8, 4)
    assert ro.leak_uw / ro.area_um2 == pytest.approx(full.leak_uw / full.area_um2)
    assert 0.024 < full.leak_uw / full.area_um2 < 0.0292


def test_the_fixed_parts_and_the_dma_engine(costs):
    """cluster_fixed.json's three parts add up to what the cost model leaves of the cluster
    (390 555 µm² at dot3_w8_cg); dma_copy.json's engine is 78.08 pJ a burst, the DMA alone;
    model_fit.json's terms."""
    from snax_forge.char.tables import FIXED_PARTS

    assert tuple(costs.fixed) == FIXED_PARTS
    assert sum(p.area_um2 for p in costs.fixed.values()) == pytest.approx(390555, abs=1)
    assert sum(p.leak_uw for p in costs.fixed.values()) == pytest.approx(6271.8, abs=0.1)
    assert sum(p.idle_pj_cycle for p in costs.fixed.values()) == pytest.approx(23.60, abs=0.01)
    assert costs.dma_burst_pj == pytest.approx(78.08, abs=0.01)
    t = costs.terms
    assert (round(t.r_strm, 3), round(t.r_xbar, 4)) == (0.151, 0.0018)
    assert t.floor(22, 16) == pytest.approx(
        {"streamers": 2.778, "crossbar": 2.343, "dma": 3.378}, abs=0.001
    )
    assert t.e_l2_pj == pytest.approx(54.48, abs=0.01)


def test_a_readers_refusal(costs):
    """streamer.csv's r_refused per reader, falling with FIFO depth; the mean without the column."""
    from dataclasses import replace

    from snax_forge.char.tables import Grid, streamer_refusal

    assert streamer_refusal(costs, 2, 1, 1) == pytest.approx(0.2615, abs=1e-4)
    assert streamer_refusal(costs, 16, 1, 8) < streamer_refusal(costs, 2, 1, 8)
    assert 0.05 < streamer_refusal(costs, 8, 3, 4) < 0.2
    bare = replace(costs, streamer=Grid(costs.streamer.axes, {k: v for k, v in costs.streamer.fields.items()
                                                              if k != "r_refused"}))  # fmt: skip
    assert streamer_refusal(bare, 8, 3, 4) == costs.terms.r_strm
