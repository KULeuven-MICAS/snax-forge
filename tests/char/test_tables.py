"""HWCOST's tables (snax_forge/char/tables.py): the files against SOURCE, the rule
against snax-forge-experiments' worked examples (its ``docs/cost_tables.md``), extrapolation
outside the grid, the bank lookup, and the crossbar's read-only and write-only inputs."""

from __future__ import annotations

import hashlib
import logging

import pytest

from snax_forge.char.tables import (
    FILES,
    HERE,
    Grid,
    bank,
    load_costs,
    read_source,
    streamer_cost,
    xbar_cost,
)


@pytest.fixture(scope="module")
def costs():
    c = load_costs("nangate45")
    assert c is not None
    return c


def test_files_match_source():
    """Each table is the file SOURCE names, byte for byte (its sha256), from one commit."""
    src = read_source(HERE / "nangate45" / "SOURCE")
    assert src["repo"].endswith("snax-forge-experiments") and len(src["commit"]) == 40
    for name in FILES:
        path, _, digest = src[name].partition(" sha256 ")
        assert path.endswith(name)
        assert hashlib.sha256((HERE / "nangate45" / name).read_bytes()).hexdigest() == digest


def test_no_tables_for_nangate15():
    """nangate15 has no memories and no streamer or crossbar tables."""
    assert load_costs("nangate15") is None


def test_example_1_reader_between_two_channel_counts(costs):
    """cost_tables.md example 1: depth 8, 2 loops, 3 channels, only channels interpolated."""
    c = streamer_cost(costs, 8, 2, 3)
    assert round(c.energy_pj, 4) == 15.3916
    assert round(c.area_um2, 1) == 15424.9
    assert c.extrapolated == ()


def test_example_2_reader_on_two_axes(costs):
    """cost_tables.md example 2: depth 6 and 3 channels between grid values, 3 loops on it."""
    assert round(streamer_cost(costs, 6, 3, 3).energy_pj, 4) == 13.5002


def test_example_3_xbar_energy_in_one_over_inputs(costs):
    """cost_tables.md example 3: 12 inputs, 32 banks; energy in 1/inputs, area in inputs."""
    c = xbar_cost(costs, 12, 32)
    assert round(c.energy_pj, 4) == 3.4030
    assert round(c.area_um2, 1) == 61450.9


def test_example_4_more_inputs_than_banks(costs):
    """cost_tables.md example 4: 40 inputs, 32 banks; the energy at inputs = banks, the
    area measured."""
    c = xbar_cost(costs, 40, 32)
    assert round(c.energy_pj, 4) == 2.7394
    assert round(c.area_um2, 1) == 210443.8


def test_grid_values_are_exact(costs):
    """A query on grid values returns the row as is (streamer.csv's 8, 1, 8)."""
    c = streamer_cost(costs, 8, 1, 8)
    assert (c.area_um2, c.energy_pj) == (38525.045548, 39.65753)


def test_the_tcdm_bank(costs):
    """The 512 × 64 bank of snax_cluster's TCDM (the RAM table's row)."""
    b = bank(costs, 512, 64)
    assert b is not None and (b.area_um2, b.e_rd_pj, b.e_wr_pj) == (13522.7, 3.53819, 6.4677)


def test_a_bank_off_the_grid_is_none(costs):
    """The bank table is never interpolated: a depth not a multiple of 32 gives None."""
    assert bank(costs, 500, 64) is None
    assert bank(costs, 8192, 64) is None


@pytest.mark.parametrize("q, axis", [
    ((8, 1, 32), "channels"),   # above 16 channels
    ((32, 1, 4), "depth"),      # above depth 16
    ((8, 9, 4), "loops"),       # above 7 loops
])  # fmt: skip
def test_streamer_outside_the_grid_is_extrapolated_with_a_warning(costs, caplog, q, axis):
    """Outside the grid an axis is extrapolated through its two nearest grid values, named in
    ``extrapolated`` and logged."""
    with caplog.at_level(logging.WARNING, logger="snax_forge.char.tables"):
        c = streamer_cost(costs, *q)
    assert c.extrapolated == (axis,)
    assert "extrapolated" in caplog.text and axis in caplog.text
    assert c.area_um2 > 0 and c.energy_pj > 0


def test_channels_extrapolate_along_the_last_two_grid_values(costs):
    """32 channels at depth 8, 1 loop: the line through 8 and 16 channels."""
    lo, hi = streamer_cost(costs, 8, 1, 8), streamer_cost(costs, 8, 1, 16)
    c = streamer_cost(costs, 8, 1, 32)
    assert c.energy_pj == pytest.approx(lo.energy_pj + 3 * (hi.energy_pj - lo.energy_pj))
    assert c.area_um2 == pytest.approx(lo.area_um2 + 3 * (hi.area_um2 - lo.area_um2))


def test_xbar_below_the_grid_extrapolates_in_one_over_inputs(costs):
    """2 inputs at 32 banks: energy on the line in 1/inputs through 4 and 8 inputs."""
    a, b = xbar_cost(costs, 4, 32), xbar_cost(costs, 8, 32)
    c = xbar_cost(costs, 2, 32)
    t = (1 / 2 - 1 / 4) / (1 / 8 - 1 / 4)
    assert c.energy_pj == pytest.approx(a.energy_pj + t * (b.energy_pj - a.energy_pj))
    assert c.extrapolated == ("inputs",)


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
    assert (k.f_ro, k.f_wo) == (0.396496, 0.41077)
    assert (k.c_rw, k.c_ro, k.c_wo, k.d, k.e, k.g) == (
        159.3358,
        95.0644,
        93.6514,
        26.3019,
        -19.4338,
        -201.0074,
    )
    assert (k.inputs, k.banks) == ((16, 32, 64), (16, 32, 64))


def test_xbar_example_5_the_clusters_interconnect(costs):
    """snax-forge-experiments' cost_tables.md example 5: 36 inputs on 32 banks, 32 read-only and 1
    write-only. Factor: the table's 187,547.0 µm² (between 32 and 40 inputs) × (1 − 0.396496 · 32/36
    − 0.41077 / 36) = 119,308; crosspoint: 115,763. The energy per access is the table's in both."""
    table = xbar_cost(costs, 36, 32)
    assert table.area_um2 == pytest.approx(187547.02857750002)
    f = xbar_cost(costs, 36, 32, 32, 1)
    assert f.area_um2 == pytest.approx(187547.02857750002 * (1 - 0.396496 * 32 / 36 - 0.41077 / 36))
    assert round(f.area_um2) == 119308 and f.energy_pj == table.energy_pj
    c = xbar_cost(costs, 36, 32, 32, 1, "crosspoint")
    assert round(c.area_um2) == 115763 and c.extrapolated == ()


def test_xbar_example_6_all_read_only(costs):
    """Example 6: 16 inputs, 16 banks, all read-only: factor 41,886.0 × (1 − 0.396496) = 25,278;
    crosspoint 95.0644 · 16 · 16 + 26.3019 · 16 − 19.4338 · 16 − 201.0074 = 24,245."""
    assert round(xbar_cost(costs, 16, 16, 16, 0).area_um2) == 25278
    assert round(xbar_cost(costs, 16, 16, 16, 0, "crosspoint").area_um2) == 24245


def test_xbar_without_kinds_is_the_table(costs):
    """No read-only or write-only input: the factor form is the table's area."""
    for q in ((27, 32), (4, 16), (64, 64)):
        assert xbar_cost(costs, *q, 0, 0).area_um2 == xbar_cost(costs, *q).area_um2
        assert (
            costs.xbar.at("area_um2", inputs=q[0], banks=q[1])[0] == xbar_cost(costs, *q).area_um2
        )


def test_xbar_crosspoint_outside_its_fit_is_extrapolated(costs, caplog):
    """The crosspoint form was fitted on 16–64 inputs and banks: 8 inputs is outside it (the
    factor form's table goes down to 4, so it is not)."""
    with caplog.at_level(logging.WARNING, logger="snax_forge.char.tables"):
        c = xbar_cost(costs, 8, 32, 8, 0, "crosspoint")
    assert c.extrapolated == ("inputs",) and "extrapolated" in caplog.text
    assert xbar_cost(costs, 8, 32, 8, 0).extrapolated == ()


@pytest.mark.parametrize("args, match", [
    ((36, 32, 30, 7), "30 read-only and 7 write-only of 36"),
    ((36, 32, -1, 0), "-1 read-only"),
    ((36, 32, 0, 0, "linear"), "crossbar form 'linear'"),
])  # fmt: skip
def test_xbar_kinds_refused(costs, args, match):
    with pytest.raises(ValueError, match=match):
        xbar_cost(costs, *args)
