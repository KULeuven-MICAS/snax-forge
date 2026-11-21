"""A copy in the estimate (copies.py): mvt_v4 (int64, 64 × 64) with data_transpose on half 2
instead of the mark, one burst per element at the DMA's measured rate, each burst one beat read
and one written on the superbank port: one read-only and one write-only crossbar access, and a
read and a write on each of the superbank's banks (D184), in the estimate and in the dynamic
energy's graph counts."""

from __future__ import annotations

from functools import cache

import pytest

from snax_forge.design import Platform
from snax_forge.estimate import DMA_CYCLES_PER_BURST, Estimate, estimate
from snax_forge.sandbox import Recipe, apply_recipe
from tests.helpers import imported, recipe_path

from .helpers import SNAX_FORGE


@cache
def mvt_v4(copy: bool, schedule: str = "serial", platform: bool = True) -> Estimate:
    """mvt_v4's estimate on snax_forge (none without ``platform``), its mark_transpose step
    data_transpose by dma (``copy``), under ``schedule``."""
    d = Recipe.load(recipe_path("mvt_v4")).to_dict()
    if copy:
        (st,) = [st for st in d["steps"] if st["transform"] == "mark_transpose"]
        st |= {"transform": "data_transpose",
               "params": {"node": "tmp2_mult_map", "container": "A", "by": "dma"}}  # fmt: skip
    r = Recipe.from_dict(d)
    return estimate(apply_recipe(r, imported(r.kernel, r.dtypes))[-1].graph,
                    Platform.load(SNAX_FORGE) if platform else None, schedule=schedule)  # fmt: skip


def test_the_copy_is_one_entry_of_4096_bursts():
    e = mvt_v4(True).to_dict()
    assert e["copies"] == [{"name": "transpose_A", "by": "dma", "bursts": 4096, "size": 8,
                            "cycles": 10922.67}]  # fmt: skip
    assert e["copy_cycles"] == round(4096 * DMA_CYCLES_PER_BURST) == 10923
    assert e["containers"]["A_T"]["placement"] == "l1"
    assert e["containers"]["A_T"]["bytes"] == 64 * 64 * 8


def test_total_cycles_grow_by_the_copy():
    mark, copy = mvt_v4(False), mvt_v4(True)
    assert mark.copies == [] and mark.copy_cycles == 0
    assert copy.accelerator_cycles == mark.accelerator_cycles
    assert copy.total_cycles == mark.total_cycles + copy.copy_cycles


def test_the_superbank_is_the_dmas_beat_over_the_banks_word():
    l1 = Platform.load(SNAX_FORGE).l1
    assert mvt_v4(True).superbank_banks == l1.wide_bits // l1.width_bits == 8
    assert mvt_v4(True, platform=False).superbank_banks is None


def test_the_banks_price_a_superbank_read_and_write_per_burst():
    """A burst of one element never crosses a superbank line: one beat each way, and each beat
    reads or writes all 8 of the superbank's banks."""
    mark, copy = mvt_v4(False).memory, mvt_v4(True).memory
    assert (copy.reads - mark.reads, copy.writes - mark.writes) == (8 * 4096, 8 * 4096)
    assert copy.energy_nj - mark.energy_nj == pytest.approx(
        8 * 4096 * (copy.e_rd_pj + copy.e_wr_pj) / 1000
    )
    assert copy.energy_nj - mark.energy_nj == pytest.approx(327.87, abs=0.005)


def test_the_crossbar_prices_each_access_at_its_kind():
    """A copy's read at a read-only input's price, its write at a write-only input's (D184)."""
    mark, copy = mvt_v4(False).xbar, mvt_v4(True).xbar
    assert (copy.reads - mark.reads, copy.writes - mark.writes) == (4096, 4096)
    assert copy.energy_nj - mark.energy_nj == pytest.approx(4096 * (copy.e_ro_pj + copy.e_wo_pj)
                                                            / 1000)  # fmt: skip
    assert copy.energy_nj - mark.energy_nj == pytest.approx(13.49, abs=0.005)


def test_the_dynamic_energy_on_the_graph_counts_prices_the_copy_as_the_estimate():
    """D177's identity with a copy: the graph's counts carry the copy's beats read and written,
    and priced they give the estimate's five parts."""
    from snax_forge.estimate.dynamic import graph_counts, port_movers, price, prices_of

    e = mvt_v4(True)
    g = apply_recipe(*_copy_recipe())[-1].graph
    c = graph_counts(g, e)
    assert c.copies == (4096, 4096)
    got = price(prices_of(g, e, port_movers(e)), c).parts
    assert got == pytest.approx({k: p.energy_nj for k, p in e.parts.items()})


def _copy_recipe():
    d = Recipe.load(recipe_path("mvt_v4")).to_dict()
    (st,) = [st for st in d["steps"] if st["transform"] == "mark_transpose"]
    st |= {"transform": "data_transpose",
           "params": {"node": "tmp2_mult_map", "container": "A", "by": "dma"}}  # fmt: skip
    r = Recipe.from_dict(d)
    return r, imported(r.kernel, r.dtypes)


def test_the_copies_round_trip():
    e = mvt_v4(True)
    back = Estimate.from_dict(e.to_dict())
    assert back.copies == e.copies and back.total_cycles == e.total_cycles
    assert back.superbank_banks == e.superbank_banks == 8
    assert back.memory == e.memory


def test_sequential_adds_up_as_serial_with_the_copy():
    serial, seq = mvt_v4(True), mvt_v4(True, "sequential")
    assert seq.accelerator_cycles == serial.accelerator_cycles == sum(g.cycles for g in seq.groups)
    assert (
        seq.total_cycles
        == serial.total_cycles
        == (seq.accelerator_cycles + seq.unmapped_cycles + 10923)
    )
    assert seq.overlap_note is None
