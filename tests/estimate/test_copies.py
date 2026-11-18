"""A copy in the estimate (copies.py): mvt_4 (int64, 64 × 64) with data_transpose on half 2
instead of the mark, one burst per element at the DMA's measured rate, two L1 accesses per
burst in the memory part."""

from __future__ import annotations

from functools import cache

import pytest

from snax_forge.design import Platform
from snax_forge.estimate import DMA_CYCLES_PER_BURST, Estimate, estimate
from snax_forge.sandbox import Recipe, apply_recipe
from tests.helpers import imported, recipe_path

from .helpers import SNAX_FORGE


@cache
def mvt_4(copy: bool, schedule: str = "serial") -> Estimate:
    """mvt_4's estimate on snax_forge, its mark_transpose step data_transpose by dma (``copy``),
    under ``schedule``."""
    d = Recipe.load(recipe_path("mvt_4")).to_dict()
    if copy:
        (st,) = [st for st in d["steps"] if st["transform"] == "mark_transpose"]
        st |= {"transform": "data_transpose",
               "params": {"node": "tmp2_mult_map", "container": "A", "by": "dma"}}  # fmt: skip
    r = Recipe.from_dict(d)
    return estimate(apply_recipe(r, imported(r.kernel, r.dtypes))[-1].graph,
                    Platform.load(SNAX_FORGE), schedule=schedule)  # fmt: skip


def test_the_copy_is_one_entry_of_4096_bursts():
    e = mvt_4(True).to_dict()
    assert e["copies"] == [{"name": "transpose_A", "by": "dma", "bursts": 4096, "size": 8,
                            "cycles": 10922.67}]  # fmt: skip
    assert e["copy_cycles"] == round(4096 * DMA_CYCLES_PER_BURST) == 10923
    assert e["containers"]["A_T"]["placement"] == "l1"
    assert e["containers"]["A_T"]["bytes"] == 64 * 64 * 8


def test_total_cycles_grow_by_the_copy():
    mark, copy = mvt_4(False), mvt_4(True)
    assert mark.copies == [] and mark.copy_cycles == 0
    assert copy.accelerator_cycles == mark.accelerator_cycles
    assert copy.total_cycles == mark.total_cycles + copy.copy_cycles


def test_the_banks_price_two_accesses_per_burst():
    mark, copy = mvt_4(False).memory, mvt_4(True).memory
    assert (copy.reads - mark.reads, copy.writes - mark.writes) == (4096, 4096)
    assert copy.energy_nj - mark.energy_nj == pytest.approx(4096 * (copy.e_rd_pj + copy.e_wr_pj)
                                                            / 1000)  # fmt: skip


def test_the_copies_round_trip():
    e = mvt_4(True)
    back = Estimate.from_dict(e.to_dict())
    assert back.copies == e.copies and back.total_cycles == e.total_cycles


def test_sequential_adds_up_as_serial_with_the_copy():
    serial, seq = mvt_4(True), mvt_4(True, "sequential")
    assert seq.accelerator_cycles == serial.accelerator_cycles == sum(g.cycles for g in seq.groups)
    assert (
        seq.total_cycles
        == serial.total_cycles
        == (seq.accelerator_cycles + seq.unmapped_cycles + 10923)
    )
    assert seq.overlap_note is None
