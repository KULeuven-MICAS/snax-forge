"""A copy in the estimate (copies.py): mvt_v4 (int64, 64 × 64) with data_transpose on half 2
instead of the mark, one burst per element at the DMA's measured rate, each burst one beat read
and one written on the superbank port: one read-only and one write-only crossbar access, and a
read and a write on each of the superbank's banks (D184), in the estimate and in the dynamic
energy's graph counts."""

from __future__ import annotations

from functools import cache

from snax_forge.design import Platform
from snax_forge.estimate import Estimate, estimate
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


def test_the_copies_round_trip():
    e = mvt_v4(True)
    back = Estimate.from_dict(e.to_dict())
    assert back.copies == e.copies and back.total_cycles == e.total_cycles
    assert back.superbank_banks == e.superbank_banks == 8
    assert back.memory == e.memory
