"""data_transpose's copy in LOWER: mvt_v4_1 at 64 × 64 on snax_forge, its
transpose_A_T one L1-to-L1 burst per element where half 2's loads are, contiguous and
``padded_rows`` at int64 and contiguous at int8; half 2's reader of A_T waits on it through the
DMA wait (c.py's ``dma_wait``); and the task list under ``sequential``. Every recipe is imported
(DaCe), hence slow."""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import pytest

from snax_forge.design import Design, DesignPoint, MemorySpec, Platform, run_checks
from snax_forge.lower import task_list
from snax_forge.lower.tasks import Configure, Start, Sync
from snax_forge.sandbox import Recipe, apply_recipe
from tests.design.helpers import SNAX_FORGE
from tests.helpers import imported, recipe_path

pytestmark = pytest.mark.slow

NAME = "mvt_v4_1"
PADDED = (("passes.layout", "padded_rows"),)


@cache
def graph(dtype: str = "int64"):
    r = Recipe.load(recipe_path(NAME))
    dtypes = {c: dtype for c in ("A", "x1", "x2", "y1", "y2")} if dtype != "int64" else r.dtypes
    return apply_recipe(r, imported(r.kernel, dtypes))[-1].graph


@cache
def point(dtype: str = "int64", memory: tuple = (), schedule: str = "serial") -> DesignPoint:
    d = Design(Platform.load(SNAX_FORGE), graph(dtype))
    d.memory_spec = MemorySpec().with_changes(list(memory))
    assert run_checks(d) == []
    return replace(DesignPoint.of(d, NAME), schedule=schedule)


def order(p: DesignPoint) -> list[str]:
    """The steps: a DMA task by name, a half's configures as ``conf half <n>`` once, starts and
    syncs by task."""
    out: list[str] = []
    for s in task_list(p).steps:
        if isinstance(s, Configure):
            half = 2 if s.task_name.startswith(("tmp2_", "sum_1_")) else 1
            word = s.task_name if s.type == "dma" else f"conf half {half}"
            if not out or out[-1] != word:
                out.append(word)
        elif isinstance(s, Start):
            out.append(f"start {s.tasks[-1]}")
        elif isinstance(s, Sync):
            out.append(f"sync {s.task}")
    return out


def test_sequential_syncs_after_every_start_the_copy_included():
    assert order(point(schedule="sequential")) == [
        "load_A", "start load_A", "sync load_A", "load_y1", "start load_y1", "sync load_y1",
        "load_x1", "start load_x1", "sync load_x1", "conf half 1", "start tmp0_mult_mvt_1",
        "sync tmp0_mult_mvt_1", "store_x1", "start store_x1", "sync store_x1", "transpose_A_T",
        "start transpose_A_T", "sync transpose_A_T", "load_y2", "start load_y2", "sync load_y2",
        "load_x2", "start load_x2", "sync load_x2", "conf half 2", "start tmp2_mult_mvt_1",
        "sync tmp2_mult_mvt_1", "store_x2", "start store_x2", "sync store_x2",
    ]  # fmt: skip
