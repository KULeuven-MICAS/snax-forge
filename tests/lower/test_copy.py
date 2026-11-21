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
from snax_forge.lower import main_c, task_list
from snax_forge.lower.tasks import Configure, Start, Sync
from snax_forge.sandbox import Recipe, apply_recipe
from tests.design.helpers import SNAX_FORGE
from tests.helpers import imported, recipe_path

from .test_c import INCLUDES, POINT, functions, parts

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


def copy_task(p: DesignPoint) -> Configure:
    return task_list(p).configured()["transpose_A_T"]


@pytest.mark.parametrize(("dtype", "memory", "src", "dst"), [
    ("int64", (), (0x10000000, [512, 8]), (0x10008000, [8, 512])),
    ("int64", PADDED, (0x10000000, [520, 8]), (0x10008200, [8, 520])),
    ("int8", (), (0x10000000, [64, 1]), (0x10001000, [1, 64])),
])  # fmt: skip
def test_the_copy_is_one_burst_per_element_with_the_layouts_strides(dtype, memory, src, dst):
    """Innermost first: along A_T's rows (A's columns), then across; size an element."""
    c = copy_task(point(dtype, memory))
    assert (c.type, c.component, c.after) == ("dma", "dma", ["load_A"])
    assert c.values == {
        "direction": "l1_to_l1", "size": 1 if dtype == "int8" else 8,
        "src": {"base": src[0], "bounds": [64, 64], "strides": src[1]},
        "dst": {"base": dst[0], "bounds": [64, 64], "strides": dst[1]},
    }  # fmt: skip


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


def test_the_copy_sits_where_half_2s_loads_are_and_its_reader_waits_on_it():
    tl = task_list(point())
    assert order(point()) == [
        "load_A", "start load_A", "load_y1", "start load_y1", "load_x1", "start load_x1",
        "conf half 1", "start tmp0_mult_mvt_1", "store_x1",
        "start store_x1", "transpose_A_T", "start transpose_A_T", "load_y2", "start load_y2",
        "load_x2", "start load_x2", "conf half 2",
        "start tmp2_mult_mvt_1", "store_x2", "start store_x2", "sync store_x1", "sync store_x2",
    ]  # fmt: skip
    assert tl.configured()["tmp2_mult_vmul_x2_a"].after == ["transpose_A_T"]


def test_sequential_syncs_after_every_start_the_copy_included():
    assert order(point(schedule="sequential")) == [
        "load_A", "start load_A", "sync load_A", "load_y1", "start load_y1", "sync load_y1",
        "load_x1", "start load_x1", "sync load_x1", "conf half 1", "start tmp0_mult_mvt_1",
        "sync tmp0_mult_mvt_1", "store_x1", "start store_x1", "sync store_x1", "transpose_A_T",
        "start transpose_A_T", "sync transpose_A_T", "load_y2", "start load_y2", "sync load_y2",
        "load_x2", "start load_x2", "sync load_x2", "conf half 2", "start tmp2_mult_mvt_1",
        "sync tmp2_mult_mvt_1", "store_x2", "start store_x2", "sync store_x2",
    ]  # fmt: skip


@pytest.mark.parametrize(("dtype", "line"), [
    ("int64", ("snrt_dma_start_2d((void *)(0x10008000 + 512 * k), (void *)(0x10000000 + 8 * k), "
               "8, 8, 512, 64);")),
    ("int8", ("snrt_dma_start_2d((void *)(0x10001000 + 64 * k), (void *)(0x10000000 + 1 * k), "
              "1, 1, 64, 64);")),
])  # fmt: skip
def test_main_c_loops_over_a_t_s_rows_and_waits_before_half_2(tmp_path, dtype, line):
    """The copy: one 2-D copy per row of A_T, its repeat the row; the default build's DMA wait
    (D135) between it and half 2's start, none between it and its load's wait."""
    p, tasks, cluster, acc, arrays = parts(tmp_path, graph=graph(dtype), name=NAME)
    core = functions(main_c(p, tasks, cluster, acc, arrays, INCLUDES, POINT))["forge_core_0"]
    at = core.index("for (uint32_t k = 0; k < 64; k++) {")
    assert core[at + 1 : at + 3] == [line, "}"]
    half_2 = [i for i, x in enumerate(core) if x == "csrw_ss(MVT_1_RW_START, 1);"][1]
    assert "snrt_dma_wait_all();" in core[at + 3 : half_2]
