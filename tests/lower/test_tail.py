"""A tail's padding in memory and in the task list (D144): the layouts over the padded shape,
the DMA's whole beats, the firing count and the rate; main.c's self-check over valid
elements."""

from __future__ import annotations

from snax_forge.design import Design, DesignPoint, MemorySpec, Platform, run_checks
from snax_forge.dfg import Graph
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.gen.data import Array
from snax_forge.lower import task_list
from snax_forge.lower.c import _compare
from snax_forge.lower.layout import moved_bytes
from snax_forge.sandbox import Recipe, apply_recipe
from tests.design.helpers import REPO, SNAX_FORGE


def _design(name: str, dtypes: dict | None = None, **params) -> Design:
    recipe = Recipe.load(REPO / "recipes" / f"{name}.json").with_params(params)
    d = apply_recipe(recipe, import_kernel(recipe.kernel))[-1].graph.to_dict()
    for c, dt in (dtypes or {}).items():
        d["containers"][c]["dtype"] = dt
    design = Design(Platform.load(SNAX_FORGE), Graph.from_dict(d), memory_spec=MemorySpec())
    assert run_checks(design) == []
    return design


def _moved(design: Design, c: str) -> int:
    ctx = design.context()
    return moved_bytes(ctx.block(c, design.memory.layouts[c]["l2"]), ctx.elem_bytes(c), 64)


def test_rowsum_int64_3x7_rows_of_8_moved_as_3_beats():
    d = _design("rowsum", W=4, M=3, N=7)
    lay = d.memory.layouts["A"]["l2"]
    assert (lay.shape, lay.strides) == ((3, 7), (64, 8))
    assert _moved(d, "A") == 192
    steps = {s.task_name: s for s in task_list(DesignPoint.of(d, "rowsum")).steps
             if hasattr(s, "task_name")}  # fmt: skip
    assert steps["load_A"].values["src"]["bounds"] == [3]
    assert steps["sum_acc"].values == {"n": 6, "T": 2}  # ceil(7 / 4) = 2 firings per row


def test_rowsum_int32_3x7_rows_of_8_moved_as_2_beats():
    """The layout only: the rows of 8 int32 move as 2 beats each."""
    d = _design("rowsum", {"A": "int32"}, W=4, M=3, N=7)
    lay = d.memory.layouts["A"]["l2"]
    assert (lay.shape, lay.strides) == ((3, 7), (32, 4))
    assert _moved(d, "A") == 128


def test_vecadd_int64_n7_moves_one_beat():
    d = _design("vecadd", W=4, N=7)
    assert _moved(d, "A") == 64
    steps = {s.task_name: s for s in task_list(DesignPoint.of(d, "vecadd")).steps
             if hasattr(s, "task_name")}  # fmt: skip
    assert steps["add_acc"].values == {"n": 2}


def test_the_self_check_compares_a_padded_matrix_row_by_row():
    a = Array("C", "C", "output", "int64_t", 21, (0,) * 24, 64, (3, 7), (3, 8))
    assert _compare(a) == [
        "for (uint32_t i0 = 0; i0 < 3; i0++) {",
        "    for (uint32_t i1 = 0; i1 < 7; i1++) {",
        "        uint32_t i = i0 * 8 + i1;",
        "        if (((volatile int64_t *)C)[i] != C_golden[i]) {",
        "            data_errors++;",
        "        }",
        "    }",
        "}",
    ]
    vector = Array("C", "C", "output", "int64_t", 7, (0,) * 8, 64, (7,), (8,))
    assert _compare(vector)[0] == "for (uint32_t i = 0; i < 7; i++) {"
