"""rowsum_2 under the ``padded_rows`` layout (design/memory.py): A's rows 65 words apart in L2
and L1. data.h writes A with each row's pad word zero, the DMA moves the padded block (4160 B,
65 beats), the streamers take the row stride from the layout, and the golden y is the one the
contiguous layout gives."""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.design import Design, DesignPoint, MemorySpec, Platform, run_checks
from snax_forge.gen.data import arrays, data_h
from snax_forge.lower import task_list
from snax_forge.sandbox import Recipe, apply_recipe
from tests.design.helpers import SNAX_FORGE
from tests.helpers import imported, recipe_path

PADDED = MemorySpec().with_changes([("passes.layout", "padded_rows")])


def point(spec: MemorySpec) -> DesignPoint:
    g = apply_recipe(Recipe.load(recipe_path("rowsum_2")), imported("rowsum"))[-1].graph
    d = Design(Platform.load(SNAX_FORGE), g, memory_spec=spec)
    assert run_checks(d) == []
    return DesignPoint.of(d, "rowsum_2")


@pytest.fixture(scope="module")
def both() -> tuple[DesignPoint, DesignPoint]:
    return point(MemorySpec()), point(PADDED)


def test_data_h_writes_a_with_its_padding(both):
    flat, padded = both
    a = {x.name: x for x in arrays(padded)}["A"]
    assert (a.elements, a.shape, a.padded, len(a.values)) == (512, (8, 64), (8, 65), 520)
    rows = np.array(a.values).reshape(8, 65)
    assert not rows[:, 64].any()  # the pad word of every row is zero
    plain = {x.name: x for x in arrays(flat)}["A"]
    assert rows[:, :64].reshape(-1).tolist() == list(plain.values)
    assert "const int64_t A[520] __attribute__((aligned(64)))" in data_h(padded, "x")


def test_the_golden_and_the_output_are_unchanged(both):
    padded = both[1]
    for name in ("y", "y_golden"):
        a, b = ({x.name: x for x in arrays(p)}[name] for p in both)
        assert a == b, name
    y = {x.name: x for x in arrays(padded)}["y_golden"]
    assert len(y.values) == 8 and y.padded == (8,)


def steps(p: DesignPoint) -> dict:
    return {s.task_name: s for s in task_list(p).steps if hasattr(s, "task_name")}


def test_the_dma_moves_the_padded_block(both):
    flat, padded = both
    for p, beats in ((flat, 64), (padded, 65)):
        load = steps(p)["load_A"].values
        assert load["src"]["bounds"] == [beats] and load["dst"]["bounds"] == [beats]
        assert load["src"]["strides"] == [64]
    assert steps(padded)["store_y"].values["src"]["bounds"] == [1]


def test_the_streamers_take_the_row_stride_from_the_layout(both):
    """A's reader: the W lanes one row apart (spatial), the column j one word, the next block of
    W rows W rows on: 512 and 2048 B contiguous, 520 and 2080 padded."""
    for p, row in zip(both, (512, 520), strict=True):
        (a,) = [s.values for s in task_list(p).steps
                if hasattr(s, "values") and s.values.get("spatial_strides")
                and s.values.get("temporal_bounds") == [64, 2]]  # fmt: skip
        assert a["spatial_strides"] == [row]
        assert a["temporal_strides"] == [8, 4 * row]
