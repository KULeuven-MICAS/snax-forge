"""rowsum_v2 under the ``padded_rows`` layout (design/memory.py): A's rows 65 words apart in L2
and L1. data.h writes A with each row's pad word zero, the DMA moves the padded block (4160 B,
65 beats), the streamers take the row stride from the layout, and the golden y is the one the
contiguous layout gives."""

from __future__ import annotations

import pytest

from snax_forge.design import Design, DesignPoint, MemorySpec, Platform, run_checks
from snax_forge.gen.data import arrays
from snax_forge.lower import task_list
from snax_forge.sandbox import Recipe, apply_recipe
from tests.design.helpers import SNAX_FORGE
from tests.helpers import imported, recipe_path

PADDED = MemorySpec().with_changes([("passes.layout", "padded_rows")])


def point(spec: MemorySpec) -> DesignPoint:
    g = apply_recipe(Recipe.load(recipe_path("rowsum_v2")), imported("rowsum"))[-1].graph
    d = Design(Platform.load(SNAX_FORGE), g, memory_spec=spec)
    assert run_checks(d) == []
    return DesignPoint.of(d, "rowsum_v2")


@pytest.fixture(scope="module")
def both() -> tuple[DesignPoint, DesignPoint]:
    return point(MemorySpec()), point(PADDED)


def test_the_golden_and_the_output_are_unchanged(both):
    padded = both[1]
    for name in ("y", "y_golden"):
        a, b = ({x.name: x for x in arrays(p)}[name] for p in both)
        assert a == b, name
    y = {x.name: x for x in arrays(padded)}["y_golden"]
    assert len(y.values) == 8 and y.padded == (8,)


def steps(p: DesignPoint) -> dict:
    return {s.task_name: s for s in task_list(p).steps if hasattr(s, "task_name")}
