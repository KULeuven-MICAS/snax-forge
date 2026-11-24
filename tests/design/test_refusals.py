"""The design package's refusals, one table: each case builds its input and makes the call that
is refused, with the error and the message it must give."""

from __future__ import annotations

from dataclasses import replace

import pytest

from snax_forge.design import (
    DesignError,
    Platform,
)
from snax_forge.design.memory import MemorySpec
from snax_forge.design.point import DesignPoint
from snax_forge.design.streamers import (
    ShareError,
    instances,
    resolve,
    shared_movers,
)
from snax_forge.lower.layout import Layout, LayoutError, dma_side
from tests.design.helpers import (
    SMALL16,
    SNAX_FORGE,
    design,
)
from tests.helpers import steps


def _the_dma_refuses_a_layout_off_a_beat():
    dma_side(Layout(8, (1,), (8,)), 8, 64)


def _saved_as_renames_and_keeps_the_record():
    w = Platform.load(SMALL16).with_changes([("streamers.default.fifo_depth", 4)])
    s = w.saved_as("small16_f4")
    assert (s.name, s.base, s.changes) == ("small16_f4", "small16", w.changes)
    assert Platform.from_dict(s.to_dict()) == s
    c = s.with_changes([("l1.rows", 128)])
    assert c.base == "small16" and list(c.changes) == ["streamers.default.fifo_depth", "l1.rows"]
    w.saved_as("small-16")


def _only_a_checked_design_makes_a_point():
    d = design()
    d.memory_spec = MemorySpec().with_changes([("A.l1.base", 0), ("B.l1.base", 256)])
    DesignPoint.of(d, "x")


def _a_port_that_does_not_fit_its_mover_is_refused():
    g = steps("dot_v1_2", W=4)[-1]
    found = resolve(Platform.load(SNAX_FORGE), g, instances(g))
    found["vsum_a"] = replace(found["vsum_a"], spatial_bounds=(2, 2))
    shared_movers(g, found)


CASES = [
    pytest.param(_the_dma_refuses_a_layout_off_a_beat, LayoutError, 'not contiguous from a 64-byte beat', id='the_dma_refuses_a_layout_off_a_beat'),
    pytest.param(_saved_as_renames_and_keeps_the_record, DesignError, 'not an identifier', id='saved_as_renames_and_keeps_the_record'),
    pytest.param(_only_a_checked_design_makes_a_point, ValueError, 'checks passed', id='only_a_checked_design_makes_a_point'),
    pytest.param(_a_port_that_does_not_fit_its_mover_is_refused, ShareError, "vsum_a's spatial bounds \\[2, 2\\] \\(in words\\) do not fit in the first channels of vmul_a's \\[4\\]", id='a_port_that_does_not_fit_its_mover_is_refused'),
]  # fmt: skip


@pytest.mark.parametrize(("case", "error", "message"), CASES)
def test_refusal(case, error, message):
    with pytest.raises(error, match=message):
        case()
