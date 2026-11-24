"""A carried fold: ``carry_seed`` and its inverse, ``bind`` giving the unit its own
output as its seed, the reference executor's identity on a first visit, and LOWER's CARRY and
its refusal when the queues could overtake it, on ``matvec_v9``."""

from __future__ import annotations

import pytest

from snax_forge.dfg import DfgError, Reduction, Seed
from snax_forge.lower.derive import LowerError
from tests.helpers import steps


def _carried():
    gs = steps("matvec_v9")
    k = next(
        i
        for i, g in enumerate(gs)
        if any(
            m.reduce is not None and m.reduce.carry for n, _ in g.walk() for m in n.outputs.values()
        )
    )
    return gs, k


def test_a_carry_reduction_round_trips_and_excludes_a_seed():
    r = Reduction("add", 0, None, True)
    assert r.to_dict() == {"op": "add", "identity": 0, "carry": True}
    assert Reduction.from_dict(r.to_dict(), "x") == r
    with pytest.raises(DfgError, match="carry and seed are mutually exclusive"):
        Reduction("add", None, Seed("q", ("m",)), True)
    with pytest.raises(DfgError, match="carry: true when given"):
        Reduction.from_dict({"op": "add", "identity": 0, "carry": False}, "x")


def test_lower_refuses_a_carry_the_queues_could_overtake(tmp_path):
    from snax_forge.design.point import DesignPoint
    from snax_forge.lower.derive import task_list
    from tests.gen.helpers import designed

    path = designed(tmp_path, "matvec_v9", "matvec_v9_m16", ("M=16",))
    with pytest.raises(LowerError, match="read back as its seed 16 jobs after it is written"):
        task_list(DesignPoint.load(tmp_path / path))
