"""The estimate's data side and its work not mapped, and the round trip."""

from __future__ import annotations

import pytest

from snax_forge.estimate import Estimate, estimate

from .helpers import estimated


def _bound(g, N=None):
    d = g.to_dict()
    d["symbols"] = {"N": N or (512 if g.name == "dot" else 64)}
    return type(g).from_dict(d)


def _totals(*graphs):
    return [(e.accelerator_cycles, e.not_mapped, e.unmapped_cycles, e.total_cycles)
            for e in (estimate(g) for g in graphs)]  # fmt: skip


@pytest.mark.slow
def test_the_totals_round_trip_and_stay_out_of_the_counter():
    """The two new fields round-trip, and a group's expected BUSY_COUNTER is its cycles plus the
    fill alone: dot 4 at N = 512 binds everything, dot 1 one group of two runs."""
    for name in ("dot_v1", "dot_v4"):
        e = estimated(name)
        back = Estimate.from_dict(e.to_dict())
        assert (back.unmapped_cycles, back.total_cycles) == (e.unmapped_cycles, e.total_cycles)
        assert back.to_dict() == e.to_dict()
        assert all(g.counter == g.cycles + g.fill for g in e.groups)
        assert e.total_cycles == e.accelerator_cycles + e.unmapped_cycles


@pytest.mark.slow
@pytest.mark.parametrize("name", ["dot_v1", "dot_v4", "vecadd_v1_3"])
def test_the_estimate_round_trips(name):
    e = estimated(name)
    assert Estimate.from_dict(e.to_dict()).to_dict() == e.to_dict()
    assert Estimate.from_dict(e.to_dict()) == e
