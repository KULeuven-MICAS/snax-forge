"""The accelerator's cycles under the design point's schedule (D179): mvt_3_split at W = 8, two
independent halves, adds them serial and takes the slower pipelined; mvt_3, the halves on one
shared group, adds them under every schedule. Every recipe is imported (DaCe), hence slow."""

from __future__ import annotations

import pytest

from snax_forge.design import Platform
from snax_forge.design.check import SCHEDULES
from snax_forge.estimate import Estimate, estimate
from snax_forge.estimate.schedule import RunEstimate, longest_path
from tests.helpers import steps

from .helpers import SNAX_FORGE

pytestmark = pytest.mark.slow


def at(name: str, schedule: str) -> Estimate:
    return estimate(steps(name, W=8)[-1], Platform.load(SNAX_FORGE), "nangate45", schedule)


def test_mvt_3_split_adds_its_halves_serial_and_takes_the_slower_pipelined():
    serial = at("mvt_3_split", "serial")
    halves = [g.cycles for g in serial.groups]
    assert [g.name for g in serial.groups] == ["mvt", "mvt_1"]
    assert serial.accelerator_cycles == sum(halves) == 1030
    assert [r.after for r in serial.runs] == [[], []]  # A is read by both, written by neither
    assert serial.overlap_note is None
    for schedule in ("pipeline", "pipeline_compute"):
        e = at("mvt_3_split", schedule)
        assert e.accelerator_cycles == max(halves) == 515
        assert e.total_cycles == 515 + e.unmapped_cycles
        # two chains of A and y at 8 words, and x's: past the 32 banks
        assert e.overlap_note == (f"mvt and mvt_1 run at once under {schedule}: their streams "
                                  "move 32.3 words a cycle over 32 banks, so bank conflicts "
                                  "stretch them beyond the ideal")  # fmt: skip
        # the control units clocked over the shorter run
        assert e.groups[0].cost.ctrl_nj < serial.groups[0].cost.ctrl_nj


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_mvt_3_is_the_same_under_every_schedule(schedule):
    e = at("mvt_3", schedule)
    assert e.accelerator_cycles == at("mvt_3", "serial").accelerator_cycles == 1030
    assert [r.after for r in e.runs] == [[], ["tmp0_mult"]] and e.overlap_note is None


def test_the_runs_round_trip_and_a_chain_is_its_sum():
    e = at("mvt_3_split", "pipeline")
    back = Estimate.from_dict(e.to_dict())
    assert (back.schedule, back.runs, back.accelerator_cycles) == ("pipeline", e.runs, 515)
    chain = [RunEstimate("a", "g", "x", 3, 0), RunEstimate("b", "h", "y", 4, 0, ["a"]),
             RunEstimate("c", "k", "z", 5, 0)]  # fmt: skip
    assert longest_path(chain) == 7
