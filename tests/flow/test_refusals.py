"""The flow package's refusals, one table: each case builds its input and makes the call that
is refused, with the error and the message it must give."""

from __future__ import annotations

import dataclasses

import pytest

from snax_forge.design import Platform
from snax_forge.dfg.reference import KernelError, kernel_inputs
from snax_forge.estimate import estimate
from snax_forge.flow import (
    FlowError,
    run_exit_code,
)
from snax_forge.flow.energy import energy_of
from snax_forge.run import RunError, cluster_from, parse_line
from snax_forge.sdfg.loader import load
from tests.design.helpers import SNAX_FORGE
from tests.flow.test_energy import MARKS, totals
from tests.flow.test_inputs import _vecadd
from tests.helpers import steps
from tests.run.helpers import files, snap_lines


def _a_run_without_snapshots_is_refused():
    cluster = cluster_from(*files())
    graph = steps("vecadd_v1")[-1]
    energy_of([], cluster, MARKS, graph, estimate(graph, Platform.load(SNAX_FORGE)))


def _a_default_build_that_wrote_marks_is_refused():
    cluster = cluster_from(*files())
    graph = steps("vecadd_v1")[-1]
    snaps = [(1, 50, totals(0, 0, 0, 0)), (-1, 140, totals(16, 0, 16, 16))]
    lines = [parse_line(x) for x in snap_lines(cluster, snaps)]
    energy_of(lines, cluster, None, graph, estimate(graph, Platform.load(SNAX_FORGE)))


def _the_runs_exit_code_from_make():
    assert run_exit_code(0, "") == 0
    assert run_exit_code(2, "Warning: ...\nmake: *** [Makefile:110: run] Error 1\n") == 1
    run_exit_code(2, "make: *** No rule to make target 'run'.  Stop.\n")


def _a_symbol_make_inputs_does_not_take_is_refused():
    spec = load("vecadd")
    other = dataclasses.replace(spec, make_inputs=lambda rng, size=8: spec.make_inputs(rng, n=size))
    kernel_inputs(other, _vecadd(12), 0)


def _inputs_of_another_size_are_refused():
    spec = load("vecadd")
    fixed = dataclasses.replace(spec, make_inputs=lambda rng, n=8: spec.make_inputs(rng, n=4))
    kernel_inputs(fixed, _vecadd(12), 0)


CASES = [
    pytest.param(_a_run_without_snapshots_is_refused, RunError, 'energy.snapshots', id='a_run_without_snapshots_is_refused'),
    pytest.param(_a_default_build_that_wrote_marks_is_refused, RunError, 'energy.marks', id='a_default_build_that_wrote_marks_is_refused'),
    pytest.param(_the_runs_exit_code_from_make, FlowError, 'gvsoc: make run failed \\(exit 2\\) before running gvrun', id='the_runs_exit_code_from_make'),
    pytest.param(_a_symbol_make_inputs_does_not_take_is_refused, KernelError, "binds N = 12, but vecadd's make_inputs has no parameter 'n'", id='a_symbol_make_inputs_does_not_take_is_refused'),
    pytest.param(_inputs_of_another_size_are_refused, KernelError, 'gives N = 4, the graph binds 12', id='inputs_of_another_size_are_refused'),
]  # fmt: skip


@pytest.mark.parametrize(("case", "error", "message"), CASES)
def test_refusal(case, error, message):
    with pytest.raises(error, match=message):
        case()
