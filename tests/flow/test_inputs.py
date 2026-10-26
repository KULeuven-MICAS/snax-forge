"""The flow's inputs at every bound size (M13 B0): kernel_inputs.

Every symbol the graph binds goes to the kernel's ``make_inputs`` by its
lower-case name, and the arrays' shapes must give the graph's values.
"""

from __future__ import annotations

import dataclasses

import pytest

from snax_forge.dfg import Graph
from snax_forge.flow import FlowError
from snax_forge.flow.run import kernel_inputs
from snax_forge.sdfg.loader import load
from tests.design.helpers import FIXTURES, bound_matadd


def _vecadd(n: int) -> Graph:
    d = Graph.load(FIXTURES / "vecadd.snaxdfg").to_dict()
    d["symbols"] = {"N": n}
    return Graph.from_dict(d)


def test_inputs_at_m_and_n():
    """matadd binds M = 8 and N = 16: make_inputs(m=8, n=16), every array 8 x 16."""
    inputs = kernel_inputs(load("matadd"), bound_matadd(), 0)
    assert {c: a.shape for c, a in inputs.items()} == {"A": (8, 16), "B": (8, 16), "C": (8, 16)}


def test_a_symbol_make_inputs_does_not_take_is_refused():
    spec = load("vecadd")
    other = dataclasses.replace(spec, make_inputs=lambda rng, size=8: spec.make_inputs(rng, n=size))
    with pytest.raises(
        FlowError, match="binds N = 12, but vecadd's make_inputs has no parameter 'n'"
    ):
        kernel_inputs(other, _vecadd(12), 0)


def test_inputs_of_another_size_are_refused():
    spec = load("vecadd")
    fixed = dataclasses.replace(spec, make_inputs=lambda rng, n=8: spec.make_inputs(rng, n=4))
    with pytest.raises(FlowError, match="gives N = 4, the graph binds 12"):
        kernel_inputs(fixed, _vecadd(12), 0)
