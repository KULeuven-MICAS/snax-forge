"""The kernel's inputs at every size the graph binds: dfg.reference.kernel_inputs.

Every symbol the graph binds goes to the kernel's ``make_inputs`` by its
lower-case name, and the arrays' shapes must give the graph's values.
"""

from __future__ import annotations

from snax_forge.dfg import Graph
from snax_forge.dfg.reference import kernel_inputs
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
