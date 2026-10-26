"""rowsum: a reduction with a restart every T < n beats (open item 81).

rowsum imports as two nested maps (``sum_map`` over the rows i, ``sum_map_1``
over the columns j) around a tasklet whose output ``y[i]`` reduces by add.
The recipes tag the row map temporal, split the column map by W and bind
``sum`` to ``accumulate``. The fold port's memlet uses the row loop but not
the column loop, so its named rate T is the column loop's count, N / W: the
accumulator restarts once per row, M times per run.
"""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.brm import load_brm
from snax_forge.design.streamers import firing_loops
from snax_forge.dfg import execute, named_rates
from snax_forge.dfg.execute import against_reference
from snax_forge.sdfg.loader import load
from tests.design.helpers import bound_rowsum


def rates(g, node="sum"):
    """The bound node's firing loops (var, count) and its named rates (D102)."""
    n = g.node(node)
    brm = load_brm(n.attrs["brm"])
    loops = [(lp.var, lp.count) for lp in firing_loops(g, n)]
    ports = {p.name: (({**n.inputs, **n.outputs})[p.name], p.rate) for p in brm.interface.ports}
    return loops, named_rates(ports, loops, node)


@pytest.mark.parametrize(
    "name, w, m, n, lanes",
    [("rowsum", 4, 8, 64, "4 * j_t:4 * j_t + 4"), ("rowsum_serial", 1, 4, 16, "j_t:j_t + 1")],
)  # fmt: skip
def test_the_fold_port_has_one_lane_per_row_and_t_is_n_over_w(name, w, m, n, lanes):
    """Two firing loops, lanes along a row, the sum written once per row: T = N / W."""
    g = bound_rowsum(name)
    assert [(nd.id, nd.attrs["var"], nd.attrs["range"], nd.attrs.get("loop.kind"))
            for nd, _ in g.walk() if nd.kind == "map"] == [
        ("sum_map", "i", "0:M", "temporal"),
        ("sum_map_1", "j_t", f"0:N // {w}", "temporal"),
    ]  # fmt: skip
    s = g.node("sum")
    assert s.attrs["params"] == {"W": w, "op": "add"}
    assert s.inputs["a"].subset == ["i", lanes]
    assert s.outputs["out"].to_dict() == {"data": "y", "subset": ["i:i + 1"]}
    loops, named = rates(g)
    assert loops == [("i", m), ("j_t", n // w)]
    assert named == {"T": n // w}


def test_two_rows_of_eight_on_four_lanes():
    """M = 2, N = 8, W = 4: the tree gives 10, 26 then 6, 6, so T = 2 and y = (36, 12)."""
    g = bound_rowsum("rowsum", M=2, N=8)
    assert rates(g)[1] == {"T": 2}
    a = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [2, 0, 1, 3, 0, 0, 5, 1]], dtype=np.int64)
    inputs = {"A": a, "y": np.zeros(2, dtype=np.int64)}
    assert execute(g, inputs)["y"].tolist() == [36, 12]
    for got, ref in against_reference(load("rowsum"), inputs, lambda x: execute(g, x)).values():
        assert np.array_equal(got, ref)
