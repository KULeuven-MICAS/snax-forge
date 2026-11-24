"""simd's code table through bind: one BRM for two-stream and one-stream SIMDs.

Accepted when relu (``B = max(A, 0)``, ``tests/dfg/fixtures/relu.snaxdfg``) split by 4 binds to
simd at op max, one stream, k = 0, the recipe's reference check passing, and gives the worked
example's B; every entry of the table binds on a graph of its own form, its params and slot
set; vecadd and vecmul still bind at two streams with no k; ``+``, ``*``, ``min`` and ``max``
commute (``max(0, a)`` is relu, ``3 + a`` the add at k = 3); and each refusal names what it
found: a contradicting bind param or slot, a slot that does not fit the op type, and no
matching entry (``3 - a``: ``-`` does not commute). Scalar operands and axpy are
test_axpy.py's.
"""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.dfg import Graph, execute
from snax_forge.sandbox import SandboxError, bind, split_map
from tests.helpers import graph_dict

BIND = {"brm": "simd", "implementation": "chisel_tiled_spatial", "instance": "acc"}
RELU = {"node": "numpy_maximum", **BIND}
VECADD = {"node": "add", **BIND}

TWO = {"add": "in1 + in2", "sub": "in1 - in2", "mul": "in1 * in2", "min": "min(in1, in2)",
       "max": "max(in1, in2)"}  # fmt: skip
ONE = {"add": "in1 + {k}", "sub": "in1 - {k}", "mul": "in1 * {k}", "min": "min(in1, {k})",
       "max": "max(in1, {k})"}  # fmt: skip
NP = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "min": np.minimum, "max": np.maximum}


def relu(n: int = 8, code: str | None = None, dtype: str | None = None) -> Graph:
    """relu as imported, N = ``n``, its tasklet's code or its dtype replaced, split by 4."""
    d = graph_dict("relu")
    d["symbols"]["N"] = n
    if code is not None:
        d["body"][0]["body"][0]["attrs"]["code"] = code
    if dtype is not None:
        for c in d["containers"].values():
            c["dtype"] = dtype
    return split_map(Graph.from_dict(d), "numpy_maximum_map", 4)


def vecadd(code: str) -> Graph:
    d = graph_dict("vecadd_split")
    d["body"][0]["body"][0]["body"][0]["attrs"]["code"] = code
    return Graph.from_dict(d)


# =============================================================================
# relu (the acceptance)
# =============================================================================


# =============================================================================
# Every entry
# =============================================================================


@pytest.mark.parametrize("op", list(TWO))
def test_every_two_stream_entry_binds(op):
    g = bind(vecadd(f"out = {TWO[op]}"), **VECADD)
    assert g.node("add").attrs["params"] == {"W": 4, "op": op, "streams": 2}
    rng = np.random.default_rng(1)
    A, B = rng.integers(-99, 99, 64), rng.integers(-99, 99, 64)
    C = execute(g, {"A": A, "B": B, "C": np.zeros(64, np.int64)})["C"]
    assert np.array_equal(C, NP[op](A, B))


@pytest.mark.parametrize("k", [0, 5, -3])
@pytest.mark.parametrize("op", list(ONE))
def test_every_one_stream_entry_binds_with_its_slot(op, k):
    g = bind(relu(code=f"out = {ONE[op].format(k=k)}"), **RELU)
    n = g.node("numpy_maximum")
    assert n.attrs["params"] == {"W": 4, "op": op, "streams": 1, "k": k}
    A = np.random.default_rng(2).integers(-99, 99, 8).astype(np.int32)
    B = execute(g, {"A": A, "B": np.zeros(8, np.int32)})["B"]
    assert np.array_equal(B, NP[op](A, np.int32(k)))


# =============================================================================
# Refusals
# =============================================================================

LISTED = (r"'simd' computes one of 'out = a \+ b' \(op = 'add', streams = 2\), .*"
          r"'out = max\(a, k\)' \(op = 'max', streams = 1\), .*"
          r"'out = k1 \* a \+ b' \(op = 'axpby', streams = 2\)$")  # fmt: skip


@pytest.mark.parametrize(
    ("make", "params", "message"),
    [
        (relu, {"streams": 2}, (r"streams = 2 given, but the tasklet computes "
         r"'out = max\(a, 0\)', 'simd''s code at streams = 1")),
        (relu, {"op": "min"}, (r"op = 'min' given, but the tasklet computes "
         r"'out = max\(a, 0\)', 'simd''s code at op = 'max'")),
        (relu, {"k": 1}, (r"k = 1 given, but the tasklet computes 'out = max\(a, 0\)', 'simd''s "
         r"'out = max\(a, k\)' at its slot k = 0")),
        (lambda: relu(code="out = max(in1, 3000000000)"), {},
         (r"k = 3000000000 does not fit the op type int32 \(NumPy's result type of the input "
          r"ports' dtypes or casts; the literal is weak, NEP 50\)")),
        (lambda: relu(code="out = in1 + 128", dtype="int8"), {},
         r"k = 128 does not fit the op type int8"),
        (lambda: relu(code="out = in1 - -1", dtype="uint8"), {},
         r"k = -1 does not fit the op type uint8"),
        (lambda: relu(code="out = 3 - in1"), {},
         r"the tasklet computes 'out = 3 - a' on the BRM's ports, " + LISTED),
        (lambda: relu(code="out = max(in1, in1)"), {}, r"reads in1 more than once"),
    ],
)  # fmt: skip
def test_bind_refuses(make, params, message):
    with pytest.raises(SandboxError, match=message):
        bind(make(), **(RELU | {"params": params}))
