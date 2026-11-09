"""simd's code table through bind: one BRM for two-stream and one-stream SIMDs (C§10).

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
from snax_forge.sandbox import Recipe, SandboxError, apply_recipe, bind, split_map
from snax_forge.sandbox.session import bind_list
from tests.helpers import FIXTURES, graph_dict, recipe_path

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


def test_relu_binds_at_max_one_stream_k_0():
    g = bind(relu(), **RELU)
    n = g.node("numpy_maximum")
    assert n.attrs["params"] == {"W": 4, "op": "max", "streams": 1, "k": 0}
    assert n.attrs["code"] == "out = max(a, k)"
    assert list(n.inputs) == ["a"] and list(n.outputs) == ["out"]
    assert n.inputs["a"].subset == ["4 * n_1_t:4 * n_1_t + 4"]
    A = np.array([-3, 5, 0, -1, 7, -8, 2, -2], np.int32)
    B = execute(g, {"A": A, "B": np.zeros(8, np.int32)})["B"]
    assert B.tolist() == [0, 5, 0, 0, 7, 0, 2, 0]


def test_relus_recipe_steps_check_against_the_reference():
    """split_map then bind, each step checked against the imported graph's result."""
    r = Recipe.from_dict({"name": "relu", "kernel": "relu", "params": {"W": 4, "N": 64},
                          "symbols": {"N": "N"}, "steps": [
        {"transform": "split_map", "params": {"map": "numpy_maximum_map", "factor": "W"}},
        {"transform": "bind", "params": RELU}]})  # fmt: skip
    results = apply_recipe(r, Graph.load(FIXTURES / "relu.snaxdfg"))
    assert results[-1].graph.node("numpy_maximum").attrs["params"]["k"] == 0


def test_the_editors_bind_list_offers_simd_on_relu():
    ok = [(e["brm"], e["implementation"]) for e in bind_list(relu(), "numpy_maximum") if e["ok"]]
    assert ok == [("simd", "chisel_tiled_spatial")]


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


def test_a_slot_takes_a_constant_expression():
    assert bind(relu(code="out = in1 + 2 * 4"), **RELU).node("numpy_maximum").attrs["params"][
        "k"] == 8  # fmt: skip


@pytest.mark.parametrize(
    ("name", "node", "op"), [("vecadd", "add", "add"), ("vecmul", "mult", "mul")]
)
def test_vecadd_and_vecmul_bind_at_two_streams_with_no_k(name, node, op):
    g = apply_recipe(Recipe.load(recipe_path(name)), Graph.load(FIXTURES / f"{name}.snaxdfg"))
    assert g[-1].graph.node(node).attrs["params"] == {"W": 4, "op": op, "streams": 2}


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
          r"ports' dtypes; the literal is weak, NEP 50\)")),
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


@pytest.mark.parametrize(
    ("code", "op", "k"),
    [("out = max(0, in1)", "max", 0), ("out = 3 + in1", "add", 3), ("out = -2 * in1", "mul", -2),
     ("out = min(7, in1)", "min", 7)],
)  # fmt: skip
def test_commuting_ops_bind_either_way_round(code, op, k):
    """``max(0, a)`` is relu: the entry ``max(a, k)`` with its operands swapped."""
    g = bind(relu(code=code), **RELU)
    n = g.node("numpy_maximum")
    assert n.attrs["params"] == {"W": 4, "op": op, "streams": 1, "k": k}
    A = np.array([-3, 5, 0, -1, 7, -8, 2, -2], np.int32)
    assert np.array_equal(execute(g, {"A": A, "B": np.zeros(8, np.int32)})["B"],
                          NP[op](A, np.int32(k)))  # fmt: skip


def test_a_one_stream_tasklet_never_binds_two_streams():
    """``out = in1 + 0`` is the one-stream add at k = 0, never a + b."""
    assert bind(relu(code="out = in1 + 0"), **RELU).node("numpy_maximum").attrs["params"] == {
        "W": 4, "op": "add", "streams": 1, "k": 0}  # fmt: skip
