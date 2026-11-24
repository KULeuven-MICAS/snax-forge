"""Scalar operands on slots, and axpy (``y += alpha * x``) as one simd unit.

A scalar operand is a tasklet input on a kernel's scalar argument (``sdfg.scalar``) whose
subset names no map variable (axpy's ``alpha[0]``): it maps to no port; ``bind`` matches it
to a slot as it matches a literal, and the accelerated node keeps it as an input connector
named after the slot (``k1: alpha[0]``), the slot left out of the params (a CSR). axpy as
imported (``tests/dfg/fixtures/axpy.snaxdfg``) is two nests, ``alpha[0] * x[n]`` into the
transient ``tmp0``, then ``y[n] + tmp0[n]`` into ``y``; split by W and fused, ``bind`` on
the fused nest composes the two codes, ``out = y + alpha * x``, and takes simd's
``out = k1 * a + b`` (op axpby; ``+`` and ``*`` commute, the array inputs tried on the ports
in every order). Accepted when the worked example (W = 2, N = 4, alpha = 3) gives y' =
(13, 14 | 7, 7) and every step checks against the reference. ``bake_scalar`` puts a value in
place of the scalar (``3 * x + y``, k1 baked), checked against the graph before it at that
value, and ``unbake_scalar`` takes it back when nothing changed since.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from snax_forge.dfg import Graph, execute
from snax_forge.sandbox import (
    SandboxError,
    bake_scalar,
    bind,
    fuse_map,
    split_map,
    unbake_scalar,
    unbind,
)
from tests.helpers import FIXTURES, graph_dict

BIND = {"brm": "simd", "implementation": "chisel_tiled_spatial", "instance": "acc"}
X = np.array([1, -2, 4, 0], np.int32)
Y = np.array([10, 20, -5, 7], np.int32)


def axpy(n: int = 4) -> Graph:
    g = Graph.load(FIXTURES / "axpy.snaxdfg")
    g.symbols["N"] = n
    return Graph.from_dict(g.to_dict())


def split(w: int = 2, n: int = 4) -> Graph:
    g = split_map(axpy(n), "mult_map", w)
    return split_map(g, "augassign_19_4_map", w)


def fused(w: int = 2, n: int = 4) -> Graph:
    return fuse_map(split(w, n), "mult_map", "augassign_19_4_map")


def run(g: Graph, alpha: int = 3) -> np.ndarray:
    return execute(g, {"alpha": np.int32(alpha), "x": X, "y": Y.copy()})["y"]


# =============================================================================
# axpy fused: one axpby unit, k1 from alpha
# =============================================================================


# =============================================================================
# One stream: k from a scalar
# =============================================================================


# =============================================================================
# Two scalars: k1 * a + k2 * b
# =============================================================================


def scaled(code: str = "out = in1 * in2 + in3 * in4", beta_dtype: str = "int32") -> Graph:
    """z = alpha * x + beta * y, one tasklet, split by 2 (N = 4)."""
    d = copy.deepcopy(graph_dict("axpy"))
    d["symbols"]["N"] = 4
    d["containers"] = {
        "alpha": d["containers"]["alpha"],
        "beta": {**d["containers"]["alpha"], "dtype": beta_dtype},
        "x": d["containers"]["x"], "y": d["containers"]["y"], "z": d["containers"]["x"]}  # fmt: skip
    t = d["body"][0]["body"][0]
    t["inputs"] = {"in1": {"data": "alpha", "subset": [0]}, "in2": {"data": "x", "subset": ["n"]},
                   "in3": {"data": "beta", "subset": [0]}, "in4": {"data": "y", "subset": ["n"]}}  # fmt: skip
    t["inputs"] = {c: m for c, m in t["inputs"].items() if c in code}
    t["outputs"] = {"out": {"data": "z", "subset": ["n"]}}
    t["attrs"]["code"] = code
    d["body"] = d["body"][:1]
    return split_map(Graph.from_dict(d), "mult_map", 2)


# =============================================================================
# Refusals and arrays
# =============================================================================


# =============================================================================
# The recipes, bake_scalar and unbake_scalar
# =============================================================================


def test_bake_scalar_puts_the_value_in_the_code_and_drops_the_container():
    g = fused()
    b = bake_scalar(g, "alpha", 3)
    t = b.node("mult")
    assert t.attrs["code"] == "out = 3 * in2" and list(t.inputs) == ["in2"]
    assert "alpha" not in b.containers
    assert run(g).tolist() == execute(b, {"x": X, "y": Y.copy()})["y"].tolist() == [13, 14, 7, 7]
    assert unbake_scalar(b, "alpha") == g
    bound = bind(b, node="mult_map_s", **BIND)
    assert bound.node("mult").attrs["params"]["k1"] == 3
    assert unbake_scalar(unbind(bound, "mult"), "alpha") == g


def test_bake_scalar_csr_names_the_scalar_and_keeps_the_container():
    """value csr: the connector takes alpha's name (``out = alpha * in2``), alpha stays, the
    values are unchanged; bind reads k1 from it as without the step; unbake undoes it, also
    after a fuse; a second bake is refused."""
    g = axpy()
    b = bake_scalar(g, "alpha", "csr")
    t = b.node("mult")
    assert t.attrs["code"] == "out = alpha * in2" and list(t.inputs) == ["alpha", "in2"]
    assert t.inputs["alpha"].data == "alpha" and "alpha" in b.containers
    assert unbake_scalar(b, "alpha") == g
    f = fuse_map(b, "mult_map", "augassign_19_4_map")
    u = unbake_scalar(f, "alpha").node("mult")  # inside the fused nest too
    assert u.attrs["code"] == "out = in1 * in2" and list(u.inputs) == ["in1", "in2"]
    n = bind(split_map(f, "mult_map", 2), node="mult_map_s", **BIND).node("mult")
    plain = bind(fused(), node="mult_map_s", **BIND).node("mult")
    assert (n.attrs["code"], n.attrs["params"]) == (plain.attrs["code"], plain.attrs["params"])
    assert {c: m.data for c, m in n.inputs.items()} == {"a": "x", "b": "y", "k1": "alpha"}
    for again in ("csr", 3):
        with pytest.raises(SandboxError, match="reads alpha as CSR already"):
            bake_scalar(b, "alpha", again)
    with pytest.raises(SandboxError, match="neither an int nor 'csr'"):
        bake_scalar(g, "alpha", "CSR")


@pytest.mark.parametrize(
    ("make", "params", "message"),
    [
        (fused, {"container": "x", "value": 3}, r"x is not a kernel's scalar argument"),
        (fused, {"container": "beta", "value": 3}, r"the graph has no container 'beta'"),
        (fused, {"container": "alpha", "value": 2**31},
         r"2147483648 does not fit alpha's dtype int32"),
        (lambda: bind(fused(), node="mult_map_s", **BIND), {"container": "alpha", "value": 3},
         r"the accelerated mult reads alpha \(unbind first\)"),
    ],
)  # fmt: skip
def test_bake_scalar_refuses(make, params, message):
    with pytest.raises(SandboxError, match=message):
        bake_scalar(make(), **params)


def test_unbake_scalar_refuses_what_changed_since():
    b = bake_scalar(fused(), "alpha", 3)
    with pytest.raises(SandboxError, match=r"no live tasklet records alpha as baked \(bake_scalar; "
                       r"a bound unit's: unbind first\), nor is it a tasklet"):  # fmt: skip
        unbake_scalar(bind(b, node="mult_map_s", **BIND), "alpha")
    d = b.to_dict()
    d["body"][0]["body"][0]["body"][0]["attrs"]["code"] = "out = 4 * in2"
    with pytest.raises(SandboxError, match="mult's code changed since bake_scalar"):
        unbake_scalar(Graph.from_dict(d), "alpha")
