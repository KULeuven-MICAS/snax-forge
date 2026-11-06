"""Scalar operands on slots, and axpy (``y += alpha * x``) as one simd unit (C§10, C§12).

A scalar operand is a tasklet input on a kernel's scalar argument (``sdfg.scalar``) whose
subset names no map variable (axpy's ``alpha[0]``): it maps to no port; ``bind`` matches it
to a slot as it matches a literal, and the accelerated node keeps it as an input connector
named after the slot (``k1: alpha[0]``), the slot left out of the params (a CSR). axpy as
imported (``tests/dfg/fixtures/axpy.snaxdfg``) is two nests, ``alpha[0] * x[i]`` into the
transient ``tmp0``, then ``y[i] + tmp0[i]`` into ``y``; split by W and fused, ``bind`` on
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
    Recipe,
    SandboxError,
    apply_recipe,
    bake_scalar,
    bind,
    fuse_map,
    split_map,
    unbake_scalar,
    unbind,
    unfuse_map,
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


def test_axpy_fused_binds_axpby_with_k1_from_alpha():
    g = bind(fused(), node="mult_map_s", **BIND)
    n = g.node("mult")
    assert n.attrs["params"] == {"W": 2, "op": "axpby", "streams": 2}
    assert n.attrs["code"] == "out = k1 * a + b"
    assert {c: (m.data, m.subset) for c, m in n.inputs.items()} == {
        "a": ("x", ["2 * i_t:2 * i_t + 2"]), "b": ("y", ["2 * i_t:2 * i_t + 2"]),
        "k1": ("alpha", [0])}  # fmt: skip
    assert {c: m.data for c, m in n.outputs.items()} == {"out": "y"}
    assert "tmp0" not in g.containers  # the local value went with the nest
    assert run(g).tolist() == [13, 14, 7, 7]
    assert run(g, 5).tolist() == (Y + 5 * X).tolist()  # alpha read at run time
    assert Graph.from_dict(g.to_dict()) == g


def test_axpys_steps_check_against_the_reference():
    """split_map twice, fuse_map, bind, each step checked against the imported graph's result."""
    r = Recipe.from_dict({"name": "axpy", "kernel": "axpy", "params": {"W": 4, "N": 64},
                          "symbols": {"N": "N"}, "steps": [
        {"transform": "split_map", "params": {"map": "mult_map", "factor": "W"}},
        {"transform": "split_map", "params": {"map": "augassign_19_4_map", "factor": "W"}},
        {"transform": "fuse_map", "params": {"first": "mult_map", "second": "augassign_19_4_map"}},
        {"transform": "bind", "params": {"node": "mult_map_s", **BIND}}]})  # fmt: skip
    results = apply_recipe(r, Graph.load(FIXTURES / "axpy.snaxdfg"))
    assert results[-1].graph.node("mult").attrs["params"]["op"] == "axpby"


def test_unbind_then_unfuse_give_the_split_graph_back():
    g = fused()
    assert unbind(bind(g, node="mult_map_s", **BIND), "mult") == g
    assert unfuse_map(g, "mult_map") == split()


def test_split_map_leaves_the_scalar_as_it_is():
    """alpha[0] names no map variable: split_map leaves it, in1 still on alpha[0]."""
    g = split()
    assert g.node("mult").inputs["in1"].data == "alpha"
    assert g.node("mult").inputs["in1"].subset == [0]
    assert g.node("mult").inputs["in2"].subset == ["2 * i_t + i_s"]


# =============================================================================
# One stream: k from a scalar
# =============================================================================


def test_a_one_stream_x_times_alpha_binds_mul_with_k_from_alpha():
    """axpy's first unit alone (no fuse_map): ``alpha * x`` is the one-stream mul, k read from
    alpha; the second nest stays a plain map."""
    g = bind(split(), node="mult", **BIND)
    n = g.node("mult")
    assert n.attrs["params"] == {"W": 2, "op": "mul", "streams": 1}
    assert n.attrs["code"] == "out = a * k"
    assert {c: m.data for c, m in n.inputs.items()} == {"a": "x", "k": "alpha"}
    assert run(g).tolist() == [13, 14, 7, 7]


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
    t["inputs"] = {"in1": {"data": "alpha", "subset": [0]}, "in2": {"data": "x", "subset": ["i"]},
                   "in3": {"data": "beta", "subset": [0]}, "in4": {"data": "y", "subset": ["i"]}}  # fmt: skip
    t["inputs"] = {c: m for c, m in t["inputs"].items() if c in code}
    t["outputs"] = {"out": {"data": "z", "subset": ["i"]}}
    t["attrs"]["code"] = code
    d["body"] = d["body"][:1]
    return split_map(Graph.from_dict(d), "mult_map", 2)


def test_axpby_binds_both_slots_from_two_scalars():
    g = bind(scaled(), node="mult", **BIND)
    n = g.node("mult")
    assert n.attrs["code"] == "out = k1 * a + k2 * b"
    assert n.attrs["params"] == {"W": 2, "op": "axpby", "streams": 2}
    assert {c: m.data for c, m in n.inputs.items()} == {
        "a": "x", "b": "y", "k1": "alpha", "k2": "beta"}  # fmt: skip
    ins = {"alpha": np.int32(3), "beta": np.int32(-2), "x": X, "y": Y, "z": np.zeros(4, np.int32)}
    assert execute(g, ins)["z"].tolist() == (3 * X - 2 * Y).tolist()


def test_a_scalar_and_a_literal_take_one_slot_each():
    """``alpha * x + 3 * y``: k1 from alpha, k2 baked at 3."""
    g = bind(scaled("out = in1 * in2 + 3 * in4"), node="mult", **BIND)
    n = g.node("mult")
    assert n.attrs["params"] == {"W": 2, "op": "axpby", "streams": 2, "k2": 3}
    assert {c: m.data for c, m in n.inputs.items()} == {"a": "x", "b": "y", "k1": "alpha"}


# =============================================================================
# Refusals and arrays
# =============================================================================


def test_a_scalar_whose_subset_uses_a_map_variable_stays_an_array_input():
    """A container marked sdfg.scalar read at ``i`` is an array input, on a port: relu's A."""
    d = graph_dict("relu")
    d["symbols"]["N"] = 8
    d["containers"]["A"]["attrs"] = {"sdfg.scalar": True}
    g = bind(split_map(Graph.from_dict(d), "numpy_maximum_map", 4), node="numpy_maximum",
             **BIND)  # fmt: skip
    assert list(g.node("numpy_maximum").inputs) == ["a"]


def test_a_non_scalar_on_a_slot_is_refused_naming_it():
    d = graph_dict("relu")
    d["symbols"]["N"] = 8
    t = d["body"][0]["body"][0]
    t["inputs"]["in2"] = {"data": "A", "subset": [0]}
    t["attrs"]["code"] = "out = max(in1, in2)"
    g = split_map(Graph.from_dict(d), "numpy_maximum_map", 4)
    with pytest.raises(SandboxError, match=r"in2 reads A\[0\], the same element in every firing; "
                       r"a slot takes a kernel's scalar argument \(sdfg.scalar\), and A is not "
                       "one"):  # fmt: skip
        bind(g, node="numpy_maximum", **BIND)


def test_a_slot_given_as_a_param_and_read_from_a_scalar_is_refused():
    with pytest.raises(SandboxError, match=r"k1 = 3 given, but the tasklet computes "
                       r"'out = b \+ alpha \* a', 'simd''s 'out = k1 \* a \+ b' with its slot k1 "
                       "from the scalar alpha"):  # fmt: skip
        bind(fused(), node="mult_map_s", **BIND, params={"k1": 3})


def test_a_scalar_wider_than_the_op_type_is_refused():
    with pytest.raises(SandboxError, match="k2 reads the scalar beta, int64, which the op type "
                       "int32 does not hold"):  # fmt: skip
        bind(scaled(beta_dtype="int64"), node="mult", **BIND)


# =============================================================================
# The recipes, bake_scalar and unbake_scalar
# =============================================================================


@pytest.mark.parametrize(("name", "k1"), [("axpy", None), ("axpy_baked", 3)])
def test_axpys_recipes_run_and_check(name, k1):
    """Each step checked against the reference (the baked one at alpha = 3, which it keeps)."""
    from tests.helpers import recipe_path

    results = apply_recipe(Recipe.load(recipe_path(name)), Graph.load(FIXTURES / "axpy.snaxdfg"))
    n = results[-1].graph.node("mult")
    assert n.attrs["params"] == {"W": 4, "op": "axpby", "streams": 2} | (
        {} if k1 is None else {"k1": k1})  # fmt: skip
    assert ("k1" in n.inputs) == (k1 is None)
    assert ("alpha" in results[-1].graph.containers) == (k1 is None)


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


def test_a_negative_value_is_baked_and_bound():
    b = bind(bake_scalar(fused(), "alpha", -7), node="mult_map_s", **BIND)
    assert b.node("mult").attrs["params"]["k1"] == -7
    assert execute(b, {"x": X, "y": Y.copy()})["y"].tolist() == (Y - 7 * X).tolist()


def test_the_reference_check_runs_at_the_baked_value():
    """The step is checked against the graph before it with alpha at the value; a step that
    changed what the graph computes there would fail, the later steps run at the value."""
    from snax_forge.sandbox.run import checked_step, reference

    g = fused()
    inputs, want = reference(g, 0, "t")
    assert int(inputs["alpha"][0]) != 3
    checked_step(g, "bake_scalar", {"container": "alpha", "value": 3}, inputs, want, "t")
    assert inputs["alpha"].tolist() == [3]
    assert np.array_equal(want["y"], inputs["y"] + 3 * inputs["x"])


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
