"""The BRM library: elementwise_add, elementwise_mul and accumulate.

Each BRM resolves to the function the reference executor runs (its kind in
brm/function.py, its params resolved), and its ``code`` computes what that
kind computes, lane by lane; accumulate's Chisel implementation declares the
Accumulator's drain, 0 (D133).
"""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.brm import LIBRARY, Brm, BrmError, function_of, library_names, load_brm, task_nest

IMPL = "chisel_tiled_spatial"


def function_params(inst) -> dict:
    """The function's params, resolved, in their order."""
    return {k: inst.value(v) for k, v in inst.brm.function.params.items()}


# =============================================================================
# The library
# =============================================================================


@pytest.mark.parametrize("name", library_names())
def test_library_files_are_canonical(name):
    """Every file loads, validates, and is written exactly as ``to_json`` writes it."""
    brm = load_brm(name)
    assert (LIBRARY / f"{name}.json").read_text() == brm.to_json()
    assert Brm.from_dict(brm.to_dict()) == brm


def test_library_has_dots_blocks():
    assert {"elementwise_add", "elementwise_mul", "accumulate"} <= set(library_names())


def test_the_brms_are_bound():
    """The bindings: elementwise_add and elementwise_mul, the same block with the op through
    ElementwiseOp; accumulate's Accumulator, counted with loopCount = n then len = T, pipelined,
    its lanes W, its port a on the block's in; the adder tree the same block."""
    from snax_forge.gen import ELEMENTWISE_OP

    from .test_brm import BINDING

    for name, op in (("elementwise_add", "add"), ("elementwise_mul", "mul")):
        brm = load_brm(name)
        assert brm.implementations[IMPL].binding == BINDING
        assert brm.resolve(IMPL).params["op"] == op
    assert ELEMENTWISE_OP["mul"] == 2
    acc = load_brm("accumulate").implementations
    assert acc["chisel_accumulator"].binding == {
        "block": "Accumulator", "args": {"lanes": "W"}, "control": "counted",
        "csrs": [{"field": "loopCount", "value": "n"}, {"field": "len", "value": "T"}],
        "wait": "pipelined", "ports": {"a": "in", "out": "out"},
    }  # fmt: skip
    assert acc["chisel_adder_tree"].binding == acc["chisel_accumulator"].binding


def test_unknown_name():
    with pytest.raises(BrmError, match="no BRM 'gemm' in the library"):
        load_brm("gemm")


# =============================================================================
# elementwise_add and elementwise_mul against the elementwise kind
# =============================================================================


def test_default_is_four_lanes_of_add():
    """No params: W = 4 by default, the elementwise kind with two inputs, latency 0, II 1."""
    inst = load_brm("elementwise_add").resolve(IMPL)
    assert inst.brm.function.accel == "elementwise"
    assert list(function_params(inst).items()) == [("lanes", 4), ("n_inputs", 2), ("op", "add")]
    assert (inst.latency, inst.initiation_interval, inst.drain) == (0, 1, 0)


def test_other_lane_counts():
    brm = load_brm("elementwise_add")
    for w in (1, 2, 8):
        inst = brm.resolve(IMPL, {"W": w})
        assert function_params(inst)["lanes"] == w
        assert [p.lanes for p in function_of(inst).ports] == [w, w, w]


def test_only_add():
    with pytest.raises(BrmError, match="not in the param's values"):
        load_brm("elementwise_add").resolve(IMPL, {"op": "mul"})


def test_ports_agree_on_element_positions():
    """The D70 assumption, checked for this BRM: a, b and out take the same element."""
    inst = load_brm("elementwise_add").resolve(IMPL)
    idx = [task_nest(inst, p, {"n": 16}).indices() for p in ("a", "b", "out")]
    assert np.array_equal(idx[0], idx[1]) and np.array_equal(idx[0], idx[2])


@pytest.mark.parametrize("w", [1, 4, 8])
def test_function_code_agrees_with_its_kind(w):
    """The BRM's code and the kind the reference executor runs compute the same lanes."""
    from snax_forge import expr

    inst = load_brm("elementwise_add").resolve("chisel_tiled_spatial", {"W": w})
    cfg = function_of(inst)
    rng = np.random.default_rng(w)
    ins = {p.name: rng.integers(-1000, 1000, p.lanes) for p in cfg.inputs}
    got = cfg.fn(0, ins, {}, {"n": 1})
    for target, e in expr.statements(inst.brm.function.code, "code"):
        assert np.array_equal(got[target], expr.evaluate(e, ins))


def test_elementwise_mul_is_add_with_mul():
    inst = load_brm("elementwise_mul").resolve(IMPL)
    assert inst.brm.function.accel == "elementwise"
    assert function_params(inst) == {"lanes": 4, "n_inputs": 2, "op": "mul"}
    assert (inst.latency, inst.initiation_interval) == (0, 1)
    assert inst.brm.pattern.attrs == {"op": "mul", "arity": 2}


@pytest.mark.parametrize("w", [1, 4, 8])
def test_mul_code_agrees_with_its_kind(w):
    """Per lane, elementwise_mul's code and its kind compute the same."""
    from snax_forge import expr

    inst = load_brm("elementwise_mul").resolve(IMPL, {"W": w})
    cfg = function_of(inst)
    rng = np.random.default_rng(w)
    ins = {p.name: rng.integers(-1000, 1000, p.lanes) for p in cfg.inputs}
    got = cfg.fn(0, ins, {}, {"n": 1})
    for target, e in expr.statements(inst.brm.function.code, "code"):
        assert np.array_equal(got[target], expr.evaluate(e, ins))


# =============================================================================
# accumulate against the reduce kind
# =============================================================================


def test_the_adder_tree_folds_four_lanes_into_one():
    """The adder tree at W = 4: the reduce kind over 4 lanes into 1, latency 1, II 1."""
    inst = load_brm("accumulate").resolve("chisel_adder_tree")
    assert inst.brm.function.accel == "reduce"
    assert list(function_params(inst).items()) == [("lanes", 4), ("lanes_out", 1), ("op", "add")]
    assert (inst.latency, inst.initiation_interval) == (1, 1)


def test_the_chisel_accumulator_declares_its_drain():
    """One lane, drain 0 (D133)."""
    brm = load_brm("accumulate")
    inst = brm.resolve("chisel_accumulator", {"W": 1})
    assert (inst.latency, inst.initiation_interval, inst.drain) == (1, 1, 0)
    assert function_params(inst) == {"lanes": 1, "lanes_out": 1, "op": "add"}
    assert brm.resolve("chisel_adder_tree").drain == 0
    with pytest.raises(BrmError, match="W = 4 not supported"):
        brm.resolve("chisel_accumulator", {"W": 4})


@pytest.mark.parametrize(("impl", "w"), [("chisel_adder_tree", 4), ("chisel_accumulator", 1)])
def test_accumulate_code_folded_agrees_with_its_kind(impl, w):
    """``out = a`` folded with the pattern's op over every lane of T beats is what the kind gives."""
    from snax_forge.dfg import REDUCE_OPS

    inst = load_brm("accumulate").resolve(impl, {"W": w})
    cfg = function_of(inst)
    T = 5
    beats = np.random.default_rng(w).integers(-1000, 1000, (T, w))
    state: dict = {}
    outs = [cfg.fn(k, {"a": beats[k]}, state, {"n": T, "T": T}) for k in range(T)]
    assert outs[:-1] == [{}] * (T - 1)
    op = REDUCE_OPS[inst.brm.pattern.attrs["op"]]
    assert inst.brm.function.code == "out = a"
    assert outs[-1]["out"].tolist() == [op.ufunc.reduce(beats.ravel())]


def test_accumulate_ports_and_beats():
    """a takes W elements per beat; out gives one element per T beats."""
    inst = load_brm("accumulate").resolve("chisel_adder_tree")
    assert task_nest(inst, "a", {"n": 16, "T": 4}).n_beats == 16
    out = task_nest(inst, "out", {"n": 16, "T": 4})
    assert (out.n_beats, out.n_lanes) == (4, 1)
