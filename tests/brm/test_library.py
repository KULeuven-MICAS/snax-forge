"""The BRM library: simd, accumulate and mac.

Each BRM resolves to the function the reference executor runs (its kind in
brm/function.py, its params resolved), and its ``code`` computes what that
kind computes, lane by lane; accumulate's Chisel implementation declares the
Accumulator's drain, 0 (D133).
"""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.brm import (
    LIBRARY,
    Brm,
    BrmError,
    function_of,
    library_names,
    load_brm,
    memlet_ordered,
)

from .helpers import IMPL


def function_params(inst) -> dict:
    """The function's params, resolved, in their order; one naming an unset param left out."""
    from snax_forge import expr

    return {k: inst.value(v) for k, v in inst.brm.function.params.items()
            if expr.names(v) <= set(inst.params)}  # fmt: skip


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
    assert {"simd", "accumulate"} <= set(library_names())
    assert "elementwise_add" not in library_names() and "elementwise_mul" not in library_names()


def test_the_brms_are_bound():
    """The bindings: simd, SimdLanes with its lanes, op, streams and slot k, at each entry of
    its code table; accumulate's TreeAccumulate, its lanes W, its port a on the datapath's in;
    the adder tree the same datapath; every one under the counted nest controller."""
    from snax_forge.gen import ELEMENTWISE_OP

    brm = load_brm("simd")
    assert brm.implementations[IMPL].binding == {
        "datapath": "SimdLanes", "args": {"lanes": "W", "op": "op", "streams": "streams", "k": "k"},
        "controller": "counted",
    }  # fmt: skip
    for e in brm.function.entries():
        given = e.params | ({"k": 0} if e.params["streams"] == 1 else {})
        assert brm.resolve(IMPL, given).params == {"W": 4, **given}
    assert ELEMENTWISE_OP["mul"] == 2
    acc = load_brm("accumulate").implementations
    assert acc["chisel_accumulator"].binding == {
        "datapath": "TreeAccumulate", "args": {"lanes": "W"}, "controller": "counted",
        "ports": {"a": "in", "out": "out"},
    }  # fmt: skip
    assert acc["chisel_adder_tree"].binding == acc["chisel_accumulator"].binding


def test_unknown_name():
    with pytest.raises(BrmError, match="no BRM 'gemm' in the library"):
        load_brm("gemm")


# =============================================================================
# simd against the simd kind: a code table by op and streams, the slot k
# =============================================================================

TWO = {"add": "a + b", "sub": "a - b", "mul": "a * b", "min": "min(a, b)", "max": "max(a, b)"}
ONE = {"add": "a + k", "sub": "a - k", "mul": "a * k", "min": "min(a, k)", "max": "max(a, k)"}


def test_default_is_four_lanes_of_add_on_two_streams():
    """No params: W = 4, op add, two streams, k unset (left out of the params and of the
    function's), latency 0, II 1."""
    inst = load_brm("simd").resolve(IMPL)
    assert inst.params == {"W": 4, "op": "add", "streams": 2}
    assert inst.brm.function.accel == "simd"
    assert list(function_params(inst).items()) == [("lanes", 4), ("n_inputs", 2), ("op", "add")]
    assert [p.name for p in inst.ports] == ["a", "b", "out"]
    assert (inst.latency, inst.initiation_interval, inst.drain) == (0, 1, 0)


def test_one_stream_has_no_b_and_takes_k():
    inst = load_brm("simd").resolve(IMPL, {"op": "max", "streams": 1, "k": 0})
    assert inst.params == {"W": 4, "op": "max", "streams": 1, "k": 0}
    assert [p.name for p in inst.ports] == ["a", "out"]
    assert function_params(inst) == {"lanes": 4, "n_inputs": 1, "op": "max", "k": 0}
    assert [p.name for p in function_of(inst).ports] == ["a", "out"]


def test_one_stream_without_k_or_two_with_one_is_refused():
    brm = load_brm("simd")
    with pytest.raises(BrmError, match="one stream takes the constant k"):
        brm.resolve(IMPL, {"streams": 1})
    with pytest.raises(BrmError, match="k = 3 at two streams"):
        brm.resolve(IMPL, {"k": 3})


def test_other_lane_counts():
    brm = load_brm("simd")
    for w in (1, 2, 8):
        inst = brm.resolve(IMPL, {"W": w})
        assert function_params(inst)["lanes"] == w
        assert [p.lanes for p in function_of(inst).ports] == [w, w, w]


def test_only_its_ops():
    with pytest.raises(BrmError, match="not in the param's values"):
        load_brm("simd").resolve(IMPL, {"op": "and"})


@pytest.mark.parametrize("name", ["simd", "accumulate"])
def test_unit_brms_are_in_the_stream_notation(name):
    """Unit blocks (D70): every port's nest is empty, the bound memlet gives the order, so a,
    b and out take the same element as the memlets do; each loads and resolves."""
    brm = load_brm(name)
    assert brm.dataflow.notation == "stream" and memlet_ordered(brm)
    assert brm.dataflow.ports == {p.name: {} for p in brm.interface.ports}
    for impl in brm.implementations:
        brm.resolve(impl, {"W": 1} if impl == "chisel_accumulator" else {})


@pytest.mark.parametrize("streams", [2, 1])
@pytest.mark.parametrize("op", list(TWO))
@pytest.mark.parametrize("w", [1, 4, 8])
def test_function_code_agrees_with_its_kind(streams, op, w):
    """Each entry's code and the kind the reference executor runs compute the same lanes (k
    random at one stream)."""
    from snax_forge import expr

    rng = np.random.default_rng(w)
    k = {"k": int(rng.integers(-50, 50))} if streams == 1 else {}
    inst = load_brm("simd").resolve(IMPL, {"W": w, "op": op, "streams": streams, **k})
    cfg = function_of(inst)
    ins = {p.name: rng.integers(-1000, 1000, p.lanes) for p in cfg.inputs}
    got = cfg.fn(0, ins, {}, {"n": 1})
    for target, e in expr.statements(inst.brm.function.code_for(inst.params), "code"):
        assert np.array_equal(got[target], expr.evaluate(e, ins | k))


def test_one_stream_casts_k_to_a_dtype():
    """k at a's dtype, the result at it too (cast into the output's when written)."""
    inst = load_brm("simd").resolve(IMPL, {"op": "add", "streams": 1, "k": -1})
    a = np.array([0, 1, 127, -128], np.int8)
    got = function_of(inst).fn(0, {"a": a}, {}, {"n": 1})["out"]
    assert got.dtype == np.int8 and got.tolist() == [-1, 0, 126, 127]


def test_simd_holds_ten_entries():
    """The code table: the five ops on two streams, then on one with the slot k; the default
    op add; the pattern leaves everything to the code (no attrs)."""
    brm = load_brm("simd")
    assert [(e.params, e.code) for e in brm.function.entries()] == [
        *(({"op": o, "streams": 2}, f"out = {c}") for o, c in TWO.items()),
        *(({"op": o, "streams": 1}, f"out = {c}") for o, c in ONE.items())]  # fmt: skip
    assert [brm.slots(e) for e in brm.function.entries()] == [[]] * 5 + [["k"]] * 5
    assert brm.function.code_for(brm.resolve(IMPL).params) == "out = a + b"
    inst = brm.resolve(IMPL, {"op": "min", "streams": 1, "k": 7})
    assert brm.function.code_for(inst.params) == "out = min(a, k)"
    assert brm.pattern.family == "simd" and brm.pattern.attrs == {}


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
    """One lane, drain 0."""
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
    """a takes W elements per beat; out gives one element per T beats (the interface's lanes
    and rates: the stream notation adds nothing to them)."""
    inst = load_brm("accumulate").resolve("chisel_adder_tree")
    assert (inst.lanes("a"), inst.brm.port("a").rate) == (4, 1)
    assert (inst.lanes("out"), inst.brm.port("out").rate) == (1, "T")
