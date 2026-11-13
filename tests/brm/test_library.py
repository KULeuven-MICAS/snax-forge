"""The BRM library: simd, accumulate and mac.

Each BRM resolves to the function the reference executor runs (its kind in
brm/function.py, its params resolved), and its ``code`` computes what that
kind computes, lane by lane; accumulate's Chisel implementation declares the
Accumulator's drain, 0.
"""

from __future__ import annotations

import json

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
    its code table; accumulate's TreeAccumulate, its lanes W, its port a on the datapath's in
    and its optional seed on seed; the adder tree the same datapath; every one under the
    counted nest controller."""
    from snax_forge.gen import ELEMENTWISE_OP

    brm = load_brm("simd")
    assert brm.implementations[IMPL].binding == {
        "datapath": "SimdLanes", "args": {"lanes": "W", "op": "op", "streams": "streams", "k": "k",
                                          "k1": "k1", "k2": "k2"},
        "controller": "counted",
    }  # fmt: skip
    for e in brm.function.entries():
        given = e.params | {s: 2 for s in brm.slots(e)}
        assert brm.resolve(IMPL, given).params == {"W": 4, **given}
    assert ELEMENTWISE_OP["mul"] == 2
    acc = load_brm("accumulate").implementations
    assert list(acc) == ["chisel_adder_tree", "chisel_serial"]  # one name per datapath
    assert acc["chisel_adder_tree"].binding == {
        "datapath": "TreeAccumulate", "args": {"lanes": "W"}, "controller": "counted",
        "ports": {"a": "in", "seed": "seed", "out": "out"},
    }  # fmt: skip
    assert acc["chisel_serial"].binding == acc["chisel_adder_tree"].binding | {
        "datapath": "SerialAccumulate"}  # fmt: skip


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


def test_an_entry_fits_when_its_slots_are_set_or_read_and_no_other_is():
    """An instance's code is the one entry whose params it has, every slot the code names set
    (baked) or read (a CSR), every other slot neither: axpby with k2 unset is ``k1 * a + b``,
    with k2 baked or read ``k1 * a + k2 * b``; one stream with k read is ``op(a, k)``."""
    brm = load_brm("simd")
    axpby = {"op": "axpby"}
    two, one = "out = k1 * a + k2 * b", "out = k1 * a + b"
    for params, read, code in [
        (axpby | {"k1": 3}, (), one),
        (axpby, ("k1",), one),
        (axpby | {"k1": 3, "k2": -2}, (), two),
        (axpby | {"k1": 3}, ("k2",), two),
        (axpby, ("k2", "k1"), two),
        ({"op": "max", "streams": 1}, ("k",), "out = max(a, k)"),
        ({}, (), "out = a + b"),
    ]:
        inst = brm.resolve(IMPL, params, read)
        assert inst.code == code and brm.code_for(inst.params, read) == code
        assert inst.read == tuple(s for s in ("k1", "k2", "k") if s in read)
    assert brm.code_for({"W": 4, "op": "axpby", "streams": 2}) is None  # k1 has no value


@pytest.mark.parametrize(
    ("params", "read", "message"),
    [
        ({"op": "axpby"}, (), "axpby takes k1, got None"),
        ({"op": "axpby", "k2": 2}, (), "axpby takes k1, got None"),
        ({"op": "max", "streams": 1}, (), "one stream takes the constant k, got None"),
        ({"op": "axpby", "k1": 3}, ("k1",), "k1 = 3 is set, and read from a container too"),
        ({"op": "add"}, ("alpha",), "'alpha' is read from a container, but it is no slot"),
        ({"op": "add"}, ("k1",), "k1 and k2 are axpby's, not add's"),
        ({"op": "axpby", "streams": 1, "k": 2}, (), "axpby at one stream"),
    ],
)
def test_an_instance_no_entry_fits_is_refused(params, read, message):
    """Refused at resolve, not at a firing: a slot the code needs with no value, one set and
    read, and a read slot the code does not name."""
    with pytest.raises(BrmError, match=message):
        load_brm("simd").resolve(IMPL, params, read)


def test_the_code_check_refuses_what_no_entry_fits():
    """resolve's own check (instance.py ``_check_code``), behind the adapter's: a BRM without
    a binding, whose datapath adapter does not run, gets it alone."""
    d = json.loads(load_brm("simd").to_json())
    for impl in d["implementations"].values():
        impl["binding"] = None
    brm = Brm.from_dict(d)
    with pytest.raises(BrmError, match=r"the code is 'out = max\(a, k\)' with k; the slots set "
                       r"here \(baked, or read from a container\): none"):  # fmt: skip
        brm.resolve(IMPL, {"op": "max", "streams": 1})
    with pytest.raises(BrmError, match="'out = k1 \\* a \\+ k2 \\* b' with k1 and k2 or "
                       "'out = k1 \\* a \\+ b' with k1; the slots set here .*: k2"):  # fmt: skip
        brm.resolve(IMPL, {"op": "axpby", "k2": 2})


def test_two_entries_that_fit_the_same_instances_are_refused():
    d = json.loads(load_brm("simd").to_json())
    d["function"]["code"].append({"params": {"op": "axpby", "streams": 2},
                                  "code": "out = a * k1 + b"})  # fmt: skip
    with pytest.raises(BrmError, match=r"code\[12\]: fits the instances code\[11\] fits"):
        Brm.from_dict(d)


@pytest.mark.parametrize("dtypes", [("int32", "int32"), ("int8", "int32"), ("int64", "int64")])
@pytest.mark.parametrize("k2", [None, -2])
def test_axpby_is_k1_a_plus_k2_b_at_the_op_type(k2, dtypes):
    """k1 * a + k2 * b (k2 unset: + b) at the inputs' result type, wrapping there."""
    rng = np.random.default_rng(3)
    k1 = int(rng.integers(-9, 9))
    params = {"op": "axpby", "k1": k1} | ({} if k2 is None else {"k2": k2})
    fn = function_of(load_brm("simd").resolve(IMPL, params)).fn
    t = np.result_type(*dtypes)
    info = np.iinfo(dtypes[0])
    a = rng.integers(info.min, info.max, 4, dtype=dtypes[0], endpoint=True)
    b = rng.integers(-1000, 1000, 4).astype(dtypes[1])
    got = fn(0, {"a": a, "b": b}, {}, {"n": 1})["out"]
    want = t.type(k1) * a.astype(t) + (b.astype(t) if k2 is None else t.type(k2) * b.astype(t))
    assert got.dtype == t and np.array_equal(got, want)


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
    """Unit blocks: every port's nest is empty, the bound memlet gives the order, so a,
    b and out take the same element as the memlets do; each loads and resolves."""
    brm = load_brm(name)
    assert brm.dataflow.notation == "stream" and memlet_ordered(brm)
    assert brm.dataflow.ports == {p.name: {} for p in brm.interface.ports}
    for impl in brm.implementations:
        brm.resolve(impl, {})


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
    for target, e in expr.statements(inst.code, "code"):
        assert np.array_equal(got[target], expr.evaluate(e, ins | k))


def test_one_stream_casts_k_to_a_dtype():
    """k at a's dtype, the result at it too (cast into the output's when written)."""
    inst = load_brm("simd").resolve(IMPL, {"op": "add", "streams": 1, "k": -1})
    a = np.array([0, 1, 127, -128], np.int8)
    got = function_of(inst).fn(0, {"a": a}, {}, {"n": 1})["out"]
    assert got.dtype == np.int8 and got.tolist() == [-1, 0, 126, 127]


def test_simd_holds_twelve_entries():
    """The code table: the five ops on two streams, then on one with the slot k, then axpby
    with k1 and k2, and with k1 alone; the default op add; the pattern leaves everything to the
    code (no attrs)."""
    brm = load_brm("simd")
    axpby = {"op": "axpby", "streams": 2}
    assert [(e.params, e.code) for e in brm.function.entries()] == [
        *(({"op": o, "streams": 2}, f"out = {c}") for o, c in TWO.items()),
        *(({"op": o, "streams": 1}, f"out = {c}") for o, c in ONE.items()),
        (axpby, "out = k1 * a + k2 * b"), (axpby, "out = k1 * a + b")]  # fmt: skip
    assert [brm.slots(e) for e in brm.function.entries()] == (
        [[]] * 5 + [["k"]] * 5 + [["k1", "k2"], ["k1"]])  # fmt: skip
    assert brm.all_slots() == ["k", "k1", "k2"]
    assert brm.resolve(IMPL).code == "out = a + b"
    assert brm.resolve(IMPL, {"op": "min", "streams": 1, "k": 7}).code == "out = min(a, k)"
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


def test_the_adder_tree_at_one_lane_declares_its_drain():
    """The adder tree at W = 1 (vsum's), drain 0 as at any W; the serial accumulator too."""
    brm = load_brm("accumulate")
    inst = brm.resolve("chisel_adder_tree", {"W": 1})
    assert (inst.latency, inst.initiation_interval, inst.drain) == (1, 1, 0)
    assert function_params(inst) == {"lanes": 1, "lanes_out": 1, "op": "add"}
    assert brm.resolve("chisel_adder_tree").drain == brm.resolve("chisel_serial").drain == 0


@pytest.mark.parametrize(("impl", "w"), [("chisel_adder_tree", 4), ("chisel_adder_tree", 1),
                                         ("chisel_serial", 4)])  # fmt: skip
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
