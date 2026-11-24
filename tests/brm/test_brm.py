"""The BRM format: parts, round trip and validation."""

from __future__ import annotations

import json

import pytest

from snax_forge.brm import Brm, BrmError

from .helpers import adder

# =============================================================================
# Round trip
# =============================================================================


def test_round_trip_every_field_written():
    d = adder()
    b = Brm.from_dict(d)
    assert b.to_dict() == d
    assert Brm.from_dict(b.to_dict()) == b


# =============================================================================
# Derived: registers
# =============================================================================


# =============================================================================
# Required and optional parts
# =============================================================================


# The binding of a simd BRM: the datapath, its args, the nest controller
BINDING = {
    "datapath": "SimdLanes",
    "args": {"lanes": "W", "op": "op"},
    "controller": "counted",
}


def _binding(**kw):
    return lambda d: _impl(d).update(binding={**BINDING, **kw})


BINDING_CASES = [
    ("binding key", lambda d: _impl(d).update(binding={**BINDING, "op": 0}), "unknown keys"),
    ("binding missing", lambda d: _impl(d).update(binding={"datapath": "X"}), "missing key 'args'"),
    ("datapath", _binding(datapath=3), "datapath: must name a datapath"),
    ("controller", _binding(controller="free"), "controller: 'free' not registered"),
    ("old keys", _binding(csrs=[], wait="lockstep"), "unknown keys"),
    ("args object", _binding(args=["W"]), "args: must be an object"),
    ("args runtime", _binding(args={"lanes": "n"}), "not design params"),
    ("args expr", _binding(args={"lanes": "'W'"}), "not allowed"),
    ("ports object", _binding(ports=["a"]), "ports: must be an object"),
    ("ports missing", _binding(ports={"a": "a", "b": "b"}), "must map every BRM port"),
    ("ports unknown", _binding(ports={"a": "a", "b": "b", "out": "o", "c": "c"}), "every BRM"),
    ("ports name", _binding(ports={"a": "a", "b": "b", "out": "o x"}), "not a datapath port name"),
    ("ports twice", _binding(ports={"a": "x", "b": "x", "out": "out"}), "two BRM ports"),
]


# =============================================================================
# Validation
# =============================================================================


def _params(d):
    return d["interface"]["params"]


def _port(d, i):
    return d["interface"]["ports"][i]


def _impl(d):
    return next(iter(d["implementations"].values()))


CASES = [
    # params
    ("stage", lambda d: _params(d)["W"].update(stage="elab"), "stage must be one of"),
    ("runtime default", lambda d: _params(d)["n"].update(default=4), "runtime param is an int"),
    ("default in values", lambda d: _params(d)["op"].update(default="sub"), "not in values"),
    ("value type", lambda d: _params(d)["W"].update(values=[1, "2"]), "of type int"),
    # ports
    ("duplicate port", lambda d: _port(d, 1).update(name="a"), "duplicate port names"),
    ("no output", lambda d: _port(d, 2).update(direction="in"), "one input and one output"),
    ("lanes runtime", lambda d: _port(d, 0).update(lanes="n"), "not design params"),
    ("lanes zero", lambda d: _port(d, 0).update(lanes=0), "lanes: must be >= 1"),
    ("rate design", lambda d: _port(d, 2).update(rate="W"), "runtime param name"),
    ("rate expr", lambda d: _port(d, 2).update(rate="n*2"), "runtime param name"),
    ("dtypes", lambda d: _port(d, 0).update(dtypes=["int65"]), "distinct names of"),
    ("dtypes empty", lambda d: _port(d, 0).update(dtypes=[]), "non-empty list"),
    ("dtypes twice", lambda d: _port(d, 0).update(dtypes=["int8", "int8"]), "distinct"),
    ("old dtype", lambda d: _port(d, 0).update(dtype="int64"), r"unknown keys \['dtype'\]"),
    ("keyword port", lambda d: _port(d, 0).update(name="in"), "'in' is a Python keyword"),
    ("underscore port", lambda d: _port(d, 0).update(name="_a"), "starts with an underscore"),
    ("extra runtime", lambda d: _params(d).update(T={"stage": "runtime"}), "exactly n and"),
    ("no n", lambda d: _params(d).pop("n"), "exactly n and"),
    # function
    ("function timing", lambda d: d["function"]["params"].update(ii=1), "from the implementation"),
    ("function name", lambda d: d["function"]["params"].update(lanes="V"), "not design params"),
    ("function expr", lambda d: d["function"]["params"].update(op="'add'"), "not allowed"),
    # implementations
    ("source", lambda d: _impl(d).update(source="systemverilog"), "source: must be one of"),
    ("supports param", lambda d: _impl(d)["supports"].update(n=[16]), "not a design param"),
    ("supports values", lambda d: _impl(d)["supports"].update(op=["sub"]), "not within"),
    ("ii zero", lambda d: _impl(d)["timing"].update(initiation_interval=0), "must be >= 1"),
    ("latency runtime", lambda d: _impl(d)["timing"].update(latency="n"), "not design params"),
    ("drain negative", lambda d: _impl(d)["timing"].update(drain=-1), "drain: must be >= 0"),
    ("drain runtime", lambda d: _impl(d)["timing"].update(drain="n"), "not design params"),
    ("binding", lambda d: _impl(d).update(binding="chisel"), "null or an object"),
    # pattern and dataflow
    ("family", lambda d: d["pattern"].update(family=""), "family: must be"),
    ("notation", lambda d: d["dataflow"].update(notation="affine9"), "not registered"),
    ("nest missing", lambda d: d["dataflow"]["ports"].pop("b"), "one nest per port"),
    ("nest invalid", lambda d: d["dataflow"]["ports"].update(out=["x"]), "list of ints"),
    ("keyword param", lambda d: _params(d).update({"in": {"stage": "design"}}), "valid name"),
    ("stream key", lambda d: d.update(dataflow=_stream({"out": {"shape": [4]}})), "has no keys"),
    ("stream list", lambda d: d.update(dataflow=_stream({"a": []})), "an empty object"),
]


def _stream(nests: dict) -> dict:
    """The stream notation: every port's nest empty but the ones given."""
    return {"notation": "stream", "ports": {"a": {}, "b": {}, "out": {}} | nests}


@pytest.mark.parametrize(
    ("case", "edit", "msg"), CASES + BINDING_CASES, ids=[c[0] for c in CASES + BINDING_CASES]
)
def test_invalid(case, edit, msg):
    d = adder()
    edit(d)
    with pytest.raises(BrmError, match=msg):
        Brm.from_dict(d)


# =============================================================================
# function.code: what one lane computes
# =============================================================================


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("out = a + c", "'c' is not an input"),
        ("y = a + b", "assigns 'y', which is not an output"),
        ("out = a\nout = b", "exactly once"),
        ("out = a / b", "Div is not allowed"),
        ("a + b", "is not 'output = expression'"),
    ],
)
def test_function_code_rejected(code, message):
    d = adder()
    d["function"]["code"] = code
    with pytest.raises(BrmError, match=message):
        Brm.from_dict(d)


# =============================================================================
# function.code as a table: one code per design-param value (simd's op)
# =============================================================================


def two_ops() -> dict:
    """adder() computing add or mul by its op, a code entry each."""
    d = adder()
    d["interface"]["params"]["op"]["values"] = ["add", "mul"]
    d["function"]["code"] = [
        {"params": {"op": "add"}, "code": "out=(a+b)"},
        {"params": {"op": "mul"}, "code": "out = a * b"},
    ]
    return d


def test_a_code_table_is_stored_canonical_and_round_trips():
    b = Brm.from_dict(two_ops())
    assert [(e.params, e.code) for e in b.function.entries()] == [
        ({"op": "add"}, "out = a + b"), ({"op": "mul"}, "out = a * b")]  # fmt: skip
    assert b.to_dict()["function"]["code"] == [
        {"params": {"op": "add"}, "code": "out = a + b"},
        {"params": {"op": "mul"}, "code": "out = a * b"},
    ]
    assert Brm.from_dict(b.to_dict()) == b and Brm.from_dict(json.loads(b.to_json())) == b
    assert b.code_for({"W": 4, "op": "mul"}) == "out = a * b"
    assert b.code_for({"W": 4, "op": "sub"}) is None


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda c: c.clear(), r"function.code: must be a string, a non-empty list"),
        (lambda c: c[0]["params"].update(W=4, K=1), r"code\[0\].params.K: not a design param"),
        (lambda c: c[1]["params"].update(op="sub"),
         r"code\[1\].params.op: 'sub' is not an allowed value \(\['add', 'mul'\]\)"),
        (lambda c: c[1]["params"].update(op=2), r"code\[1\].params.op: 2 is not an allowed value"),
        (lambda c: c[1].update(code="out = a / b"), r"code\[1\].code.*Div is not allowed"),
        (lambda c: c[0].pop("params"), r"code\[0\]"),
        (lambda c: c[0].update(when=1), r"code\[0\]"),
        (lambda c: c[0].update(params=[]), r"code\[0\].params: must be an object"),
    ],
)  # fmt: skip
def test_a_bad_code_table_is_rejected(edit, message):
    d = two_ops()
    edit(d["function"]["code"])
    with pytest.raises(BrmError, match=message):
        Brm.from_dict(d)


# =============================================================================
# Ports with when, unset params, slots
# =============================================================================


def one_or_two() -> dict:
    """adder() on one stream or two: b only at two, the one-stream entry over the slot k."""
    d = adder()
    params = d["interface"]["params"]
    params["streams"] = {"stage": "design", "type": "int", "default": 2, "values": [1, 2]}
    params["k"] = {"stage": "design", "type": "int", "default": None, "values": None}
    d["interface"]["ports"][1]["when"] = {"streams": 2}
    d["function"]["code"] = [
        {"params": {"op": "add", "streams": 2}, "code": "out = a + b"},
        {"params": {"op": "add", "streams": 1}, "code": "out = a + k"},
    ]
    return d


def test_a_port_with_when_round_trips_and_is_written_only_when_set():
    d = one_or_two()
    b = Brm.from_dict(d)
    assert b.to_dict() == d and Brm.from_dict(json.loads(b.to_json())) == b
    assert b.interface.ports[1].when == {"streams": 2}
    assert "when" not in b.to_dict()["interface"]["ports"][0]
    assert [p.name for p in b.present({"streams": 1})] == ["a", "out"]
    assert [p.name for p in b.present({"streams": 2})] == ["a", "b", "out"]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d["interface"]["ports"][1].update(when={"n": 2}),
         r"port 'b'.when.n: not a design param"),
        (lambda d: d["interface"]["ports"][1].update(when={"lanes": 2}),
         r"port 'b'.when.lanes: not a design param"),
        (lambda d: d["interface"]["ports"][1].update(when={"streams": 3}),
         r"port 'b'.when.streams: 3 is not an allowed value \(\[1, 2\]\)"),
        (lambda d: d["interface"]["ports"][1].update(when={}), r"port 'b'.when: must be a non-empty"),
        (lambda d: d["function"]["code"][1].update(code="out = a + b"),
         r"code\[1\].code: reads 'b', a port not present at op = 'add', streams = 1"),
        (lambda d: d["function"]["code"][0].update(code="out = a + q"),
         r"code\[0\].code: 'q' is not an input present at op = 'add', streams = 2, nor a slot"),
        (lambda d: d["function"]["code"][1].update(code="out = a + op"),
         r"code\[1\].code: 'op' is not an input present at .*nor a slot"),
    ],
)  # fmt: skip
def test_a_bad_when_or_slot_is_rejected(edit, message):
    d = one_or_two()
    edit(d)
    with pytest.raises(BrmError, match=message):
        Brm.from_dict(d)
