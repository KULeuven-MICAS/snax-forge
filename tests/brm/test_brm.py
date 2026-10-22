"""The BRM format: parts, round trip and validation."""

from __future__ import annotations

import pytest

from snax_forge.brm import Brm, BrmError

from .helpers import adder, reducer

# =============================================================================
# Round trip
# =============================================================================


def test_round_trip_every_field_written():
    d = adder()
    b = Brm.from_dict(d)
    assert b.to_dict() == d
    assert Brm.from_dict(b.to_dict()) == b


def test_sparse_file_takes_defaults_and_writes_them():
    b = Brm.from_dict(reducer())
    d = b.to_dict()
    assert d["interface"]["params"]["n"] == {
        "stage": "runtime",
        "type": "int",
        "default": None,
        "values": None,
    }
    assert d["interface"]["ports"][0]["rate"] == 1
    assert d["interface"]["ports"][0]["dtype"] == "int64"
    assert d["pattern"] == {"family": "reduce", "attrs": {}, "predicate": None}
    assert d["function"]["code"] is None  # a reduction cannot say it per lane yet (D82)
    impl = d["implementations"]["chisel_accumulator"]
    assert impl["supports"] == {} and impl["binding"] is None
    assert Brm.from_dict(d) == b


def test_save_load(tmp_path):
    b = Brm.from_dict(adder())
    b.save(tmp_path / "test_add.json")
    assert Brm.load(tmp_path / "test_add.json") == b
    assert (tmp_path / "test_add.json").read_text() == b.to_json()


# =============================================================================
# Derived: registers
# =============================================================================


def test_registers_are_n_then_named_rates():
    assert Brm.from_dict(adder()).registers == ["n"]
    assert Brm.from_dict(reducer()).registers == ["n", "T"]


# =============================================================================
# Required and optional parts
# =============================================================================


@pytest.mark.parametrize(
    "part", ["name", "interface", "function", "dataflow", "pattern", "implementations"]
)
def test_missing_part_is_rejected(part):
    d = adder()
    del d[part]
    with pytest.raises(BrmError, match=f"missing part '{part}'"):
        Brm.from_dict(d)


@pytest.mark.parametrize("part", ["source", "timing"])
def test_missing_implementation_part_is_rejected(part):
    d = adder()
    del d["implementations"]["chisel_tiled_spatial"][part]
    with pytest.raises(BrmError, match=f"missing part '{part}'"):
        Brm.from_dict(d)


def test_no_implementation_is_rejected():
    d = adder()
    d["implementations"] = {}
    with pytest.raises(BrmError, match="at least one implementation"):
        Brm.from_dict(d)


# The binding of ElementwiseTiledSpatial
BINDING = {
    "block": "ElementwiseTiledSpatial",
    "args": {"lanes": "W", "supportedOps": "op"},
    "control": "counted",
    "csrs": [{"field": "loopCount", "value": "n"}],
    "wait": "lockstep",
}


def test_binding_is_optional():
    d = adder()
    del d["implementations"]["chisel_tiled_spatial"]["binding"]
    assert Brm.from_dict(d).implementations["chisel_tiled_spatial"].binding is None
    d["implementations"]["chisel_tiled_spatial"]["binding"] = BINDING
    assert Brm.from_dict(d).to_dict()["implementations"]["chisel_tiled_spatial"]["binding"] == (
        BINDING
    )


def test_binding_expressions_are_stored_canonical():
    d = adder()
    _impl(d)["binding"] = {**BINDING, "args": {"lanes": "W*1", "supportedOps": "op"},
                           "csrs": [{"field": "loopCount", "value": "(n)"}]}  # fmt: skip
    got = Brm.from_dict(d).implementations["chisel_tiled_spatial"].binding
    assert got["args"]["lanes"] == "W * 1" and got["csrs"] == [{"field": "loopCount", "value": "n"}]


def test_binding_ports_are_kept_in_port_order_when_given():
    d = adder()
    _impl(d)["binding"] = {**BINDING, "ports": {"out": "y", "b": "x1", "a": "x0"}}
    got = Brm.from_dict(d).implementations["chisel_tiled_spatial"].binding
    assert list(got["ports"].items()) == [("a", "x0"), ("b", "x1"), ("out", "y")]
    _impl(d)["binding"] = BINDING
    assert "ports" not in Brm.from_dict(d).implementations["chisel_tiled_spatial"].binding


def _binding(**kw):
    return lambda d: _impl(d).update(binding={**BINDING, **kw})


BINDING_CASES = [
    ("binding key", lambda d: _impl(d).update(binding={**BINDING, "op": 0}), "unknown keys"),
    ("binding missing", lambda d: _impl(d).update(binding={"block": "X"}), "missing key 'args'"),
    ("block", _binding(block=3), "block: must name a Chisel block"),
    ("control", _binding(control="free"), "control: 'free' not registered"),
    ("wait", _binding(wait="buffered"), "wait: 'buffered' not registered"),
    ("args object", _binding(args=["W"]), "args: must be an object"),
    ("args runtime", _binding(args={"lanes": "n"}), "not design params"),
    ("args expr", _binding(args={"lanes": "'W'"}), "not allowed"),
    ("csrs list", _binding(csrs={"loopCount": "n"}), "csrs: must be a list"),
    ("csr keys", _binding(csrs=[{"field": "loopCount"}]), "must be {field, value}"),
    ("csr name", _binding(csrs=[{"field": "T", "value": "m"}]), "uses \\['m'\\]"),
    ("csr field", _binding(csrs=[{"field": "a b", "value": "n"}]), "not a field name"),
    ("csr twice", _binding(csrs=[{"field": "x", "value": "n"}] * 2), "duplicate fields"),
    ("ports object", _binding(ports=["a"]), "ports: must be an object"),
    ("ports missing", _binding(ports={"a": "a", "b": "b"}), "must map every BRM port"),
    ("ports unknown", _binding(ports={"a": "a", "b": "b", "out": "o", "c": "c"}), "every BRM"),
    ("ports name", _binding(ports={"a": "a", "b": "b", "out": "o x"}), "not a block port name"),
    ("ports twice", _binding(ports={"a": "x", "b": "x", "out": "out"}), "two BRM ports"),
]


@pytest.mark.parametrize(("case", "edit", "msg"), BINDING_CASES, ids=[c[0] for c in BINDING_CASES])
def test_invalid_binding(case, edit, msg):
    d = adder()
    edit(d)
    with pytest.raises(BrmError, match=msg):
        Brm.from_dict(d)


def test_control_and_wait_are_registered_kinds():
    from snax_forge.brm import CONTROLS, WAITS, register_wait

    assert list(CONTROLS) == ["counted"] and list(WAITS) == ["lockstep", "pipelined"]
    with pytest.raises(ValueError, match="already registered"):
        register_wait("lockstep", "again")


def test_unknown_keys_are_rejected():
    d = adder()
    d["timing"] = {}
    with pytest.raises(BrmError, match="unknown keys \\['timing'\\]"):
        Brm.from_dict(d)
    d = adder()
    d["interface"]["ports"][0]["width"] = 64
    with pytest.raises(BrmError, match="unknown keys \\['width'\\]"):
        Brm.from_dict(d)


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
    ("dtype", lambda d: _port(d, 0).update(dtype="int65"), "is not a dtype"),
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
]


@pytest.mark.parametrize(("case", "edit", "msg"), CASES, ids=[c[0] for c in CASES])
def test_invalid(case, edit, msg):
    d = adder()
    edit(d)
    with pytest.raises(BrmError, match=msg):
        Brm.from_dict(d)


def test_python_construction_is_validated():
    b = Brm.from_dict(adder())
    b.interface.ports[0].lanes = "n"
    with pytest.raises(BrmError, match="not design params"):
        Brm(b.name, b.interface, b.function, b.dataflow, b.pattern, b.implementations)


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


def test_function_code_is_stored_canonical():
    d = adder()
    d["function"]["code"] = "out=(a+b)"
    assert Brm.from_dict(d).function.code == "out = a + b"
