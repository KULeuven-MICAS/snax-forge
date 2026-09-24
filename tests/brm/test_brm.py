"""The BRM format (BRM1, D68): parts, round trip and validation."""

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


def test_binding_is_optional():
    d = adder()
    del d["implementations"]["chisel_tiled_spatial"]["binding"]
    assert Brm.from_dict(d).implementations["chisel_tiled_spatial"].binding is None
    binding = {"module": "snax.forge.elementwise.ElementwiseTiledSpatial", "params": {}}
    d["implementations"]["chisel_tiled_spatial"]["binding"] = binding
    assert (
        Brm.from_dict(d).to_dict()["implementations"]["chisel_tiled_spatial"]["binding"] == binding
    )


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
