"""BRM instances and their accelerator entry (BRM1, D68)."""

from __future__ import annotations

import pytest

from snax_forge.brm import Brm, BrmError
from snax_forge.snax_model.accel import elementwise_stub, reduce_stub

from .helpers import adder, reducer

IMPL = "chisel_tiled_spatial"


def _brm(d=None) -> Brm:
    return Brm.from_dict(d or adder())


# =============================================================================
# Resolving and the entry
# =============================================================================


def test_entry_of_the_adder():
    inst = _brm().resolve(IMPL)
    assert inst.params == {"W": 4, "op": "add"}  # defaults, in declaration order
    kind, params = inst.accel_entry()
    assert kind == "elementwise"
    assert list(params.items()) == [
        ("lanes", 4),
        ("n_inputs", 2),
        ("op", "add"),
        ("latency", 0),
        ("ii", 1),
    ]


def test_entry_follows_the_design_params():
    inst = _brm().resolve(IMPL, {"W": 8})
    assert inst.accel_entry()[1]["lanes"] == 8
    assert inst.lanes("out") == 8


def test_entry_builds_the_stub():
    cfg = _brm().resolve(IMPL).accel_config()
    stub = elementwise_stub(lanes=4, n_inputs=2)
    assert (cfg.ports, cfg.latency, cfg.ii, cfg.kind) == (
        stub.ports,
        stub.latency,
        stub.ii,
        stub.kind,
    )


def test_reducer_matches_the_reduce_stub():
    inst = Brm.from_dict(reducer()).resolve("chisel_accumulator", {"W": 4})
    assert inst.accel_entry() == (
        "reduce",
        {"lanes": 4, "lanes_out": 1, "latency": 1, "ii": 1},
    )
    stub = reduce_stub(lanes=4, lanes_out=1)
    assert inst.accel_config().ports == stub.ports


def test_timing_expressions_are_resolved():
    d = adder()
    d["implementations"][IMPL]["timing"] = {"latency": "W // 2", "initiation_interval": "1"}
    inst = _brm(d).resolve(IMPL, {"W": 8})
    assert (inst.latency, inst.initiation_interval) == (4, 1)
    assert inst.accel_entry()[1]["latency"] == 4


# =============================================================================
# Rejected values
# =============================================================================

VALUE_CASES = [
    ("impl", "chisel_loop", {}, "no implementation 'chisel_loop'"),
    ("unknown", IMPL, {"V": 4}, "unknown param 'V'"),
    ("runtime", IMPL, {"n": 16}, "runtime param, given per task"),
    ("type", IMPL, {"W": "4"}, "not of type int"),
    ("values", IMPL, {"op": "sub"}, "not in the param's values"),
    ("supports", IMPL, {"W": 3}, "not supported"),
]


@pytest.mark.parametrize(
    ("case", "impl", "params", "msg"), VALUE_CASES, ids=[c[0] for c in VALUE_CASES]
)
def test_rejected_values(case, impl, params, msg):
    with pytest.raises(BrmError, match=msg):
        _brm().resolve(impl, params)


def test_missing_value_without_default():
    with pytest.raises(BrmError, match="no value for 'W'"):
        Brm.from_dict(reducer()).resolve("chisel_accumulator")


def test_resolved_timing_out_of_range():
    d = adder()
    d["implementations"][IMPL]["timing"]["latency"] = "W - 8"
    with pytest.raises(BrmError, match="latency = 'W - 8' gives -4"):
        _brm(d).resolve(IMPL)


# =============================================================================
# Mismatches with the model's kind
# =============================================================================


def _rename_ports(d):
    for p, new in zip(d["interface"]["ports"], ["x", "y", "out"], strict=True):
        p["name"] = new
    d["dataflow"]["ports"] = {k: list(range(4)) for k in ("x", "y", "out")}
    d["function"]["code"] = "out = x + y"


MODEL_CASES = [
    ("kind", lambda d: d["function"].update(accel="gemm"), "not registered"),
    ("factory", lambda d: d["function"]["params"].update(width="W"), "rejects"),
    ("names", _rename_ports, "differ from kind"),
    ("lanes", lambda d: d["function"]["params"].update(lanes="W*2"), "differ from kind"),
    ("inputs", lambda d: d["function"]["params"].update(n_inputs=3), "differ from kind"),
    (
        "latency",
        lambda d: d["function"].update(accel="test_fixed_latency", params={"lanes": "W"}),
        "builds latency 3",
    ),
]


@pytest.mark.parametrize(("case", "edit", "msg"), MODEL_CASES, ids=[c[0] for c in MODEL_CASES])
def test_mismatch_with_the_model(case, edit, msg):
    d = adder()
    edit(d)
    with pytest.raises(BrmError, match=msg):
        _brm(d).resolve(IMPL)


def test_rate_mismatch():
    d = reducer()
    d["interface"]["params"]["K"] = d["interface"]["params"].pop("T")
    d["interface"]["ports"][1]["rate"] = "K"
    with pytest.raises(BrmError, match="differ from kind 'reduce'"):
        Brm.from_dict(d).resolve("chisel_accumulator", {"W": 4})
