"""BRM instances, their function and their block."""

from __future__ import annotations

import pytest

from snax_forge.brm import Brm, BrmError, function_of, load_brm
from snax_forge.brm.function import elementwise, reduce_kind

from .helpers import IMPL, adder, reducer


def _brm(d=None) -> Brm:
    return Brm.from_dict(d or adder())


# =============================================================================
# Resolving and the function
# =============================================================================


def test_the_adder_resolves_with_its_defaults():
    inst = _brm().resolve(IMPL)
    assert inst.params == {"W": 4, "op": "add"}  # defaults, in declaration order
    assert (inst.latency, inst.initiation_interval, inst.drain) == (0, 1, 0)


def test_the_function_follows_the_design_params():
    inst = _brm().resolve(IMPL, {"W": 8})
    assert [p.lanes for p in function_of(inst).ports] == [8, 8, 8]
    assert inst.lanes("out") == 8


def test_the_function_is_the_elementwise_kind():
    f = function_of(_brm().resolve(IMPL))
    want = elementwise(lanes=4, n_inputs=2)
    assert (f.ports, f.kind) == (want.ports, want.kind)


def test_the_reducer_is_the_reduce_kind():
    inst = Brm.from_dict(reducer()).resolve("chisel_accumulator", {"W": 4})
    assert function_of(inst).ports == reduce_kind(lanes=4, lanes_out=1).ports


def test_timing_expressions_are_resolved():
    d = adder()
    d["implementations"][IMPL]["timing"] = {"latency": "W // 2", "initiation_interval": "1"}
    inst = _brm(d).resolve(IMPL, {"W": 8})
    assert (inst.latency, inst.initiation_interval) == (4, 1)


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
# The function against its kind
# =============================================================================


def _rename_ports(d):
    for p, new in zip(d["interface"]["ports"], ["x", "y", "out"], strict=True):
        p["name"] = new
    d["dataflow"]["ports"] = {k: list(range(4)) for k in ("x", "y", "out")}
    d["function"]["code"] = "out = x + y"


KIND_CASES = [
    ("kind", lambda d: d["function"].update(accel="gemm"), "not registered"),
    ("factory", lambda d: d["function"]["params"].update(width="W"), "rejects"),
    ("names", _rename_ports, "differ from kind"),
    ("lanes", lambda d: d["function"]["params"].update(lanes="W*2"), "differ from kind"),
    ("inputs", lambda d: d["function"]["params"].update(n_inputs=3), "differ from kind"),
]


@pytest.mark.parametrize(("case", "edit", "msg"), KIND_CASES, ids=[c[0] for c in KIND_CASES])
def test_a_function_its_kind_does_not_build(case, edit, msg):
    d = adder()
    edit(d)
    inst = _brm(d).resolve(IMPL)
    with pytest.raises(BrmError, match=msg):
        function_of(inst)


def test_rate_mismatch():
    d = reducer()
    d["interface"]["params"]["K"] = d["interface"]["params"].pop("T")
    d["interface"]["ports"][1]["rate"] = "K"
    inst = Brm.from_dict(d).resolve("chisel_accumulator", {"W": 4})
    with pytest.raises(BrmError, match="differ from kind 'reduce'"):
        function_of(inst)


# =============================================================================
# The datapath a binding names
# =============================================================================


def _bound(binding_edit=None, timing=None):
    d = adder()
    d["implementations"][IMPL]["binding"] = load_brm("elementwise_add").to_dict()[
        "implementations"
    ][IMPL]["binding"]
    if binding_edit:
        binding_edit(d["implementations"][IMPL]["binding"])
    if timing:
        d["implementations"][IMPL]["timing"] = timing
    return _brm(d)


def test_a_bound_implementation_resolves_when_its_datapath_takes_it():
    assert _bound().resolve(IMPL).impl.binding["datapath"] == "ElementwiseLanes"


def test_a_datapath_the_generator_does_not_build_is_refused():
    brm = _bound(lambda b: b.update(datapath="Gemm"))
    with pytest.raises(BrmError, match="datapath 'Gemm' is not one the generator builds"):
        brm.resolve(IMPL)


def test_a_datapath_that_refuses_the_instance_is_refused():
    brm = _bound(lambda b: b.update(args={"lanes": "W * 2", "op": "op"}))
    with pytest.raises(BrmError, match=r"refuses the instance: \[gen.lanes\]"):
        brm.resolve(IMPL)
