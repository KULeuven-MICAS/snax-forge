"""BRM instances, their function and their block."""

from __future__ import annotations

import pytest

from snax_forge.brm import Brm, BrmError, function_of

from .helpers import IMPL, adder


def _brm(d=None) -> Brm:
    return Brm.from_dict(d or adder())


# =============================================================================
# Resolving and the function
# =============================================================================


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


# =============================================================================
# The datapath a binding names
# =============================================================================


def _bound(binding_edit=None, timing=None):
    d = adder()
    d["implementations"][IMPL]["binding"] = {  # simd's, at two streams and no slot
        "datapath": "SimdLanes", "args": {"lanes": "W", "op": "op"}, "controller": "counted"}  # fmt: skip
    if binding_edit:
        binding_edit(d["implementations"][IMPL]["binding"])
    if timing:
        d["implementations"][IMPL]["timing"] = timing
    return _brm(d)
