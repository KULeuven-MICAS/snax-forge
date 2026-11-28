"""The brm package's refusals, one table: each case builds its input and makes the call that
is refused, with the error and the message it must give."""

from __future__ import annotations

import json

import pytest

from snax_forge.brm import (
    Brm,
    BrmError,
    load_brm,
    task_nest,
)
from tests.brm.helpers import IMPL, adder, affine_adder, reducer
from tests.brm.test_affine import _nest
from tests.brm.test_instance import _bound, _brm


def _spatial_lanes_checked_at_resolve():
    d = affine_adder()
    _nest(d, "out")["loops"][1]["bound"] = "W // 2"
    b = Brm.from_dict(d)
    b.resolve(IMPL, {"W": 4})


def _indices_must_stay_in_the_shape():
    d = affine_adder()
    _nest(d)["offset"] = [1]
    task_nest(Brm.from_dict(d).resolve(IMPL), "a", {"n": 16})


def _no_implementation_is_rejected():
    d = adder()
    d["implementations"] = {}
    Brm.from_dict(d)


def _the_controller_is_a_registered_kind():
    from snax_forge.brm import CONTROLLERS, register_controller

    assert list(CONTROLLERS) == ["counted"]
    register_controller("counted", "again")


def _missing_value_without_default():
    Brm.from_dict(reducer()).resolve("chisel_adder_tree")


def _resolved_timing_out_of_range():
    d = adder()
    d["implementations"][IMPL]["timing"]["latency"] = "W - 8"
    _brm(d).resolve(IMPL)


def _a_datapath_the_generator_does_not_build_is_refused():
    brm = _bound(lambda b: b.update(datapath="Gemm"))
    brm.resolve(IMPL)


def _a_datapath_that_refuses_the_instance_is_refused():
    brm = _bound(lambda b: b.update(args={"lanes": "W * 2", "op": "op"}))
    brm.resolve(IMPL)


def _two_entries_that_fit_the_same_instances_are_refused():
    d = json.loads(load_brm("simd").to_json())
    d["function"]["code"].append(
        {"params": {"op": "axpby", "streams": 2}, "code": "out = a * k1 + b"}
    )
    Brm.from_dict(d)


def _an_unknown_port_is_refused():
    from snax_forge.brm import load_brm

    load_brm("simd").port("zz")


CASES = [
    pytest.param(_spatial_lanes_checked_at_resolve, BrmError, 'give 2 lanes, the port has 4', id='spatial_lanes_checked_at_resolve'),
    pytest.param(_indices_must_stay_in_the_shape, ValueError, 'run over \\[1, 64\\], outside the shape \\[64\\]', id='indices_must_stay_in_the_shape'),
    pytest.param(_no_implementation_is_rejected, BrmError, 'at least one implementation', id='no_implementation_is_rejected'),
    pytest.param(_the_controller_is_a_registered_kind, ValueError, 'already registered', id='the_controller_is_a_registered_kind'),
    pytest.param(_missing_value_without_default, BrmError, "no value for 'W'", id='missing_value_without_default'),
    pytest.param(_resolved_timing_out_of_range, BrmError, "latency = 'W - 8' gives -4", id='resolved_timing_out_of_range'),
    pytest.param(_a_datapath_the_generator_does_not_build_is_refused, BrmError, "datapath 'Gemm' is not one the generator builds", id='a_datapath_the_generator_does_not_build_is_refused'),
    pytest.param(_a_datapath_that_refuses_the_instance_is_refused, BrmError, 'refuses the instance: \\[gen.lanes\\]', id='a_datapath_that_refuses_the_instance_is_refused'),
    pytest.param(_two_entries_that_fit_the_same_instances_are_refused, BrmError, 'code\\[22\\]: fits the instances code\\[11\\] fits', id='two_entries_that_fit_the_same_instances_are_refused'),
    pytest.param(_an_unknown_port_is_refused, KeyError, "no port 'zz'", id='an_unknown_port_is_refused'),
]  # fmt: skip


@pytest.mark.parametrize(("case", "error", "message"), CASES)
def test_refusal(case, error, message):
    with pytest.raises(error, match=message):
        case()
