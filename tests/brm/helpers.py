"""Shared helpers for the SNAX-BRM tests (BRM1).

BRMs in the tests are written as plain dicts, the way a person or an LLM
writes the JSON file. Until BRM2 registers ``affine``, the tests use a
notation of their own, ``test_list``: a nest is a list of logical indices,
and a port's list must hold a multiple of its lanes when they are constant.

``affine_adder`` and ``affine_reducer`` are the same BRMs with ``affine``
nests (D70): a vector walked W elements per beat.

``test_fixed_latency`` is an accelerator kind registered here: the
elementwise stub with latency 3 whatever it is given, so a BRM whose
implementation declares another latency is caught.
"""

from __future__ import annotations

import copy
from typing import Any

from snax_forge import expr
from snax_forge.brm import NOTATIONS, register_notation
from snax_forge.snax_model.accel import elementwise_stub
from snax_forge.snax_model.scenario import ACCEL_KINDS, register_accel


def _test_list(nest: Any, port: Any, brm: Any) -> None:
    if not isinstance(nest, list) or not all(type(i) is int for i in nest):
        raise ValueError("a test_list nest is a list of ints")
    lanes = expr.constant(port.lanes)
    if lanes is not None and len(nest) % lanes:
        raise ValueError(f"{len(nest)} indices is not a multiple of {lanes} lanes")


if "test_list" not in NOTATIONS:
    register_notation("test_list", _test_list)


def _fixed_latency(lanes: int = 1, latency: int = 0, ii: int = 1, **_: object):
    return elementwise_stub(lanes=lanes, latency=3, ii=ii)


if "test_fixed_latency" not in ACCEL_KINDS:
    register_accel("test_fixed_latency", _fixed_latency)


def adder() -> dict[str, Any]:
    """A two-input adder shaped like the elementwise stub, every field written."""
    nest = list(range(8))
    return copy.deepcopy(
        {
            "name": "test_add",
            "interface": {
                "params": {
                    "W": {"stage": "design", "type": "int", "default": 4, "values": None},
                    "op": {"stage": "design", "type": "str", "default": "add", "values": ["add"]},
                    "n": {"stage": "runtime", "type": "int", "default": None, "values": None},
                },
                "ports": [
                    {"name": "a", "direction": "in", "lanes": "W", "rate": 1, "dtype": "int64"},
                    {"name": "b", "direction": "in", "lanes": "W", "rate": 1, "dtype": "int64"},
                    {"name": "out", "direction": "out", "lanes": "W", "rate": 1, "dtype": "int64"},
                ],
            },
            "function": {
                "accel": "elementwise",
                "params": {"lanes": "W", "n_inputs": 2, "op": "op"},
            },
            "dataflow": {"notation": "test_list", "ports": {"a": nest, "b": nest, "out": nest}},
            "pattern": {
                "family": "elementwise",
                "attrs": {"op": "add", "arity": 2},
                "predicate": None,
            },
            "implementations": {
                "chisel_tiled_spatial": {
                    "source": "chisel",
                    "supports": {"W": [1, 2, 4, 8]},
                    "timing": {"latency": 0, "initiation_interval": 1},
                    "binding": None,
                }
            },
        }
    )


def reducer() -> dict[str, Any]:
    """A reduce shaped like the reduce stub: output rate is the runtime param T (D25)."""
    return copy.deepcopy(
        {
            "name": "test_reduce",
            "interface": {
                "params": {
                    "W": {"stage": "design"},
                    "lanes_out": {"stage": "design", "default": 1, "values": [1]},
                    "n": {"stage": "runtime"},
                    "T": {"stage": "runtime"},
                },
                "ports": [
                    {"name": "in", "direction": "in", "lanes": "W"},
                    {"name": "out", "direction": "out", "lanes": "lanes_out", "rate": "T"},
                ],
            },
            "function": {"accel": "reduce", "params": {"lanes": "W", "lanes_out": "lanes_out"}},
            "dataflow": {"notation": "test_list", "ports": {"in": [0, 1], "out": [0]}},
            "pattern": {"family": "reduce"},
            "implementations": {
                "chisel_accumulator": {
                    "source": "chisel",
                    "timing": {"latency": 1, "initiation_interval": 1},
                }
            },
        }
    )


def vector_nest(length: str, lanes: str) -> dict[str, Any]:
    """``length`` elements, ``lanes`` per beat, in order: element t*lanes + j."""
    return {
        "shape": [f"{length} * {lanes}"],
        "loops": [
            {"bound": length, "strides": [lanes]},
            {"bound": lanes, "strides": [1], "spatial": True},
        ],
    }


def affine_adder() -> dict[str, Any]:
    d = adder()
    nest = vector_nest("n", "W")
    d["dataflow"] = {"notation": "affine", "ports": {"a": nest, "b": nest, "out": nest}}
    return copy.deepcopy(d)


def affine_reducer() -> dict[str, Any]:
    d = reducer()
    d["dataflow"] = {
        "notation": "affine",
        "ports": {"in": vector_nest("n", "W"), "out": vector_nest("n // T", "lanes_out")},
    }
    return copy.deepcopy(d)
