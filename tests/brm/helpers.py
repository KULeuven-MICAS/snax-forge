"""Shared helpers for the SNAX-BRM tests (BRM1).

BRMs in the tests are written as plain dicts, the way a person or an LLM
writes the JSON file. Until BRM2 registers ``affine``, the tests use a
notation of their own, ``test_list``: a nest is a list of logical indices,
and a port's list must hold a multiple of its lanes when they are constant.
"""

from __future__ import annotations

import copy
from typing import Any

from snax_forge.brm import NOTATIONS, expr, register_notation


def _test_list(nest: Any, port: Any, brm: Any) -> None:
    if not isinstance(nest, list) or not all(type(i) is int for i in nest):
        raise ValueError("a test_list nest is a list of ints")
    lanes = expr.constant(port.lanes)
    if lanes is not None and len(nest) % lanes:
        raise ValueError(f"{len(nest)} indices is not a multiple of {lanes} lanes")


if "test_list" not in NOTATIONS:
    register_notation("test_list", _test_list)


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
                    "lanes_out": {"stage": "design", "values": [1]},
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
