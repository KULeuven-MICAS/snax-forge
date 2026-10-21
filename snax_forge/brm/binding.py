"""An implementation's binding: its Chisel block and how the BRM fills it (D129).

    {"block":   "ElementwiseTiledSpatial",            the Chisel block's name
     "args":    {"lanes": "W", "supportedOps": "op"}, its parameters, each an
                                                      expression over design params
     "control": "counted",                            who counts the firings
     "csrs":    [{"field": "loopCount", "value": "n"}],
                                                      the configuration fields in
                                                      CSR order (start follows), each
                                                      value an expression over params;
                                                      several for a block with more
                                                      than a count (the Accumulator's
                                                      loopCount then len)
     "wait":    "lockstep"}                           when the block takes a beat

``control`` and ``wait`` are registered kinds (D19): ``register_control``
and ``register_wait``. ``counted`` is the only control: every block counts
its own firings (D129). Which blocks exist is the generators' business
(snax_forge/gen registers the GVSoC adapters; the Chisel side is not
generated yet), so ``block`` is only a name here. Every expression is
checked and stored canonical (snax_forge/expr.py). A binding may be null:
the implementation is not generated yet.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from snax_forge import expr


class BindingError(ValueError):
    """A binding that is malformed; brm.py turns it into a BrmError."""


KEYS = ("block", "args", "control", "csrs", "wait")
CSR_KEYS = ("field", "value")

# name -> what it means; extended by registration (D19)
CONTROLS: dict[str, str] = {}
WAITS: dict[str, str] = {}


def register_control(name: str, meaning: str) -> None:
    """Add a ``control`` kind."""
    if name in CONTROLS:
        raise ValueError(f"control {name!r} already registered")
    CONTROLS[name] = meaning


def register_wait(name: str, meaning: str) -> None:
    """Add a ``wait`` kind."""
    if name in WAITS:
        raise ValueError(f"wait {name!r} already registered")
    WAITS[name] = meaning


register_control(
    "counted", "the block counts its firings; the start's configuration holds the count"
)
register_wait(
    "lockstep", "a firing takes a beat on every input and room on every output in one cycle"
)
register_wait("pipelined", "an L-stage pipeline or output register, one firing per II (D129)")


def check_binding(
    binding: Any, design: Mapping[str, Any], params: Mapping[str, Any], what: str
) -> dict[str, Any]:
    """The binding with its expressions canonical; ``BindingError`` naming what is wrong.

    ``design`` are the BRM's design params (``args`` may use them), ``params``
    all of its params (``csrs`` values may use them; ``n`` is a runtime one).
    """
    if not isinstance(binding, Mapping):
        raise BindingError(f"{what}: must be null or an object")
    unknown = sorted(set(binding) - set(KEYS))
    if unknown:
        raise BindingError(f"{what}: unknown keys {unknown} (allowed: {list(KEYS)})")
    missing = [k for k in KEYS if k not in binding]
    if missing:
        raise BindingError(f"{what}: missing key {missing[0]!r}")
    block = binding["block"]
    if not isinstance(block, str) or not block.isidentifier():
        raise BindingError(f"{what}.block: must name a Chisel block, got {block!r}")
    for k, kinds in (("control", CONTROLS), ("wait", WAITS)):
        if binding[k] not in kinds:
            raise BindingError(f"{what}.{k}: {binding[k]!r} not registered ({sorted(kinds)})")

    args = binding["args"]
    if not isinstance(args, Mapping):
        raise BindingError(f"{what}.args: must be an object")
    out_args = {}
    for k, v in args.items():
        if not isinstance(k, str) or not k.isidentifier():
            raise BindingError(f"{what}.args: {k!r} is not a parameter name")
        out_args[k] = _value(v, design, f"{what}.args.{k}", "design params")

    csrs = binding["csrs"]
    if not isinstance(csrs, list):
        raise BindingError(f"{what}.csrs: must be a list of {{field, value}}")
    out_csrs = []
    for i, c in enumerate(csrs):
        w = f"{what}.csrs[{i}]"
        if not isinstance(c, Mapping) or set(c) != set(CSR_KEYS):
            raise BindingError(f"{w}: must be {{field, value}}, got {c!r}")
        if not isinstance(c["field"], str) or not c["field"].isidentifier():
            raise BindingError(f"{w}.field: {c['field']!r} is not a field name")
        out_csrs.append({"field": c["field"], "value": _value(c["value"], params, w, "params")})
    fields = [c["field"] for c in out_csrs]
    if len(set(fields)) != len(fields):
        raise BindingError(f"{what}.csrs: duplicate fields {fields}")
    return {"block": block, "args": out_args, "control": binding["control"], "csrs": out_csrs,
            "wait": binding["wait"]}  # fmt: skip


def _value(v: Any, allowed: Mapping[str, Any], what: str, noun: str) -> expr.Value:
    expr.check(v, what)
    unknown = sorted(expr.names(v) - set(allowed))
    if unknown:
        raise BindingError(f"{what}: {v!r} uses {unknown}, not {noun} ({sorted(allowed)})")
    return expr.canonical(v)
