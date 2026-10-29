"""An implementation's binding: its datapath and how the BRM fills it (D129).

    {"datapath":   "ElementwiseLanes",          a datapath of the catalogue
                                                (hw/chisel/blocks.json, D147)
     "args":       {"lanes": "W", "op": "op"},  its parameters, each an
                                                expression over design params,
                                                or a bool (a design-time variant:
                                                Mac's accumulate)
     "controller": "counted",                   the nest controller that drives
                                                it (D147)
     "ports":      {"a": "in", "out": "out"}}   optional: every BRM port -> the
                                                datapath's port of that role (by
                                                default the same name)

``controller`` is a registered kind (D19): ``register_controller``.
``counted`` is the only one: the shell's nest controller counts the firings,
JOBS outermost and each fold's beats inside (D146, D147). The CSRs are not
the binding's: they follow from the unit's group (snax_forge/gen/accelerator.py).
Which datapaths exist is the generators' business (snax_forge/gen/blocks.py
registers them), so ``datapath`` is only a name here. Every expression is
checked and stored canonical (snax_forge/expr.py). ``ports`` is written only
when given, in the BRM's port order; whether its names are the datapath's
ports is the generator's check. A binding may be null: the implementation is
not generated yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from snax_forge import expr
from snax_forge.common import check_keys


class BindingError(ValueError):
    """A binding that is malformed; brm.py turns it into a BrmError."""


KEYS = ("datapath", "args", "controller")
OPTIONAL = ("ports",)

# name -> what it means; extended by registration (D19)
CONTROLLERS: dict[str, str] = {}


def register_controller(name: str, meaning: str) -> None:
    """Add a ``controller`` kind."""
    if name in CONTROLLERS:
        raise ValueError(f"controller {name!r} already registered")
    CONTROLLERS[name] = meaning


register_controller(
    "counted", "the shell's nest controller counts the firings: JOBS outermost, each fold's "
    "beats inside (D147)",
)  # fmt: skip


def check_binding(
    binding: Any, design: Mapping[str, Any], ports: Sequence[str], what: str
) -> dict[str, Any]:
    """The binding with its expressions canonical; ``BindingError`` naming what is wrong.

    ``design`` are the BRM's design params (``args`` may use them), ``ports`` the BRM's port
    names in order (``ports`` maps each of them).
    """
    if not isinstance(binding, Mapping):
        raise BindingError(f"{what}: must be null or an object")
    try:
        check_keys(binding, [*KEYS, *OPTIONAL], what, KEYS)
    except ValueError as e:
        raise BindingError(str(e)) from None
    datapath = binding["datapath"]
    if not isinstance(datapath, str) or not datapath.isidentifier():
        raise BindingError(f"{what}.datapath: must name a datapath, got {datapath!r}")
    if binding["controller"] not in CONTROLLERS:
        raise BindingError(
            f"{what}.controller: {binding['controller']!r} not registered ({sorted(CONTROLLERS)})"
        )

    args = binding["args"]
    if not isinstance(args, Mapping):
        raise BindingError(f"{what}.args: must be an object")
    out_args = {}
    for k, v in args.items():
        if not isinstance(k, str) or not k.isidentifier():
            raise BindingError(f"{what}.args: {k!r} is not a parameter name")
        out_args[k] = _value(v, design, f"{what}.args.{k}", "design params")

    out = {"datapath": datapath, "args": out_args, "controller": binding["controller"]}
    if "ports" in binding:
        out["ports"] = _ports(binding["ports"], ports, f"{what}.ports")
    return out


def _ports(m: Any, ports: Sequence[str], what: str) -> dict[str, str]:
    """``ports`` checked: every BRM port once, each to a distinct datapath port name; in the
    BRM's port order."""
    if not isinstance(m, Mapping):
        raise BindingError(f"{what}: must be an object, BRM port -> datapath port")
    if set(m) != set(ports):
        raise BindingError(f"{what}: maps {sorted(m)}; it must map every BRM port {list(ports)}")
    for k, v in m.items():
        if not isinstance(v, str) or not v.isidentifier():
            raise BindingError(f"{what}.{k}: {v!r} is not a datapath port name")
    if len(set(m.values())) != len(m):
        raise BindingError(f"{what}: two BRM ports on one datapath port ({dict(m)})")
    return {p: m[p] for p in ports}


def _value(v: Any, allowed: Mapping[str, Any], what: str, noun: str) -> expr.Value:
    if isinstance(v, bool):  # a design-time variant, kept as it is
        return v
    unknown = expr.outside(v, allowed, what)
    if unknown:
        raise BindingError(f"{what}: {v!r} uses {unknown}, not {noun} ({sorted(allowed)})")
    return expr.canonical(v)
