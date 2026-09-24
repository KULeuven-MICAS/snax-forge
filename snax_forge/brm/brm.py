"""The block runtime model (BRM1, D3, D68): one accelerator as plain data.

A BRM is a hand-written JSON file. It has a shared part, which every
implementation of the accelerator has in common, and a map of
implementations, which differ in how they are built:

    {
     "name": ...,
     "interface":       params (design / runtime) and ports
     "function":        a registered accelerator kind and its params (D43)
     "dataflow":        a notation and one nest per port (notation.py)
     "pattern":         the DFG subgraph it can replace; descriptive until DFG2
     "implementations": {name: {source, supports, timing, binding}}
    }

Designs that differ only in timing, the param values they support or their
RTL source are implementations of one BRM. Different ports, rates or data
order make a different BRM. Only ``source: chisel`` is accepted for now
(open item 28), and ``binding`` may be null until M10.

Which part serves which component: interface, function and the chosen
implementation's timing give the accelerator entry of the cluster file
(SNAX-MODEL, through LOW1c); the dataflow gives the streamer values
(SNAX-LOWER, through LOW1a); the pattern serves SNAX-DFG (DFG2) and the
binding the HW generator (GEN1).

Params have a stage. ``design`` params (lanes, op) are fixed per instance
and end up in the cluster file. ``runtime`` params are the accelerator's
start parameters: exactly ``n`` and every named port rate (D25, CONTRACTS.md
section 4). The register map is derived from them, and a port's interconnect
ports from its lanes (D12, D36); neither is written in the BRM.

Value fields (lanes, rate, timing, function params) hold an int or an
expression over params (expr.py). Lanes, timing and function params may use
design params only; a rate is an int or the name of a runtime param.

Every field is written by ``to_dict``; a missing required part is an error
that names it, optional fields take their defaults, and unknown keys are
errors (D26, D41). A BRM is validated when it is made, whether from a file
or in Python. Nothing here reads the model; the link to the accelerator
entry comes with BRM1's second patch.
"""

from __future__ import annotations

import json
import keyword
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from snax_forge.snax_model.config import check_keys, plain, to_json

from . import expr
from .expr import Value
from .notation import NOTATIONS

STAGES = ("design", "runtime")
TYPES = {"int": int, "str": str}
DIRECTIONS = ("in", "out")
SOURCES = ("chisel",)  # SystemVerilog and HLS: open item 28
TIMING_IN_FUNCTION = ("latency", "ii")  # come from the implementation's timing


class BrmError(ValueError):
    """A BRM that is malformed or inconsistent."""


def _keys(
    d: Any, required: tuple[str, ...], optional: tuple[str, ...], what: str, noun: str = "key"
) -> None:
    """``d`` is an object with every required key and no unknown one."""
    if not isinstance(d, Mapping):
        raise BrmError(f"{what}: must be an object, got {d!r}")
    try:
        check_keys(d, [*required, *optional], what)
    except ValueError as e:
        raise BrmError(str(e)) from None
    missing = [k for k in required if k not in d]
    if missing:
        raise BrmError(f"{what}: missing {noun} {missing[0]!r}")


def _ident(name: Any, what: str, in_expr: bool = False) -> None:
    """A valid name; one used in expressions (a param) may not be a Python keyword."""
    bad = not isinstance(name, str) or not name.isidentifier()
    if bad or (in_expr and keyword.iskeyword(name)):
        raise BrmError(f"{what}: {name!r} is not a valid name")


# =============================================================================
# Parts
# =============================================================================


@dataclass
class Param:
    """One BRM param. ``values`` restricts a design param; runtime params are ints."""

    stage: str
    type: str = "int"
    default: int | str | None = None
    values: list[int | str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "type": self.type,
            "default": self.default,
            "values": plain(self.values),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Param:
        _keys(d, ("stage",), ("type", "default", "values"), what)
        values = d.get("values")
        return cls(
            stage=d["stage"],
            type=d.get("type", "int"),
            default=d.get("default"),
            values=list(values) if isinstance(values, (list, tuple)) else values,
        )


@dataclass
class Port:
    """One data port: ``lanes`` elements per beat, one beat every ``rate`` firings (D25)."""

    name: str
    direction: str
    lanes: Value
    rate: Value = 1
    dtype: str = "int64"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction,
            "lanes": self.lanes,
            "rate": self.rate,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Port:
        _keys(d, ("name", "direction", "lanes"), ("rate", "dtype"), what)
        return cls(d["name"], d["direction"], d["lanes"], d.get("rate", 1), d.get("dtype", "int64"))


@dataclass
class Interface:
    """Params and ports. The register map and interconnect ports are derived."""

    params: dict[str, Param]
    ports: list[Port]

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": {k: p.to_dict() for k, p in self.params.items()},
            "ports": [p.to_dict() for p in self.ports],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Interface:
        _keys(d, ("params", "ports"), (), what)
        if not isinstance(d["params"], Mapping):
            raise BrmError(f"{what}.params: must be an object")
        if not isinstance(d["ports"], (list, tuple)):
            raise BrmError(f"{what}.ports: must be a list")
        params = {k: Param.from_dict(v, f"{what}.params.{k}") for k, v in d["params"].items()}
        ports = [Port.from_dict(p, f"{what}.ports[{i}]") for i, p in enumerate(d["ports"])]
        return cls(params, ports)


@dataclass
class Function:
    """A registered accelerator kind (D43) and its factory params, without timing."""

    accel: str
    params: dict[str, Value] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"accel": self.accel, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Function:
        _keys(d, ("accel",), ("params",), what)
        return cls(d["accel"], dict(d.get("params", {})))


@dataclass
class Dataflow:
    """A notation (notation.py) and one nest per port, in logical indices."""

    notation: str
    ports: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"notation": self.notation, "ports": plain(self.ports)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Dataflow:
        _keys(d, ("notation", "ports"), (), what)
        if not isinstance(d["ports"], Mapping):
            raise BrmError(f"{what}.ports: must be an object")
        return cls(d["notation"], dict(d["ports"]))


@dataclass
class Pattern:
    """The DFG subgraph the BRM can replace. Descriptive only; the predicate comes in DFG2."""

    family: str
    attrs: dict[str, Any] = field(default_factory=dict)
    predicate: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "attrs": plain(self.attrs), "predicate": self.predicate}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Pattern:
        _keys(d, ("family",), ("attrs", "predicate"), what)
        return cls(d["family"], dict(d.get("attrs", {})), d.get("predicate"))


@dataclass
class Timing:
    """``latency`` (L) and ``initiation_interval`` (II, written ``ii`` in the cluster file)."""

    latency: Value
    initiation_interval: Value

    def to_dict(self) -> dict[str, Any]:
        return {"latency": self.latency, "initiation_interval": self.initiation_interval}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Timing:
        _keys(d, ("latency", "initiation_interval"), (), what)
        return cls(d["latency"], d["initiation_interval"])


@dataclass
class Implementation:
    """One way to build the accelerator: its source, supported values, timing and binding."""

    source: str
    timing: Timing
    supports: dict[str, list[int | str]] = field(default_factory=dict)
    binding: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "supports": plain(self.supports),
            "timing": self.timing.to_dict(),
            "binding": plain(self.binding),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Implementation:
        _keys(d, ("source", "timing"), ("supports", "binding"), what, "part")
        supports = d.get("supports", {})
        if not isinstance(supports, Mapping):
            raise BrmError(f"{what}.supports: must be an object")
        binding = d.get("binding")
        return cls(
            source=d["source"],
            timing=Timing.from_dict(d["timing"], f"{what}.timing"),
            supports={
                k: list(v) if isinstance(v, (list, tuple)) else v for k, v in supports.items()
            },
            binding=dict(binding) if isinstance(binding, Mapping) else binding,
        )


# =============================================================================
# The BRM
# =============================================================================

_PARTS = ("name", "interface", "function", "dataflow", "pattern", "implementations")


@dataclass
class Brm:
    """One accelerator: the shared part and its implementations. Validated when made."""

    name: str
    interface: Interface
    function: Function
    dataflow: Dataflow
    pattern: Pattern
    implementations: dict[str, Implementation]

    def __post_init__(self) -> None:
        _validate(self)

    # -- derived ------------------------------------------------------------

    def params(self, stage: str) -> dict[str, Param]:
        """The params of one stage, in declaration order."""
        return {k: p for k, p in self.interface.params.items() if p.stage == stage}

    @property
    def registers(self) -> list[str]:
        """The start parameters, in register order: ``n``, then named rates in port order."""
        named = [p.rate for p in self.interface.ports if isinstance(p.rate, str)]
        return ["n", *dict.fromkeys(named)]

    def port(self, name: str) -> Port:
        for p in self.interface.ports:
            if p.name == name:
                return p
        raise KeyError(f"brm {self.name!r}: no port {name!r}")

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "interface": self.interface.to_dict(),
            "function": self.function.to_dict(),
            "dataflow": self.dataflow.to_dict(),
            "pattern": self.pattern.to_dict(),
            "implementations": {k: i.to_dict() for k, i in self.implementations.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Brm:
        what = f"brm {d.get('name', '?')!r}" if isinstance(d, Mapping) else "brm"
        _keys(d, _PARTS, (), what, "part")
        impls = d["implementations"]
        if not isinstance(impls, Mapping):
            raise BrmError(f"{what}.implementations: must be an object")
        return cls(
            name=d["name"],
            interface=Interface.from_dict(d["interface"], f"{what}.interface"),
            function=Function.from_dict(d["function"], f"{what}.function"),
            dataflow=Dataflow.from_dict(d["dataflow"], f"{what}.dataflow"),
            pattern=Pattern.from_dict(d["pattern"], f"{what}.pattern"),
            implementations={
                k: Implementation.from_dict(v, f"{what}.implementation {k!r}")
                for k, v in impls.items()
            },
        )

    def to_json(self) -> str:
        return to_json(self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> Brm:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())


# =============================================================================
# Validation
# =============================================================================


def _validate(b: Brm) -> None:
    what = f"brm {b.name!r}"
    _ident(b.name, what)
    design = _check_params(b, what)
    _check_ports(b, design, what)
    _check_function(b, design, what)
    _check_pattern(b, what)
    _check_implementations(b, design, what)
    _check_dataflow(b, what)  # last: a notation may rely on everything above


def _uses(v: Value, allowed: dict[str, Param], what: str) -> None:
    """``v`` is a valid expression over ``allowed`` params only."""
    try:
        expr.check(v, what)
    except expr.ExprError as e:
        raise BrmError(str(e)) from None
    unknown = sorted(expr.names(v) - set(allowed))
    if unknown:
        raise BrmError(f"{what}: {v!r} uses {unknown}, not design params ({sorted(allowed)})")


def _check_params(b: Brm, what: str) -> dict[str, Param]:
    for name, p in b.interface.params.items():
        w = f"{what}.params.{name}"
        _ident(name, w, in_expr=True)
        if p.stage not in STAGES:
            raise BrmError(f"{w}: stage must be one of {list(STAGES)}, got {p.stage!r}")
        if p.type not in TYPES:
            raise BrmError(f"{w}: type must be one of {list(TYPES)}, got {p.type!r}")
        if p.stage == "runtime":
            if p.type != "int" or p.default is not None or p.values is not None:
                raise BrmError(f"{w}: a runtime param is an int with no default or values")
            continue
        t = TYPES[p.type]
        if p.values is not None:
            if not isinstance(p.values, list) or not p.values:
                raise BrmError(f"{w}: values must be a non-empty list")
            if any(type(v) is not t for v in p.values):
                raise BrmError(f"{w}: every value must be of type {p.type}")
        if p.default is not None:
            if type(p.default) is not t:
                raise BrmError(f"{w}: default {p.default!r} is not of type {p.type}")
            if p.values is not None and p.default not in p.values:
                raise BrmError(f"{w}: default {p.default!r} is not in values {p.values}")
    return b.params("design")


def _check_ports(b: Brm, design: dict[str, Param], what: str) -> None:
    ports = b.interface.ports
    names = [p.name for p in ports]
    if len(set(names)) != len(names):
        raise BrmError(f"{what}: duplicate port names {names}")
    for p in ports:
        w = f"{what}.port {p.name!r}"
        _ident(p.name, w)
        if p.direction not in DIRECTIONS:
            raise BrmError(f"{w}: direction must be 'in' or 'out', got {p.direction!r}")
        _uses(p.lanes, design, f"{w}.lanes")
        if (c := expr.constant(p.lanes)) is not None and c < 1:
            raise BrmError(f"{w}.lanes: must be >= 1, got {c}")
        runtime = b.params("runtime")
        if isinstance(p.rate, str):
            if not expr.is_name(p.rate) or p.rate not in runtime:
                raise BrmError(f"{w}.rate: {p.rate!r} must be an int or a runtime param name")
        elif type(p.rate) is not int or p.rate < 1:
            raise BrmError(f"{w}.rate: must be an int >= 1 or a runtime param name")
        try:
            np.dtype(p.dtype)
        except TypeError:
            raise BrmError(f"{w}.dtype: {p.dtype!r} is not a dtype") from None
    dirs = {p.direction for p in ports}
    if dirs != set(DIRECTIONS):
        raise BrmError(f"{what}: needs at least one input and one output port")
    runtime = list(b.params("runtime"))
    if sorted(runtime) != sorted(b.registers):
        raise BrmError(
            f"{what}: runtime params must be exactly n and the named rates {b.registers}, "
            f"got {runtime}"
        )


def _check_function(b: Brm, design: dict[str, Param], what: str) -> None:
    f, w = b.function, f"{what}.function"
    if not isinstance(f.accel, str) or not f.accel:
        raise BrmError(f"{w}.accel: must name a registered accelerator kind")
    for k, v in f.params.items():
        if k in TIMING_IN_FUNCTION:
            raise BrmError(f"{w}.params.{k}: timing comes from the implementation, not here")
        _uses(v, design, f"{w}.params.{k}")


def _check_pattern(b: Brm, what: str) -> None:
    p, w = b.pattern, f"{what}.pattern"
    if not isinstance(p.family, str) or not p.family:
        raise BrmError(f"{w}.family: must be a non-empty string")
    if not isinstance(p.attrs, dict):
        raise BrmError(f"{w}.attrs: must be an object")
    if p.predicate is not None and not isinstance(p.predicate, str):
        raise BrmError(f"{w}.predicate: must be null or a registered name")


def _check_implementations(b: Brm, design: dict[str, Param], what: str) -> None:
    if not b.implementations:
        raise BrmError(f"{what}: needs at least one implementation")
    for name, impl in b.implementations.items():
        w = f"{what}.implementation {name!r}"
        _ident(name, w)
        if impl.source not in SOURCES:
            raise BrmError(f"{w}.source: must be one of {list(SOURCES)}, got {impl.source!r}")
        for k, vals in impl.supports.items():
            if k not in design:
                raise BrmError(f"{w}.supports: {k!r} is not a design param")
            p = design[k]
            if not isinstance(vals, list) or not vals:
                raise BrmError(f"{w}.supports.{k}: must be a non-empty list")
            if any(type(v) is not TYPES[p.type] for v in vals):
                raise BrmError(f"{w}.supports.{k}: every value must be of type {p.type}")
            if p.values is not None and not set(vals) <= set(p.values):
                raise BrmError(f"{w}.supports.{k}: {vals} not within the param's values {p.values}")
        t = impl.timing
        _uses(t.latency, design, f"{w}.timing.latency")
        _uses(t.initiation_interval, design, f"{w}.timing.initiation_interval")
        if (c := expr.constant(t.latency)) is not None and c < 0:
            raise BrmError(f"{w}.timing.latency: must be >= 0, got {c}")
        if (c := expr.constant(t.initiation_interval)) is not None and c < 1:
            raise BrmError(f"{w}.timing.initiation_interval: must be >= 1, got {c}")
        if impl.binding is not None and not isinstance(impl.binding, dict):
            raise BrmError(f"{w}.binding: must be null or an object")


def _check_dataflow(b: Brm, what: str) -> None:
    d, w = b.dataflow, f"{what}.dataflow"
    if d.notation not in NOTATIONS:
        raise BrmError(f"{w}.notation: {d.notation!r} not registered ({sorted(NOTATIONS)})")
    ports = [p.name for p in b.interface.ports]
    if set(d.ports) != set(ports):
        raise BrmError(f"{w}.ports: need one nest per port {ports}, got {list(d.ports)}")
    for p in b.interface.ports:
        try:
            NOTATIONS[d.notation](d.ports[p.name], p, b)
        except ValueError as e:
            raise BrmError(f"{w}.ports.{p.name}: {e}") from None
