"""The block runtime model: one accelerator as plain data (CONTRACTS, "Block runtime model").

``Brm`` holds a hand-written JSON file: a shared part (``Interface`` with its ``Param``s and
``Port``s, ``Function`` with its ``Code``, ``Dataflow``, ``Pattern``) and a map of
``Implementation``s (source, supports, ``Timing``, binding). Designs that differ only in
timing, supported values or RTL source are implementations of one BRM; different ports, rates
or data order make another. A port's ``dtypes`` are a rule (dtypes.py), every integer dtype
unless the file narrows them, and ``to_dict`` writes them only then; its streamer width comes
from its lanes. Lanes, timing and function params use design params only; a rate is an int or a
runtime param's name. A BRM is validated when it is made, from a file or in Python; a missing
part or an unknown key is a ``BrmError`` naming it. ``Brm.resolve`` builds an instance
(instance.py).
"""

from __future__ import annotations

import keyword
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from snax_forge import expr
from snax_forge.common import JsonFile, check_keys, plain
from snax_forge.expr import Value

from .binding import check_binding
from .dtypes import INT_DTYPES
from .notation import NOTATIONS

if TYPE_CHECKING:
    from .instance import Instance

STAGES = ("design", "runtime")
TYPES = {"int": int, "str": str}
DIRECTIONS = ("in", "out")
SOURCES = ("chisel",)  # SystemVerilog and HLS: not yet
TIMING_IN_FUNCTION = ("latency", "ii")  # come from the implementation's timing


class BrmError(ValueError):
    """A BRM that is malformed or inconsistent."""


def _keys(
    d: Any, required: tuple[str, ...], optional: tuple[str, ...], what: str, noun: str = "key"
) -> None:
    """``d`` is an object with every required key and no unknown one."""
    try:
        check_keys(d, [*required, *optional], what, required, noun)
    except (TypeError, ValueError) as e:
        raise BrmError(str(e)) from None


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


# the optional input of a fold's seed (``accumulate``'s, at its design param ``seed`` 1): no
# tasklet connector maps onto it, ``bind`` puts a seeded fold's seed element there
SEED = "seed"


@dataclass
class Port:
    """One data port: ``lanes`` elements per beat, one beat every ``rate`` firings; present only
    at the design params ``when`` gives (every one holding; always when None)."""

    name: str
    direction: str
    lanes: Value
    rate: Value = 1
    dtypes: list[str] = field(default_factory=lambda: list(INT_DTYPES))  # the rule: dtypes.py
    when: dict[str, int | str] | None = None

    def present(self, params: Mapping[str, Any]) -> bool:
        """The port exists at design ``params``: each of its ``when`` holds there."""
        return all(k in params and params[k] == v for k, v in (self.when or {}).items())

    def to_dict(self) -> dict[str, Any]:
        d = {"name": self.name, "direction": self.direction, "lanes": self.lanes, "rate": self.rate}
        if self.when is not None:  # written only when set
            d["when"] = dict(self.when)
        if list(self.dtypes) != list(INT_DTYPES):  # every integer dtype unless the file says
            d["dtypes"] = list(self.dtypes)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Port:
        _keys(d, ("name", "direction", "lanes"), ("rate", "when", "dtypes"), what)
        dtypes = d.get("dtypes", INT_DTYPES)
        dtypes = list(dtypes) if isinstance(dtypes, (list, tuple)) else dtypes
        when = d.get("when")
        when = dict(when) if isinstance(when, Mapping) else when
        return cls(d["name"], d["direction"], d["lanes"], d.get("rate", 1), dtypes, when)


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
class Code:
    """One entry of a function's code table: ``code`` when the design params are ``params``
    (each a design param at one of its allowed values)."""

    params: dict[str, int | str]
    code: str

    def to_dict(self) -> dict[str, Any]:
        return {"params": dict(self.params), "code": self.code}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Code:
        _keys(d, ("params", "code"), (), what)
        if not isinstance(d["params"], Mapping):
            raise BrmError(f"{what}.params: must be an object")
        return cls(dict(d["params"]), d["code"])


@dataclass
class Function:
    """A registered accelerator kind, its factory params, and what it computes.

    ``code`` says what the accelerator computes, symbolically: one
    ``output = expression`` per output port over the input ports, per lane
    of one firing, in the expression grammar (``"out = a + b"``). It is what
    SNAX-SANDBOX's ``bind`` compares a tasklet against and what an
    accelerated node carries, so a bound graph says what each accelerator
    does without its BRM. The registered kind (function.py) is how the
    reference executor computes it; the two must agree (tests/brm). A
    reduction says what one element contributes (``accumulate``'s ``out = a``);
    that it folds, and with which op, is its pattern's and kind's. A
    BRM with a null ``code`` cannot be bound.

    ``code`` is one string, or a table of ``Code`` entries when one BRM computes several
    things by design params (``simd``'s ``op`` and ``streams``: ``out = a + b`` at add and two
    streams, ``out = max(a, k)`` at max and one). An entry's code uses the ports present at
    its params (``Port.when``) and may name a slot: an int design param that is no port and
    that the entry does not fix (``simd``'s ``k``, ``axpby``'s ``k1`` and ``k2``), a constant
    baked into the hardware, or, matched to a kernel's scalar argument, a value the program
    writes at run time (a CSR; the instance leaves the slot unset and lists it in ``read``).
    An entry fits an instance when the instance has every param the entry fixes, every slot
    its code names has a value (baked in the params, or read from a container) and every other
    slot is unset, so at most one entry fits (``Brm.entry_for``; two entries that would fit the
    same instances are refused at load): ``simd``'s axpby with ``k2`` unset is
    ``out = k1 * a + b``, with ``k2`` set ``out = k1 * a + k2 * b``. ``bind`` takes the first
    entry whose code is the tasklet's up to its slots and sets its params and slot values
    (sandbox/transforms.py, ``expr.unify``); ``Brm.code_for`` gives an instance's code back.
    """

    accel: str
    params: dict[str, Value] = field(default_factory=dict)
    code: str | list[Code] | None = None

    def entries(self) -> list[Code]:
        """The code table: one entry without params for a string, none for a null code."""
        if self.code is None:
            return []
        if isinstance(self.code, str):
            return [Code({}, self.code)]
        return list(self.code)

    def to_dict(self) -> dict[str, Any]:
        code = self.code if not isinstance(self.code, list) else [e.to_dict() for e in self.code]
        return {"accel": self.accel, "params": dict(self.params), "code": code}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Function:
        _keys(d, ("accel",), ("params", "code"), what)
        code = d.get("code")
        if isinstance(code, (list, tuple)):
            code = [Code.from_dict(e, f"{what}.code[{i}]") for i, e in enumerate(code)]
        return cls(d["accel"], dict(d.get("params", {})), code)


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
    """The DFG subgraph the BRM can replace: a family, matched by SNAX-SANDBOX.

    ``family`` names a matcher registered in snax_forge/sandbox/patterns.py
    and ``attrs`` are its parameters.
    """

    family: str
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "attrs": plain(self.attrs)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Pattern:
        _keys(d, ("family",), ("attrs",), what)
        return cls(d["family"], dict(d.get("attrs", {})))


@dataclass
class Timing:
    """``latency`` (L), ``initiation_interval`` (II, ``ii`` in accelerator.json), ``drain``
    and ``span``.

    L: cycles from a firing to its push. II: the least number of cycles
    between two firings. ``drain``: the cycles, starting with the
    cycle an output is pushed, in which no input is taken (1 for a block
    whose ``in.ready`` is low while its result waits; the generator takes 0
    only, as the nest controller has none). Optional, default 0;
    always written here. ``span``: the cycles the unit reads one input
    beat in place, taking it in the last, so its throughput is one beat per
    max(II, span) (``SerialAccumulate``'s lanes). Optional, default 1; always
    written here.
    """

    latency: Value
    initiation_interval: Value
    drain: Value = 0
    span: Value = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency": self.latency,
            "initiation_interval": self.initiation_interval,
            "drain": self.drain,
            "span": self.span,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Timing:
        _keys(d, ("latency", "initiation_interval"), ("drain", "span"), what)
        return cls(d["latency"], d["initiation_interval"], d.get("drain", 0), d.get("span", 1))


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
class Brm(JsonFile):
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

    def resolve(self, implementation: str, params: Mapping[str, Any] | None = None,
                read: Iterable[str] = ()) -> Instance:  # fmt: skip
        """The ``Instance`` built as ``implementation`` with design ``params`` and the slots
        ``read`` from a container (instance.py)."""
        from .instance import resolve

        return resolve(self, implementation, params, read)

    def present(self, params: Mapping[str, Any]) -> list[Port]:
        """The ports present at design ``params`` (``Port.when``), in order."""
        return [p for p in self.interface.ports if p.present(params)]

    def defaults(self) -> dict[str, int | str]:
        """The design params that have a default, at it."""
        return {k: p.default for k, p in self.params("design").items() if p.default is not None}

    def all_slots(self) -> list[str]:
        """Every code entry's slots, in table order."""
        return list(dict.fromkeys(s for e in self.function.entries() for s in self.slots(e)))

    def entry_for(self, params: Mapping[str, Any], read: Iterable[str] = ()) -> Code | None:
        """The code entry that fits an instance at design ``params`` with the slots ``read``
        from a container (``Function``): the entry's params all at ``params``, every slot its
        code names set or read, every other slot neither. None when none fits."""
        read = set(read)
        has = {s for s in self.all_slots() if params.get(s) is not None or s in read}
        for e in self.function.entries():
            if set(self.slots(e)) == has and all(params.get(k) == v for k, v in e.params.items()):
                return e
        return None

    def code_for(self, params: Mapping[str, Any], read: Iterable[str] = ()) -> str | None:
        """The code of the entry that fits (``entry_for``); None if none does."""
        e = self.entry_for(params, read)
        return None if e is None else e.code

    def slots(self, entry: Code) -> list[str]:
        """The slots ``entry``'s code names (``Function``), in the order they appear."""
        design = self.params("design")
        ports = {p.name for p in self.interface.ports}
        found = [n for _, e in expr.statements(entry.code, "code") for n in expr.ordered_names(e)]
        return list(dict.fromkeys(n for n in found if n in design and n not in ports
                                  and n not in entry.params))  # fmt: skip

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
        unknown = expr.outside(v, allowed, what)
    except expr.ExprError as e:
        raise BrmError(str(e)) from None
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


def _port_name(name: Any, what: str) -> None:
    """A port name: used in ``function.code`` and in the streamer name ``<instance>_<port>``.

    So not a Python keyword (``out = in`` does not parse) and no leading
    underscore (``acc__in``). The convention is ``a``, ``b``, ... for
    inputs and ``out`` for the output.
    """
    _ident(name, what)
    if keyword.iskeyword(name):
        raise BrmError(
            f"{what}: {name!r} is a Python keyword and cannot appear in code; "
            "name the port 'a', 'b', ... or 'out'"
        )
    if name.startswith("_"):
        raise BrmError(
            f"{what}: {name!r} starts with an underscore, which the streamer name "
            "<instance>_<port> would double; name the port 'a', 'b', ... or 'out'"
        )


def _check_ports(b: Brm, design: dict[str, Param], what: str) -> None:
    ports = b.interface.ports
    names = [p.name for p in ports]
    if len(set(names)) != len(names):
        raise BrmError(f"{what}: duplicate port names {names}")
    runtime = list(b.params("runtime"))
    for p in ports:
        w = f"{what}.port {p.name!r}"
        _port_name(p.name, w)
        if p.when is not None:
            if not isinstance(p.when, dict) or not p.when:
                raise BrmError(f"{w}.when: must be a non-empty object, design param -> value")
            for k, v in p.when.items():
                _allowed_value(design, k, v, f"{w}.when.{k}")
        if p.direction not in DIRECTIONS:
            raise BrmError(f"{w}: direction must be 'in' or 'out', got {p.direction!r}")
        _uses(p.lanes, design, f"{w}.lanes")
        if (c := expr.constant(p.lanes)) is not None and c < 1:
            raise BrmError(f"{w}.lanes: must be >= 1, got {c}")
        if isinstance(p.rate, str):
            if not expr.is_name(p.rate) or p.rate not in runtime:
                raise BrmError(f"{w}.rate: {p.rate!r} must be an int or a runtime param name")
        elif type(p.rate) is not int or p.rate < 1:
            raise BrmError(f"{w}.rate: must be an int >= 1 or a runtime param name")
        if (
            not isinstance(p.dtypes, list)
            or not p.dtypes
            or any(d not in INT_DTYPES for d in p.dtypes)
            or len(set(p.dtypes)) != len(p.dtypes)
        ):
            raise BrmError(
                f"{w}.dtypes: must be a non-empty list of distinct names of {list(INT_DTYPES)}, "
                f"got {p.dtypes!r}"
            )
    dirs = {p.direction for p in ports}
    if dirs != set(DIRECTIONS):
        raise BrmError(f"{what}: needs at least one input and one output port")
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
    if f.code is None:
        return
    if isinstance(f.code, str):
        f.code = _check_entry(b, design, Code({}, f.code), f"{w}.code").code
        return
    if not isinstance(f.code, list) or not f.code:
        raise BrmError(f"{w}.code: must be a string, a non-empty list of {{params, code}} or null")
    for i, e in enumerate(f.code):
        wc = f"{w}.code[{i}]"
        if not isinstance(e, Code):
            raise BrmError(f"{wc}: must be an entry {{params, code}}")
        for k, v in e.params.items():
            _allowed_value(design, k, v, f"{wc}.params.{k}")
        e.code = _check_entry(b, design, e, f"{wc}.code").code
    seen: dict[tuple, int] = {}
    for i, e in enumerate(f.code):  # at most one entry fits an instance (Function)
        key = (tuple(sorted(e.params.items())), frozenset(b.slots(e)))
        if key in seen:
            raise BrmError(f"{w}.code[{i}]: fits the instances code[{seen[key]}] fits (the same "
                           "params, and the same slots set)")  # fmt: skip
        seen[key] = i


def _allowed_value(design: dict[str, Param], k: str, v: Any, what: str) -> None:
    """``k`` is a design param and ``v`` one of its values."""
    if k not in design:
        raise BrmError(f"{what}: not a design param ({sorted(design)})")
    p = design[k]
    if type(v) is not TYPES[p.type] or (p.values is not None and v not in p.values):
        allowed = p.values if p.values is not None else f"any {p.type}"
        raise BrmError(f"{what}: {v!r} is not an allowed value ({allowed})")


def _check_entry(b: Brm, design: dict[str, Param], e: Code, what: str) -> Code:
    """``e`` with its code canonical: each output present at its params (over the defaults)
    assigned once, and every name read a present input or a slot (an int design param that is
    no port and that the entry does not fix)."""
    at = b.defaults() | e.params
    present = b.present(at)
    ins = [p.name for p in present if p.direction == "in"]
    outs = [p.name for p in present if p.direction == "out"]
    absent = {p.name for p in b.interface.ports} - {p.name for p in present}
    shown = ", ".join(f"{k} = {v!r}" for k, v in e.params.items()) or "the defaults"
    try:
        code = expr.canonical_code(e.code, what)
        lines = expr.statements(code, what)
    except expr.ExprError as err:
        raise BrmError(str(err)) from None
    targets = [t for t, _ in lines]
    for t in targets:
        if t in absent:
            raise BrmError(f"{what}: assigns {t!r}, a port not present at {shown}")
        if t not in outs:
            raise BrmError(f"{what}: assigns {t!r}, which is not an output")
    for o in outs:
        if targets.count(o) != 1:
            raise BrmError(f"{what}: output {o!r} must be assigned exactly once")
    for _, rhs in lines:
        for n in sorted(expr.names(rhs)):
            if n in ins:
                continue
            if n in absent:
                raise BrmError(f"{what}: reads {n!r}, a port not present at {shown}")
            slot = n in design and design[n].type == "int" and n not in e.params
            if not slot:
                raise BrmError(f"{what}: {n!r} is not an input present at {shown}, nor a slot "
                               "(an int design param the entry does not fix)")  # fmt: skip
    return Code(e.params, code)


def _check_pattern(b: Brm, what: str) -> None:
    p, w = b.pattern, f"{what}.pattern"
    if not isinstance(p.family, str) or not p.family:
        raise BrmError(f"{w}.family: must be a non-empty string")
    if not isinstance(p.attrs, dict):
        raise BrmError(f"{w}.attrs: must be an object")


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
        _uses(t.drain, design, f"{w}.timing.drain")
        _uses(t.span, design, f"{w}.timing.span")
        if (c := expr.constant(t.span)) is not None and c < 1:
            raise BrmError(f"{w}.timing.span: must be >= 1, got {c}")
        if (c := expr.constant(t.drain)) is not None and c < 0:
            raise BrmError(f"{w}.timing.drain: must be >= 0, got {c}")
        if (c := expr.constant(t.latency)) is not None and c < 0:
            raise BrmError(f"{w}.timing.latency: must be >= 0, got {c}")
        if (c := expr.constant(t.initiation_interval)) is not None and c < 1:
            raise BrmError(f"{w}.timing.initiation_interval: must be >= 1, got {c}")
        if impl.binding is not None:
            try:
                impl.binding = check_binding(
                    impl.binding, design, [p.name for p in b.interface.ports], f"{w}.binding"
                )
            except ValueError as e:
                raise BrmError(str(e)) from None


def _check_dataflow(b: Brm, what: str) -> None:
    d, w = b.dataflow, f"{what}.dataflow"
    if d.notation not in NOTATIONS:
        raise BrmError(f"{w}.notation: {d.notation!r} not registered ({sorted(NOTATIONS)})")
    ports = [p.name for p in b.interface.ports]
    if set(d.ports) != set(ports):
        raise BrmError(f"{w}.ports: need one nest per port {ports}, got {list(d.ports)}")
    notation = NOTATIONS[d.notation]
    for p in b.interface.ports:
        try:
            notation.check(d.ports[p.name], p, b)
        except (TypeError, ValueError) as e:
            raise BrmError(f"{w}.ports.{p.name}: {e}") from None
    if notation.normalize is not None:  # every field written
        d.ports = {p.name: notation.normalize(d.ports[p.name]) for p in b.interface.ports}
