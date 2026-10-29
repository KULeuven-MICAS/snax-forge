"""The datapaths the generator fills: a binding's ``datapath`` -> a unit's ports and params (D147).

An adapter reads an instance (its BRM, implementation and design params,
snax_forge/brm/instance.py) and its ports' dtypes (the node's containers',
``brm.port_dtypes``) and gives what ``accelerator.json`` holds of the unit
beyond its name, BRM, timing and CSRs (accelerator.py): per BRM port its
datapath port, dtype and lanes, ``opType`` for an elementwise unit, and
``params``; or it refuses with a ``GenError`` naming what does not fit.
Adapters are registered by the datapath's name with its latency L and span
(``register_datapath``, D19), each an int or an ``expr.py`` expression over the
params the adapter gives and ``lanes`` (its first input's; a bool as 0 or 1,
``timing``); the implementation's timing must be L, II 1, that span and drain
0 (accelerator.py). Every datapath is driven by the shell's nest
controller (``counted``, D129): data and enables only.

**ElementwiseLanes** (L 0): one op over every lane.

    ports        two inputs and one output (SnaxUnit's op table is binary),
                 each with args.lanes lanes; each port's lanes * N whole
                 64-bit channels (D142), N its dtype's bits (intN or uintN,
                 N in 8, 16, 32, 64)
    opType       the inputs' NumPy result type (brm/dtypes.py's
                 ``result_type``, D143), the type the op runs at; the result
                 is then extended or wrapped into the output's
    params       ``op``: args.op, one of ElementwiseOp's names (add, sub, mul,
                 and, or, xor, min, max; ElementwiseAlu.scala's encoding is
                 ``ELEMENTWISE_OP``, which tests/gen reads from the source)

**TreeAccumulate** (L 1, D133): a fold, one sum every T inputs, the input's
lanes summed by an adder tree in front of its register.

    ports        one input of rate 1 with args.lanes lanes, whole 64-bit
                 channels, and one output of a named rate (T) of one lane,
                 one word whatever its N (the lanes are cast to it, D143)
    op           add: the BRM's op param, the only one an adder tree runs
                 (``gen.op`` until a datapath takes another)
    params       ``op``: add, as ElementwiseLanes' ``op``; args are lanes only

**SerialAccumulate** (L 1, span lanes, D149): TreeAccumulate's ports, op and
args, its lanes added one a cycle (lane k in cycle k) into one sum; params
none (its op is add).

**Mac** (L ``max(accumulate, regOut)``, D149): args lanes, ``accumulate`` and
``regOut`` (bools). With ``accumulate``: inputs ``a`` and ``b`` of rate 1, an
output of a named rate (T), each lane summing T products; otherwise the plain
``a * b + c``, ``c`` an input of rate 1 with ``out``'s dtype and lanes, every
rate 1. Every port args.lanes lanes of whole 64-bit channels; ``opType`` the
result type of ``a`` and ``b``; params ``accumulate`` and ``regOut``.

An instance judged without a graph (brm/instance.py's check of its datapath)
has no dtypes: its adapter checks everything but the widths and gives ports
without dtypes, which are never written.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from snax_forge import expr
from snax_forge.brm import Instance
from snax_forge.brm.dtypes import result_type
from snax_forge.common import CodedError


class GenError(CodedError):
    """A design point the generator refuses; ``code`` names the rule (``gen.<rule>``)."""


@dataclass(frozen=True)
class Port:
    """A unit's BRM port in ``accelerator.json``: the datapath's port, its direction (``in``
    or ``out``, the BRM port's), its dtype, its lanes."""

    port: str
    dir: str
    dtype: str | None  # None: judged without a graph (module doc)
    lanes: int

    def to_dict(self) -> dict[str, Any]:
        return {"port": self.port, "dir": self.dir, "dtype": self.dtype, "lanes": self.lanes}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Port:
        return cls(d["port"], d["dir"], d["dtype"], d["lanes"])


@dataclass(frozen=True)
class Fill:
    """What an adapter gives: the unit's ports (per BRM port, in the BRM's order), ``opType``
    (an elementwise unit's, else None) and ``params``."""

    ports: dict[str, Port]
    op_type: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


Adapter = Callable[..., Fill]  # (inst, dtypes=None) -> Fill


@dataclass(frozen=True)
class Datapath:
    """A registered datapath: its latency L, its adapter and its span (module doc)."""

    latency: int | str
    fill: Adapter
    span: int | str = 1

    def timing(self, filled: Fill) -> tuple[int, int]:
        """``(L, span)`` for the unit ``filled`` gives (module doc)."""
        env = {k: int(v) for k, v in filled.params.items() if isinstance(v, (bool, int))}
        first = next((p for p in filled.ports.values() if p.dir == "in"), None)
        env["lanes"] = 1 if first is None else first.lanes
        return tuple(v if isinstance(v, int) else expr.evaluate(v, env)
                     for v in (self.latency, self.span))  # type: ignore[return-value]  # fmt: skip


DATAPATHS: dict[str, Datapath] = {}


def register_datapath(name: str, latency: int | str, adapter: Adapter, span: int | str = 1) -> None:
    """Add the datapath ``name`` (a binding's ``datapath``) of latency ``latency`` and span
    ``span`` (module doc)."""
    if name in DATAPATHS:
        raise ValueError(f"datapath {name!r} already registered")
    DATAPATHS[name] = Datapath(latency, adapter, span)


# ElementwiseOp in ElementwiseAlu.scala
ELEMENTWISE_OP = {"add": 0, "sub": 1, "mul": 2, "and": 3, "or": 4, "xor": 5, "min": 6, "max": 7}
DTYPE = re.compile(r"(u?)int(8|16|32|64)")


def port_dtype(dtype: str, where: str) -> tuple[int, bool]:
    """``(bits, signed)`` of a port's dtype: intN or uintN, N in 8, 16, 32, 64."""
    m = DTYPE.fullmatch(dtype)
    if m is None:
        raise GenError(
            "gen.dtype", f"{where}: dtype {dtype!r} is not intN or uintN with N in 8, 16, 32, 64"
        )
    return int(m.group(2)), not m.group(1)


def _ports(inst: Instance, n_in: int, n_out: int, who: str, what: str) -> tuple[list, list]:
    """The BRM's input and output ports, ``n_in`` and ``n_out`` of them."""
    ins = [p for p in inst.brm.interface.ports if p.direction == "in"]
    outs = [p for p in inst.brm.interface.ports if p.direction == "out"]
    if (len(ins), len(outs)) != (n_in, n_out):
        raise GenError(
            "gen.ports",
            f"{what}: {len(ins)} inputs and {len(outs)} outputs; {who} takes {n_in} and {n_out}",
        )
    return ins, outs


def _lanes(inst: Instance, want: dict[str, int], what: str) -> None:
    """Each port has the lanes ``want`` gives it."""
    for port, n in want.items():
        if inst.lanes(port) != n:
            raise GenError(
                "gen.lanes", f"{what}: port {port} has {inst.lanes(port)} lanes, the datapath {n}"
            )


def _fill_ports(inst: Instance, dtypes: Mapping[str, str] | None, words: set[str],
                what: str) -> dict[str, Port]:  # fmt: skip
    """Every BRM port in its order: the binding's datapath port (by default the same name), its
    dtype and its lanes; a port in ``words`` must be whole 64-bit channels (``gen.lanes``),
    the others are one word (a fold's output)."""
    mapping = inst.impl.binding.get("ports") or {}
    out = {}
    for p in inst.brm.interface.ports:
        n, dtype = inst.lanes(p.name), None
        if dtypes is not None:
            dtype = dtypes[p.name]
            bits, _ = port_dtype(dtype, f"{what} port {p.name}")
            if p.name in words and n * bits % 64:
                raise GenError(
                    "gen.lanes",
                    f"{what}: port {p.name}'s {n} lanes of {bits} bits are not whole 64-bit "
                    "channels",
                )
        out[p.name] = Port(mapping.get(p.name, p.name), p.direction, dtype, n)
    return out


def _lanes_arg(inst: Instance, what: str) -> int:
    lanes = expr.evaluate(inst.impl.binding["args"]["lanes"], inst.params)
    if not isinstance(lanes, int) or lanes < 1:
        raise GenError("gen.lanes", f"{what}: {lanes!r} lanes")
    return lanes


def elementwise_lanes(inst: Instance, dtypes: Mapping[str, str] | None = None) -> Fill:
    """ElementwiseLanes' fill; see the module doc."""
    brm, b = inst.brm, inst.impl.binding
    what = f"{brm.name}.{inst.implementation}"
    if set(b["args"]) != {"lanes", "op"}:
        raise GenError(
            "gen.args", f"{what}: args {sorted(b['args'])}; ElementwiseLanes takes lanes and op"
        )
    lanes = _lanes_arg(inst, what)
    op = expr.evaluate(b["args"]["op"], inst.params)
    if op not in ELEMENTWISE_OP:
        raise GenError(
            "gen.op", f"{what}: op {op!r} is not one of ElementwiseOp's {list(ELEMENTWISE_OP)}"
        )
    ins, _ = _ports(inst, 2, 1, "ElementwiseLanes", what)
    names = [p.name for p in brm.interface.ports]
    _lanes(inst, {p: lanes for p in names}, what)
    ports = _fill_ports(inst, dtypes, set(names), what)
    op_type = None if dtypes is None else result_type([dtypes[p.name] for p in ins])
    return Fill(ports, op_type, {"op": op})


register_datapath("ElementwiseLanes", 0, elementwise_lanes)


def tree_accumulate(inst: Instance, dtypes: Mapping[str, str] | None = None,
                    who: str = "TreeAccumulate") -> Fill:  # fmt: skip
    """TreeAccumulate's fill (SerialAccumulate's as ``who``); see the module doc."""
    brm, b = inst.brm, inst.impl.binding
    what = f"{brm.name}.{inst.implementation}"
    if set(b["args"]) != {"lanes"}:
        raise GenError("gen.args", f"{what}: args {sorted(b['args'])}; {who} takes lanes only")
    lanes = _lanes_arg(inst, what)
    op = inst.params.get("op")
    if op != "add":
        raise GenError("gen.op", f"{what}: op {op!r}; {who} adds")
    (a,), (out,) = _ports(inst, 1, 1, who, what)
    if a.rate != 1 or not isinstance(out.rate, str):
        raise GenError(
            "gen.rate",
            f"{what}: rates {a.rate!r} in and {out.rate!r} out; {who} takes 1 in and "
            "a named rate out (one sum per T inputs)",
        )
    _lanes(inst, {a.name: lanes, out.name: 1}, what)
    params = {"op": op} if who == "TreeAccumulate" else {}
    return Fill(_fill_ports(inst, dtypes, {a.name}, what), params=params)


register_datapath("TreeAccumulate", 1, tree_accumulate)


def serial_accumulate(inst: Instance, dtypes: Mapping[str, str] | None = None) -> Fill:
    """SerialAccumulate's fill; see the module doc."""
    return tree_accumulate(inst, dtypes, "SerialAccumulate")


register_datapath("SerialAccumulate", 1, serial_accumulate, span="lanes")


def mac(inst: Instance, dtypes: Mapping[str, str] | None = None) -> Fill:
    """Mac's fill; see the module doc."""
    brm, b = inst.brm, inst.impl.binding
    what = f"{brm.name}.{inst.implementation}"
    if set(b["args"]) != {"lanes", "accumulate", "regOut"}:
        raise GenError("gen.args", f"{what}: args {sorted(b['args'])}; Mac takes lanes, "
                       "accumulate and regOut")  # fmt: skip
    accumulate, reg_out = b["args"]["accumulate"], b["args"]["regOut"]
    if type(accumulate) is not bool or type(reg_out) is not bool:
        raise GenError("gen.args", f"{what}: accumulate {accumulate!r}, regOut {reg_out!r}; "
                       "each is true or false")  # fmt: skip
    lanes = _lanes_arg(inst, what)
    ins, (out,) = _ports(inst, 2 if accumulate else 3, 1, "Mac", what)
    if any(p.rate != 1 for p in ins) or isinstance(out.rate, str) != accumulate:
        raise GenError("gen.rate", f"{what}: rates {[p.rate for p in ins]} in and {out.rate!r} "
                       "out; Mac takes 1 in and, accumulating, a named rate out (one sum per T "
                       "products), else 1")  # fmt: skip
    names = [p.name for p in brm.interface.ports]
    _lanes(inst, {p: lanes for p in names}, what)
    ports = _fill_ports(inst, dtypes, set(names), what)
    if not accumulate and dtypes is not None and dtypes[ins[2].name] != dtypes[out.name]:
        raise GenError("gen.dtype", f"{what}: c is {dtypes[ins[2].name]}, out "
                       f"{dtypes[out.name]}; the plain form's c is out's dtype")  # fmt: skip
    op_type = None if dtypes is None else result_type([dtypes[p.name] for p in ins[:2]])
    return Fill(ports, op_type, {"accumulate": accumulate, "regOut": reg_out})


register_datapath("Mac", "max(accumulate, regOut)", mac)
