"""The datapaths the generator fills: a binding's ``datapath`` -> a unit's ports and params.

An adapter reads an instance (brm/instance.py) and its ports' dtypes (``brm.port_dtypes``) and
gives what ``accelerator.json`` holds of the unit beyond its name, BRM, timing and CSRs: per BRM
port its datapath port, dtype and lanes, ``opType`` for a SIMD or Mac unit, and ``params``; or
it refuses with a ``GenError``. ``register_datapath`` registers an adapter by the datapath's
name with its latency and span (an int or an ``expr.py`` expression over its params and
``lanes``). The datapaths, their ports and the refusals are CONTRACTS, "Accelerator RTL spec".

The args each adapter reads: ``simd_lanes`` lanes, op, streams (2 by default) and k, or k1 and
k2 for axpby, each baked (an int that fits opType, else ``gen.k``) or read from a container (a
CSR); ``tree_accumulate`` and ``serial_accumulate`` lanes, op add only, a seed port when the
BRM's design param ``seed`` is 1; ``mac`` lanes, ``accumulate`` and ``regOut``;
``lane_accumulate`` lanes. Every port is whole 64-bit channels. An instance judged without a
graph (brm/instance.py's check) has no dtypes: its adapter checks all but the widths.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from snax_forge import expr
from snax_forge.brm import SEED, Instance
from snax_forge.brm.dtypes import fits, result_type
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
    (a SIMD unit's, else None), ``params`` and ``scalars``: each slot read from a container,
    in slot order, at its dtype (an R/W CSR the program writes, the datapath's pin)."""

    ports: dict[str, Port]
    op_type: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    scalars: dict[str, str] = field(default_factory=dict)


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
    ins = [p for p in inst.ports if p.direction == "in"]
    outs = [p for p in inst.ports if p.direction == "out"]
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
    """Every port of the instance in its order: the binding's datapath port (by default the same
    name), its
    dtype and its lanes; a port in ``words`` must be whole 64-bit channels (``gen.lanes``),
    the others are one word (a fold's output)."""
    mapping = inst.impl.binding.get("ports") or {}
    out = {}
    for p in inst.ports:
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


AXPBY = "axpby"  # SimdLanes' one ternary op, k1 * a + k2 * b (k2 unset: + b)


def simd_lanes(inst: Instance, dtypes: Mapping[str, str] | None = None) -> Fill:
    """SimdLanes' fill; see the module doc."""
    brm, b = inst.brm, inst.impl.binding
    what = f"{brm.name}.{inst.implementation}"
    if not {"lanes", "op"} <= set(b["args"]) <= {"lanes", "op", "streams", "k", "k1", "k2"}:
        raise GenError("gen.args", f"{what}: args {sorted(b['args'])}; SimdLanes takes lanes and "
                       "op, and optionally streams, k, k1 and k2")  # fmt: skip
    lanes = _lanes_arg(inst, what)
    op = expr.evaluate(b["args"]["op"], inst.params)
    if op not in ELEMENTWISE_OP and op != AXPBY:
        raise GenError("gen.op", f"{what}: op {op!r} is not one of ElementwiseOp's "
                       f"{list(ELEMENTWISE_OP)} nor {AXPBY}")  # fmt: skip
    streams = expr.evaluate(b["args"].get("streams", 2), inst.params)
    if streams not in (1, 2):
        raise GenError("gen.args", f"{what}: streams {streams!r}; SimdLanes takes 1 or 2")
    slots = {s: _set_arg(inst, s) for s in ("k", "k1", "k2")}
    k = slots["k"]
    if streams == 2 and k is not None:
        raise GenError("gen.args", f"{what}: k = {k} at two streams; k is one stream's constant")
    if op == AXPBY and streams != 2:
        raise GenError("gen.args", f"{what}: {AXPBY} at one stream; it takes two")
    read = {s for s in slots if _read_arg(inst, s)}
    if op != AXPBY and ({"k1", "k2"} & read or (slots["k1"], slots["k2"]) != (None, None)):
        raise GenError("gen.args", f"{what}: k1 and k2 are {AXPBY}'s, not {op}'s")
    if streams == 1 and k is None and "k" not in read:
        raise GenError("gen.args", f"{what}: one stream takes the constant k, got None (neither "
                       "baked nor read from a container)")  # fmt: skip
    if op == AXPBY and slots["k1"] is None and "k1" not in read:
        raise GenError("gen.args", f"{what}: {AXPBY} takes k1, got None (neither baked nor read "
                       "from a container)")  # fmt: skip
    ins, _ = _ports(inst, streams, 1, "SimdLanes", what)
    names = [p.name for p in inst.ports]
    _lanes(inst, {p: lanes for p in names}, what)
    ports = _fill_ports(inst, dtypes, set(names), what)
    op_type = None if dtypes is None else result_type([dtypes[p.name] for p in ins])
    baked = {s: v for s, v in slots.items() if v is not None}
    for s, v in baked.items():
        if type(v) is not int:
            raise GenError("gen.args", f"{what}: {s} = {v!r}; a slot is an int")
        if op_type is not None and not fits(v, op_type):
            raise GenError("gen.k", f"{what}: {s} = {v} does not fit the op type {op_type}")
    scalars = {s: op_type for s in slots if s in read and op_type is not None}
    return Fill(ports, op_type, {"op": op} | baked, scalars)


def _set_arg(inst: Instance, name: str) -> Any:
    """Arg ``name`` at the instance's params; None when it is absent or names an unset param."""
    arg = inst.impl.binding["args"].get(name)
    if arg is None or not expr.names(arg) <= set(inst.params):  # unset: not evaluated
        return None
    return expr.evaluate(arg, inst.params)


def _read_arg(inst: Instance, name: str) -> bool:
    """Whether arg ``name`` names a slot the instance reads from a container (a CSR)."""
    arg = inst.impl.binding["args"].get(name)
    return arg is not None and bool(expr.names(arg) & set(inst.read))


register_datapath("SimdLanes", 0, simd_lanes)


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
    seeded = any(p.name == SEED for p in inst.ports)
    ins, (out,) = _ports(inst, 2 if seeded else 1, 1, who, what)
    a = next(p for p in ins if p.name != SEED)
    if a.rate != 1 or not isinstance(out.rate, str):
        raise GenError(
            "gen.rate",
            f"{what}: rates {a.rate!r} in and {out.rate!r} out; {who} takes 1 in and "
            "a named rate out (one sum per T inputs)",
        )
    if seeded and inst.brm.port(SEED).rate != out.rate:
        raise GenError("gen.rate", f"{what}: the seed's rate {inst.brm.port(SEED).rate!r}, the "
                       f"output's {out.rate!r}; {who} reads one seed per sum")  # fmt: skip
    _lanes(inst, {a.name: lanes, out.name: 1, **({SEED: 1} if seeded else {})}, what)
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
    names = [p.name for p in inst.ports]
    _lanes(inst, {p: lanes for p in names}, what)
    ports = _fill_ports(inst, dtypes, set(names), what)
    if not accumulate and dtypes is not None and dtypes[ins[2].name] != dtypes[out.name]:
        raise GenError("gen.dtype", f"{what}: c is {dtypes[ins[2].name]}, out "
                       f"{dtypes[out.name]}; the plain form's c is out's dtype")  # fmt: skip
    op_type = None if dtypes is None else result_type([dtypes[p.name] for p in ins[:2]])
    return Fill(ports, op_type, {"accumulate": accumulate, "regOut": reg_out})


register_datapath("Mac", "max(accumulate, regOut)", mac)


def lane_accumulate(inst: Instance, dtypes: Mapping[str, str] | None = None) -> Fill:
    """LaneAccumulate's fill; see the module doc."""
    brm, b = inst.brm, inst.impl.binding
    what = f"{brm.name}.{inst.implementation}"
    if set(b["args"]) != {"lanes"}:
        raise GenError("gen.args", f"{what}: args {sorted(b['args'])}; LaneAccumulate takes "
                       "lanes only")  # fmt: skip
    lanes = _lanes_arg(inst, what)
    op = inst.params.get("op")
    if op != "add":
        raise GenError("gen.op", f"{what}: op {op!r}; LaneAccumulate adds")
    (a,), (out,) = _ports(inst, 1, 1, "LaneAccumulate", what)
    if a.rate != 1 or not isinstance(out.rate, str):
        raise GenError("gen.rate", f"{what}: rates {a.rate!r} in and {out.rate!r} out; "
                       "LaneAccumulate takes 1 in and a named rate out (one sum per lane per T "
                       "inputs)")  # fmt: skip
    _lanes(inst, {a.name: lanes, out.name: lanes}, what)
    return Fill(_fill_ports(inst, dtypes, {a.name, out.name}, what))


register_datapath("LaneAccumulate", 1, lane_accumulate)
