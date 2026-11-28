"""The datapaths the generator fills: a binding's ``datapath`` -> a unit's ports and params.

An adapter reads an instance (brm/instance.py) and its ports' dtypes (``brm.port_dtypes``) and
gives what ``accelerator.json`` holds of the unit beyond its name, BRM, timing and CSRs: per BRM
port its datapath port, dtype and lanes, ``opType`` for a SIMD or Mac unit, and ``params``; or
it refuses with a ``GenError``. ``register_datapath`` registers an adapter by the datapath's
name with its latency and span (an int or an ``expr.py`` expression over its params and
``lanes``). The datapaths, their ports and the refusals are CONTRACTS, "Accelerator RTL spec".

The args each adapter reads: ``simd_lanes`` lanes, op, streams (2 by default) and k, or the
slots of an op that has its own (``NAMED_OPS``: axpby's k1 and k2; axpbypcz's k1, k2, k3 and d;
requant's d, lo and hi; threshold's k and hi), each baked (an int that fits opType, else
``gen.k``) or read from a container (a CSR) where the op allows, d a power of two; ``tree_accumulate`` and ``serial_accumulate`` lanes, op add only, a seed port when the
BRM's design param ``seed`` is 1; ``mac`` lanes, ``accumulate`` and ``regOut``;
``lane_accumulate`` lanes. ``mac`` takes a seed port of its lanes when the BRM's design param
``seed`` is 1. Every port is whole 64-bit channels, a packed output or a sliced
input. A datapath registered ``lanewise`` (SimdLanes, Mac: each lane on its own) takes
a broadcast input, one lane its reader moves given to every lane: the port keeps the
unit's lanes and carries ``broadcast``. An instance judged without a graph (brm/instance.py's check) has no
dtypes: its adapter checks all but the widths. A datapath registered with ``product`` (Mac, and
SimdLanes at mul, axpby and an axpbypcz with a k) takes ``narrowMul`` true in its params when
the design point's ``narrow_mul`` is (D207); no other unit, and no default design, has it.
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
from snax_forge.design.streamers import per_word


class GenError(CodedError):
    """A design point the generator refuses; ``code`` names the rule (``gen.<rule>``)."""


@dataclass(frozen=True)
class Port:
    """A unit's BRM port in ``accelerator.json``: the datapath's port, its direction (``in``
    or ``out``, the BRM port's), its dtype, its lanes; a packed outer output's d_out, a sliced
    outer input's d_in; a broadcast, a held input and a carried seed marked."""

    port: str
    dir: str
    dtype: str | None  # None: judged without a graph (module doc)
    lanes: int
    d_out: int = 1  # an outer output's results per word through its packer
    d_in: int = 1  # an outer input's pops per word through its slicer
    broadcast: bool = False  # an outer input whose reader moves one lane for all
    hold: bool = False  # an input read once a bank row and kept
    carry: bool = False  # a seed that is its unit's own output, 0 before CARRY jobs

    @property
    def stream_lanes(self) -> int:
        """The lanes its reader or writer moves a beat: one for a broadcast, else its lanes."""
        return 1 if self.broadcast else self.lanes

    def to_dict(self) -> dict[str, Any]:
        d = {"port": self.port, "dir": self.dir, "dtype": self.dtype, "lanes": self.lanes}
        d |= {"d_out": self.d_out} if self.d_out > 1 else {}
        d |= {"d_in": self.d_in} if self.d_in > 1 else {}
        d |= {"broadcast": True} if self.broadcast else {}
        d |= {"hold": True} if self.hold else {}
        return d | ({"carry": True} if self.carry else {})

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Port:
        return cls(d["port"], d["dir"], d["dtype"], d["lanes"], d.get("d_out", 1),
                   d.get("d_in", 1), d.get("broadcast", False), d.get("hold", False),
                   d.get("carry", False))  # fmt: skip


@dataclass(frozen=True)
class Fill:
    """What an adapter gives: the unit's ports (per BRM port, in the BRM's order), ``opType``
    (a SIMD unit's, else None), ``params`` and ``scalars``: each slot read from a container,
    in slot order, at its dtype (an R/W CSR the program writes, the datapath's pin)."""

    ports: dict[str, Port]
    op_type: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    scalars: dict[str, str] = field(default_factory=dict)


Adapter = Callable[..., Fill]  # (inst, dtypes=None, op_type=None) -> Fill


@dataclass(frozen=True)
class Datapath:
    """A registered datapath: its latency L, its adapter, its span, whether each lane
    computes on its own (``lanewise``, so an input may be a broadcast) and whether a fill of it
    has a product (``product``, so it takes ``narrowMul``; module doc)."""

    latency: int | str
    fill: Adapter
    span: int | str = 1
    lanewise: bool = False
    product: Callable[[Fill], bool] | None = None

    def timing(self, filled: Fill) -> tuple[int, int]:
        """``(L, span)`` for the unit ``filled`` gives (module doc)."""
        env = {k: int(v) for k, v in filled.params.items() if isinstance(v, (bool, int))}
        first = next((p for p in filled.ports.values() if p.dir == "in"), None)
        env["lanes"] = 1 if first is None else first.lanes
        return tuple(v if isinstance(v, int) else expr.evaluate(v, env)
                     for v in (self.latency, self.span))  # type: ignore[return-value]  # fmt: skip


DATAPATHS: dict[str, Datapath] = {}
NARROW_MUL = "narrowMul"  # the param a product's datapath takes for design.narrow_mul (D207)


def fill_node(datapath: str, inst: Instance, graph: Any, node: Any,
              narrow_mul: bool = False) -> Fill:  # fmt: skip
    """``datapath``'s adapter on the instance bound at ``node``, its op type the node's
    ``opType`` where the kernel's casts set one, else its inputs' result type; with
    ``narrow_mul``, ``narrowMul`` true in its params where it has a product (module doc)."""
    from dataclasses import replace

    from snax_forge.brm import port_dtypes

    dp = DATAPATHS[datapath]
    op_type = node.attrs.get("opType")
    f = dp.fill(inst, port_dtypes(graph, node), op_type=op_type)
    if narrow_mul and dp.product is not None and dp.product(f):
        f = replace(f, params=f.params | {NARROW_MUL: True})
    if op_type is None or f.op_type is None:
        return f
    return replace(f, op_type=op_type, scalars={s: op_type for s in f.scalars})


def register_datapath(name: str, latency: int | str, adapter: Adapter, span: int | str = 1,
                      lanewise: bool = False,
                      product: Callable[[Fill], bool] | None = None) -> None:  # fmt: skip
    """Add the datapath ``name`` (a binding's ``datapath``) of latency ``latency``, span
    ``span``, ``lanewise``, lanes that compute on their own, and ``product``, whether a fill
    has a product (module doc)."""
    if name in DATAPATHS:
        raise ValueError(f"datapath {name!r} already registered")
    DATAPATHS[name] = Datapath(latency, adapter, span, lanewise, product)


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
    name), its dtype and its lanes; a port in ``words`` must be whole 64-bit channels
    (``gen.lanes``), a packed output or a sliced input; the others are one word
    (a fold's output)."""
    mapping = inst.impl.binding.get("ports") or {}
    out = {}
    for p in inst.ports:
        n, dtype = inst.lanes(p.name), None
        if dtypes is not None:
            dtype = dtypes[p.name]
            bits, _ = port_dtype(dtype, f"{what} port {p.name}")
            packs = per_word(dtype, n, 64) > 1  # packed or sliced: one word on one channel
            if p.name in words and n * bits % 64 and not packs:
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


AXPBY = "axpby"  # SimdLanes' k1 * a + k2 * b (k2 unset: + b)
# SimdLanes' ops with slots of their own: streams, the slots it needs, those it may take, and
# those that must be baked (the unit shifts by d; requant's bounds are wires)
NAMED_OPS = {
    AXPBY: (2, {"k1"}, {"k1", "k2"}, set()),
    "axpbypcz": (3, {"d"}, {"k1", "k2", "k3", "d"}, {"d"}),
    "requant": (1, {"d", "lo", "hi"}, {"d", "lo", "hi"}, {"d", "lo", "hi"}),
    "threshold": (1, {"k", "hi"}, {"k", "hi"}, {"hi"}),
}
SLOTS = ("k", "k1", "k2", "k3", "d", "lo", "hi")


def simd_lanes(inst: Instance, dtypes: Mapping[str, str] | None = None,
               op_type: str | None = None) -> Fill:  # fmt: skip
    """SimdLanes' fill; see the module doc. ``op_type`` is the node's where the kernel's casts
    set one (gray's uint16 over uint8 inputs), so a baked slot is checked against it."""
    brm, b = inst.brm, inst.impl.binding
    what = f"{brm.name}.{inst.implementation}"
    if not {"lanes", "op"} <= set(b["args"]) <= {"lanes", "op", "streams", *SLOTS}:
        raise GenError("gen.args", f"{what}: args {sorted(b['args'])}; SimdLanes takes lanes and "
                       f"op, and optionally streams and the slots {', '.join(SLOTS)}")  # fmt: skip
    lanes = _lanes_arg(inst, what)
    op = expr.evaluate(b["args"]["op"], inst.params)
    if op not in ELEMENTWISE_OP and op not in NAMED_OPS:
        raise GenError("gen.op", f"{what}: op {op!r} is not one of ElementwiseOp's "
                       f"{list(ELEMENTWISE_OP)} nor {', '.join(NAMED_OPS)}")  # fmt: skip
    streams = expr.evaluate(b["args"].get("streams", 2), inst.params)
    if streams not in (1, 2, 3):
        raise GenError("gen.args", f"{what}: streams {streams!r}; SimdLanes takes 1, 2 or 3")
    slots = {s: _set_arg(inst, s) for s in SLOTS}
    read = {s for s in slots if _read_arg(inst, s)}
    have = {s for s, v in slots.items() if v is not None} | read
    if op in NAMED_OPS:
        n, needs, takes, baked_only = NAMED_OPS[op]
        if streams != n:
            raise GenError("gen.args", f"{what}: {op} at {streams} stream(s); it takes {n}")
        if have - takes:
            raise GenError("gen.args", f"{what}: {', '.join(sorted(have - takes))} not {op}'s; "
                           f"it takes {', '.join(sorted(takes))}")  # fmt: skip
        for s in sorted(needs - have):
            raise GenError("gen.args", f"{what}: {op} takes {s}, got None (neither baked nor "
                           "read from a container)")  # fmt: skip
        if read & baked_only:
            raise GenError("gen.args", f"{what}: {', '.join(sorted(read & baked_only))} read "
                           f"from a container; {op} bakes it into the hardware")  # fmt: skip
        d = slots["d"]
        if type(d) is int and (d <= 0 or d & (d - 1)):
            raise GenError("gen.args", f"{what}: d = {d} is not a power of two: the unit divides "
                           "by shifting")  # fmt: skip
    else:
        k = slots["k"]
        if streams == 3:
            raise GenError("gen.args", f"{what}: {op} at three streams; only axpbypcz takes "
                           "three")  # fmt: skip
        if streams == 2 and k is not None:
            raise GenError("gen.args", f"{what}: k = {k} at two streams; k is one stream's "
                           "constant")  # fmt: skip
        if have - {"k"}:
            owners = sorted(o for o, (_, _, t, _) in NAMED_OPS.items() if have & t - {"k"})
            raise GenError("gen.args", f"{what}: {', '.join(sorted(have - {'k'}))} "
                           f"{'are' if len(have - {'k'}) > 1 else 'is'} "
                           f"{' and '.join(owners)}'s, not {op}'s")  # fmt: skip
        if streams == 1 and k is None and "k" not in read:
            raise GenError("gen.args", f"{what}: one stream takes the constant k, got None "
                           "(neither baked nor read from a container)")  # fmt: skip
    ins, _ = _ports(inst, streams, 1, "SimdLanes", what)
    names = [p.name for p in inst.ports]
    _lanes(inst, {p: lanes for p in names}, what)
    ports = _fill_ports(inst, dtypes, set(names), what)
    if op_type is None and dtypes is not None:
        op_type = result_type([dtypes[p.name] for p in ins])
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


def _simd_product(f: Fill) -> bool:
    """A SimdLanes fill multiplies: mul, axpby (k1 always), or axpbypcz with a k baked or read."""
    op = f.params["op"]
    ks = {"k1", "k2", "k3"} & (set(f.params) | set(f.scalars))
    return op in ("mul", AXPBY) or (op == "axpbypcz" and bool(ks))


register_datapath("SimdLanes", 0, simd_lanes, lanewise=True, product=_simd_product)


def tree_accumulate(inst: Instance, dtypes: Mapping[str, str] | None = None,
                    who: str = "TreeAccumulate", op_type: str | None = None) -> Fill:  # fmt: skip
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


def serial_accumulate(inst: Instance, dtypes: Mapping[str, str] | None = None,
                      op_type: str | None = None) -> Fill:  # fmt: skip
    """SerialAccumulate's fill; see the module doc."""
    return tree_accumulate(inst, dtypes, "SerialAccumulate")


register_datapath("SerialAccumulate", 1, serial_accumulate, span="lanes")


def mac(inst: Instance, dtypes: Mapping[str, str] | None = None,
        op_type: str | None = None) -> Fill:  # fmt: skip
    """Mac's fill; see the module doc."""
    brm, b = inst.brm, inst.impl.binding
    what = f"{brm.name}.{inst.implementation}"
    if (
        not {"lanes", "accumulate", "regOut"}
        <= set(b["args"])
        <= {"lanes", "accumulate", "regOut", "rows"}
    ):
        raise GenError("gen.args", f"{what}: args {sorted(b['args'])}; Mac takes lanes, "
                       "accumulate and regOut, and rows")  # fmt: skip
    accumulate, reg_out = b["args"]["accumulate"], b["args"]["regOut"]
    rows = inst.value(b["args"]["rows"]) if "rows" in b["args"] else 1
    if type(rows) is not int or rows < 1:
        raise GenError("gen.args", f"{what}: rows {rows!r}; a bank has one row or more")
    if type(accumulate) is not bool or type(reg_out) is not bool:
        raise GenError("gen.args", f"{what}: accumulate {accumulate!r}, regOut {reg_out!r}; "
                       "each is true or false")  # fmt: skip
    lanes = _lanes_arg(inst, what)
    seeded = any(p.name == SEED for p in inst.ports)
    if seeded and not accumulate:
        raise GenError("gen.seed", f"{what}: a seed port, but accumulate is false; Mac takes a "
                       "seed only as a fold")  # fmt: skip
    ins, (out,) = _ports(inst, 3 if seeded or not accumulate else 2, 1, "Mac", what)
    if seeded and inst.brm.port(SEED).rate != out.rate:
        raise GenError("gen.rate", f"{what}: the seed's rate {inst.brm.port(SEED).rate!r}, the "
                       f"output's {out.rate!r}; Mac reads one seed per sum")  # fmt: skip
    ins = [p for p in ins if p.name != SEED]
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
    if rows > 1 and not accumulate:
        raise GenError("gen.args", f"{what}: rows {rows}, but accumulate is false; only an "
                       "accumulating Mac has a bank (set accumulate true, or rows 1)")  # fmt: skip
    op_type = None if dtypes is None else result_type([dtypes[p.name] for p in ins[:2]])
    params = {"accumulate": accumulate, "regOut": reg_out} | ({"rows": rows} if rows > 1 else {})
    return Fill(ports, op_type, params)


register_datapath("Mac", "max(accumulate, regOut)", mac, lanewise=True,
                  product=lambda f: True)  # fmt: skip


def lane_accumulate(inst: Instance, dtypes: Mapping[str, str] | None = None,
                    op_type: str | None = None) -> Fill:  # fmt: skip
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
