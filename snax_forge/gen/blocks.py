"""The datapaths the generator fills: a binding's ``datapath`` -> a unit's ports and params (D147).

An adapter reads an instance (its BRM, implementation and design params,
snax_forge/brm/instance.py) and its ports' dtypes (the node's containers',
``brm.port_dtypes``) and gives what ``accelerator.json`` holds of the unit
beyond its name, BRM, timing and CSRs (accelerator.py): per BRM port its
datapath port, dtype and lanes, ``opType`` for an elementwise unit, and
``params``; or it refuses with a ``GenError`` naming what does not fit.
Adapters are registered by the datapath's name with its latency L
(``register_datapath``, D19); the implementation's timing must be L, II 1 and
drain 0 (accelerator.py). Every datapath is driven by the shell's nest
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
    """A unit's BRM port in ``accelerator.json``: the datapath's port, its dtype, its lanes."""

    port: str
    dtype: str | None  # None: judged without a graph (module doc)
    lanes: int

    def to_dict(self) -> dict[str, Any]:
        return {"port": self.port, "dtype": self.dtype, "lanes": self.lanes}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Port:
        return cls(d["port"], d["dtype"], d["lanes"])


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
    """A registered datapath: its latency L and its adapter."""

    latency: int
    fill: Adapter


DATAPATHS: dict[str, Datapath] = {}


def register_datapath(name: str, latency: int, adapter: Adapter) -> None:
    """Add the datapath ``name`` (a binding's ``datapath``) of latency ``latency``."""
    if name in DATAPATHS:
        raise ValueError(f"datapath {name!r} already registered")
    DATAPATHS[name] = Datapath(latency, adapter)


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
        out[p.name] = Port(mapping.get(p.name, p.name), dtype, n)
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


def tree_accumulate(inst: Instance, dtypes: Mapping[str, str] | None = None) -> Fill:
    """TreeAccumulate's fill; see the module doc."""
    brm, b = inst.brm, inst.impl.binding
    what = f"{brm.name}.{inst.implementation}"
    if set(b["args"]) != {"lanes"}:
        raise GenError(
            "gen.args", f"{what}: args {sorted(b['args'])}; TreeAccumulate takes lanes only"
        )
    lanes = _lanes_arg(inst, what)
    op = inst.params.get("op")
    if op != "add":
        raise GenError("gen.op", f"{what}: op {op!r}; TreeAccumulate adds")
    (a,), (out,) = _ports(inst, 1, 1, "TreeAccumulate", what)
    if a.rate != 1 or not isinstance(out.rate, str):
        raise GenError(
            "gen.rate",
            f"{what}: rates {a.rate!r} in and {out.rate!r} out; TreeAccumulate takes 1 in and "
            "a named rate out (one sum per T inputs)",
        )
    _lanes(inst, {a.name: lanes, out.name: 1}, what)
    return Fill(_fill_ports(inst, dtypes, {a.name}, what), params={"op": op})


register_datapath("TreeAccumulate", 1, tree_accumulate)
