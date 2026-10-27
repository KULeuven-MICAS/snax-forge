"""The blocks the GVSoC generator can build: a binding's ``block`` -> a SnaxUnit (D134).

An adapter reads an instance (its BRM, implementation and design params,
snax_forge/brm/instance.py) and its ports' dtypes (the node's containers',
``brm.port_dtypes``; int64 on every port when only the instance is judged,
brm/instance.py) and gives the properties of snax-gvsoc's
``SnaxUnit`` and the fields of its ``SnaxShell`` (``Unit``), or refuses
with a ``GenError`` naming what does not fit. Adapters are registered by
the block's name (``register_block``, D19). Both blocks are ``counted``
(the only control, D129): the start's configuration holds the count.

**ElementwiseTiledSpatial** (snax-forge
``hw/chisel/.../elementwise/ElementwiseTiledSpatial.scala``):

    lanes        args.lanes
    op           args.supportedOps, one op name, through ElementwiseOp's
                 encoding (add 0, sub 1, mul 2, and 3, or 4, xor 5, min 6,
                 max 7); tests/gen reads the table from the Chisel source
    data_width   N of the ports' dtypes (intN signed, uintN unsigned); every
    signed       port the same dtype, N in 8, 16, 32, 64, lanes * N a
                 multiple of 64 (a streamer beat is whole 64-bit channels)
    inputs       the BRM's input and output port names, exactly 2 in and 1 out
    outputs      (SnaxUnit's op table is binary)
    control      counted only (the block's trip counter, loopCount)
    wait         lockstep only, with the implementation's latency 0, II 1
                 and drain 0
    fields       the binding's csrs[].field, in order

``lanes`` must be every port's lanes: the unit's beat is the streamer's.

**Accumulator** (snax-forge ``hw/chisel/.../samples/Accumulator.scala``,
D133): a fold, one sum every T inputs, each input's lanes summed by the
adder tree in front of its register.

    inputs       one input port of rate 1 and one output port of a named
    outputs      rate (T), the output of one lane
    lanes        args.lanes, the input port's lanes
    data_width   the ports' dtype, as above (the output's one lane whole
    signed       64-bit channels, so N is 64)
    op           add (0): the BRM's op param
    control      counted
    wait         pipelined, with the implementation's latency 1, II 1 and
                 drain 0 (the block's one register)
    fields       loopCount then len, their values n and the output's rate
    config       ["count", "rate"]: SnaxUnit reads the first word as the
                 run's firings and the second as firings per output
    args         lanes only: the block's widths come from the dtype

``config`` is a SnaxUnit property only when it is not the default
``["count"]``, so an elementwise block's datapath block has none.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from snax_forge import expr
from snax_forge.brm import Instance
from snax_forge.common import CodedError

# SnaxUnit's config when none is given: one word, the count
DEFAULT_CONFIG = ["count"]


class GenError(CodedError):
    """A design point the GVSoC generator refuses; ``code`` names the rule (``gen.<rule>``)."""


@dataclass(frozen=True)
class Unit:
    """What an adapter gives: SnaxUnit's properties (csr_base aside) and SnaxShell's fields."""

    inputs: list[str]
    outputs: list[str]
    lanes: int
    data_width: int
    op: int
    signed: bool
    control: str
    wait: str
    latency: int
    ii: int
    fields: list[str]
    config: list[str] = field(default_factory=lambda: list(DEFAULT_CONFIG))

    def properties(self) -> dict[str, Any]:
        """SnaxUnit's keyword arguments, in snax_unit.py's order (csr_base aside); ``config``
        only when it is not the default."""
        props = {
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "lanes": self.lanes,
            "data_width": self.data_width,
            "op": self.op,
            "signed": self.signed,
            "control": self.control,
            "config": list(self.config),
            "wait": self.wait,
            "latency": self.latency,
            "ii": self.ii,
        }
        if self.config == DEFAULT_CONFIG:
            del props["config"]
        return props


Adapter = Callable[..., Unit]  # (inst, dtypes=None) -> Unit
BLOCKS: dict[str, Adapter] = {}


def register_block(name: str, adapter: Adapter) -> None:
    """Add the adapter of the Chisel block ``name`` (a binding's ``block``)."""
    if name in BLOCKS:
        raise ValueError(f"block {name!r} already registered")
    BLOCKS[name] = adapter


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


def _control(inst: Instance, wait: str, why: str, timing: tuple[int, int, int], who: str,
             what: str) -> None:  # fmt: skip
    """The binding counts and waits ``wait`` (else ``why``), and the timing is ``timing``."""
    b = inst.impl.binding
    if b["control"] != "counted":
        raise GenError(
            "gen.control", f"{what}: control {b['control']!r}; the block counts (counted)"
        )
    if b["wait"] != wait:
        raise GenError("gen.wait", f"{what}: wait {b['wait']!r}; {why}")
    got = (inst.latency, inst.initiation_interval, inst.drain)
    if got != timing:
        raise GenError(
            "gen.timing",
            f"{what}: latency {got[0]}, II {got[1]}, drain {got[2]}; "
            f"{who} is latency {timing[0]}, II {timing[1]}, drain {timing[2]}",
        )


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


def instance_dtypes(inst: Instance) -> dict[str, str]:
    """The dtypes an instance alone is judged at: int64 on every port, one element per word
    (the node's containers give the real ones, ``brm.port_dtypes``)."""
    return {p.name: "int64" for p in inst.brm.interface.ports}


def _dtype(inst: Instance, dtypes: Mapping[str, str] | None, what: str) -> tuple[int, bool]:
    """The one dtype of every port (``dtypes``, port -> dtype), as bits and signedness."""
    dtypes = instance_dtypes(inst) if dtypes is None else dtypes
    found = {dtypes[p.name] for p in inst.brm.interface.ports}
    if len(found) != 1:
        raise GenError("gen.dtype", f"{what}: ports of dtypes {sorted(found)}; one is needed")
    return port_dtype(found.pop(), what)


def _lanes(inst: Instance, want: dict[str, int], what: str) -> None:
    """Each port has the lanes ``want`` gives it."""
    for port, n in want.items():
        if inst.lanes(port) != n:
            raise GenError(
                "gen.lanes", f"{what}: port {port} has {inst.lanes(port)} lanes, the block {n}"
            )


def elementwise_tiled_spatial(inst: Instance, dtypes: Mapping[str, str] | None = None) -> Unit:
    """ElementwiseTiledSpatial's unit; see the module doc."""
    brm, b = inst.brm, inst.impl.binding
    what = f"{brm.name}.{inst.implementation}"
    _control(inst, "lockstep", "only lockstep is generated yet", (0, 1, 0), "lockstep", what)
    args = b["args"]
    for k in ("lanes", "supportedOps"):
        if k not in args:
            raise GenError("gen.args", f"{what}: binding args lack {k!r}")
    lanes = expr.evaluate(args["lanes"], inst.params)
    op = expr.evaluate(args["supportedOps"], inst.params)
    if op not in ELEMENTWISE_OP:
        raise GenError(
            "gen.op",
            f"{what}: supportedOps {op!r} is not one of ElementwiseOp's {list(ELEMENTWISE_OP)}",
        )
    ins, outs = _ports(inst, 2, 1, "SnaxUnit", what)
    bits, signed = _dtype(inst, dtypes, what)
    if not isinstance(lanes, int) or lanes < 1 or lanes * bits % 64:
        raise GenError(
            "gen.lanes", f"{what}: {lanes} lanes of {bits} bits is not whole 64-bit channels"
        )
    _lanes(inst, {p.name: lanes for p in brm.interface.ports}, what)
    return Unit(
        inputs=[p.name for p in ins], outputs=[p.name for p in outs], lanes=lanes,
        data_width=bits, op=ELEMENTWISE_OP[op], signed=signed, control="counted",
        wait="lockstep", latency=0, ii=1, fields=[c["field"] for c in b["csrs"]],
    )  # fmt: skip


register_block("ElementwiseTiledSpatial", elementwise_tiled_spatial)


def accumulator(inst: Instance, dtypes: Mapping[str, str] | None = None) -> Unit:
    """The Accumulator's unit; see the module doc."""
    brm, b = inst.brm, inst.impl.binding
    what = f"{brm.name}.{inst.implementation}"
    _control(inst, "pipelined", "the block is pipelined", (1, 1, 0), "the Accumulator", what)
    if set(b["args"]) != {"lanes"}:
        raise GenError(
            "gen.args", f"{what}: args {sorted(b['args'])}; the Accumulator takes lanes only"
        )
    lanes = expr.evaluate(b["args"]["lanes"], inst.params)
    op = inst.params.get("op")
    if op != "add":
        raise GenError("gen.op", f"{what}: op {op!r}; the Accumulator adds")
    (a,), (out,) = _ports(inst, 1, 1, "the Accumulator", what)
    if a.rate != 1 or not isinstance(out.rate, str):
        raise GenError(
            "gen.rate",
            f"{what}: rates {a.rate!r} in and {out.rate!r} out; the Accumulator takes 1 in and "
            "a named rate out (one sum per T inputs)",
        )
    bits, signed = _dtype(inst, dtypes, what)
    _lanes(inst, {a.name: lanes, out.name: 1}, what)
    if bits % 64:
        raise GenError("gen.lanes", f"{what}: 1 lane of {bits} bits is not whole 64-bit channels")
    csrs = [(c["field"], c["value"]) for c in b["csrs"]]
    if csrs != [("loopCount", "n"), ("len", out.rate)]:
        raise GenError(
            "gen.fields",
            f"{what}: csrs {csrs}; the Accumulator's are loopCount = n, then len = {out.rate}",
        )
    return Unit(
        inputs=[a.name], outputs=[out.name], lanes=lanes, data_width=bits, op=ELEMENTWISE_OP["add"],
        signed=signed, control="counted", wait="pipelined", latency=1, ii=1,
        fields=["loopCount", "len"], config=["count", "rate"],
    )  # fmt: skip


register_block("Accumulator", accumulator)
