"""A BRM port's dtypes as a rule (D143): which container dtypes a node may bind.

A port lists the NumPy integer dtypes it takes (``Port.dtypes``, all eight
by default); the dtype a node puts on a port is its container's, read through
the port's memlet (``port_dtypes``), never stored in the graph. A node's
dtypes pass ``check_dtypes`` when

- each port's dtype is one its ``dtypes`` lists;
- every output is at least as wide as every input (the op runs at the
  inputs' result type and is extended into the output, D143);
- no signed input meets a ``uint64`` input: NumPy's result type of the two
  is float64, and SnaxUnit refuses it (snax-gvsoc ``snax/snax_unit.hpp``).

``output_dtypes`` lists, per output port, the dtypes it accepts for given
input dtypes: what a refusal's fix names. ``result_type`` is the type an op
runs at: NumPy's result type of its inputs (the Chisel blocks' ``opType``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from snax_forge.dfg import Graph, Node

# Every dtype a port may take, narrowest first, the signed one of a width first
INT_DTYPES = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")


def bits(dtype: str) -> int:
    return np.dtype(dtype).itemsize * 8


def result_type(dtypes: Iterable[str]) -> str:
    """NumPy's result type of integer ``dtypes``, the type an op on them runs at (D143);
    ValueError when it is not an integer (a signed dtype with uint64)."""
    dts = list(dtypes)
    out = np.result_type(*dts).name
    if out not in INT_DTYPES:
        raise ValueError(f"{dts} have no integer result type (NumPy gives {out})")
    return out


def port_dtypes(graph: Graph, node: Node) -> dict[str, str]:
    """The dtype on each connector of ``node``: its memlet's container's."""
    return {c: graph.containers[m.data].dtype for c, m in (node.inputs | node.outputs).items()}


def _signed_with_uint64(dtypes: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Two inputs, a signed one and a uint64 one, as ``(signed, uint64)`` ports; or None."""
    signed = [p for p, d in dtypes if d in INT_DTYPES and d.startswith("int")]
    wide = [p for p, d in dtypes if d == "uint64"]
    return (signed[0], wide[0]) if signed and wide else None


def check_dtypes(brm: Any, dtypes: Mapping[str, str]) -> list[str]:
    """Why ``dtypes`` (port -> dtype) do not fit ``brm``'s ports; empty when they do."""
    ports = {p.name: p for p in brm.interface.ports}
    out = []
    for name, dt in dtypes.items():
        if name in ports and dt not in ports[name].dtypes:
            out.append(f"port {name} is {dt}; {brm.name} takes {list(ports[name].dtypes)} there")
    ins = [(n, d) for n, d in dtypes.items() if n in ports and ports[n].direction == "in"]
    outs = [(n, d) for n, d in dtypes.items() if n in ports and ports[n].direction == "out"]
    ints = [(n, d) for n, d in ins if d in INT_DTYPES]
    for o, do in outs:
        if do not in INT_DTYPES:
            continue
        for i, di in ints:
            if bits(do) < bits(di):
                out.append(
                    f"output {o} is {do}, narrower than input {i} ({di}): an output is at "
                    "least as wide as every input (D143)"
                )
    pair = _signed_with_uint64(ints)
    if pair is not None:
        s, u = pair
        out.append(
            f"inputs {s} ({dtypes[s]}) and {u} (uint64): a signed input with a uint64 one has no "
            "integer result type (NumPy gives float64)"
        )
    return out


def output_dtypes(brm: Any, inputs: Mapping[str, str]) -> dict[str, list[str]]:
    """Per output port, the dtypes it accepts with these input dtypes (empty when the inputs
    themselves cannot be taken)."""
    ports = [p for p in brm.interface.ports]
    ins = [(p.name, inputs[p.name]) for p in ports if p.direction == "in" and p.name in inputs]
    if any(not ok for ok in (_ok_input(brm, n, d) for n, d in ins)) or _signed_with_uint64(ins):
        return {p.name: [] for p in ports if p.direction == "out"}
    widest = max((bits(d) for _, d in ins), default=0)
    return {
        p.name: [d for d in p.dtypes if bits(d) >= widest] for p in ports if p.direction == "out"
    }


def _ok_input(brm: Any, port: str, dtype: str) -> bool:
    return dtype in brm.port(port).dtypes


def accepted(brm: Any, dtypes: Mapping[str, str]) -> str:
    """The accepted output dtypes for ``dtypes``' inputs, as a refusal's fix: ``out: int64,
    uint64``, one output after another."""
    ins = {n: d for n, d in dtypes.items() if brm.port(n).direction == "in"}
    outs = output_dtypes(brm, ins)
    return "; ".join(f"{o}: {', '.join(ds) if ds else 'none'}" for o, ds in outs.items())
