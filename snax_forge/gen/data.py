"""A design point -> the program's data, ``data.h``.

    arrays(point, seed=0, kernel=None) -> list[Array]
    data_h(point, point_path, data=None, seed=0) -> str

Every container the DMA moves (it lives in L2 and L1), in the graph's
container order, as one C array or two:

    input    the kernel's ``make_inputs`` at the size the graph binds,
             seeded (the flow's own path, ``flow.run.kernel_inputs``),
             ``const``
    output   zeroed (the kernel's ``inout``), the array the DMA stores into,
             then ``<name>_golden``: the reference executor
             (``dfg/execute.py``) on the bound graph and those
             inputs, as the flow checks a run, ``const``

The C type is the container's dtype (``int64_t`` for every design so far).
Each array, golden included, is padded with zeros to whole DMA beats exactly
as the memory plan's L2 layout is (``lower/layout.py``'s ``moved_bytes``,
D105) and aligned to the beat (``l2.beat_bits``, 64 bytes on snax_forge).
main.c (lower/c.py) reads the arrays by name: a DMA's L2 side is the array its task names
(``symbol``).

The file starts with a ``//`` comment naming the design point by its path
from the root, as the other generated files do, so one design point gives
byte-identical text wherever it is written.

**Refused**, each a ``GenError``: a dtype that is not intN or uintN
(``gen.dtype``), an L2 layout that is not the array's contiguous elements
(``gen.layout``), a graph named after no kernel, an input ``make_inputs`` does
not give or a moved container that is neither an input nor an output
(``gen.inputs``), and an output whose
``make_inputs`` value is not zero (``gen.inout``: an in-place update, open
item 47).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from snax_forge.common import generated_header
from snax_forge.design.point import DesignPoint
from snax_forge.dfg.execute import execute
from snax_forge.flow.run import FlowError, kernel_inputs, kernel_spec
from snax_forge.lower.layout import Layout, moved_bytes

from .blocks import GenError, port_dtype

DATA_H = "data.h"
# Values per line of an array's initialiser
PER_LINE = 8


@dataclass(frozen=True)
class Array:
    """One C array of data.h: ``role`` is ``input``, ``output`` or ``golden``; ``values`` are
    padded to ``len(values)``, the container's own ``elements`` first."""

    name: str
    container: str
    role: str
    ctype: str
    elements: int
    values: tuple[int, ...]
    align: int

    @property
    def const(self) -> bool:
        return self.role != "output"


def ctype(dtype: str, where: str) -> str:
    """The C type of an intN or uintN dtype."""
    bits, signed = port_dtype(dtype, where)
    return f"{'' if signed else 'u'}int{bits}_t"


def moved(point: DesignPoint) -> list[str]:
    """The containers the DMA moves (an L2 and an L1 layout), in the graph's order."""
    lays = point.memory.layouts
    return [c for c in point.graph.containers if {"l1", "l2"} <= set(lays.get(c, {}))]


def arrays(point: DesignPoint, seed: int = 0, kernel: str | None = None) -> list[Array]:
    """data.h's arrays; see the module doc. ``kernel`` names the kernel when the graph's name
    does not (the flow's ``--kernel``)."""
    graph = point.graph
    try:
        spec = kernel_spec(kernel or graph.name, "--kernel" if kernel else "the graph's name")
        inputs = kernel_inputs(spec, graph, seed)
    except FlowError as e:
        raise GenError("gen.inputs", str(e)) from None
    golden = execute(graph, {k: np.copy(v) for k, v in inputs.items()})
    beat = point.platform.l2.beat_bits // 8
    out = []
    for c in moved(point):
        cont = graph.containers[c]
        t = ctype(cont.dtype, f"container {c}")
        size = np.dtype(cont.dtype).itemsize
        lay = point.memory.layouts[c]["l2"]
        if lay != Layout.contiguous(lay.base, lay.shape, size):
            raise GenError(
                "gen.layout", f"{c}'s L2 layout {lay} is not its {size}-byte elements in order"
            )
        padded = moved_bytes(lay, size, beat) // size
        if c in spec.inout:
            if c in inputs and np.any(inputs[c]):
                raise GenError(
                    "gen.inout", f"output {c} starts nonzero in {spec.name}'s make_inputs: an "
                    "in-place update (open item 47)",
                )  # fmt: skip
            out.append(_array(c, c, "output", t, np.zeros(lay.shape, cont.dtype), padded, beat))
            out.append(_array(f"{c}_golden", c, "golden", t, golden[c], padded, beat))
        elif c in inputs:
            out.append(_array(c, c, "input", t, inputs[c], padded, beat))
        else:
            raise GenError(
                "gen.inputs", f"{c} is moved by the DMA but is neither given by {spec.name}'s "
                "make_inputs nor one of its outputs",
            )  # fmt: skip
    return out


def _array(
    name: str, c: str, role: str, t: str, values: np.ndarray, padded: int, align: int
) -> Array:
    flat = [int(v) for v in np.asarray(values).reshape(-1)]
    return Array(name, c, role, t, len(flat), tuple(flat + [0] * (padded - len(flat))), align)


def literal(v: int, t: str) -> str:
    """``v`` as a C literal of type ``t`` (the most negative int64 has none of its own)."""
    if t == "int64_t" and v == -(2**63):
        return "(-INT64_C(9223372036854775807) - 1)"
    if t == "uint64_t":
        return f"UINT64_C({v})"
    if t == "int64_t":
        return f"INT64_C({v})" if abs(v) > 2**31 - 1 else str(v)
    return str(v)


def array_text(a: Array) -> str:
    n = f"{a.elements} element{'s' if a.elements != 1 else ''}"
    what = f"// {a.container}: {a.role}, {n}, {len(a.values)} padded to beats"
    head = f"{'const ' if a.const else ''}{a.ctype} {a.name}[{len(a.values)}] "
    head += f"__attribute__((aligned({a.align})))"
    if a.role == "output":
        return f"{what}\n{head} = {{0}};\n"
    vals = [literal(v, a.ctype) for v in a.values]
    lines = [", ".join(vals[i : i + PER_LINE]) for i in range(0, len(vals), PER_LINE)]
    return f"{what}\n{head} = {{\n" + "".join(f"    {x},\n" for x in lines) + "};\n"


def data_h(point: DesignPoint, point_path: str, data: Sequence[Array] | None = None,
           seed: int = 0) -> str:  # fmt: skip
    """The text of data.h (``data`` the arrays when already made); see the module doc."""
    head = generated_header(
        point_path,
        f"main.c's data: the inputs from make_inputs (seed {seed}), the outputs zeroed and "
        "their goldens from the reference executor, each padded to whole DMA beats and "
        "aligned to one.",
    )
    data = arrays(point, seed) if data is None else data
    return head + "\n#include <stdint.h>\n\n" + "\n".join(array_text(a) for a in data)
