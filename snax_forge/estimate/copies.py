"""The copies a graph makes outside its accelerators: data_transpose's ``transpose`` nodes.

``copies(graph)`` gives a ``CopyEstimate`` per transpose node (CONTRACTS, "Estimate"): by the
DMA one burst per element of its input, ``DMA_CYCLES_PER_BURST`` cycles each, and two L1
accesses per burst (one read, one write) whatever the burst's size, priced at the banks'
per-access energy in the memory part (``Estimate.memory``). The DMA engine itself is not priced.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

from snax_forge import expr
from snax_forge.dfg import Graph

# The DMA's cycles per burst of a copy inside L1, one element a burst: measured on the RTL,
# busy = 8/3 · bursts + 2 (three bursts in flight, 8 cycles each three); the 2 left out.
# NB (D182): a measured rate held in one constant, not a model of the DMA; replace it when one is.
DMA_CYCLES_PER_BURST = 8 / 3


@dataclass
class CopyEstimate:
    """One copy: its node, what makes it, its bursts and their bytes, and its cycles."""

    name: str
    by: str
    bursts: int
    size: int  # bytes per burst: one element

    @property
    def cycles(self) -> float:
        return self.bursts * DMA_CYCLES_PER_BURST

    @property
    def accesses(self) -> int:
        """Its L1 accesses: one read and one write per burst."""
        return 2 * self.bursts

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "by": self.by, "bursts": self.bursts, "size": self.size,
                "cycles": round(self.cycles, 2)}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CopyEstimate:
        return cls(d["name"], d["by"], d["bursts"], d["size"])


def copies(graph: Graph) -> list[CopyEstimate]:
    """Each transpose node of ``graph`` (its symbols bound), in walk order."""
    symbols = graph.bound_symbols
    out = []
    for n, _ in graph.walk():
        if n.kind != "transpose":
            continue
        (src,) = n.inputs.values()
        c = graph.containers[src.data]
        bursts = prod(int(expr.evaluate(s, symbols)) for s in c.shape)
        out.append(CopyEstimate(n.id, n.attrs["by"], bursts, np.dtype(c.dtype).itemsize))
    return out
