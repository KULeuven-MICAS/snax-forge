"""The data side of the ideal estimate (D151, C§19): where each container lives, what it
occupies and what moves, per container, per outer port and per tailed split.

    containers(graph) -> dict[str, ContainerData]
    ports(graph) -> dict[str, PortData]
    splits(graph) -> dict[str, SplitData]

**Placement**: ``link`` (with its depth) when a link node names the container, ``local`` when
it is a fused nest's local value (``loop.local``), else ``l1``. **Bytes**: the padded shape
(D144, dfg/pad.py) times the itemsize. **Traffic**, in 64-bit words: a bound outer port moves
beats · num_ch per task, beats from the firing loops its streamer runs over
(design/streamers.py's ``port_loops``), num_ch = ceil(lanes / num_slc) with num_slc the
elements of its container's dtype in a 64-bit word (as the DFG viewer's ``packing``); a
tasklet no unit runs moves one element per memlet per iteration of every map around it,
ceil(iterations · b / 64) words (b the dtype's bits). A link's and a local value's traffic is
not L1 traffic: none is counted.

**Splits**: per map whose split left a tail (``loop.split.valid`` n), its W (the inner map's
length, or once bound the lanes, dfg/pad.py's ``split_factor``) and lane use
n / (ceil(n / W) · W).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any

import numpy as np

from snax_forge import expr
from snax_forge.design.streamers import (
    firing_loops,
    instance_of,
    linked_ports,
    links,
    port_loops,
    slices_per_word,
    unplaced,
)
from snax_forge.dfg import Graph, Node
from snax_forge.dfg.pad import padded_shape, split_factor, tailed_split
from snax_forge.dfg.subset import map_range

WORD_BITS = 64


@dataclass
class ContainerData:
    placement: str
    bytes: int
    reads: int = 0
    writes: int = 0
    depth: int | None = None  # a link's

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"placement": self.placement}
        if self.depth is not None:
            d["depth"] = self.depth
        return d | {"bytes": self.bytes, "reads": self.reads, "writes": self.writes}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContainerData:
        return cls(d["placement"], d["bytes"], d["reads"], d["writes"], d.get("depth"))


@dataclass
class PortData:
    container: str
    loops: list[tuple[str, int]] = field(default_factory=list)
    beats: int = 0
    num_ch: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"container": self.container, "loops": [list(lp) for lp in self.loops],
                "beats": self.beats, "num_ch": self.num_ch}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PortData:
        return cls(d["container"], [(v, c) for v, c in d["loops"]], d["beats"], d["num_ch"])


@dataclass
class SplitData:
    n: int
    w: int

    @property
    def lane_use(self) -> float:
        return self.n / (-(-self.n // self.w) * self.w)

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "w": self.w, "lane_use": self.lane_use}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SplitData:
        return cls(d["n"], d["w"])


def _bits(graph: Graph, c: str) -> int:
    return np.dtype(graph.containers[c].dtype).itemsize * 8


def _num_ch(graph: Graph, c: str, lanes: int) -> int:
    return -(-lanes // slices_per_word(graph.containers[c].dtype, WORD_BITS))


def outer_ports(graph: Graph) -> list[tuple[Node, str, PortData]]:
    """Every bound node's ports on no link, each with what it moves in one task."""
    linked = linked_ports(graph)
    out = []
    for n, _ in graph.walk():
        if n.kind != "accelerated":
            continue
        inst, loops = instance_of(n), firing_loops(graph, n)
        for p in inst.brm.interface.ports:
            if f"{n.attrs['instance']}.{p.name}" in linked:
                continue
            m = (n.inputs | n.outputs)[p.name]
            mine = port_loops(inst, p.name, m, loops)
            out.append((n, p.name, PortData(m.data, [(lp.var, lp.count) for lp in mine],
                                            prod(lp.count for lp in mine),
                                            _num_ch(graph, m.data, inst.lanes(p.name)))))  # fmt: skip
    return out


def containers(graph: Graph) -> dict[str, ContainerData]:
    """Every container's placement, bytes and L1 traffic (module doc)."""
    linked = {k.attrs["container"]: k.attrs["depth"] for k in links(graph)}
    out = {}
    for c, cont in graph.containers.items():
        size = prod(padded_shape(graph, c)) * np.dtype(cont.dtype).itemsize
        if c in linked:
            out[c] = ContainerData("link", size, depth=linked[c])
        else:
            out[c] = ContainerData("local" if "loop.local" in cont.attrs else "l1", size)
    off_l1 = unplaced(graph)
    for n, name, port in outer_ports(graph):
        words = port.beats * port.num_ch
        if name in n.inputs:
            out[port.container].reads += words
        else:
            out[port.container].writes += words
    symbols = graph.bound_symbols
    for n, outer in graph.walk():
        if n.kind != "tasklet":
            continue
        iterations = prod(len(map_range(m.attrs["range"], symbols, m.id)) for m in outer
                          if m.kind == "map")  # fmt: skip
        for side, memlets in (("reads", n.inputs), ("writes", n.outputs)):
            for m in memlets.values():
                if m.data not in off_l1:
                    words = -(-iterations * _bits(graph, m.data) // WORD_BITS)
                    setattr(out[m.data], side, getattr(out[m.data], side) + words)
    return out


def ports(graph: Graph) -> dict[str, PortData]:
    """Per outer port, ``<instance>.<port>``: its first node's loops, beats and num_ch."""
    out: dict[str, PortData] = {}
    for n, name, port in outer_ports(graph):
        out.setdefault(f"{n.attrs['instance']}.{name}", port)
    return out


def splits(graph: Graph) -> dict[str, SplitData]:
    """Per map with a tail, its n and W (module doc)."""
    out = {}
    for n, _ in graph.walk():
        split = tailed_split(n)
        if split is None:
            continue
        kids = n.body or []
        if len(kids) == 1 and kids[0].kind == "map":
            w = len(map_range(kids[0].attrs["range"], graph.bound_symbols, kids[0].id))
        else:
            w = split_factor(n)
        out[n.id] = SplitData(int(expr.evaluate(split["valid"], graph.bound_symbols)), w)
    return out
