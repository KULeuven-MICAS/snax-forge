"""A run's regions (C§7, D137): where each container lives in each memory, so an address on
an event can be named.

    regions_of(point, elf_symbols) -> [Region]
    padding_of(point, region, beat_bytes) -> Padding

A region is a container's layout in one memory: ``container``, ``memory`` (``l1``, ``l2``),
``base``, ``shape``, ``strides`` (bytes) and the container's ``dtype``. Both memories are known
before the run, nothing is inferred from it (D86): in L1 the design point's memory plan gives
the layout; in L2 the plan gives the shape and the program's link the base, the address of the
container's array in the ELF (``run.json``'s ``elf_symbols``, elf.py), the strides the plan's
(contiguous, as data.h declares it). Without ``elf_symbols`` there are no L2 regions. ``Region.elems(addr, size)`` gives the flat indexes (row-major over
``shape``) of the elements whose first byte lies in ``[addr, addr + size)``, and
``Region.vals(addr, data)`` their values, read from the bytes ``data`` that start at
``addr`` by the region's dtype. ``region_at`` finds the region an address is in. Regions come
in the design point's order (its containers, each memory as its plan lists them). The
report's memory tables and notes read the same regions (``report/design.py``).

``padding_of`` says what of a region is padding (D144), as the memory checks count it
(design/check.py): the plan lays a container out over its padded shape (``dfg/pad.py``'s
``padded_shape``, ``lower.layout.over``), so the elements of the padded shape outside the
declared one are pads, each at its address (``pads``: its row-major index in the padded
shape and its address); and for a container the DMA moves (it lives in L1 and L2), the bytes
past the padded block up to a whole DMA beat (``lower.layout.moved_bytes``) are kept free
(``beat_pad``). ``data_end`` is the padded block's end, ``end`` the extent the checks use.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from snax_forge.lower.layout import Layout, moved_bytes, over


@dataclass(frozen=True)
class Region:
    container: str
    memory: str
    base: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str

    @property
    def layout(self) -> Layout:
        return Layout(self.base, self.shape, self.strides)

    @property
    def itemsize(self) -> int:
        return np.dtype(self.dtype).itemsize

    @cached_property
    def _index(self) -> dict[int, int]:
        """Each element's address -> its flat index."""
        idx = np.indices(self.shape).reshape(len(self.shape), -1).T
        return {int(a): i for i, a in enumerate(self.layout.address(idx))}

    def span(self) -> tuple[int, int]:
        """``[lo, hi)``: from its lowest element's first byte to past its highest's last."""
        lo, hi = self.layout.span()
        return lo, hi + self.itemsize

    def contains(self, addr: int) -> bool:
        lo, hi = self.span()
        return lo <= addr < hi

    def elems(self, addr: int, size: int) -> list[int]:
        """The flat indexes of the elements whose first byte is in ``[addr, addr + size)``."""
        return [self._index[a] for a in range(addr, addr + size) if a in self._index]

    def vals(self, addr: int, data: bytes) -> list[Any]:
        """Those elements' values, read from ``data`` (the bytes from ``addr``) by dtype."""
        out = []
        for a in range(addr, addr + len(data)):
            if a in self._index and a - addr + self.itemsize <= len(data):
                raw = data[a - addr : a - addr + self.itemsize]
                out.append(np.frombuffer(raw, dtype=self.dtype)[0].item())
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"container": self.container, "memory": self.memory, "base": self.base,
                "shape": list(self.shape), "strides": list(self.strides), "dtype": self.dtype}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Region:
        return cls(d["container"], d["memory"], d["base"], tuple(d["shape"]),
                   tuple(d["strides"]), d["dtype"])  # fmt: skip


def regions_of(point: Any, elf_symbols: dict[str, Any] | None) -> list[Region]:
    """Every container's region in every memory of a run (module doc): a design point
    (``DesignPoint``) and the ELF's symbols of its L2 containers (container -> ``addr``,
    ``size``), in the point's order."""
    containers = point.graph.containers
    out = []
    for name, lays in point.memory.layouts.items():
        for mem, lay in lays.items():
            if mem == "l2":
                sym = (elf_symbols or {}).get(name)
                if sym is None:
                    continue
                base = sym["addr"]
            else:
                base = lay.base
            out.append(Region(name, mem, base, tuple(lay.shape), tuple(lay.strides),
                              containers[name].dtype))  # fmt: skip
    return out


def region_at(regions: list[Region], memory: str, addr: int) -> Region | None:
    """The region of ``memory`` that ``addr`` is in, or None."""
    for r in regions:
        if r.memory == memory and r.contains(addr):
            return r
    return None


@dataclass(frozen=True)
class Padding:
    """What of a region is padding (module doc)."""

    valid_shape: tuple[int, ...]
    padded_shape: tuple[int, ...]
    pads: tuple[tuple[int, int], ...]  # (row-major index in padded_shape, address) per pad
    data_end: int
    beat_pad: tuple[int, int] | None  # [from, to) bytes the DMA's whole beats add

    @property
    def end(self) -> int:
        return self.beat_pad[1] if self.beat_pad else self.data_end


def padding_of(point: Any, region: Region, beat_bytes: int) -> Padding:
    """The padding of ``region``, a region of ``point`` (a DesignPoint; module doc). Without
    the point's graph, or the container in it, the region has none."""
    from snax_forge.dfg.pad import padded_shape

    shape = tuple(region.shape)
    graph = getattr(point, "graph", None)
    if graph is None or region.container not in graph.containers:
        return Padding(shape, shape, (), region.span()[1], None)
    padded = padded_shape(graph, region.container)
    block = over(region.layout, padded)
    pads: tuple[tuple[int, int], ...] = ()
    if padded != shape:
        idx = np.indices(padded).reshape(len(padded), -1).T
        out = (idx >= np.array(shape)).any(axis=1)
        flats = np.nonzero(out)[0]
        pads = tuple(zip(flats.tolist(), block.address(idx[out]).tolist(), strict=True))
    lo, hi = block.extent(region.itemsize)
    beat_pad = None
    if {"l1", "l2"} <= set(point.memory.layouts.get(region.container, {})):
        end = lo + moved_bytes(block, region.itemsize, beat_bytes)
        if end > hi:
            beat_pad = (hi, end)
    return Padding(shape, padded, pads, hi, beat_pad)
