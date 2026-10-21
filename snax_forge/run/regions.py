"""A run's regions (C§7, D137): where each container lives in each memory, so an address on
an event can be named.

    regions_of(point, elf_symbols) -> [Region]

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
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from snax_forge.lower.layout import Layout


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
