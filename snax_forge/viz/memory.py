"""A run's memory layout for the memory tab: where each region lies, as folded rows of words.

``layout_of`` places the run's regions (run/regions.py): L1 as bank columns by rows, through the
cluster's word-interleaved map; L2 as flat rows of one DMA beat. A cell is one word, a list of
``[region, flat]`` pairs, one per element whose first byte is in it (``flat`` its row-major
index in the region's shape). Padding is apart: a tail's pad element p (row-major in the
``padded_shape``) is ``[region, -2 - p]``, a word of beat padding ``[region, -1]``.

``fold`` sends a run of alike rows (the same regions per column, each flat index moving by one
common step) as its first row, a ``fold`` line and its last row, and a run of empty rows as one
``empty`` line; ``rows`` serves hidden rows on request, at most ``MAX_ROWS`` at a time. The work
grows with the elements and the occupied rows, never with the memory's size. A mark of the run
index (index.py) adds a value per word: a count (``conflicts``) folds only on equal values, a
time (``arrival``, ``use``, ``wait``) only when every word's time moves by one common step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from snax_forge.run import Cluster, Padding, Region, padding_of

MAX_ROWS = 256  # rows per request

Cell = list[tuple[int, int]]  # (region, flat) per element of the word
BEAT_PAD = -1  # the flat of a word of a region's beat padding (module doc)


def tail_pad(p: int) -> int:
    """The flat of pad ``p`` (row-major in the padded shape), and back (module doc)."""
    return -2 - p


class MemoryViewError(ValueError):
    """A request the memory layout cannot answer; the message says why."""


def unravel(flat: int, shape: Sequence[int]) -> tuple[int, ...]:
    """A row-major flat index as an index tuple of ``shape``."""
    out = []
    for e in reversed(shape):
        out.append(flat % e)
        flat //= e
    return tuple(reversed(out))


def element_name(container: str, shape: Sequence[int], flat: int) -> str:
    """Element ``flat`` of a container by name: ``A[5]``, ``M[1,2]``."""
    return f"{container}[{','.join(str(i) for i in unravel(flat, shape))}]"


@dataclass
class RowMarks:
    """A mark placed on one memory's rows: per row one value per column."""

    kind: str  # count or time (module doc)
    rows: dict[int, list[Any]] = field(default_factory=dict)

    @property
    def default(self) -> int | None:
        return 0 if self.kind == "count" else None

    def row(self, r: int, columns: int) -> list[Any]:
        return list(self.rows.get(r) or [self.default] * columns)


@dataclass
class MemoryLayout:
    """One memory of a run: its geometry, its regions and the words they occupy."""

    mem: str
    columns: int
    rows: int
    base_addr: int
    word_bytes: int
    size_bytes: int
    group: int  # columns per group: a superbank in L1, one beat in L2
    column_label: str
    row_label: str
    regions: list[Region]
    padding: list[Padding] = field(default_factory=list)  # per region (module doc)
    occupied: dict[int, list[Cell]] = field(default_factory=dict)  # row -> cells
    bank_row: Callable[[int], tuple[int, int]] | None = None  # L1's map (Cluster.bank_row)

    def holds(self, addr: int) -> bool:
        return self.base_addr <= addr < self.base_addr + self.size_bytes

    def place(self, addr: int) -> tuple[int, int]:
        """Byte address -> (column, row). In L1 a column is a bank and a row the bank's word,
        word-interleaved (``Cluster.bank_row``); in L2 a row is a DMA beat."""
        if self.bank_row is not None:
            return self.bank_row(addr)
        word = (addr - self.base_addr) // self.word_bytes
        return word % self.columns, word // self.columns

    def addr(self, column: int, row: int) -> int:
        return self.base_addr + (row * self.columns + column) * self.word_bytes

    def cells(self, row: int) -> list[Cell]:
        return self.occupied.get(row) or [[] for _ in range(self.columns)]

    def row_line(self, row: int, marks: RowMarks | None = None) -> dict[str, Any]:
        line = {
            "kind": "row",
            "row": row,
            "addr": [self.addr(c, row) for c in range(self.columns)],
            "cells": [[list(p) for p in cell] for cell in self.cells(row)],
        }
        if marks is not None:
            line["marks"] = marks.row(row, self.columns)
        return line


def layout_of(mem: str, cluster: Cluster, point: Any, regions: list[Region]) -> MemoryLayout:
    """The layout of ``mem`` (``l1`` or ``l2``) with the regions that live in it (module doc);
    ``point`` is the run's DesignPoint (its platform's L2)."""
    mine = [r for r in regions if r.memory == mem]
    word = cluster.bank_bytes
    if mem == "l1":
        lay = MemoryLayout(
            "l1", cluster.banks, cluster.l1_size // (word * cluster.banks), cluster.l1_base, word,
            cluster.l1_size, cluster.dma_beat_bytes // word,
            "bank", "row", mine, bank_row=cluster.bank_row,
        )  # fmt: skip
    elif mem == "l2":  # the caller lists l2 only for a run with one (api.memories)
        l2 = point.platform.l2
        per_beat = cluster.dma_beat_bytes // word
        lay = MemoryLayout(
            "l2", per_beat, l2.size_bytes // cluster.dma_beat_bytes, l2.base_addr, word,
            l2.size_bytes, per_beat, "word", "beat", mine,
        )  # fmt: skip
    else:
        raise MemoryViewError(f"the run has no memory {mem!r}")
    lay.padding = [padding_of(point, r, cluster.dma_beat_bytes) for r in mine]
    _occupy(lay)
    return lay


def _occupy(lay: MemoryLayout) -> None:
    """Fill ``occupied`` from the regions: every element at its word, in byte order."""
    words: dict[int, list[tuple[int, int, int]]] = {}  # word address -> (byte, region, flat)
    for ri, r in enumerate(lay.regions):
        idx = np.indices(r.shape).reshape(len(r.shape), -1).T
        for flat, a in enumerate(r.layout.address(idx).tolist()):
            if not lay.holds(a):
                raise MemoryViewError(f"{r.container} in {lay.mem}: element {flat} at {a:#x} is "
                                      f"outside [{lay.base_addr:#x}, "
                                      f"{lay.base_addr + lay.size_bytes:#x})")  # fmt: skip
            w = a - (a - lay.base_addr) % lay.word_bytes
            words.setdefault(w, []).append((a, ri, flat))
        for p, a in lay.padding[ri].pads if ri < len(lay.padding) else ():
            w = a - (a - lay.base_addr) % lay.word_bytes
            words.setdefault(w, []).append((a, ri, tail_pad(p)))
        pad = lay.padding[ri].beat_pad if ri < len(lay.padding) else None
        if pad is not None:  # the beat padding's words that hold no element
            first = pad[0] + (-(pad[0] - lay.base_addr)) % lay.word_bytes
            for w in range(first, pad[1], lay.word_bytes):
                if lay.holds(w):
                    words.setdefault(w, []).append((w, ri, BEAT_PAD))
    for w, elems in words.items():
        col, row = lay.place(w)
        cells = lay.occupied.setdefault(row, [[] for _ in range(lay.columns)])
        cells[col] = [(ri, flat) for _, ri, flat in sorted(elems)]


def row_marks(lay: MemoryLayout, marks: dict[tuple[str, int], Any], kind: str) -> RowMarks:
    """A mark of the index (``{(mem, addr): value}``) on this memory's rows; words outside the
    memory are left out."""
    out = RowMarks(kind)
    for (mem, addr), v in marks.items():
        if mem != lay.mem or not lay.holds(addr):
            continue
        col, row = lay.place(addr)
        out.rows.setdefault(row, [out.default] * lay.columns)[col] = v
    return out


def _steps(a: list[Cell], b: list[Cell]) -> dict[int, int] | None:
    """The step per region from row ``a`` to row ``b``, or None if they are not alike."""
    steps: dict[int, int] = {}
    for ca, cb in zip(a, b, strict=True):
        if len(ca) != len(cb):
            return None
        for (ra, fa), (rb, fb) in zip(ca, cb, strict=True):
            if ra != rb or steps.setdefault(ra, fb - fa) != fb - fa:
                return None
    return steps


def _time_step(a: list[int | None], b: list[int | None]) -> tuple[bool, int | None]:
    """Whether rows with times ``a`` and ``b`` are alike, and their step (None: no times)."""
    d = None
    for x, y in zip(a, b, strict=True):
        if (x is None) != (y is None):
            return False, None
        if x is not None and y is not None:
            if d is None:
                d = y - x
            elif y - x != d:
                return False, None
    return True, d


def fold(lay: MemoryLayout, marks: RowMarks | None = None) -> list[dict[str, Any]]:
    """The folded lines of a memory (module doc), top row first."""
    lines: list[dict[str, Any]] = []
    occupied = sorted(lay.occupied)
    times = marks is not None and marks.kind == "time"

    def mark(r: int) -> list[Any]:
        return [] if marks is None else marks.row(r, lay.columns)

    row = 0
    i = 0
    while row < lay.rows:
        if i >= len(occupied) or occupied[i] > row:  # empty up to the next occupied row
            end = occupied[i] - 1 if i < len(occupied) else lay.rows - 1
            lines.append({"kind": "empty", "from": row, "to": end})
            row = end + 1
            continue
        # a run of occupied rows starting at `row`
        first, steps = row, None
        tstep: list[int | None] = []  # the run's time step, once two rows set it
        while i + 1 < len(occupied) and occupied[i + 1] == row + 1:
            s = _steps(lay.occupied[row], lay.occupied[row + 1])
            if s is None or (steps is not None and s != steps):
                break
            if times:
                ok, d = _time_step(mark(row), mark(row + 1))
                if not ok or (tstep and d != tstep[0]):
                    break
                tstep = [d]
            elif mark(row) != mark(row + 1):
                break
            steps, row, i = s, row + 1, i + 1
        lines.append(lay.row_line(first, marks))
        if row - first >= 2:
            line = _fold_line(lay, first + 1, row - 1, steps or {})
            hidden = [x for r in range(first + 1, row) for x in mark(r) if x is not None]
            if times:
                line["mark_step"] = tstep[0] if tstep else None
                line["mark_span"] = [min(hidden), max(hidden)] if hidden else None
            elif marks is not None:
                line["marked"] = sum(hidden)
            lines.append(line)
        if row > first:
            lines.append(lay.row_line(row, marks))
        row, i = row + 1, i + 1
    return lines


def _fold_line(lay: MemoryLayout, a: int, b: int, steps: dict[int, int]) -> dict[str, Any]:
    span: dict[int, list[int]] = {}
    for r in (a, b):  # with one step per region the extremes are in the end rows
        for cell in lay.occupied[r]:
            for ri, flat in cell:
                if flat < 0:  # padding (module doc)
                    continue
                lo, hi = span.setdefault(ri, [flat, flat])
                span[ri] = [min(lo, flat), max(hi, flat)]
    return {
        "kind": "fold",
        "from": a,
        "to": b,
        "hidden": b - a + 1,
        "step": {str(k): v for k, v in sorted(steps.items())},
        "span": {str(k): v for k, v in sorted(span.items())},
    }


def summary(lay: MemoryLayout, marks: RowMarks | None = None) -> dict[str, Any]:
    """Geometry, regions (extent, bytes, share, columns and rows touched) and use."""
    regions = []
    touched: dict[int, tuple[set[int], set[int]]] = {
        i: (set(), set()) for i in range(len(lay.regions))
    }
    used = 0
    for row, cells in lay.occupied.items():
        for col, cell in enumerate(cells):
            used += bool(cell)
            for ri, _ in cell:
                touched[ri][0].add(col)
                touched[ri][1].add(row)
    for ri, r in enumerate(lay.regions):
        lo, _ = r.span()
        pad = lay.padding[ri]
        cols, rows = touched[ri]
        n = int(np.prod(r.shape))
        regions.append({
            "name": r.container, "shape": list(r.shape), "strides": list(r.strides),
            "start": lo, "end": pad.end, "bytes": n * r.itemsize,
            lay.column_label + "s": sorted(cols), "rows": [min(rows), max(rows)],
            "data_end": pad.data_end, "padded_shape": list(pad.padded_shape),
            "pad_elements": len(pad.pads),
            "beat_pad": list(pad.beat_pad) if pad.beat_pad else None,
        })  # fmt: skip
    n_words = lay.columns * lay.rows
    geometry = {
        "columns": lay.columns, "rows": lay.rows, "column_label": lay.column_label,
        "row_label": lay.row_label, "group": lay.group, "base_addr": lay.base_addr,
        "word_bytes": lay.word_bytes, "size_bytes": lay.size_bytes,
    }  # fmt: skip
    out: dict[str, Any] = {
        "mem": lay.mem,
        "geometry": geometry,
        "regions": regions,
        "used": {"words": used, "share": used / n_words},
        "lines": fold(lay, marks),
    }
    if marks is not None:
        vals = [x for row in marks.rows.values() for x in row if x is not None]
        out["mark_range"] = [min(vals), max(vals)] if vals else None
    return out


def rows(lay: MemoryLayout, start: int, stop: int, marks: RowMarks | None = None
         ) -> list[dict[str, Any]]:  # fmt: skip
    """Rows ``start <= r < stop`` in full, at most MAX_ROWS of them."""
    if not 0 <= start < stop <= lay.rows:
        raise MemoryViewError(f"rows [{start}, {stop}) are not inside [0, {lay.rows})")
    if stop - start > MAX_ROWS:
        raise MemoryViewError(f"at most {MAX_ROWS} rows per request, asked {stop - start}")
    return [lay.row_line(r, marks) for r in range(start, stop)]


__all__ = [
    "MAX_ROWS",
    "MemoryLayout",
    "MemoryViewError",
    "RowMarks",
    "element_name",
    "fold",
    "layout_of",
    "row_marks",
    "rows",
    "summary",
    "unravel",
]
