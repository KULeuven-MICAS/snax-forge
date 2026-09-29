"""The memory layout of a run, for the memory tab (VIS4a, D95, D96).

What this is
------------
Where each region of a run (``run.json``'s ``regions``, D95) lies in its
memory, as rows of words, folded so that the answer grows with the regions
and not with the depth of the memory. Everything is computed here, in
Python, as the DFG viewer's rows are (D81); the viewer only draws.

Geometry
--------
L1 is ``n_banks`` columns by ``rows`` rows, a word placed through the
model's own address map (``WordInterleaved``, the only one a scenario can
have, open item 18), so the tab shows what the banks hold. L2 is flat, so
it is drawn as rows of one DMA beat (``words_per_beat`` words, 8 by
default). A cell is one word; it holds a list of ``[region, flat]`` pairs,
one per element in the word (one element per word for now, D13), where
``region`` indexes that memory's region list and ``flat`` is the element's
row-major index in its region's shape. An empty cell is ``[]``.

Folding
-------
Two neighbouring rows are *alike* when every column holds the same regions
in the same order in both, and each region's flat index moves by one step
that is the same in every column (so A[0..15] and A[16..31] are alike, with
A's step 16). A run of rows is a maximal stretch in which each row is alike
to the next with the same steps. A run of occupied rows is sent as its first
row, one fold line for the rows between (if any) and its last row; a run of
empty rows is one line, however long. Lines:

    {"kind": "row", "row": r, "addr": [...], "cells": [...]}
    {"kind": "fold", "from": a, "to": b, "hidden": b - a + 1,
     "step": {region: step}, "span": {region: [first flat, last flat]}}
    {"kind": "empty", "from": a, "to": b}

``from`` and ``to`` are inclusive. A fold's rows, or an empty run's, are
served on request (``rows``), at most ``MAX_ROWS`` per request, so a large
memory is never sent whole. Only occupied rows are visited: the cost is the
number of elements plus the occupied rows, not the size of the memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from snax_forge.snax_model.mem import L1Config, WordInterleaved
from snax_forge.snax_model.scenario import ClusterConfig, Region

MAX_ROWS = 256  # rows per request

Cell = list[tuple[int, int]]  # (region, flat) per element of the word


class MemoryViewError(ValueError):
    """A request the memory layout cannot answer; the message says why."""


@dataclass
class MemoryLayout:
    """One memory of a run: its geometry, its regions and the words they occupy."""

    mem: str
    columns: int
    rows: int
    base_addr: int
    word_bytes: int
    elems_per_word: int
    size_bytes: int
    group: int  # columns per group: a superbank in L1, one beat in L2
    column_label: str
    row_label: str
    regions: list[Region]
    occupied: dict[int, list[Cell]] = field(default_factory=dict)  # row -> cells

    @property
    def elem_bytes(self) -> int:
        return self.word_bytes // self.elems_per_word

    def place(self, word: int) -> tuple[int, int]:
        """Word index -> (column, row)."""
        if self.mem == "l1":
            return WordInterleaved(self.columns).decode(word)
        return word % self.columns, word // self.columns

    def word_of(self, column: int, row: int) -> int:
        if self.mem == "l1":
            return WordInterleaved(self.columns).encode(column, row)
        return row * self.columns + column

    def addr(self, column: int, row: int) -> int:
        return self.base_addr + self.word_of(column, row) * self.word_bytes

    def cells(self, row: int) -> list[Cell]:
        return self.occupied.get(row) or [[] for _ in range(self.columns)]

    def row_line(self, row: int) -> dict[str, Any]:
        return {
            "kind": "row",
            "row": row,
            "addr": [self.addr(c, row) for c in range(self.columns)],
            "cells": [[list(p) for p in cell] for cell in self.cells(row)],
        }


def layout_of(mem: str, cluster: ClusterConfig, regions: list[Region]) -> MemoryLayout:
    """The layout of ``mem`` (``l1`` or ``l2``) with the regions that live in it."""
    mine = [r for r in regions if r.mem == mem]
    if mem == "l1":
        c: L1Config = cluster.l1
        lay = MemoryLayout(
            "l1", c.n_banks, c.rows, c.base_addr, c.word_bytes, c.elems_per_word, c.size_bytes,
            c.wide_bits // c.width_bits, "bank", "row", mine,
        )  # fmt: skip
    elif mem == "l2" and cluster.l2 is not None:
        c2 = cluster.l2
        wpb = c2.words_per_beat
        lay = MemoryLayout(
            "l2", wpb, c2.n_words // wpb, c2.base_addr, c2.word_bytes, c2.elems_per_word,
            c2.size_bytes, wpb, "word", "beat", mine,
        )  # fmt: skip
    else:
        raise MemoryViewError(f"the run has no memory {mem!r}")
    _occupy(lay)
    return lay


def _occupy(lay: MemoryLayout) -> None:
    """Fill ``occupied`` from the regions: every element at its word, in sub-word order."""
    words: dict[int, list[tuple[int, int, int]]] = {}  # word -> (sub, region, flat)
    for ri, r in enumerate(lay.regions):
        idx = np.indices(r.shape).reshape(len(r.shape), -1).T
        off = r.base + idx @ np.asarray(r.strides, dtype=np.int64) - lay.base_addr
        for flat, o in enumerate(off.tolist()):
            w, sub = divmod(o, lay.word_bytes)
            words.setdefault(w, []).append((sub // lay.elem_bytes, ri, flat))
    for w, elems in words.items():
        col, row = lay.place(w)
        cells = lay.occupied.setdefault(row, [[] for _ in range(lay.columns)])
        cells[col] = [(ri, flat) for _, ri, flat in sorted(elems)]


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


def fold(lay: MemoryLayout) -> list[dict[str, Any]]:
    """The folded lines of a memory (module doc), top row first."""
    lines: list[dict[str, Any]] = []
    occupied = sorted(lay.occupied)
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
        while i + 1 < len(occupied) and occupied[i + 1] == row + 1:
            s = _steps(lay.occupied[row], lay.occupied[row + 1])
            if s is None or (steps is not None and s != steps):
                break
            steps, row, i = s, row + 1, i + 1
        lines.append(lay.row_line(first))
        if row - first >= 2:
            lines.append(_fold_line(lay, first + 1, row - 1, steps or {}))
        if row > first:
            lines.append(lay.row_line(row))
        row, i = row + 1, i + 1
    return lines


def _fold_line(lay: MemoryLayout, a: int, b: int, steps: dict[int, int]) -> dict[str, Any]:
    span: dict[int, list[int]] = {}
    for r in (a, b):  # with one step per region the extremes are in the end rows
        for cell in lay.occupied[r]:
            for ri, flat in cell:
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


def summary(lay: MemoryLayout) -> dict[str, Any]:
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
        lo, hi = r.span()
        cols, rows = touched[ri]
        nbytes = r.size * lay.elem_bytes
        regions.append({
            "index": ri, "name": r.name, "base": r.base, "shape": list(r.shape),
            "strides": list(r.strides), "start": lo, "end": hi + lay.elem_bytes,
            "elements": r.size, "bytes": nbytes, "share": nbytes / lay.size_bytes,
            lay.column_label + "s": sorted(cols), "rows": [min(rows), max(rows)],
        })  # fmt: skip
    n_words = lay.columns * lay.rows
    geometry = {
        "columns": lay.columns, "rows": lay.rows, "column_label": lay.column_label,
        "row_label": lay.row_label, "group": lay.group, "base_addr": lay.base_addr,
        "word_bytes": lay.word_bytes, "elem_bytes": lay.elem_bytes,
        "elems_per_word": lay.elems_per_word, "size_bytes": lay.size_bytes,
    }  # fmt: skip
    return {
        "mem": lay.mem,
        "geometry": geometry,
        "regions": regions,
        "used": {"words": used, "share": used / n_words},
        "lines": fold(lay),
    }


def rows(lay: MemoryLayout, start: int, stop: int) -> list[dict[str, Any]]:
    """Rows ``start <= r < stop`` in full, at most MAX_ROWS of them."""
    if not 0 <= start < stop <= lay.rows:
        raise MemoryViewError(f"rows [{start}, {stop}) are not inside [0, {lay.rows})")
    if stop - start > MAX_ROWS:
        raise MemoryViewError(f"at most {MAX_ROWS} rows per request, asked {stop - start}")
    return [lay.row_line(r) for r in range(start, stop)]


__all__ = ["MAX_ROWS", "MemoryLayout", "MemoryViewError", "fold", "layout_of", "rows", "summary"]
