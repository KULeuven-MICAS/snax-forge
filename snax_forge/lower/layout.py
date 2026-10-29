"""Buffer layout: where an operand's elements live in a memory (D70, D86, D87).

A layout is affine, like the nests it serves:

    addr(index) = base + sum_d index[d] * strides[d]

``base`` is the byte address of index (0, ..., 0), ``shape`` the operand's
extent per dimension, ``strides`` the bytes per step of each dimension. One
form covers storage order (row- or column-major) and padding. Choosing it is
SNAX-DSE's decision (section 5.4); SNAX-LOWER only reads it.

This is the form of the design point's memory plan (D86):
snax_forge/design/memory.py makes one per container and memory, and
SNAX-LOWER reads them. ``check`` holds the L1 rules streams.py needs; the
design checks judge a whole plan, in L1 and L2. A layout places each element
at its dtype's itemsize, so a word holds several narrow ones (D142). A
container a tail pads (dfg/pad.py, D144) keeps its logical shape, its strides
row-major over the padded shape; ``over`` gives its padded block, what the
DMA moves and what memory it takes. ``l2_bytes`` is a container's array in L2,
that block in whole DMA beats: data.h declares it so, the ELF check expects
it and the DMA's copies touch it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from snax_forge.common import Config
from snax_forge.dfg.pad import padded_shape

if TYPE_CHECKING:  # design imports lower.layout
    from snax_forge.design.platform import L1Config
    from snax_forge.design.point import DesignPoint


class LayoutError(ValueError):
    """A layout that does not fit the operand or the L1."""


@dataclass(frozen=True)
class Layout(Config):
    """``base`` + one byte stride per dimension, over an operand of ``shape``."""

    base: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(x) for x in self.shape))
        object.__setattr__(self, "strides", tuple(int(x) for x in self.strides))
        if not self.shape or len(self.shape) != len(self.strides):
            raise LayoutError(f"{len(self.shape)} extents but {len(self.strides)} strides")
        if any(e < 1 for e in self.shape):
            raise LayoutError(f"extents must be >= 1, got {list(self.shape)}")

    @classmethod
    def contiguous(cls, base: int, shape: tuple[int, ...], elem_bytes: int = 8) -> Layout:
        """Row-major (last dimension fastest), no padding."""
        strides, s = [], elem_bytes
        for e in reversed(shape):
            strides.append(s)
            s *= e
        return cls(base, tuple(shape), tuple(reversed(strides)))

    def address(self, index: np.ndarray) -> np.ndarray:
        """Byte addresses of ``index`` (last axis = dimensions)."""
        return self.base + np.asarray(index, dtype=np.int64) @ np.asarray(self.strides)

    def span(self) -> tuple[int, int]:
        """The lowest and highest byte address of an element of the operand."""
        lo = hi = self.base
        for e, s in zip(self.shape, self.strides, strict=True):
            lo += min(0, (e - 1) * s)
            hi += max(0, (e - 1) * s)
        return lo, hi

    def extent(self, elem_bytes: int) -> tuple[int, int]:
        """The bytes ``[lo, hi)`` the operand covers, each element ``elem_bytes`` bytes."""
        lo, hi = self.span()
        return lo, hi + elem_bytes

    def check(self, l1: L1Config, elem_bytes: int) -> None:
        """The layout fits ``l1``: its base on a word, its strides whole elements of
        ``elem_bytes`` (several may share a word, D142), inside L1."""
        word = l1.width_bits // 8
        if self.base % word:
            raise LayoutError(f"base {self.base} must be a multiple of the {word}-byte word")
        if any(s % elem_bytes for s in self.strides):
            raise LayoutError(
                f"strides {list(self.strides)} must be multiples of the {elem_bytes}-byte element"
            )
        lo, hi = self.extent(elem_bytes)
        start = l1.base_addr
        end = start + l1.size_bytes
        if lo < start or hi > end:
            raise LayoutError(f"the operand spans bytes [{lo}, {hi}), L1 is [{start}, {end})")


def over(lay: Layout, shape: tuple[int, ...]) -> Layout:
    """``lay``'s base and strides over ``shape``: with a container's padded shape (dfg/pad.py,
    D144), the block it takes in memory, its pads included."""
    return lay if tuple(shape) == lay.shape else Layout(lay.base, tuple(shape), lay.strides)


def round_up(x: int, a: int) -> int:
    """``x`` rounded up to a multiple of ``a``."""
    return -(-x // a) * a


def moved_bytes(lay: Layout, elem_bytes: int, beat: int) -> int:
    """The bytes the DMA moves for a layout of ``elem_bytes`` elements: its size padded to
    whole beats (D105), e.g. 16 int32 or 64 int8 elements per 64-byte beat."""
    return round_up(int(np.prod(lay.shape)) * elem_bytes, beat)


def l2_bytes(point: DesignPoint, c: str) -> int:
    """The bytes of container ``c``'s array in L2: its L2 layout over its padded shape (a
    tail's pads included, D144) in whole DMA beats of the platform's ``l2.beat_bits`` (D105).
    gen/data.py declares the array so, run/elf.py checks its symbol and run/convert.py its
    copies against it."""
    elem = np.dtype(point.graph.containers[c].dtype).itemsize
    block = over(point.memory.layouts[c]["l2"], padded_shape(point.graph, c))
    return moved_bytes(block, elem, point.platform.l2.beat_bits // 8)


def dma_side(lay: Layout, elem_bytes: int, beat: int) -> dict[str, int | list[int]]:
    """One side of a DMA task moving a whole layout: ``base``, ``bounds``, ``strides``.

    A contiguous layout starting on a wide beat is one loop of beats (open
    item 38). A size that is not whole beats is padded (D105): the last beat
    also moves the bytes after the layout, up to the beat's end, which the
    memory plan keeps free in both memories. Anything else raises
    LayoutError, which the design check ``memory.align`` reports first.
    """
    if lay != Layout.contiguous(lay.base, lay.shape, elem_bytes) or lay.base % beat:
        raise LayoutError(f"layout {lay} is not contiguous from a {beat}-byte beat")
    return {
        "base": lay.base,
        "bounds": [moved_bytes(lay, elem_bytes, beat) // beat],
        "strides": [beat],
    }
