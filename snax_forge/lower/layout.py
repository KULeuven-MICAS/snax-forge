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
design checks judge a whole plan, in L1 and L2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from snax_forge.common import Config

if TYPE_CHECKING:  # design imports lower.layout
    from snax_forge.design.platform import L1Config


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

    def extent(self, word: int) -> tuple[int, int]:
        """The bytes ``[lo, hi)`` the operand covers, each element ``word`` bytes."""
        lo, hi = self.span()
        return lo, hi + word

    def check(self, l1: L1Config) -> None:
        """The layout fits ``l1``: word-aligned, inside L1."""
        word = l1.width_bits // 8
        if self.base % word or any(s % word for s in self.strides):
            raise LayoutError(
                f"base {self.base} and strides {list(self.strides)} must be multiples of the "
                f"{word}-byte word"
            )
        lo, hi = self.extent(word)
        start = l1.base_addr
        end = start + l1.size_bytes
        if lo < start or hi > end:
            raise LayoutError(f"the operand spans bytes [{lo}, {hi}), L1 is [{start}, {end})")


def round_up(x: int, a: int) -> int:
    """``x`` rounded up to a multiple of ``a``."""
    return -(-x // a) * a


def moved_bytes(lay: Layout, word: int, beat: int) -> int:
    """The bytes the DMA moves for a layout: its size padded to whole beats (D105)."""
    return round_up(int(np.prod(lay.shape)) * word, beat)


def dma_side(lay: Layout, word: int, beat: int) -> dict[str, int | list[int]]:
    """One side of a DMA task moving a whole layout: ``base``, ``bounds``, ``strides``.

    A contiguous layout starting on a wide beat is one loop of beats (open
    item 38). A size that is not whole beats is padded (D105): the last beat
    also moves the bytes after the layout, up to the beat's end, which the
    memory plan keeps free in both memories. Anything else raises
    LayoutError, which the design check ``memory.align`` reports first.
    """
    if lay != Layout.contiguous(lay.base, lay.shape, word) or lay.base % beat:
        raise LayoutError(f"layout {lay} is not contiguous from a {beat}-byte beat")
    return {"base": lay.base, "bounds": [moved_bytes(lay, word, beat) // beat], "strides": [beat]}
