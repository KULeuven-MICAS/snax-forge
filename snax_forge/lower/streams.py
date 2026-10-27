"""A BRM port's nest, through a layout, onto a streamer's values (D70).

The nest (``snax_forge.brm.affine``) says in which order the accelerator
takes an operand's elements, in logical indices; the layout (layout.py) says
where each element lives. Both are affine, so their composition is too, and
it is exactly what a streamer runs (CONTRACTS.md section 9), in words
(D142): a streamer channel moves one bank word, which holds k elements of
the port's dtype (the streamer's ``k``):

    base              layout.base + sum_d offset[d] * layout.strides[d]
    loop byte stride  sum_d loop.strides[d] * layout.strides[d]: the loop's
                      element stride times the itemsize, the layout's
                      strides being in bytes
    temporal loops    the nest's temporal loops, innermost first
    spatial loops     the nest's spatial loops, fastest first; their bounds
                      must be the streamer's design-time spatial bounds (in
                      elements). The fastest one runs S_0 / k channels, each
                      a word on: its byte stride is k times the element's,
                      8 bytes for a packed port (whose fastest element stride
                      is 1, ``connect.pack``); the others as above

At k = 1 (int64) the fastest stride is the element's: nothing changes.

The result is the ``values`` of a streamer task in the task list (D64), in
the streamer type's own terms; only the loops the nest uses are given and
the adapter pads the rest. Nothing is decided here (principle 4): the layout
is SNAX-DSE's, the nest the BRM's.

The streamer is the design point's (design/streamers.py: its ``write``, its
spatial bounds and its ``temporal_dims``), the L1 the platform's.
Everything that does not fit is an error, never adjusted: a reader serves an
output port or a writer an input port, the spatial bounds differ from the
streamer's, there are more temporal loops than the streamer has, the
layout's shape differs from the nest's, or the layout does not fit the L1
(layout.py).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from snax_forge.brm import Instance, task_nest

from .layout import Layout, LayoutError

if TYPE_CHECKING:
    from snax_forge.design.platform import L1Config
    from snax_forge.design.streamers import Streamer


class StreamError(ValueError):
    """A nest that cannot be mapped onto the given streamer and layout."""


def _dot(strides: tuple[int, ...], layout: Layout) -> int:
    return sum(s * b for s, b in zip(strides, layout.strides, strict=True))


def streamer_values(
    inst: Instance,
    port: str,
    task: Mapping[str, int],
    layout: Layout,
    streamer: Streamer,
    l1: L1Config,
) -> dict[str, Any]:
    """The ``values`` of ``streamer``'s task for ``port`` of ``inst``; see the module doc.

    ``task`` holds the accelerator task's start parameters (``n`` and any
    named rate), as in its own ``values``.
    """
    what = f"{inst.brm.name}.{port} -> {streamer.name}"
    try:
        nest = task_nest(inst, port, task)
    except ValueError as e:
        raise StreamError(str(e)) from None
    if layout.shape != nest.shape:
        raise StreamError(
            f"{what}: layout shape {list(layout.shape)} is not the operand's {list(nest.shape)}"
        )
    try:
        layout.check(l1, np.dtype(streamer.dtype).itemsize)
    except LayoutError as e:
        raise StreamError(f"{what}: {e}") from None

    write = streamer.write
    direction = inst.brm.port(port).direction
    if write != (direction == "out"):
        side = "writer" if write else "reader"
        raise StreamError(f"{what}: a {side} cannot serve an {direction!r} port")

    spatial = [b for b, _ in reversed(nest.spatial)]  # fastest first
    if tuple(spatial) != tuple(streamer.spatial_bounds):
        raise StreamError(
            f"{what}: spatial bounds {spatial} (fastest first) are not the streamer's "
            f"design-time {list(streamer.spatial_bounds)}"
        )
    temporal = list(reversed(nest.temporal))  # innermost first
    dims = streamer.temporal_dims
    if len(temporal) > dims:
        raise StreamError(f"{what}: {len(temporal)} temporal loops, the streamer has {dims}")
    if not temporal:
        temporal = [(1, (0,) * nest.rank)]  # one beat
    return {
        "base": layout.base + _dot(nest.offset, layout),
        "temporal_bounds": [b for b, _ in temporal],
        "temporal_strides": [_dot(s, layout) for _, s in temporal],
        "spatial_strides": in_words(
            [_dot(s, layout) for _, s in reversed(nest.spatial)], streamer.k
        ),
    }


def in_words(spatial_strides: list[int], k: int) -> list[int]:
    """Spatial byte strides, fastest first, with the fastest a channel's: k elements on, one
    word for a packed port (D142)."""
    return [spatial_strides[0] * k, *spatial_strides[1:]] if spatial_strides else []
