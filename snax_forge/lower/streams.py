"""A BRM port's nest, through a layout, onto a streamer's values.

The nest (``snax_forge.brm.affine``) gives the order in which the accelerator takes an
operand's elements; the layout (layout.py) gives where each element lives. Both are affine, so
their composition is too: ``streamer_values`` turns it into the ``values`` of a streamer task
(CONTRACTS, "Task list"), temporal loops innermost first and spatial loops fastest first. Only
the loops the nest uses are given; the adapter pads the rest. ``in_words`` turns element
strides into bank-word strides.

Nothing is decided here: the layout is SNAX-DSE's, the nest the BRM's, the streamer the design
point's (design/streamers.py) and the L1 the platform's. What does not fit is a ``StreamError``,
never adjusted: a reader on an output port or a writer on an input port, spatial bounds that
differ from the streamer's, more temporal loops than it has, a layout whose shape differs from
the nest's, or a layout outside the L1.
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
            [_dot(s, layout) for _, s in reversed(nest.spatial)], streamer.num_slc
        ),
    }


def in_words(spatial_strides: list[int], num_slc: int) -> list[int]:
    """Spatial byte strides, fastest first, with the fastest a channel's: num_slc elements on,
    one word for a packed port."""
    return [spatial_strides[0] * num_slc, *spatial_strides[1:]] if spatial_strides else []
