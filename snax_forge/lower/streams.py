"""A BRM port's nest, through a layout, onto a streamer's values (BRM2, D70).

The nest (``snax_forge.brm.affine``) says in which order the accelerator
takes an operand's elements, in logical indices; the layout (layout.py) says
where each element lives. Both are affine, so their composition is too, and
it is exactly what a streamer runs (CONTRACTS.md section 3):

    base              layout.base + sum_d offset[d] * layout.strides[d]
    loop byte stride  sum_d loop.strides[d] * layout.strides[d]
    temporal loops    the nest's temporal loops, innermost first
    spatial loops     the nest's spatial loops, fastest first; their bounds
                      must be the streamer's design-time spatial bounds

The result is the ``values`` of a streamer task in the task list (D64), in
the streamer type's own terms; only the loops the nest uses are given and
the adapter pads the rest. Nothing is decided here (principle 4): the layout
is SNAX-DSE's, the nest the BRM's.

Everything that does not fit is an error, never adjusted: the component is
not a streamer, a reader serves an output port or a writer an input port,
the spatial bounds differ from the streamer's, there are more temporal loops
than the streamer has, the layout's shape differs from the nest's, or the
layout does not fit the L1 (layout.py).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from snax_forge.brm import Instance, task_nest
from snax_forge.snax_model.scenario import ClusterConfig, register_map_of
from snax_forge.snax_model.streamer import Streamer

from .layout import Layout, LayoutError


class StreamError(ValueError):
    """A nest that cannot be mapped onto the given streamer and layout."""


def _dot(strides: tuple[int, ...], layout: Layout) -> int:
    return sum(s * b for s, b in zip(strides, layout.strides, strict=True))


def streamer_values(
    inst: Instance,
    port: str,
    task: Mapping[str, int],
    layout: Layout,
    cluster: ClusterConfig,
    streamer: str,
) -> dict[str, Any]:
    """The ``values`` of ``streamer``'s task for ``port`` of ``inst``; see the module doc.

    ``task`` holds the accelerator task's start parameters (``n`` and any
    named rate), as in its own ``values``.
    """
    what = f"{inst.brm.name}.{port} -> {streamer}"
    try:
        nest = task_nest(inst, port, task)
    except ValueError as e:
        raise StreamError(str(e)) from None
    if layout.shape != nest.shape:
        raise StreamError(
            f"{what}: layout shape {list(layout.shape)} is not the operand's {list(nest.shape)}"
        )
    try:
        layout.check(cluster.l1)
    except LayoutError as e:
        raise StreamError(f"{what}: {e}") from None

    blocks = register_map_of(cluster).blocks
    if streamer not in blocks or not isinstance(blocks[streamer].comp, Streamer):
        raise StreamError(f"{what}: {streamer!r} is not a streamer of the cluster")
    block = blocks[streamer]
    write = block.comp.cfg.write
    direction = inst.brm.port(port).direction
    if write != (direction == "out"):
        side = "writer" if write else "reader"
        raise StreamError(f"{what}: a {side} cannot serve an {direction!r} port")

    spatial = [b for b, _ in reversed(nest.spatial)]  # fastest first
    if tuple(spatial) != tuple(block.adapter.spatial_bounds):
        raise StreamError(
            f"{what}: spatial bounds {spatial} (fastest first) are not the streamer's "
            f"design-time {list(block.adapter.spatial_bounds)}"
        )
    temporal = list(reversed(nest.temporal))  # innermost first
    if len(temporal) > block.adapter.d:
        raise StreamError(
            f"{what}: {len(temporal)} temporal loops, the streamer has {block.adapter.d}"
        )
    if not temporal:
        temporal = [(1, (0,) * nest.rank)]  # one beat
    return {
        "base": layout.base + _dot(nest.offset, layout),
        "temporal_bounds": [b for b, _ in temporal],
        "temporal_strides": [_dot(s, layout) for _, s in temporal],
        "spatial_strides": [_dot(s, layout) for _, s in reversed(nest.spatial)],
    }
