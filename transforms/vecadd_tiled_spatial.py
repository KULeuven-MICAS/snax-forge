"""Tile vecadd, declare the tile spatial, then raise it to a SnaxVectorOp.

Extends vecadd_opt with a third step. The first two are the allocation
decision -- 64 elements per hardware instance, spatially replicated -- and the
third reads that decision off the graph and commits it to a library node.

Run under verify_each (the default) this is where bit-exactness is checked at
the moment a map scope is swapped for a node, which is the only point in the
flow where the golden reference and the raised graph can still be compared
directly.
"""

from dace.transformation.dataflow import MapTiling

from snax_forge.libnodes.libnodes import raise_vector_ops
from snax_forge.sdfg.recipes import Step, TransformRecipe, set_map_property

RECIPE = TransformRecipe(
    name="vecadd_forge",
    kernel="vecadd",
    steps=(
        Step(
            MapTiling,
            target="_Add__map",
            options={"tile_sizes": (64,), "divides_evenly": True},
        ),
        # unroll=True is what makes the inner dimension spatial rather than a
        # sequential walk. TiledElementwise requires it: no unroll, no datapath.
        Step(set_map_property, options={"params": ["__i0"], "unroll": True}),
        Step(raise_vector_ops),
    ),
    tags=("tiling", "allocation", "libnode"),
)