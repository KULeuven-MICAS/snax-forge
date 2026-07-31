"""Tile vecadd's map scope with a tile size of 64.

Under one-map-scope-per-accelerator this can improve data locality and performance.
"""

from dace.transformation.dataflow import MapTiling

from snax_forge.sdfg.recipes import Step, TransformRecipe, set_map_property

RECIPE = TransformRecipe(
    name="vecadd_opt",
    kernel="vecadd",
    steps=(
        Step(
            MapTiling,
            target="_Add__map",
            options={"tile_sizes": (64,), "divides_evenly": True},
        ),
        Step(set_map_property, options={"params": ["__i0"], "unroll": True}),
    ),
    tags=("tiling", "allocation"),
)
