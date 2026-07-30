"""Tile matadd's 2-D map by 64x64.

One tile_sizes entry per map dimension. The map is (__i0 over M, __i1 over N),
so (64, 64) tiles both; a shorter tuple would tile only the leading dims.

divides_evenly asserts M % 64 == 0 and N % 64 == 0. Every sweep size is a
multiple of 64, so this holds. DaCe does NOT verify it -- violating the
assertion silently drops the clamp and produces wrong results.
"""

from dace.transformation.dataflow import MapExpansion, MapTiling

from snax_forge.sdfg.transforms import Step, TransformRecipe, set_map_property

RECIPE = TransformRecipe(
    name="matadd_opt",
    kernel="matadd",
    steps=(
        Step(
            MapTiling, target="_Add__map", options={"tile_sizes": (64, 64), "divides_evenly": True}
        ),
        Step(MapExpansion, repeat=True),
        Step(set_map_property, options={"params": ["__i0"], "unroll": True}),
        Step(set_map_property, options={"params": ["__i1"], "unroll": True}),
    ),
    tags=("tiling", "spatial", "unroll"),
)
