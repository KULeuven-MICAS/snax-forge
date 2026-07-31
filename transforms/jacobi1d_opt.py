"""Fuse jacobi1d's six map scopes down to two.

Under one-map-scope-per-accelerator this is the difference between six
generated datapaths and two. Same result auto_optimize reaches, but explicit.
"""

from dace.transformation.dataflow import MapFusion

from snax_forge.sdfg.recipes import Step, TransformRecipe

RECIPE = TransformRecipe(
    name="jacobi1d_opt",
    kernel="jacobi1d",
    steps=(Step(MapFusion, repeat=True),),
    tags=("fusion", "scope-reduction"),
)
