"""Raise vecadd straight to a SnaxVectorOp, with no tiling first.

The counterpart to vecadd_forge: same kernel, same library node, no allocation
decision. The simplified SDFG has one map over 0:N and does not declare it
unrolled, so this raises a datapath of ONE lane fed N times -- correct, and
the least parallel member of the family.

Read the two recipes side by side and the split is the only difference:

    vecadd_flat    lanes = 1     trips = N
    vecadd_forge   lanes = 64    trips = ceil(N/64)

Both satisfy lanes * trips = N.
"""

from snax_forge.libnodes.libnodes import raise_vector_ops
from snax_forge.sdfg.recipes import Step, TransformRecipe

RECIPE = TransformRecipe(
    name="vecadd_loop",
    kernel="vecadd",
    steps=(Step(raise_vector_ops),),
    tags=("libnode", "no-allocation"),
)