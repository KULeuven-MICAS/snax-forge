"""Declare vecadd's single map fully spatial, then raise it.

The other extreme from vecadd_flat: one lane per element, fed once.

Two things this recipe exists to demonstrate.

First, `schedule=Sequential` is not decoration. A top-level map is inferred as
CPU_Multicore at codegen time, and DaCe refuses to unroll an OpenMP map
("An OpenMP map cannot be unrolled"). The tiled recipe never hits this because
its inner map is nested and therefore inferred Sequential; a flat map has no
outer scope to be nested in, so the schedule must be stated.

Second, this passes verification while describing hardware that cannot be
built. N is symbolic, so the CPU backend emits a `#pragma unroll` over an
unknown bound and gets the right answer, but a width of N lanes is not a
width. The descriptor records bounded=False; nothing in the CPU flow would
have told us.
"""

import dace

from snax_forge.libnodes.libnodes import raise_vector_ops
from snax_forge.sdfg.recipes import Step, TransformRecipe, set_map_property

RECIPE = TransformRecipe(
    name="vecadd_spatial",
    kernel="vecadd",
    steps=(
        Step(
            set_map_property,
            options={
                "params": ["__i0"],
                "unroll": True,
                "schedule": dace.ScheduleType.Sequential,
            },
        ),
        Step(raise_vector_ops),
    ),
    tags=("libnode", "spatial", "unbounded"),
)