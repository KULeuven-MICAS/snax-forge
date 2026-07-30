"""BLAS-1 axpy: y += alpha * x.

First kernel with a SCALAR argument and a two-op tasklet -- multiply then add,
i.e. a MAC. That fused pair is the atom of every datapath downstream, so this
is the smallest kernel that exercises operator chaining inside one tasklet.
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N = dace.symbol("N")


def axpy(alpha, x, y):
    y += alpha * x


def make_inputs(rng, n=1024):
    # Small ranges: y accumulates across profiling reps. int32 wraps rather
    # than faults, so timing stays valid, but values drift.
    return {
        "alpha": np.int32(3),
        "x": rng.integers(-100, 100, size=n, dtype=np.int32),
        "y": rng.integers(-100, 100, size=n, dtype=np.int32),
    }


SPEC = KernelSpec(
    name="axpy",
    func=axpy,
    domain="elementwise",
    descriptors={"alpha": dace.int32, "x": dace.int32[N], "y": dace.int32[N]},
    make_inputs=make_inputs,
    inout=("y",),
    flops=lambda n=1024: 2 * n,  # one multiply + one add per element
    bytes_moved=lambda n=1024: 3 * 4 * n,  # read x, read y, write y
    sweep_sizes=(1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18),
    tags=("scalar-arg", "mac", "streamable"),
)
