"""PolyBench jacobi-1d: 3-point 1D stencil, two half-steps per timestep.

The only kernel here that stays multi-state after simplify (4 states), so it is
the one that exercises per-state instrumentation. Pure maps, no library nodes.
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N = dace.symbol("N")
TSTEPS = 8  # baked into the SDFG as a loop bound, not a symbol


def jacobi1d(A, B):
    for _ in range(TSTEPS):
        B[1:-1] = (A[:-2] + A[1:-1] + A[2:]) // 3
        A[1:-1] = (B[:-2] + B[1:-1] + B[2:]) // 3


def make_inputs(rng, n=1024):
    # Strictly positive: numpy floor-divides, C truncates. They agree only
    # for non-negative operands -- keeping inputs > 0 preserves bit-exactness.
    return {
        "A": rng.integers(1, 1000, size=n, dtype=np.int32),
        "B": rng.integers(1, 1000, size=n, dtype=np.int32),
    }


SPEC = KernelSpec(
    name="jacobi1d",
    func=jacobi1d,
    domain="stencil",
    descriptors={"A": dace.int32[N], "B": dace.int32[N]},
    make_inputs=make_inputs,
    inout=("A", "B"),
    flops=lambda n=1024: 6 * TSTEPS * (n - 2),  # 2 adds + 1 div per output
    bytes_moved=lambda n=1024: 16 * TSTEPS * (n - 2),
    tags=("stencil", "multi-state"),
    sweep_sizes=(1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18),
)
