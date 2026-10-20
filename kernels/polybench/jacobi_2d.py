# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause (kernels/NPBENCH_LICENSE)
# From NPBench (github.com/spcl/npbench, f2d7f27): npbench/benchmarks/polybench/jacobi_2d/jacobi_2d_dace.py
"""PolyBench jacobi-2d: TSTEPS - 1 sweeps of a 5-point average, A into B, then B into A.

Tier D (PAPER §6.1). Staged: not yet linked to the flow. int64 (D113):
NPBench's 0.2 * (...) becomes (...) // 5, until the tier's wider
fixed-point type.
"""

import dace as dc
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N = dc.symbol("N", dtype=dc.int64)


def jacobi_2d(TSTEPS, A, B):
    for t in range(1, TSTEPS):
        B[1:-1, 1:-1] = (
            A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1]
        ) // 5
        A[1:-1, 1:-1] = (
            B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1]
        ) // 5


def make_inputs(rng, tsteps=20, n=30):
    return {
        "TSTEPS": np.int64(tsteps),
        "A": rng.integers(0, 1000, size=(n, n), dtype=np.int64),
        "B": rng.integers(0, 1000, size=(n, n), dtype=np.int64),
    }


SPEC = KernelSpec(
    name="jacobi_2d",
    func=jacobi_2d,
    domain="stencil",
    descriptors={"TSTEPS": dc.int64, "A": dc.int64[N, N], "B": dc.int64[N, N]},
    make_inputs=make_inputs,
    inout=("A", "B"),
    tags=("staged", "stencil", "time-loop", "divide"),
)
