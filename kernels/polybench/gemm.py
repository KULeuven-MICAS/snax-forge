# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause (kernels/NPBENCH_LICENSE)
# From NPBench (github.com/spcl/npbench, f2d7f27): npbench/benchmarks/polybench/gemm/gemm_dace.py
"""PolyBench gemm: C = alpha A B + beta C.

Tier D (PAPER §6.1). Staged: not yet linked to the flow. int64 (D113).
"""

import dace as dc
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

NI, NJ, NK = (dc.symbol(s, dtype=dc.int64) for s in ("NI", "NJ", "NK"))


def gemm(alpha, beta, C, A, B):
    C[:] = alpha * A @ B + beta * C


def make_inputs(rng, ni=20, nj=25, nk=30):
    return {
        "alpha": np.int64(3),
        "beta": np.int64(2),
        "C": rng.integers(-8, 8, size=(ni, nj), dtype=np.int64),
        "A": rng.integers(-8, 8, size=(ni, nk), dtype=np.int64),
        "B": rng.integers(-8, 8, size=(nk, nj), dtype=np.int64),
    }


SPEC = KernelSpec(
    name="gemm",
    func=gemm,
    domain="linalg",
    descriptors={
        "alpha": dc.int64,
        "beta": dc.int64,
        "C": dc.int64[NI, NJ],
        "A": dc.int64[NI, NK],
        "B": dc.int64[NK, NJ],
    },
    make_inputs=make_inputs,
    inout=("C",),
    tags=("staged", "matmul", "in-place", "scalars"),
)
