# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause (kernels/NPBENCH_LICENSE)
# From NPBench (github.com/spcl/npbench, f2d7f27): npbench/benchmarks/polybench/k2mm/k2mm_dace.py
"""PolyBench 2mm: D = alpha A B C + beta D.

Tier D (PAPER §6.1). Staged: not yet linked to the flow. int64 (D113).
"""

import dace as dc
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

NI, NJ, NK, NL = (dc.symbol(s, dtype=dc.int64) for s in ("NI", "NJ", "NK", "NL"))


def k2mm(alpha, beta, A, B, C, D):
    D[:] = alpha * A @ B @ C + beta * D


def make_inputs(rng, ni=16, nj=18, nk=22, nl=24):
    return {
        "alpha": np.int64(3),
        "beta": np.int64(2),
        "A": rng.integers(-4, 4, size=(ni, nk), dtype=np.int64),
        "B": rng.integers(-4, 4, size=(nk, nj), dtype=np.int64),
        "C": rng.integers(-4, 4, size=(nj, nl), dtype=np.int64),
        "D": rng.integers(-4, 4, size=(ni, nl), dtype=np.int64),
    }


SPEC = KernelSpec(
    name="k2mm",
    func=k2mm,
    domain="linalg",
    descriptors={
        "alpha": dc.int64,
        "beta": dc.int64,
        "A": dc.int64[NI, NK],
        "B": dc.int64[NK, NJ],
        "C": dc.int64[NJ, NL],
        "D": dc.int64[NI, NL],
    },
    make_inputs=make_inputs,
    inout=("D",),
    tags=("staged", "matmul", "chaining-barrier", "in-place", "scalars"),
)
