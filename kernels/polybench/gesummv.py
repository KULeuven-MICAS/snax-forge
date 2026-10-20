# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause (kernels/NPBENCH_LICENSE)
# From NPBench (github.com/spcl/npbench, f2d7f27): npbench/benchmarks/polybench/gesummv/gesummv_dace.py
"""PolyBench gesummv: y = alpha A x + beta B x.

Tier A (PAPER §6.1). Staged: not yet linked to the flow. int64 (D113);
NPBench's return becomes the output argument out.
"""

import dace as dc
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N = dc.symbol("N", dtype=dc.int64)


def gesummv(alpha, beta, A, B, x, out):
    out[:] = alpha * A @ x + beta * B @ x


def make_inputs(rng, n=30):
    return {
        "alpha": np.int64(3),
        "beta": np.int64(2),
        "A": rng.integers(-8, 8, size=(n, n), dtype=np.int64),
        "B": rng.integers(-8, 8, size=(n, n), dtype=np.int64),
        "x": rng.integers(-8, 8, size=n, dtype=np.int64),
        "out": np.zeros(n, dtype=np.int64),
    }


SPEC = KernelSpec(
    name="gesummv",
    func=gesummv,
    domain="linalg",
    descriptors={
        "alpha": dc.int64,
        "beta": dc.int64,
        "A": dc.int64[N, N],
        "B": dc.int64[N, N],
        "x": dc.int64[N],
        "out": dc.int64[N],
    },
    make_inputs=make_inputs,
    inout=("out",),
    tags=("staged", "matvec", "scalars"),
)
