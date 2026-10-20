# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause (kernels/NPBENCH_LICENSE)
# From NPBench (github.com/spcl/npbench, f2d7f27): npbench/benchmarks/polybench/covariance/
# covariance_dace.py, and covariance_numpy.py as the reference
"""PolyBench covariance: the column means, the centred data, then cov = data^T data / (n - 1).

Tier C (PAPER §6.1): three phases (mean, centre, product); its mean is tier
B's integer divide. Staged: not yet linked to the flow. int64 (D113): NPBench's
np.mean and / (float_n - 1.0) become integer divides (// float_n,
// (float_n - 1)), and float_n holds N as an int64. NPBench's return
becomes the output argument out. The body allocates and loops over the
symbol M, so the reference is NPBench's NumPy version.
"""

import dace as dc
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

M, N = (dc.symbol(s, dtype=dc.int64) for s in ("M", "N"))


def covariance(float_n, data, out):
    mean = np.sum(data, axis=0) // float_n
    # data -= mean
    np.subtract(data, mean, out=data)
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i, i:M] = data[:, i] @ data[:, i:M] // (float_n - 1)
        cov[i:M, i] = cov[i, i:M]

    out[:] = cov


def covariance_ref(float_n, data, out):
    M = data.shape[1]
    mean = np.sum(data, axis=0) // float_n
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] // (float_n - 1)

    out[:] = cov


def make_inputs(rng, m=28, n=32):
    return {
        "float_n": np.int64(n),
        "data": rng.integers(-8, 8, size=(n, m), dtype=np.int64),
        "out": np.zeros((m, m), dtype=np.int64),
    }


SPEC = KernelSpec(
    name="covariance",
    func=covariance,
    domain="linalg",
    descriptors={"float_n": dc.int64, "data": dc.int64[N, M], "out": dc.int64[M, M]},
    make_inputs=make_inputs,
    inout=("data", "out"),
    ref=covariance_ref,
    tags=("staged", "multi-phase", "divide", "in-place"),
)
