# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause (kernels/NPBENCH_LICENSE)
# From NPBench (github.com/spcl/npbench, f2d7f27): npbench/benchmarks/polybench/gemver/gemver_dace.py
"""PolyBench gemver: a rank-2 update of A, then x += beta A^T y + z, then w += alpha A x.

Tier D (PAPER §6.1): the rank-1 updates read u and v with stride 0 (D70).
Staged: not yet linked to the flow. int64 (D113).
"""

import dace as dc
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

M, N = (dc.symbol(s, dtype=dc.int64) for s in ("M", "N"))


def gemver(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    A += np.multiply.outer(u1, v1) + np.multiply.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x


def make_inputs(rng, n=40):
    vec = lambda: rng.integers(-4, 4, size=n, dtype=np.int64)
    return {
        "alpha": np.int64(3),
        "beta": np.int64(2),
        "A": rng.integers(-4, 4, size=(n, n), dtype=np.int64),
        "u1": vec(),
        "v1": vec(),
        "u2": vec(),
        "v2": vec(),
        "w": vec(),
        "x": vec(),
        "y": vec(),
        "z": vec(),
    }


SPEC = KernelSpec(
    name="gemver",
    func=gemver,
    domain="linalg",
    descriptors={
        "alpha": dc.int64,
        "beta": dc.int64,
        "A": dc.int64[N, N],
        **{v: dc.int64[N] for v in ("u1", "v1", "u2", "v2", "w", "x", "y", "z")},
    },
    make_inputs=make_inputs,
    inout=("A", "x", "w"),
    tags=("staged", "rank-1-update", "in-place", "scalars"),
)
