# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause (kernels/NPBENCH_LICENSE)
# From NPBench (github.com/spcl/npbench, f2d7f27): npbench/benchmarks/polybench/k3mm/k3mm_dace.py
"""PolyBench 3mm: out = A B C D, as ((A B) C) D.

Tier D (PAPER §6.1). Staged: not yet linked to the flow. int64 (D113);
NPBench's return becomes the output argument out.
"""

import dace as dc
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

NI, NJ, NK, NL, NM = (dc.symbol(s, dtype=dc.int64) for s in ("NI", "NJ", "NK", "NL", "NM"))


def k3mm(A, B, C, D, out):
    out[:] = A @ B @ C @ D


def make_inputs(rng, ni=16, nj=18, nk=20, nl=22, nm=24):
    return {
        "A": rng.integers(-4, 4, size=(ni, nk), dtype=np.int64),
        "B": rng.integers(-4, 4, size=(nk, nj), dtype=np.int64),
        "C": rng.integers(-4, 4, size=(nj, nm), dtype=np.int64),
        "D": rng.integers(-4, 4, size=(nm, nl), dtype=np.int64),
        "out": np.zeros((ni, nl), dtype=np.int64),
    }


SPEC = KernelSpec(
    name="k3mm",
    func=k3mm,
    domain="linalg",
    descriptors={
        "A": dc.int64[NI, NK],
        "B": dc.int64[NK, NJ],
        "C": dc.int64[NJ, NM],
        "D": dc.int64[NM, NL],
        "out": dc.int64[NI, NL],
    },
    make_inputs=make_inputs,
    inout=("out",),
    tags=("staged", "matmul", "chaining-barrier"),
)
