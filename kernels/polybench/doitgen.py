# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause (kernels/NPBENCH_LICENSE)
# From NPBench (github.com/spcl/npbench, f2d7f27): npbench/benchmarks/polybench/doitgen/
# doitgen_dace.py, and doitgen_numpy.py as the reference
"""PolyBench doitgen: A[r, q, :] = A[r, q, :] C4 for every r and q, in place.

Tier D (PAPER §6.1). Staged: not yet linked to the flow. int64 (D113). The
body loops over and reshapes by the symbols, so the reference is NPBench's
NumPy version.
"""

import dace as dc
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

NR, NQ, NP = (dc.symbol(s, dtype=dc.int64) for s in ("NR", "NQ", "NP"))


def doitgen(A, C4):
    # Ideal - not working becayse Matmul with dim > 3 unsupported
    # A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))
    for r in range(NR):
        A[r, :, :] = np.reshape(np.reshape(A[r], (NQ, 1, NP)) @ C4, (NQ, NP))


def doitgen_ref(A, C4):
    NR, NQ, NP = A.shape
    A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))


def make_inputs(rng, nr=10, nq=8, np_=12):
    return {
        "A": rng.integers(-8, 8, size=(nr, nq, np_), dtype=np.int64),
        "C4": rng.integers(-8, 8, size=(np_, np_), dtype=np.int64),
    }


SPEC = KernelSpec(
    name="doitgen",
    func=doitgen,
    domain="linalg",
    descriptors={"A": dc.int64[NR, NQ, NP], "C4": dc.int64[NP, NP]},
    make_inputs=make_inputs,
    inout=("A",),
    ref=doitgen_ref,
    tags=("staged", "matmul", "in-place", "reshape"),
)
