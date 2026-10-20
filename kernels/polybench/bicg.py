"""PolyBench bicg: q = A p, s = A^T r.

The control for atax. Same ingredients -- one matvec over A and one over A^T --
but the two results are INDEPENDENT, so there is no transient between them and
no chaining barrier. If the chaining analysis flags atax and not bicg, it works.

Tier A (PAPER §6.1). int64 (D113).
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

M, N = dace.symbol("M"), dace.symbol("N")


def bicg(A, p, q, r, s):
    q[:] = A @ p
    s[:] = A.T @ r


def make_inputs(rng, m=256, n=256):
    return {
        "A": rng.integers(-8, 8, size=(m, n), dtype=np.int64),
        "p": rng.integers(-8, 8, size=n, dtype=np.int64),
        "q": np.zeros(m, dtype=np.int64),  # assigned, not accumulated
        "r": rng.integers(-8, 8, size=m, dtype=np.int64),
        "s": np.zeros(n, dtype=np.int64),
    }


SPEC = KernelSpec(
    name="bicg",
    func=bicg,
    domain="linalg",
    descriptors={
        "A": dace.int64[M, N],
        "p": dace.int64[N],
        "q": dace.int64[M],
        "r": dace.int64[M],
        "s": dace.int64[N],
    },
    make_inputs=make_inputs,
    inout=("q", "s"),
    flops=lambda m=256, n=256: 4 * m * n,
    bytes_moved=lambda m=256, n=256: 16 * m * n,  # A read twice
    sweep_sizes=(
        {"m": 128, "n": 128},
        {"m": 256, "n": 256},
        {"m": 512, "n": 512},
        {"m": 128, "n": 512},
    ),
    tags=("matvec", "two-scopes", "no-chaining"),
)
