"""Matrix-vector product into q: q += A p, matvec with a seed.

matvec's multiply into a transient and sum over j, then the row sums added to
q's old values: an elementwise map q = q + t after the fold, which
``fuse_seed`` takes into the fold as its per-job seed (M13 B2 R5).

int64 (D113).
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

M, N = dace.symbol("M"), dace.symbol("N")


def matvec_acc(A, p, q):
    q += np.sum(A * p, axis=1)


def make_inputs(rng, m=8, n=512):
    return {
        "A": rng.integers(-1000, 1000, size=(m, n), dtype=np.int64),
        "p": rng.integers(-1000, 1000, size=n, dtype=np.int64),
        "q": rng.integers(-1000, 1000, size=m, dtype=np.int64),
    }


SPEC = KernelSpec(
    name="matvec_acc",
    func=matvec_acc,
    domain="linear-algebra",
    descriptors={"A": dace.int64[M, N], "p": dace.int64[N], "q": dace.int64[M]},
    make_inputs=make_inputs,
    inout=("q",),
    flops=lambda m=8, n=512: 2 * m * n + m,
    bytes_moved=lambda m=8, n=512: 8 * m * n + 8 * n + 2 * 8 * m,
    sweep_sizes=({"m": 8, "n": 512},),
    tags=("reduction", "partial-reduction", "seed"),
)
