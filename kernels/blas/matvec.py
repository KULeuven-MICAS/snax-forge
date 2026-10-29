"""Matrix-vector product: q = A p, the row form of dot.

Each row of A dotted with p: an elementwise product into a transient
``tmp0[M, N]``, then a sum over j into ``q``, so the import mirrors dot's
(a multiply map, then a sum map whose output memlet folds with ``add``) with
the row loop i around each. M13's B1x binds it as dot's variants with the
row loop as the jobs (``recipes/matvec_<k>.json``).

int64 (D113).
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

M, N = dace.symbol("M"), dace.symbol("N")


def matvec(A, p, q):
    q[:] = np.sum(A * p, axis=1)


def make_inputs(rng, m=16, n=512):
    return {
        "A": rng.integers(-1000, 1000, size=(m, n), dtype=np.int64),
        "p": rng.integers(-1000, 1000, size=n, dtype=np.int64),
        "q": np.zeros(m, dtype=np.int64),
    }


SPEC = KernelSpec(
    name="matvec",
    func=matvec,
    domain="linear-algebra",
    descriptors={"A": dace.int64[M, N], "p": dace.int64[N], "q": dace.int64[M]},
    make_inputs=make_inputs,
    inout=("q",),
    flops=lambda m=16, n=512: 2 * m * n,
    bytes_moved=lambda m=16, n=512: 8 * m * n + 8 * n + 8 * m,
    sweep_sizes=({"m": 16, "n": 512}, {"m": 8, "n": 512}),
    tags=("reduction", "partial-reduction"),
)
