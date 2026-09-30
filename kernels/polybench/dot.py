"""
Dot product via reduction. Yields a Reduce library node, not a WCR memlet.

int64, one element per 64-bit L1 word, as vecadd (open item 30, D78): the
importer turns the Reduce into a sum map whose output memlet folds with
``add`` (D102), and integer sums match exactly in any order (D28).
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N = dace.symbol("N")


def dot(A, B, out):
    out[0] = np.sum(A * B)


def make_inputs(rng, n=1024):
    return {
        "A": rng.integers(-1000, 1000, size=n, dtype=np.int64),
        "B": rng.integers(-1000, 1000, size=n, dtype=np.int64),
        "out": np.zeros(1, dtype=np.int64),
    }


SPEC = KernelSpec(
    name="dot",
    func=dot,
    domain="reduction",
    descriptors={"A": dace.int64[N], "B": dace.int64[N], "out": dace.int64[1]},
    make_inputs=make_inputs,
    inout=("out",),
    tags=("reduction",),
    flops=lambda n=1024: 2 * n,
    bytes_moved=lambda n=1024: 2 * 8 * n + 8,
    sweep_sizes=(1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18),
)
