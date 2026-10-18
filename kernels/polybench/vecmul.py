"""
Elementwise vector multiply.

int64, one element per 64-bit L1 word, as vecadd. A test kernel: it exists
to exercise the elementwise_mul BRM on its own (GEN3 step 6a), since dot
binds two instances (elementwise_mul and accumulate) and snax_alu's board
has one accelerator slot.
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N = dace.symbol("N")


def vecmul(A, B, C):
    C[:] = A * B


def make_inputs(rng, n=1024):
    return {
        "A": rng.integers(-1000, 1000, size=n, dtype=np.int64),
        "B": rng.integers(-1000, 1000, size=n, dtype=np.int64),
        "C": np.zeros(n, dtype=np.int64),
    }


SPEC = KernelSpec(
    name="vecmul",
    func=vecmul,
    domain="elementwise",
    descriptors={"A": dace.int64[N], "B": dace.int64[N], "C": dace.int64[N]},
    make_inputs=make_inputs,
    inout=("C",),
    tags=("streamable",),
    flops=lambda n=1024: n,
    bytes_moved=lambda n=1024: 3 * 8 * n,
    sweep_sizes=(1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18),
)
