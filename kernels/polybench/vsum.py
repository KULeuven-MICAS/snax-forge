"""
Vector sum.

int64, one element per 64-bit L1 word, as dot. A test kernel: it exists to
exercise accumulate's Chisel Accumulator on its own (GEN3 step 6b), since dot
binds two instances (elementwise_mul and accumulate) and snax_alu's board has
one accelerator slot. The importer turns the Reduce into a sum map folding
into ``out[0]`` with ``add`` (D102).

Tier done (PAPER §6.1).
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N = dace.symbol("N")


def vsum(A, out):
    out[0] = np.sum(A)


def make_inputs(rng, n=1024):
    return {
        "A": rng.integers(-1000, 1000, size=n, dtype=np.int64),
        "out": np.zeros(1, dtype=np.int64),
    }


SPEC = KernelSpec(
    name="vsum",
    func=vsum,
    domain="reduction",
    descriptors={"A": dace.int64[N], "out": dace.int64[1]},
    make_inputs=make_inputs,
    inout=("out",),
    tags=("reduction",),
    flops=lambda n=1024: n,
    bytes_moved=lambda n=1024: 8 * n + 8,
    sweep_sizes=(1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18),
)
