"""
Dot product via reduction. Yields a Reduce library node, not a WCR memlet.
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N = dace.symbol("N")


def dot(A, B, out):
    out[0] = np.sum(A * B)


def make_inputs(rng, n=1024):
    return {
        "A": rng.integers(-100, 100, size=n, dtype=np.int32),
        "B": rng.integers(-100, 100, size=n, dtype=np.int32),
        "out": np.zeros(1, dtype=np.int32),
    }


SPEC = KernelSpec(
    name="dot",
    func=dot,
    domain="reduction",
    descriptors={"A": dace.int32[N], "B": dace.int32[N], "out": dace.int32[1]},
    make_inputs=make_inputs,
    inout=("out",),
    tags=("reduction",),
)
