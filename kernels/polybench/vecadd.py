"""
Elementwise vector add.
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N = dace.symbol("N")


def vecadd(A, B, C):
    C[:] = A + B


def make_inputs(rng, n=1024):
    return {
        "A": rng.integers(-1000, 1000, size=n, dtype=np.int32),
        "B": rng.integers(-1000, 1000, size=n, dtype=np.int32),
        "C": np.zeros(n, dtype=np.int32),
    }


SPEC = KernelSpec(
    name="vecadd",
    func=vecadd,
    domain="elementwise",
    descriptors={"A": dace.int32[N], "B": dace.int32[N], "C": dace.int32[N]},
    make_inputs=make_inputs,
    inout=("C",),
    tags=("gate", "streamable"),
)
