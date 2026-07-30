"""2-D elementwise add: C = A + B.

The 2-D analogue of vecadd, changing exactly one variable. Every other 2-D
kernel here also carries a reduction or library nodes; this isolates
dimensionality so a failure points at map rank and nothing else.
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

M, N = dace.symbol("M"), dace.symbol("N")


def matadd(A, B, C):
    C[:] = A + B


def make_inputs(rng, m=256, n=256):
    return {
        "A": rng.integers(-1000, 1000, size=(m, n), dtype=np.int32),
        "B": rng.integers(-1000, 1000, size=(m, n), dtype=np.int32),
        "C": np.zeros((m, n), dtype=np.int32),
    }


SPEC = KernelSpec(
    name="matadd",
    func=matadd,
    domain="elementwise",
    descriptors={"A": dace.int32[M, N], "B": dace.int32[M, N], "C": dace.int32[M, N]},
    make_inputs=make_inputs,
    inout=("C",),
    flops=lambda m=256, n=256: m * n,
    bytes_moved=lambda m=256, n=256: 3 * 4 * m * n,  # AI 0.083, same as vecadd
    sweep_sizes=(
        {"m": 128, "n": 128},
        {"m": 256, "n": 256},
        {"m": 512, "n": 512},
        {"m": 128, "n": 512},
    ),
    tags=("2d", "streamable"),
)
