"""2-D elementwise add: C = A + B.

The 2-D analogue of vecadd, changing exactly one variable. Every other 2-D
kernel here also carries a reduction or library nodes; this isolates
dimensionality so a failure points at map rank and nothing else.

int32 by default, packed two per 64-bit word, the sum at ``ACC``; the type
presets ``i8``, ``i16`` and ``i32`` are in ``SPEC`` (kernels/README.md).
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

M, N = dace.symbol("M"), dace.symbol("N")
ACC = np.int32


def matadd(A, B, C):
    C[:] = A.astype(ACC) + B.astype(ACC)


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
    bytes_moved=lambda m=256, n=256: 3 * 4 * m * n,  # AI 0.083
    sweep_sizes=(
        {"m": 128, "n": 128},
        {"m": 256, "n": 256},
        {"m": 512, "n": 512},
        {"m": 128, "n": 512},
    ),
    tags=("2d", "streamable"),
    types={"ACC": "int32"},
    presets={
        "i8": {"A": "int8", "B": "int8", "C": "int32", "ACC": "int32"},
        "i16": {"A": "int16", "B": "int16", "C": "int32", "ACC": "int32"},
        "i32": {"A": "int32", "B": "int32", "C": "int64", "ACC": "int64"},
    },
)
