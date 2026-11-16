"""
Elementwise vector multiply.

int64 by default, as vecadd, the product at ``ACC``; the type
presets ``i8``, ``i16`` and ``i32`` are in ``SPEC`` (kernels/README.md). A test kernel: it exists
to exercise the simd BRM on its own, since dot binds two
instances (simd and accumulate) and snax_forge's board has one
accelerator slot.

Tier done (PAPER §6.1).
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N = dace.symbol("N")
ACC = np.int64


def vecmul(A, B, C):
    C[:] = A.astype(ACC) * B.astype(ACC)


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
    types={"ACC": "int64"},
    presets={
        "i8": {"A": "int8", "B": "int8", "C": "int32", "ACC": "int32"},
        "i16": {"A": "int16", "B": "int16", "C": "int32", "ACC": "int32"},
        "i32": {"A": "int32", "B": "int32", "C": "int64", "ACC": "int64"},
    },
)
