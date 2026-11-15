"""Row-wise reduction: y = A.sum(axis=1).

Partial reduction: the reduced axis differs from the output axis, unlike dot
where they coincide. This is the shape softmax's normalisation pass takes,
so it is worth having the structure characterised early.

int64;
the type presets ``i8``, ``i16`` and ``i32`` are in ``SPEC`` (kernels/README.md).
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

M, N = dace.symbol("M"), dace.symbol("N")


def rowsum(A, y):
    y[:] = np.sum(A, axis=1)


def make_inputs(rng, m=256, n=256):
    return {
        "A": rng.integers(-100, 100, size=(m, n), dtype=np.int64),
        "y": np.zeros(m, dtype=np.int64),
    }


SPEC = KernelSpec(
    name="rowsum",
    func=rowsum,
    domain="reduction",
    descriptors={"A": dace.int64[M, N], "y": dace.int64[M]},
    make_inputs=make_inputs,
    inout=("y",),
    flops=lambda m=256, n=256: m * n,
    bytes_moved=lambda m=256, n=256: 8 * m * n + 8 * m,
    sweep_sizes=(
        {"m": 128, "n": 128},
        {"m": 256, "n": 256},
        {"m": 512, "n": 512},
        {"m": 128, "n": 512},
    ),
    tags=("reduction", "partial-reduction"),
    presets={
        "i8": {"A": "int8", "y": "int32"},
        "i16": {"A": "int16", "y": "int32"},
        "i32": {"A": "int32", "y": "int64"},
    },
)
