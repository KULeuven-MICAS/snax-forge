"""PolyBench atax with type presets: atax's chain, x cast to ``OP_TYPE``.

int64 by default, atax's graph but for the cast's temporaries. One cast is enough: NumPy runs
``A @ x`` at its operands' result type, so A is widened inside the op, and the transient t takes
that type, which ``A.T @ t`` keeps (at ``i8``: A int8 times t int32 into y int32). x is cast, not
``A.T`` (a cast there is a Transpose the importer refuses). Tier A (PAPER §6.1).
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

M, N, OP_TYPE = dace.symbol("M"), dace.symbol("N"), np.int64


def atax_types(A, x, y):
    y[:] = A.T @ (A @ x.astype(OP_TYPE))


def make_inputs(rng, m=256, n=256):
    return {
        "A": rng.integers(-8, 8, size=(m, n), dtype=np.int64),
        "x": rng.integers(-8, 8, size=n, dtype=np.int64),
        "y": np.zeros(n, dtype=np.int64),  # assigned, not accumulated
    }


SPEC = KernelSpec(
    name="atax_types",
    func=atax_types,
    domain="linalg",
    descriptors={"A": dace.int64[M, N], "x": dace.int64[N], "y": dace.int64[N]},
    make_inputs=make_inputs,
    inout=("y",),
    flops=lambda m=256, n=256: 4 * m * n,
    bytes_moved=lambda m=256, n=256: 16 * m * n,  # A read twice
    tags=("chaining-barrier", "matvec"),
    types={"OP_TYPE": "int64"},
    presets={
        "i8": {"A": "int8", "x": "int8", "y": "int32", "OP_TYPE": "int32"},
        "i16": {"A": "int16", "x": "int16", "y": "int32", "OP_TYPE": "int32"},
        "i32": {"A": "int32", "x": "int32", "y": "int64", "OP_TYPE": "int64"},
    },
)
