"""PolyBench mvt with type presets: mvt's two products, y cast to ``OP_TYPE``.

int64 by default, mvt's graph but for the cast's temporaries (``tmp1_*`` and ``tmp4_*``
where mvt has ``tmp0_*`` and ``tmp2_*``). One cast per product is enough: NumPy runs ``@``
at its operands' result type, so A is widened inside the op. y is cast, not ``A.T`` (a cast
there is a Transpose the importer refuses). Tier A (PAPER §6.1).
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

M, N, OP_TYPE = dace.symbol("M"), dace.symbol("N"), np.int64


def mvt_types(A, x1, x2, y1, y2):  # line 17, as mvt's: the += maps are named by their line
    x1 += A @ y1.astype(OP_TYPE)
    x2 += A.T @ y2.astype(OP_TYPE)


def make_inputs(rng, m=256, n=256):
    # mvt's: small, as x1 and x2 accumulate across profiling reps.
    return {
        "A": rng.integers(-8, 8, size=(m, n), dtype=np.int64),
        "x1": rng.integers(-8, 8, size=m, dtype=np.int64),
        "x2": rng.integers(-8, 8, size=n, dtype=np.int64),
        "y1": rng.integers(-8, 8, size=n, dtype=np.int64),
        "y2": rng.integers(-8, 8, size=m, dtype=np.int64),
    }


SPEC = KernelSpec(
    name="mvt_types",
    func=mvt_types,
    domain="linalg",
    descriptors={
        "A": dace.int64[M, N],
        "x1": dace.int64[M],
        "x2": dace.int64[N],
        "y1": dace.int64[N],
        "y2": dace.int64[M],
    },
    make_inputs=make_inputs,
    inout=("x1", "x2"),
    flops=lambda m=256, n=256: 4 * m * n,
    bytes_moved=lambda m=256, n=256: 16 * m * n,
    tags=("matvec", "two-scopes", "two-symbols"),
    types={"OP_TYPE": "int64"},
    presets={
        "i8": {
            "A": "int8",
            "y1": "int8",
            "y2": "int8",
            "x1": "int32",
            "x2": "int32",
            "OP_TYPE": "int32",
        },
        "i16": {
            "A": "int16",
            "y1": "int16",
            "y2": "int16",
            "x1": "int32",
            "x2": "int32",
            "OP_TYPE": "int32",
        },
        "i32": {
            "A": "int32",
            "y1": "int32",
            "y2": "int32",
            "x1": "int64",
            "x2": "int64",
            "OP_TYPE": "int64",
        },
    },
)
