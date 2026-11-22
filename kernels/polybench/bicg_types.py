"""PolyBench bicg with type presets: bicg's two products, p and r cast to ``OP_TYPE``.

int64 by default, bicg's graph: the products keep bicg's names (``q_mult_map``,
``s_mult_map``, ...), as they come from the assigned outputs. One cast per product is enough:
NumPy runs ``@`` at its operands' result type, so A is widened inside the op. p and r are cast,
not ``A.T`` (a cast there is a Transpose the importer refuses). Tier A (PAPER §6.1).
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

M, N, OP_TYPE = dace.symbol("M"), dace.symbol("N"), np.int64


def bicg_types(A, p, q, r, s):
    q[:] = A @ p.astype(OP_TYPE)
    s[:] = A.T @ r.astype(OP_TYPE)


def make_inputs(rng, m=256, n=256):
    return {
        "A": rng.integers(-8, 8, size=(m, n), dtype=np.int64),
        "p": rng.integers(-8, 8, size=n, dtype=np.int64),
        "q": np.zeros(m, dtype=np.int64),  # assigned, not accumulated
        "r": rng.integers(-8, 8, size=m, dtype=np.int64),
        "s": np.zeros(n, dtype=np.int64),
    }


SPEC = KernelSpec(
    name="bicg_types",
    func=bicg_types,
    domain="linalg",
    descriptors={
        "A": dace.int64[M, N],
        "p": dace.int64[N],
        "q": dace.int64[M],
        "r": dace.int64[M],
        "s": dace.int64[N],
    },
    make_inputs=make_inputs,
    inout=("q", "s"),
    flops=lambda m=256, n=256: 4 * m * n,
    bytes_moved=lambda m=256, n=256: 16 * m * n,  # A read twice
    tags=("matvec", "two-scopes", "no-chaining"),
    types={"OP_TYPE": "int64"},
    presets={
        "i8": {
            "A": "int8",
            "p": "int8",
            "r": "int8",
            "q": "int32",
            "s": "int32",
            "OP_TYPE": "int32",
        },
        "i16": {
            "A": "int16",
            "p": "int16",
            "r": "int16",
            "q": "int32",
            "s": "int32",
            "OP_TYPE": "int32",
        },
        "i32": {
            "A": "int32",
            "p": "int32",
            "r": "int32",
            "q": "int64",
            "s": "int64",
            "OP_TYPE": "int64",
        },
    },
)
