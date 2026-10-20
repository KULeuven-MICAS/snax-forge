"""PolyBench mvt: two independent matrix-vector products over A and A.T.

Two symbols (M, N) and two top-level map scopes -- the first kernel implying
more than one accelerator from a single SDFG.

Tier A (PAPER §6.1). int64 (D113).
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

M, N = dace.symbol("M"), dace.symbol("N")


def mvt(A, x1, x2, y1, y2):
    x1 += A @ y1
    x2 += A.T @ y2


def make_inputs(rng, m=256, n=256):
    # Small range: x1/x2 accumulate across profiling reps. int64 wraps rather
    # than faults, so timing stays valid, but values go junk after many calls.
    return {
        "A": rng.integers(-8, 8, size=(m, n), dtype=np.int64),
        "x1": rng.integers(-8, 8, size=m, dtype=np.int64),
        "x2": rng.integers(-8, 8, size=n, dtype=np.int64),
        "y1": rng.integers(-8, 8, size=n, dtype=np.int64),
        "y2": rng.integers(-8, 8, size=m, dtype=np.int64),
    }


SPEC = KernelSpec(
    name="mvt",
    func=mvt,
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
    flops=lambda m=256, n=256: 4 * m * n,  # two matvecs, MAC each
    bytes_moved=lambda m=256, n=256: 16 * m * n,  # A read twice
    tags=("matvec", "two-scopes", "two-symbols"),
    sweep_sizes=(
        {"m": 128, "n": 128},
        {"m": 256, "n": 256},
        {"m": 512, "n": 512},
        {"m": 128, "n": 512},  # non-square: M != N exercises bind_symbols
        {"m": 512, "n": 128},  # transposed aspect ratio
    ),
)
