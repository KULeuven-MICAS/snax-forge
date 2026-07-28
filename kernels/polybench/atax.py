"""PolyBench atax: y = A.T (A x).

Produces ZERO map entries -- the whole kernel becomes Transpose + MatMul +
MatMul library nodes. A map-scope walker finds nothing here, which is why
library-node expansion is mandatory rather than optional. Also the canonical
chaining barrier: the second reduction cannot start until the first completes.
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

M, N = dace.symbol("M"), dace.symbol("N")


def atax(A, x, y):
    y[:] = A.T @ (A @ x)


def make_inputs(rng, m=256, n=256):
    return {
        "A": rng.integers(-4, 4, size=(m, n), dtype=np.int32),
        "x": rng.integers(-4, 4, size=n, dtype=np.int32),
        "y": np.zeros(n, dtype=np.int32),  # assigned, not accumulated
    }


SPEC = KernelSpec(
    name="atax",
    func=atax,
    domain="linalg",
    descriptors={"A": dace.int32[M, N], "x": dace.int32[N], "y": dace.int32[N]},
    make_inputs=make_inputs,
    inout=("y",),
    flops=lambda m=256, n=256: 4 * m * n,
    bytes_moved=lambda m=256, n=256: 8 * m * n,
    tags=("chaining-barrier", "no-maps", "library-nodes"),
    sweep_sizes=(
        {"m": 128, "n": 128},
        {"m": 256, "n": 256},
        {"m": 512, "n": 512},
        {"m": 128, "n": 512},  # non-square: M != N exercises bind_symbols
        {"m": 512, "n": 128},  # transposed aspect ratio
    ),
)
