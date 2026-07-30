"""ReLU activation: B = max(A, 0).

First non-arithmetic kernel -- a comparator and select rather than an adder,
which is different hardware. The op shows up in the tasklet `calls` histogram
rather than `ops`, so it also tests that side of the compute extractor.
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N = dace.symbol("N")


def relu(A, B):
    B[:] = np.maximum(A, 0)


def make_inputs(rng, n=1024):
    return {
        "A": rng.integers(-1000, 1000, size=n, dtype=np.int32),  # both signs
        "B": np.zeros(n, dtype=np.int32),
    }


SPEC = KernelSpec(
    name="relu",
    func=relu,
    domain="activation",
    descriptors={"A": dace.int32[N], "B": dace.int32[N]},
    make_inputs=make_inputs,
    inout=("B",),
    flops=lambda n=1024: n,  # one compare-select per element
    bytes_moved=lambda n=1024: 2 * 4 * n,
    sweep_sizes=(1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18),
    tags=("activation", "select", "streamable"),
)
