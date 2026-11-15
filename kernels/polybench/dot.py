"""
Dot product via reduction. DaCe yields a Reduce library node, not a WCR memlet.

int64 by default: the importer turns the Reduce into a sum map whose output memlet folds with
``add``, and integer sums match exactly in any order. The products run at ``ACC``; the type
presets ``i8``, ``i16`` and ``i32`` are in ``SPEC`` (kernels/README.md).

Tier done (PAPER §6.1).
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N = dace.symbol("N")
ACC = np.int64


def dot(A, B, out):
    out[0] = np.sum(A.astype(ACC) * B.astype(ACC))


def make_inputs(rng, n=1024):
    return {
        "A": rng.integers(-1000, 1000, size=n, dtype=np.int64),
        "B": rng.integers(-1000, 1000, size=n, dtype=np.int64),
        "out": np.zeros(1, dtype=np.int64),
    }


SPEC = KernelSpec(
    name="dot",
    func=dot,
    domain="reduction",
    descriptors={"A": dace.int64[N], "B": dace.int64[N], "out": dace.int64[1]},
    make_inputs=make_inputs,
    inout=("out",),
    tags=("reduction",),
    flops=lambda n=1024: 2 * n,
    bytes_moved=lambda n=1024: 2 * 8 * n + 8,
    sweep_sizes=(1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18),
    types={"ACC": "int64"},
    presets={
        "i8": {"A": "int8", "B": "int8", "out": "int32", "ACC": "int32"},
        "i16": {"A": "int16", "B": "int16", "out": "int32", "ACC": "int32"},
        "i32": {"A": "int32", "B": "int32", "out": "int64", "ACC": "int64"},
    },
)
