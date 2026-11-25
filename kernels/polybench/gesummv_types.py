# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause (kernels/NPBENCH_LICENSE)
# From NPBench (github.com/spcl/npbench, f2d7f27): npbench/benchmarks/polybench/gesummv/gesummv_dace.py
"""PolyBench gesummv with type presets: NPBench's order, x cast to ``OP_TYPE``.

int64 by default, gesummv's graph but for the casts. The cast makes each matvec's type explicit
in the source, so moving the scale past the sum (``factor_out``) changes no type: at ``i8`` A
and B int8 times x int8 at int32, alpha and beta int32, out int32. Tier A (PAPER §6.1).
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N, OP_TYPE = dace.symbol("N"), np.int64


def gesummv_types(alpha, beta, A, B, x, out):
    out[:] = alpha * A @ x.astype(OP_TYPE) + beta * B @ x.astype(OP_TYPE)


def make_inputs(rng, n=64):
    return {
        "alpha": np.int64(3),
        "beta": np.int64(2),
        "A": rng.integers(-8, 8, size=(n, n), dtype=np.int64),
        "B": rng.integers(-8, 8, size=(n, n), dtype=np.int64),
        "x": rng.integers(-8, 8, size=n, dtype=np.int64),
        "out": np.zeros(n, dtype=np.int64),
    }


SPEC = KernelSpec(
    name="gesummv_types",
    func=gesummv_types,
    domain="linalg",
    descriptors={
        "alpha": dace.int64,
        "beta": dace.int64,
        "A": dace.int64[N, N],
        "B": dace.int64[N, N],
        "x": dace.int64[N],
        "out": dace.int64[N],
    },
    make_inputs=make_inputs,
    inout=("out",),
    tags=("matvec", "scalars"),
    types={"OP_TYPE": "int64"},
    presets={
        "i8": {
            "A": "int8",
            "B": "int8",
            "x": "int8",
            "alpha": "int32",
            "beta": "int32",
            "out": "int32",
            "OP_TYPE": "int32",
        },
        "i16": {
            "A": "int16",
            "B": "int16",
            "x": "int16",
            "alpha": "int32",
            "beta": "int32",
            "out": "int32",
            "OP_TYPE": "int32",
        },
        "i32": {
            "A": "int32",
            "B": "int32",
            "x": "int32",
            "alpha": "int64",
            "beta": "int64",
            "out": "int64",
            "OP_TYPE": "int64",
        },
    },
)
