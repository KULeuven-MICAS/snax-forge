# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause (kernels/NPBENCH_LICENSE)
# From NPBench (github.com/spcl/npbench, f2d7f27): npbench/benchmarks/deep_learning/mlp/mlp_dace.py,
# and mlp_numpy.py as the reference
"""A 3-layer MLP: two dense layers with ReLU, then a dense layer and softmax.

Tier E (PAPER §6.1). Staged: not yet linked to the flow. NPBench's float32,
kept until tier E decides its integer form. NPBench's return becomes the
output argument out. The body calls DaCe programs, so the reference is
NPBench's NumPy version.
"""

import dace as dc
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

C_in, N, S0, S1, S2, N1, N2 = (
    dc.symbol(s, dtype=dc.int64) for s in ("C_in", "N", "S0", "S1", "S2", "N1", "N2")
)


@dc.program
def relu(x: dc.float32[N1, N2]):
    return np.maximum(x, 0)


# Numerically-stable version of softmax
@dc.program
def softmax(x: dc.float32[N1, N2]):
    # tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_max = np.maximum.reduce(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    # tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    tmp_sum = np.add.reduce(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


# 3-layer MLP
def mlp(input, w1, b1, w2, b2, w3, b3, out):
    x1 = relu(input @ w1 + b1)
    x2 = relu(x1 @ w2 + b2)
    x3 = softmax(x2 @ w3 + b3)  # Softmax call can be omitted if necessary
    out[:] = x3


def mlp_ref(input, w1, b1, w2, b2, w3, b3, out):
    def relu(x):
        return np.maximum(x, 0)

    def softmax(x):
        tmp_max = np.max(x, axis=-1, keepdims=True)
        tmp_out = np.exp(x - tmp_max)
        tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
        return tmp_out / tmp_sum

    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    out[:] = softmax(x @ w3 + b3)


def make_inputs(rng, n=8, c_in=32, s0=32, s1=32, s2=16):
    f32 = lambda *shape: rng.random(shape, dtype=np.float32)
    return {
        "input": f32(n, c_in),
        "w1": f32(c_in, s0),
        "b1": f32(s0),
        "w2": f32(s0, s1),
        "b2": f32(s1),
        "w3": f32(s1, s2),
        "b3": f32(s2),
        "out": np.zeros((n, s2), dtype=np.float32),
    }


SPEC = KernelSpec(
    name="mlp",
    func=mlp,
    domain="dl",
    descriptors={
        "input": dc.float32[N, C_in],
        "w1": dc.float32[C_in, S0],
        "b1": dc.float32[S0],
        "w2": dc.float32[S0, S1],
        "b2": dc.float32[S1],
        "w3": dc.float32[S1, S2],
        "b3": dc.float32[S2],
        "out": dc.float32[N, S2],
    },
    make_inputs=make_inputs,
    inout=("out",),
    ref=mlp_ref,
    tags=("staged", "float-staged", "matmul", "multi-phase", "nested-programs"),
)
