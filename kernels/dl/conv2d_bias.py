# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause (kernels/NPBENCH_LICENSE)
# From NPBench (github.com/spcl/npbench, f2d7f27): npbench/benchmarks/deep_learning/conv2d_bias/
# conv2d_dace.py, and conv2d_numpy.py as the reference
"""A 2-D convolution, stride 1, no padding, NHWC, plus a bias per output channel.

Tier E (PAPER §6.1). Staged: not yet linked to the flow. int64 (D113): plain
integer arithmetic, so NPBench's float32 becomes int64. NPBench's return
becomes the output argument out. The body allocates by the symbols and
calls a DaCe program, so the reference is NPBench's NumPy version.
"""

import dace as dc
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

C_in, C_out, H, K, N, W = (dc.symbol(s, dc.int64) for s in ("C_in", "C_out", "H", "K", "N", "W"))


# Deep learning convolutional operator (stride = 1)
@dc.program
def conv2d(input: dc.int64[N, H, W, C_in], weights: dc.int64[K, K, C_in, C_out]):
    # K = weights.shape[0]  # Assuming square kernel
    # N = input.shape[0]
    # H_out = input.shape[1] - K + 1
    # W_out = input.shape[2] - K + 1
    # C_out = weights.shape[3]
    # output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)
    output = np.ndarray((N, H - K + 1, W - K + 1, C_out), dtype=np.int64)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    # for i, j in dc.map[0:H-K+1, 0:W-K+1]:
    for i in range(H - K + 1):
        for j in range(W - K + 1):
            output[:, i, j, :] = np.sum(
                input[:, i : i + K, j : j + K, :, np.newaxis] * weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


def conv2d_bias(input, weights, bias, out):
    out[:] = conv2d(input, weights) + bias


def conv2d_bias_ref(input, weights, bias, out):
    K = weights.shape[0]  # Assuming square kernel
    N = input.shape[0]
    H_out = input.shape[1] - K + 1
    W_out = input.shape[2] - K + 1
    C_out = weights.shape[3]
    output = np.empty((N, H_out, W_out, C_out), dtype=np.int64)
    for i in range(H_out):
        for j in range(W_out):
            output[:, i, j, :] = np.sum(
                input[:, i : i + K, j : j + K, :, np.newaxis] * weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )
    out[:] = output + bias


def make_inputs(rng, n=1, h=10, w=10, c_in=3, c_out=8, k=3):
    return {
        "input": rng.integers(-8, 8, size=(n, h, w, c_in), dtype=np.int64),
        "weights": rng.integers(-8, 8, size=(k, k, c_in, c_out), dtype=np.int64),
        "bias": rng.integers(-8, 8, size=c_out, dtype=np.int64),
        "out": np.zeros((n, h - k + 1, w - k + 1, c_out), dtype=np.int64),
    }


SPEC = KernelSpec(
    name="conv2d_bias",
    func=conv2d_bias,
    domain="dl",
    descriptors={
        "input": dc.int64[N, H, W, C_in],
        "weights": dc.int64[K, K, C_in, C_out],
        "bias": dc.int64[C_out],
        "out": dc.int64[N, H - K + 1, W - K + 1, C_out],
    },
    make_inputs=make_inputs,
    inout=("out",),
    ref=conv2d_bias_ref,
    tags=("staged", "conv", "reduction", "nested-programs"),
)
