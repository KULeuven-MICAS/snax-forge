# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause (kernels/NPBENCH_LICENSE)
# From NPBench (github.com/spcl/npbench, f2d7f27): npbench/benchmarks/deep_learning/resnet/resnet_dace.py,
# and resnet_numpy.py as the reference
"""ResNet-50's bottleneck block, inference: 1x1, 3x3 and 1x1 convs with batchnorm and ReLU, plus the skip.

Tier E (PAPER §6.1). Staged: not yet linked to the flow. NPBench's float32,
kept until tier E decides its integer form. NPBench's return becomes the
output argument out; its resnet_basicblock_gpu is left out. The body
calls DaCe programs, so the reference is NPBench's NumPy version. N = 2 by
default: batchnorm averages over the batch, and one image would make it zero.
"""

import dace as dc
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N, H, W, C1, C2, S0, S1, S2, S3, S4, S5 = (
    dc.symbol(s, dtype=dc.int64)
    for s in ("N", "H", "W", "C1", "C2", "S0", "S1", "S2", "S3", "S4", "S5")
)


@dc.program
def relu(x: dc.float32[S0, S1, S2, S3]):
    return np.maximum(x, 0)


# Deep learning convolutional operator (stride = 1)
@dc.program
def conv2d(input: dc.float32[S0, S1, S2, S3], weights: dc.float32[S4, S4, S3, S5]):
    # K = weights.shape[0]  # Assuming square kernel
    # N = input.shape[0]
    # H_out = input.shape[1] - K + 1
    # W_out = input.shape[2] - K + 1
    # C_out = weights.shape[3]
    # output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)
    output = np.ndarray((S0, S1 - S4 + 1, S2 - S4 + 1, S5), dtype=np.float32)

    # Loop structure adapted from https://github.com/SkalskiP/ILearnDeepLearning.py/blob/ba0b5ba589d4e656141995e8d1a06d44db6ce58d/01_mysteries_of_neural_networks/06_numpy_convolutional_neural_net/src/layers/convolutional.py#L88
    # for i, j in dc.map[0:H-K+1, 0:W-K+1]:
    for i in range(S1 - S4 + 1):
        for j in range(S2 - S4 + 1):
            output[:, i, j, :] = np.sum(
                input[:, i : i + S4, j : j + S4, :, np.newaxis] * weights[np.newaxis, :, :, :],
                axis=(1, 2, 3),
            )

    return output


# Batch normalization operator, as used in ResNet
@dc.program
def batchnorm2d(x: dc.float32[S0, S1, S2, S3]):
    # mean = np.mean(x, axis=0, keepdims=True)
    mean = np.ndarray((1, S1, S2, S3), dtype=np.float32)
    mean[:] = np.mean(x, axis=0)
    # std = np.std(x, axis=0, keepdims=True)
    std = np.ndarray((1, S1, S2, S3), dtype=np.float32)
    # std[:] = np.sqrt(np.sum((x - mean) ** 2, axis=0) / np.float32(S0))
    std[:] = np.sqrt(np.sum((x - mean) * (x - mean), axis=0) / np.float32(S0))
    # return (x - mean) / np.sqrt(std + eps)
    return (x - mean) / np.sqrt(std + 1e-5)


# Bottleneck residual block (after initial convolution, without downsampling)
# in the ResNet-50 CNN (inference)
def resnet_basicblock(input, conv1, conv2, conv3, out):
    # Pad output of first convolution for second convolution
    # padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2,
    #                    conv1.shape[3]))
    padded = np.zeros((N, H + 2, W + 2, C2), dtype=np.float32)

    padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1)
    x = batchnorm2d(padded)
    # x = relu(x)

    # x = conv2d(x, conv2)
    # x = batchnorm2d(x)
    # x = relu(x)
    # x = conv2d(x, conv3)
    # x = batchnorm2d(x)
    # return relu(x + input)
    x1 = relu(x)

    x2 = conv2d(x1, conv2)
    x3 = batchnorm2d(x2)
    x4 = relu(x3)
    x5 = conv2d(x4, conv3)
    x6 = batchnorm2d(x5)
    out[:] = relu(x6 + input)


def resnet_basicblock_ref(input, conv1, conv2, conv3, out):
    def relu(x):
        return np.maximum(x, 0)

    def conv2d(input, weights):
        K = weights.shape[0]  # Assuming square kernel
        N = input.shape[0]
        H_out = input.shape[1] - K + 1
        W_out = input.shape[2] - K + 1
        C_out = weights.shape[3]
        output = np.empty((N, H_out, W_out, C_out), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                output[:, i, j, :] = np.sum(
                    input[:, i : i + K, j : j + K, :, np.newaxis] * weights[np.newaxis, :, :, :],
                    axis=(1, 2, 3),
                )
        return output

    def batchnorm2d(x, eps=1e-5):
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        return (x - mean) / np.sqrt(std + eps)

    # Pad output of first convolution for second convolution
    padded = np.zeros((input.shape[0], input.shape[1] + 2, input.shape[2] + 2, conv1.shape[3]))

    padded[:, 1:-1, 1:-1, :] = conv2d(input, conv1)
    x = batchnorm2d(padded)
    x = relu(x)

    x = conv2d(x, conv2)
    x = batchnorm2d(x)
    x = relu(x)
    x = conv2d(x, conv3)
    x = batchnorm2d(x)
    out[:] = relu(x + input)


def make_inputs(rng, n=2, h=4, w=4, c1=8, c2=4):
    f32 = lambda *shape: rng.random(shape, dtype=np.float32)
    return {
        "input": f32(n, h, w, c1),
        "conv1": f32(1, 1, c1, c2),
        "conv2": f32(3, 3, c2, c2),
        "conv3": f32(1, 1, c2, c1),
        "out": np.zeros((n, h, w, c1), dtype=np.float32),
    }


SPEC = KernelSpec(
    name="resnet",
    func=resnet_basicblock,
    domain="dl",
    descriptors={
        "input": dc.float32[N, H, W, C1],
        "conv1": dc.float32[1, 1, C1, C2],
        "conv2": dc.float32[3, 3, C2, C2],
        "conv3": dc.float32[1, 1, C2, C1],
        "out": dc.float32[N, H, W, C1],
    },
    make_inputs=make_inputs,
    inout=("out",),
    ref=resnet_basicblock_ref,
    tags=("staged", "float-staged", "conv", "batchnorm", "skip", "nested-programs"),
)
