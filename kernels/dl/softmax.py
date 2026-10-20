# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause (kernels/NPBENCH_LICENSE)
# From NPBench (github.com/spcl/npbench, f2d7f27): npbench/benchmarks/deep_learning/softmax/softmax_dace.py
"""Softmax over the last axis, numerically stable: exp(x - max) / sum(exp(x - max)).

Tier C (PAPER §6.1): three phases (max, then exp and sum, then divide).
Staged: not yet linked to the flow. NPBench's float32, kept until tier C
decides its integer form; NPBench's softmax_gpu, which already writes
into out, is the form taken.
"""

import dace as dc
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N, H, SM = (dc.symbol(s, dc.int64) for s in ("N", "H", "SM"))


# Numerically-stable version of softmax
def softmax(x, out):
    # tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_max = np.maximum.reduce(x, axis=-1, keepdims=True, initial=-9999)
    tmp_out = np.exp(x - tmp_max)
    # tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    tmp_sum = np.add.reduce(tmp_out, axis=-1, keepdims=True)
    out[:] = tmp_out / tmp_sum


def make_inputs(rng, n=1, h=2, sm=16):
    return {
        "x": rng.random((n, h, sm, sm), dtype=np.float32),
        "out": np.zeros((n, h, sm, sm), dtype=np.float32),
    }


SPEC = KernelSpec(
    name="softmax",
    func=softmax,
    domain="dl",
    descriptors={"x": dc.float32[N, H, SM, SM], "out": dc.float32[N, H, SM, SM]},
    make_inputs=make_inputs,
    inout=("out",),
    tags=("staged", "float-staged", "multi-phase", "reduction"),
)
