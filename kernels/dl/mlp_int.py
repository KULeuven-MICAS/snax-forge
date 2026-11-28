"""NPBench's MLP at batch 1 in integers: three dense layers, requantized between them.

Layers 1 and 2: the int32 sum ``z = w @ a + b``, then ``a = min(max(z // 1024, 0), 127)`` as
int8 (a shift by 10 and a clamp whose lower end is the relu); layer 3's sums are the logits,
``out``. Every matrix is stored as (D_out, D_in) and read along its rows, the row form;
``mlp_int_col3`` stores layer 3 the other way, for the column form. The shift is 10 at both
sizes, so one requant serves both layers. Casts are written out and literals typed: NumPy wraps
int8 arithmetic, and DaCe widens an untyped literal's result to 64 bits. The biases are drawn
from ``BIAS`` (``ranges``), the order of the sums. Sizes: 64-64-64-32 (the defaults) and
256-256-128-10. ``kernels/dl/README.md`` has the equations and a worked example.
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

D0, D1, D2, D3 = (dace.symbol(s) for s in ("D0", "D1", "D2", "D3"))
BIAS = (-(2**15), 2**15 - 1)
SMALL = {"d0": 64, "d1": 64, "d2": 64, "d3": 32}
LARGE = {"d0": 256, "d1": 256, "d2": 128, "d3": 10}


def mlp_int(x, w1, b1, w2, b2, w3, b3, out):
    z1 = w1 @ x.astype(np.int32) + b1
    a1 = np.minimum(np.maximum(z1 // np.int32(1024), 0), 127).astype(np.int8)
    z2 = w2 @ a1.astype(np.int32) + b2
    a2 = np.minimum(np.maximum(z2 // np.int32(1024), 0), 127).astype(np.int8)
    out[:] = w3 @ a2.astype(np.int32) + b3


def make_inputs(rng, d0=64, d1=64, d2=64, d3=32):
    """x and the weights over all of int8, the biases over ``BIAS``; W_l as (D_l, D_{l-1})."""

    def i8(*shape):
        return rng.integers(-128, 127, size=shape, dtype=np.int8, endpoint=True)

    def bias(n):
        return rng.integers(*BIAS, size=n, dtype=np.int32, endpoint=True)

    return {
        "x": i8(d0),
        "w1": i8(d1, d0),
        "b1": bias(d1),
        "w2": i8(d2, d1),
        "b2": bias(d2),
        "w3": i8(d3, d2),
        "b3": bias(d3),
        "out": np.zeros(d3, dtype=np.int32),
    }


def macs(d0=64, d1=64, d2=64, d3=32):
    return d0 * d1 + d1 * d2 + d2 * d3


SPEC = KernelSpec(
    name="mlp_int",
    func=mlp_int,
    domain="dl",
    descriptors={
        "x": dace.int8[D0],
        "w1": dace.int8[D1, D0],
        "b1": dace.int32[D1],
        "w2": dace.int8[D2, D1],
        "b2": dace.int32[D2],
        "w3": dace.int8[D3, D2],
        "b3": dace.int32[D3],
        "out": dace.int32[D3],
    },
    make_inputs=make_inputs,
    inout=("out",),
    flops=lambda **s: 2 * macs(**s),
    bytes_moved=lambda d0=64, d1=64, d2=64, d3=32: (
        d0 + macs(d0, d1, d2, d3) + 4 * (d1 + d2 + 2 * d3)
    ),
    tags=("matvec", "multi-phase"),
    sweep_sizes=(SMALL, LARGE),
    ranges={"b1": BIAS, "b2": BIAS, "b3": BIAS},
)
