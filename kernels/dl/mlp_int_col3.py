"""``mlp_int`` with layer 3's matrix stored as (D2, D3), for the column form.

Layer 3 reads ``w3.T @ a2``: one input of a2 at a time, broadcast over the D3 output lanes,
each row of w3 that input's weights. The weights are constants, so the layout is free (no
transpose at run time). Same inputs as ``mlp_int`` but w3 transposed, so both give the same
logits on the same seed.
"""

import dace
import numpy as np

from snax_forge.sdfg.loader import load
from snax_forge.sdfg.spec import KernelSpec

BASE = load("mlp_int")
D2, D3 = dace.symbol("D2"), dace.symbol("D3")


def mlp_int_col3(x, w1, b1, w2, b2, w3, b3, out):
    z1 = w1 @ x.astype(np.int32) + b1
    a1 = np.minimum(np.maximum(z1 // np.int32(1024), 0), 127).astype(np.int8)
    z2 = w2 @ a1.astype(np.int32) + b2
    a2 = np.minimum(np.maximum(z2 // np.int32(1024), 0), 127).astype(np.int8)
    out[:] = w3.T @ a2.astype(np.int32) + b3


def make_inputs(rng, d0=64, d1=64, d2=64, d3=32):
    """``mlp_int``'s inputs, w3 as (D2, D3)."""
    inputs = BASE.make_inputs(rng, d0, d1, d2, d3)
    inputs["w3"] = np.ascontiguousarray(inputs["w3"].T)
    return inputs


SPEC = KernelSpec(
    name="mlp_int_col3",
    func=mlp_int_col3,
    domain="dl",
    descriptors={**BASE.descriptors, "w3": dace.int8[D2, D3]},
    make_inputs=make_inputs,
    inout=("out",),
    flops=BASE.flops,
    bytes_moved=BASE.bytes_moved,
    tags=BASE.tags,
    sweep_sizes=BASE.sweep_sizes,
    ranges=BASE.ranges,
)
