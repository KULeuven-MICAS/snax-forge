"""Elementwise integer divide: C = A // B, B nonzero.

Tier B (PAPER §6.1): the multi-cycle unit, a divider with II > 1. Staged:
not yet linked to the flow. int64 (D113). A and B take both signs, so the
reference's floor (NumPy, Python) and C's truncation differ on some
elements; the divider's BRM states which one it computes.
"""

import dace
import numpy as np

from snax_forge.sdfg.spec import KernelSpec

N = dace.symbol("N")


def vdiv(A, B, C):
    C[:] = A // B


def make_inputs(rng, n=1024):
    return {
        "A": rng.integers(-1000, 1000, size=n, dtype=np.int64),
        "B": rng.integers(1, 16, size=n, dtype=np.int64) * rng.choice([-1, 1], size=n),
        "C": np.zeros(n, dtype=np.int64),
    }


SPEC = KernelSpec(
    name="vdiv",
    func=vdiv,
    domain="elementwise",
    descriptors={"A": dace.int64[N], "B": dace.int64[N], "C": dace.int64[N]},
    make_inputs=make_inputs,
    inout=("C",),
    tags=("staged", "divide"),
)
