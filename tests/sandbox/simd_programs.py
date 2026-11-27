"""Small programs for the simd options the workloads need, written as their kernels write them.

DaCe reads a program's source, so they live in this file: ``requant`` (an MLP layer's int32
sums shifted and clamped into int8), ``requant_by_1000`` (a divisor that is no power of two),
``gray`` (planar RGB to luma, a tree of three products), ``blur_rows`` and ``blur_cols`` (the
1-2-1 blur along a row and down a column, three reads of one container) and ``threshold`` (255
where V >= t, t a scalar).

No ``from __future__ import annotations`` here: DaCe reads a program's type hints as objects.
"""

import dace
import numpy as np

N = dace.symbol("N")
H, W = dace.symbol("H"), dace.symbol("W")


@dace.program
def requant(z: dace.int32[N], a: dace.int8[N]):
    a[:] = np.minimum(np.maximum(z // np.int32(1024), 0), 127).astype(np.int8)


@dace.program
def requant_by_1000(z: dace.int32[N], a: dace.int8[N]):
    a[:] = np.minimum(np.maximum(z // np.int32(1000), 0), 127).astype(np.int8)


@dace.program
def gray(R: dace.uint8[N], G: dace.uint8[N], B: dace.uint8[N], Y: dace.uint8[N]):
    Y[:] = (
        (np.uint16(77) * R.astype(np.uint16) + np.uint16(150) * G.astype(np.uint16)
         + np.uint16(29) * B.astype(np.uint16)) // np.uint16(256)
    ).astype(np.uint8)  # fmt: skip


@dace.program
def blur_rows(Y: dace.uint8[H, W], X: dace.uint8[H, W - 2]):
    X[:] = ((Y[:, :-2].astype(np.uint16) + np.uint16(2) * Y[:, 1:-1] + Y[:, 2:])
            // np.uint16(4)).astype(np.uint8)  # fmt: skip


@dace.program
def blur_cols(X: dace.uint8[H, W], V: dace.uint8[H - 2, W]):
    V[:] = ((X[:-2, :].astype(np.uint16) + np.uint16(2) * X[1:-1, :] + X[2:, :])
            // np.uint16(4)).astype(np.uint8)  # fmt: skip


@dace.program
def threshold(V: dace.uint8[N], t: dace.int16, T: dace.uint8[N]):
    T[:] = np.int16(255) * np.minimum(np.maximum(V.astype(np.int16) - t + np.int16(1), 0), 1)
