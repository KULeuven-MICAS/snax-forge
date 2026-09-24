"""fmul: c = a * b over 5 tiles of 16 on mul1, double buffered, the DMA hidden behind compute.

Layout (all in tasks.json): a buffer lives in one superbank, 2 rows of 8
words, one wide beat per row, so a DMA pattern steps one L1 row (32 words,
256 bytes) per beat and the streamers walk 8 words, then the next row. Tiles
alternate between two tile sets: set 0 uses superbanks 0 (a) and 1 (b, and c
4 rows down), set 1 superbanks 2 and 3. While tile k runs in its set, the
DMA stores c of tile k - 1 and loads a and b of tile k + 1 in the other set,
and the controller programs the streamers for tile k + 1 (their registers
are buffered, D36). A wide DMA grant only blocks its own superbank (D33), so
the streamers never wait for it. In L2, a, b and c are contiguous, 640 bytes
apart. Integer data (D28).

The program is lowered from tasks.json (D64, D65) and is the one fmul was
scheduled by hand with before (open item 26, closed by D66). Six of its syncs
are there only to keep that program: the one before starting load_a_1, the
one before each store_c_k start, and the second sync on store_c_3. Each is a
wait on a DMA that was already waited for, one cycle each; without them fmul
runs 519 cycles instead of 525.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from clusters.clusters import mul1

from snax_forge.lower import TaskList, lower_program
from snax_forge.snax_model.scenario import MemInit, Scenario

HERE = Path(__file__).resolve().parent
N = 5 * 16  # 5 tiles of 16 elements
L2_A, L2_B = 0, N * 8  # byte addresses of a and b in L2; c is stored at 2 * N * 8


def make() -> tuple[Scenario, dict[str, np.ndarray]]:
    rng = np.random.default_rng(13)
    a, b = rng.integers(-1000, 1000, N), rng.integers(-1000, 1000, N)
    cl = mul1()
    sc = Scenario(
        name="fmul",
        cluster=cl,
        cluster_ref="../clusters/mul1.json",
        memory=[MemInit("l2", L2_A, npy="a.npy"), MemInit("l2", L2_B, npy="b.npy")],
        program=lower_program(TaskList.load(HERE / "tasks.json"), cl),
        max_cycles=5000,
    )
    return sc, {"a.npy": a, "b.npy": b}
