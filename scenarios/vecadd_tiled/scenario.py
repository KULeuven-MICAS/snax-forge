"""vecadd_tiled: vecadd over 576 elements in 3 tiles of 192, for a longer run (471 cycles).

alu4's L1 (1024 words) holds one tile each of a, b and c, so the tiles run
one after another with no overlap (double buffering is fmul's). In L1, a is
at byte 0, b at 1600 (8 banks after a, so the readers do not collide) and c
at 3200. a, b and c are contiguous in L2, 4608 bytes apart; tile k is at
+1536 * k. Each tile's tasks also name in ``after`` the tasks of the
previous tile that use the same L1 buffer; the sync on the previous store
has covered those, so they add no wait. The program is lowered from
tasks.json (D64, D65).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from clusters.clusters import alu4

from snax_forge.lower import TaskList, lower_program
from snax_forge.snax_model.scenario import MemInit, Scenario

HERE = Path(__file__).resolve().parent
N = 576
L2_A, L2_B = 0, 4608  # byte addresses of a and b in L2; c is stored at 9216


def make() -> tuple[Scenario, dict[str, np.ndarray]]:
    rng = np.random.default_rng(7)
    a, b = rng.integers(-1000, 1000, N), rng.integers(-1000, 1000, N)
    cl = alu4()
    sc = Scenario(
        name="vecadd_tiled",
        cluster=cl,
        cluster_ref="../clusters/alu4.json",
        memory=[MemInit("l2", L2_A, npy="a.npy"), MemInit("l2", L2_B, npy="b.npy")],
        program=lower_program(TaskList.load(HERE / "tasks.json"), cl),
        max_cycles=5000,
    )
    return sc, {"a.npy": a, "b.npy": b}
