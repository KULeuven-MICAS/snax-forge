"""vecadd_conflict: vecadd with b in the same banks as a (VIS3's conflict case).

b sits at L1 word 64, so it starts in bank 0 like a and acc_a and acc_b
collide. Only b's L1 place differs from vecadd: load_B's ``dst`` base and add_acc_b's
``base`` in tasks.json (576 -> 512 bytes). Data, L2 layout and cluster file
are vecadd's. The program is lowered from tasks.json (D64, D65).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from clusters.clusters import alu4

from snax_forge.lower import TaskList, lower_program
from snax_forge.snax_model.scenario import MemInit, Region, Scenario

HERE = Path(__file__).resolve().parent
N = 64
L2_A, L2_B = 0, 512  # byte addresses of a and b in L2, packed; c is stored at 1024
# Where A, B and C live, as the memory plan gives them (D95): per container its L2, then L1.
PLACES = {"A": (0, 0), "B": (512, 512), "C": (1024, 1152)}


def make() -> tuple[Scenario, dict[str, np.ndarray]]:
    rng = np.random.default_rng(3)
    a, b = rng.integers(-1000, 1000, N), rng.integers(-1000, 1000, N)
    cl = alu4()
    sc = Scenario(
        name="vecadd_conflict",
        cluster=cl,
        cluster_ref="../clusters/alu4.json",
        memory=[MemInit("l2", L2_A, npy="a.npy"), MemInit("l2", L2_B, npy="b.npy")],
        program=lower_program(TaskList.load(HERE / "tasks.json"), cl),
        max_cycles=5000,
        regions=[
            Region(c, mem, base, (N,), (8,))
            for c, bases in PLACES.items()
            for mem, base in zip(("l2", "l1"), bases, strict=True)
        ],
    )
    return sc, {"a.npy": a, "b.npy": b}
