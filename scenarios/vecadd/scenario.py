"""vecadd: c = a + b over 64 elements on alu4, one tile; the M3 target.

The MOD7 vecadd (test_profile.run_vecadd(mode="poll")): same data, program
and cluster, with the controller costs of clusters.CTL. The program is
lowered from the hand-written tasks.json (D64, D65).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from clusters.clusters import alu4

from snax_forge.lower import TaskList, lower_program
from snax_forge.snax_model.scenario import MemInit, Scenario

HERE = Path(__file__).resolve().parent
N = 64
L2_A, L2_B = 0, 1024  # byte addresses of a and b in L2; c is stored at 2048


def make() -> tuple[Scenario, dict[str, np.ndarray]]:
    rng = np.random.default_rng(3)
    a, b = rng.integers(-1000, 1000, N), rng.integers(-1000, 1000, N)
    cl = alu4()
    sc = Scenario(
        name="vecadd",
        cluster=cl,
        cluster_ref="../clusters/alu4.json",
        memory=[MemInit("l2", L2_A, npy="a.npy"), MemInit("l2", L2_B, npy="b.npy")],
        program=lower_program(TaskList.load(HERE / "tasks.json"), cl),
        max_cycles=5000,
    )
    return sc, {"a.npy": a, "b.npy": b}
