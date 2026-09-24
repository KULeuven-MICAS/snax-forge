"""reduce: 64 elements at L1 word 0 summed in 4 groups of 16 (16 beats, T = 4) to word 128.

On red4, L1 only. The program is lowered from tasks.json (D64, D65): the
three tasks start together, then the program syncs on the writer by signal
and on the accelerator by poll, and reads the accelerator's busy_cycles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from clusters.clusters import red4

from snax_forge.lower import TaskList, lower_program
from snax_forge.snax_model.scenario import MemInit, Scenario

HERE = Path(__file__).resolve().parent


def make() -> tuple[Scenario, dict[str, np.ndarray]]:
    x = np.random.default_rng(5).integers(-1000, 1000, 64)
    cl = red4()
    sc = Scenario(
        name="reduce",
        cluster=cl,
        cluster_ref="../clusters/red4.json",
        memory=[MemInit("l1", 0, npy="x.npy")],
        program=lower_program(TaskList.load(HERE / "tasks.json"), cl),
        max_cycles=1000,
    )
    return sc, {"x.npy": x}
