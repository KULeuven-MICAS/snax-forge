"""dma: 16 beats from L2 into L1 with a 2D pattern, then back to L2 at byte 4096, on alu4.

The L1 pattern is 8 beats 1 KiB apart, twice, 64 B apart. The program is
lowered from tasks.json (D64, D65): the load is synced by signal, the store
by poll.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from clusters.clusters import alu4
from common import BEAT, WORD

from snax_forge.lower import TaskList, lower_program
from snax_forge.snax_model.scenario import MemInit, Scenario

HERE = Path(__file__).resolve().parent
BEATS = 16


def make() -> tuple[Scenario, dict[str, np.ndarray]]:
    src = np.random.default_rng(11).integers(-1000, 1000, BEATS * BEAT // WORD)
    cl = alu4()
    sc = Scenario(
        name="dma",
        cluster=cl,
        cluster_ref="../clusters/alu4.json",
        memory=[MemInit("l2", 0, npy="src.npy")],
        program=lower_program(TaskList.load(HERE / "tasks.json"), cl),
        max_cycles=2000,
    )
    return sc, {"src.npy": src}
