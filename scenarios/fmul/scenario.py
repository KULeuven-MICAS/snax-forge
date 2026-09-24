"""fmul: c = a * b over 5 tiles of 16 on mul1, double buffered, the DMA hidden behind compute.

A buffer lives in one superbank: FMUL_TILE / 8 rows of 8 words, one wide
beat per row, so the DMA pattern steps one L1 row (MUL_BANKS words) per beat
and the streamers walk 8 words, then the next row. Tiles alternate between
the two tile sets. While tile k runs in its set, the DMA stores c of tile
k - 1 and loads a and b of tile k + 1 in the other set, and the controller
programs the streamers for tile k + 1 (their registers are buffered, D36). A
wide DMA grant only blocks its own superbank (D33), so the streamers never
wait for it. Integer data (D28).

Scheduled by hand with ``Program``, not as a task list: its program has two
waits a task list would not emit and configures the last store after its
waits (open item 26), so lowering a task list would change its cycles.
"""

from __future__ import annotations

import numpy as np
from clusters.clusters import MUL_BANKS, mul1
from common import WORD, contiguous

from snax_forge.lower import Program
from snax_forge.snax_model import DmaDescriptor, DmaPattern, StreamerRegs
from snax_forge.snax_model.scenario import MemInit, Scenario

FMUL_TILES = 5
FMUL_TILE = 16  # elements = words; a multiple of 8, so no padding in a beat
SB_WORDS = 8  # words per superbank row (one wide beat)
# L1 buffers per tile set: (superbank, first row of the set's rows in it).
# Set 0 uses superbanks 0 and 1, set 1 superbanks 2 and 3; a alone in one,
# b and c (c 4 rows further down) in the other.
FMUL_SETS = (
    {"a": (0, 0), "b": (1, 0), "c": (1, 4)},
    {"a": (2, 0), "b": (3, 0), "c": (3, 4)},
)


def make() -> tuple[Scenario, dict[str, np.ndarray]]:
    n = FMUL_TILES * FMUL_TILE
    rows = FMUL_TILE // SB_WORDS
    row_bytes = MUL_BANKS * WORD  # one L1 row
    rng = np.random.default_rng(13)
    a, b = rng.integers(-1000, 1000, n), rng.integers(-1000, 1000, n)
    l2 = {"a": 0, "b": n * WORD, "c": 2 * n * WORD}

    def l1_base(k: int, buf: str) -> int:
        sb, row = FMUL_SETS[k % 2][buf]
        return row * row_bytes + sb * SB_WORDS * WORD

    def l1_beats(k: int, buf: str) -> DmaPattern:
        return DmaPattern(l1_base(k, buf), (rows,), (row_bytes,))

    def l2_beats(k: int, buf: str) -> DmaPattern:
        return contiguous(l2[buf] + k * FMUL_TILE * WORD, rows)

    def walk(k: int, buf: str) -> StreamerRegs:
        return StreamerRegs(l1_base(k, buf), (SB_WORDS, rows), (WORD, row_bytes), (1,), (WORD,))

    cl = mul1()
    p = Program(cl)

    def load(k: int) -> None:
        """a and b of tile k into its set; ends with the DMA running b's load."""
        p.config("dma", DmaDescriptor("l2_to_l1", l2_beats(k, "a"), l1_beats(k, "a")))
        if k:  # tile 0's load is the DMA's first task: nothing to wait for
            p.wait("dma", "poll")
        p.start("dma")
        p.config("dma", DmaDescriptor("l2_to_l1", l2_beats(k, "b"), l1_beats(k, "b")))
        p.wait("dma", "poll")
        p.start("dma")

    def store(k: int) -> None:
        p.config("dma", DmaDescriptor("l1_to_l2", l1_beats(k, "c"), l2_beats(k, "c")))
        p.wait("dma", "poll")
        p.start("dma")

    def program(k: int) -> None:
        p.config("ra", walk(k, "a"))
        p.config("rb", walk(k, "b"))
        p.config("wr", walk(k, "c"))
        p.config("acc", {"n": FMUL_TILE})

    load(0)
    program(0)
    for k in range(FMUL_TILES):
        p.wait("dma", "poll")  # a and b of tile k are in L1 (and c of k - 2 is out)
        for blk in ("ra", "rb", "wr", "acc"):
            p.start(blk)
        # While tile k computes, the other set: c of k - 1 out, a and b of k + 1 in.
        if k:
            store(k - 1)
        if k + 1 < FMUL_TILES:
            load(k + 1)
            program(k + 1)
        p.wait("wr", "poll")
    p.wait("dma", "poll")
    store(FMUL_TILES - 1)
    p.wait("dma", "poll")
    sc = Scenario(
        name="fmul",
        cluster=cl,
        cluster_ref="../clusters/mul1.json",
        memory=[MemInit("l2", l2["a"], npy="a.npy"), MemInit("l2", l2["b"], npy="b.npy")],
        program=p.cmds,
        max_cycles=5000,
    )
    return sc, {"a.npy": a, "b.npy": b}
