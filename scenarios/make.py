"""Generate the checked-in SNAX-MODEL clusters and scenarios (MOD9, D41, D42).

    python scenarios/make.py            write every file below scenarios/
    python scenarios/make.py --check    exit 1 if a checked-in file differs

The scenario files hold plain command lists (D42). This script is where the
block-level helpers belong: it programs each block with
``RegisterMap.config_writes`` / ``start_writes`` and writes the resulting
commands in name form. test_scenario.py checks that the checked-in files
equal what this script generates, so they cannot drift apart.

Files:

    clusters/alu4.json   DMA, readers ra and rb, writer wr, elementwise add, 4 lanes
    clusters/red4.json   reader ra (4 lanes), writer wr (1 lane), reduce; no L2
    clusters/mul1.json   32 banks, 1-lane ra, rb, wr (2 loops), multiplier L = II = 5, DMA
    vecadd/              the MOD7 vecadd (test_profile.run_vecadd), the M3 target
    vecadd_conflict/     vecadd with b in the same banks as a (VIS3's conflict case)
    vecadd_tiled/        vecadd over 576 elements in 3 tiles of 192 (471 cycles)
    fmul/                a * b in 4 tiles of 24 on mul1, double buffered: DMA behind compute
    reduce/              64 elements in L1 summed in groups of 16
    dma/                 L2 -> L1 with a 2D pattern and back, on alu4
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

from snax_forge.snax_model import (
    ControllerConfig,
    DmaConfig,
    DmaDescriptor,
    DmaPattern,
    L1Config,
    L2Config,
    StreamerConfig,
    StreamerRegs,
    Wait,
)
from snax_forge.snax_model.scenario import (
    ClusterConfig,
    ComponentSpec,
    MemInit,
    NamedRead,
    RegisterMapSpec,
    Scenario,
    named,
    register_map_of,
    to_json,
)

ROOT = Path(__file__).resolve().parent
WORD = 8  # bytes per bank word (64-bit banks)
BEAT = 64  # bytes per wide beat (512 bits)
LANES = 4

# Controller costs of every scenario: one cycle per csr_write and csr_read on
# every block kind, a poll every 4 cycles. Declared defaults, not measured
# (D51, open item 10). test_profile.VECADD_CFG keeps its own non-default costs
# (DMA writes and reads 2) to exercise the D37 formulas; test_scenario runs the
# hand-built vecadd with these instead.
CTL = ControllerConfig(write_cost=1, read_cost=1, poll_interval=4)


# =============================================================================
# Clusters
# =============================================================================


def _streamer(name: str, write: bool, lanes: int, temporal_dims: int = 1) -> ComponentSpec:
    cfg = StreamerConfig(write=write, n_ports=lanes, fifo_depth=2, temporal_dims=temporal_dims)
    return ComponentSpec(name, "streamer", cfg.to_dict())


def alu4() -> ClusterConfig:
    """The cluster of test_profile's run_vecadd, in its registration order."""
    return ClusterConfig(
        l1=L1Config(n_banks=16, rows=64, read_latency=1),
        l2=L2Config(size_bytes=1 << 15, read_latency=1),
        components=[
            ComponentSpec("xbar", "xbar", {"check_hold": True}),
            ComponentSpec("dma", "dma", DmaConfig().to_dict()),
            _streamer("ra", False, LANES),
            _streamer("rb", False, LANES),
            _streamer("wr", True, LANES),
            ComponentSpec(
                "acc",
                "accel",
                accel="elementwise",
                params={"lanes": LANES, "n_inputs": 2, "op": "add", "latency": 0, "ii": 1},
                attach={"a": "ra", "b": "rb", "out": "wr"},
            ),
            ComponentSpec("ctl", "controller", CTL.to_dict()),
        ],
        register_map=RegisterMapSpec(blocks=["dma", "ra", "rb", "wr", "acc"]),
    )


def red4() -> ClusterConfig:
    """A reduce over 4 lanes into one lane (lanes_out = 1), L1 only."""
    return ClusterConfig(
        l1=L1Config(n_banks=16, rows=64, read_latency=1),
        components=[
            ComponentSpec("xbar", "xbar", {"check_hold": True}),
            _streamer("ra", False, LANES),
            _streamer("wr", True, 1),
            ComponentSpec(
                "acc",
                "accel",
                accel="reduce",
                params={"lanes": LANES, "lanes_out": 1, "op": "add", "latency": 1, "ii": 1},
                attach={"in": "ra", "out": "wr"},
            ),
            ComponentSpec("ctl", "controller", CTL.to_dict()),
        ],
        register_map=RegisterMapSpec(blocks=["ra", "wr", "acc"]),
    )


# =============================================================================
# Block values (the helpers of test_ctrl / test_profile)
# =============================================================================


# mul1: a multi-cycle multiplier with 1-lane streamers, for fmul.
MUL_BANKS = 32
MUL_LATENCY = 5
MUL_II = 5


def mul1() -> ClusterConfig:
    """One multiplier (1 lane, L = II = 5) between readers ra, rb and writer wr, and a DMA.

    32 banks = 4 superbanks of 8, so one tile's buffers fit in two
    superbanks and the DMA can work on the other two (fmul). The streamers
    have 2 temporal loops, to walk a buffer that lives in one superbank.
    """
    L, II = MUL_LATENCY, MUL_II
    return ClusterConfig(
        l1=L1Config(n_banks=MUL_BANKS, rows=32, read_latency=1),
        l2=L2Config(size_bytes=1 << 15, read_latency=1),
        components=[
            ComponentSpec("xbar", "xbar", {"check_hold": True}),
            ComponentSpec("dma", "dma", DmaConfig().to_dict()),
            _streamer("ra", False, 1, temporal_dims=2),
            _streamer("rb", False, 1, temporal_dims=2),
            _streamer("wr", True, 1, temporal_dims=2),
            ComponentSpec(
                "acc",
                "accel",
                accel="elementwise",
                params={"lanes": 1, "n_inputs": 2, "op": "mul", "latency": L, "ii": II},
                attach={"a": "ra", "b": "rb", "out": "wr"},
            ),
            ComponentSpec("ctl", "controller", CTL.to_dict()),
        ],
        register_map=RegisterMapSpec(blocks=["dma", "ra", "rb", "wr", "acc"]),
    )


def contiguous(base: int, n: int) -> DmaPattern:
    """n consecutive wide beats from byte ``base``."""
    return DmaPattern(base, (n,), (BEAT,))


def unit(word: int, n_beats: int, lanes: int) -> StreamerRegs:
    """Streamer task over n_beats contiguous beats of ``lanes`` words from word ``word``."""
    return StreamerRegs(word * WORD, (n_beats,), (lanes * WORD,), (lanes,), (WORD,))


class Program:
    """Collects commands in name form for one cluster's register map."""

    def __init__(self, cluster: ClusterConfig) -> None:
        self.map = register_map_of(cluster)
        self.cmds: list = []

    def config(self, block: str, arg) -> None:
        self.cmds += [named(c, self.map) for c in self.map.config_writes(block, arg)]

    def start(self, block: str) -> None:
        self.cmds.append(named(self.map.start_write(block), self.map))

    def wait(self, block: str, mode: str) -> None:
        self.cmds.append(Wait(block, mode))

    def read(self, reg: str) -> None:
        self.cmds.append(NamedRead(reg))


# =============================================================================
# Scenarios: (scenario, {npy file name: array})
# =============================================================================


def vecadd() -> tuple[Scenario, dict[str, np.ndarray]]:
    """test_profile.run_vecadd(mode="poll"): same data, program and cluster, costs of CTL."""
    return _vecadd("vecadd", wb=72)


def vecadd_conflict() -> tuple[Scenario, dict[str, np.ndarray]]:
    """vecadd with b at L1 word 64: b starts in bank 0 like a, so ra and rb collide (VIS3).

    Only b's L1 place changes: its DMA ``dst_base`` and ``rb.base`` (576 -> 512
    bytes). Data, L2 layout and cluster file are vecadd's.
    """
    return _vecadd("vecadd_conflict", wb=64)


def _vecadd(name: str, wb: int) -> tuple[Scenario, dict[str, np.ndarray]]:
    """The vecadd body; ``wb`` is b's L1 word address (72 in vecadd, bank 8)."""
    n_elems = 64
    nb, n_dma = n_elems // LANES, n_elems // 8
    rng = np.random.default_rng(3)
    a, b = rng.integers(-1000, 1000, n_elems), rng.integers(-1000, 1000, n_elems)
    l2a, l2b, l2c = 0, 1024, 2048
    wa, wc = 0, 144  # L1 word addresses of a and c

    def to_l1(src: int, word: int) -> DmaDescriptor:
        return DmaDescriptor("l2_to_l1", contiguous(src, n_dma), contiguous(word * WORD, n_dma))

    def to_l2(word: int, dst: int) -> DmaDescriptor:
        return DmaDescriptor("l1_to_l2", contiguous(word * WORD, n_dma), contiguous(dst, n_dma))

    cl = alu4()
    p = Program(cl)
    p.config("dma", to_l1(l2a, wa))
    p.start("dma")
    p.config("dma", to_l1(l2b, wb))
    p.wait("dma", "poll")
    p.start("dma")
    p.config("ra", unit(wa, nb, LANES))
    p.config("rb", unit(wb, nb, LANES))
    p.config("wr", unit(wc, nb, LANES))
    p.config("acc", {"n": nb})
    p.wait("dma", "poll")
    for blk in ("ra", "rb", "wr", "acc"):
        p.start(blk)
    p.config("dma", to_l2(wc, l2c))
    p.wait("wr", "poll")
    p.start("dma")
    p.wait("dma", "poll")
    sc = Scenario(
        name=name,
        cluster=cl,
        cluster_ref="../clusters/alu4.json",
        memory=[MemInit("l2", l2a, npy="a.npy"), MemInit("l2", l2b, npy="b.npy")],
        program=p.cmds,
        max_cycles=5000,
    )
    return sc, {"a.npy": a, "b.npy": b}


# vecadd_tiled: TILED_N elements in tiles of TILE, one tile of a, b and c in L1 at a time.
TILED_N = 576
TILE = 192


def vecadd_tiled() -> tuple[Scenario, dict[str, np.ndarray]]:
    """vecadd over TILED_N elements in tiles of TILE, for a run of several hundred cycles.

    alu4's L1 (1024 words) holds one tile each of a, b and c, so the tiles
    run one after another: load a, load b, add, store c, with no overlap
    (double buffering is J1's). b sits 8 banks after a as in vecadd, so the
    readers do not collide. a, b and c are contiguous in L2 like vecadd's.
    """
    n_tiles = TILED_N // TILE
    nb, n_dma = TILE // LANES, TILE // 8
    rng = np.random.default_rng(7)
    a, b = rng.integers(-1000, 1000, TILED_N), rng.integers(-1000, 1000, TILED_N)
    l2a, l2b, l2c = 0, TILED_N * WORD, 2 * TILED_N * WORD
    wa, wb, wc = 0, TILE + 8, 2 * TILE + 16  # L1 words: banks 0, 8 and 0

    def load(src: int, word: int) -> DmaDescriptor:
        return DmaDescriptor("l2_to_l1", contiguous(src, n_dma), contiguous(word * WORD, n_dma))

    def store(word: int, dst: int) -> DmaDescriptor:
        return DmaDescriptor("l1_to_l2", contiguous(word * WORD, n_dma), contiguous(dst, n_dma))

    cl = alu4()
    p = Program(cl)
    for k in range(n_tiles):
        off = k * TILE * WORD  # byte offset of tile k in each L2 array
        p.config("dma", load(l2a + off, wa))
        p.start("dma")
        p.config("dma", load(l2b + off, wb))
        p.wait("dma", "poll")
        p.start("dma")
        p.config("ra", unit(wa, nb, LANES))
        p.config("rb", unit(wb, nb, LANES))
        p.config("wr", unit(wc, nb, LANES))
        p.config("acc", {"n": nb})
        p.wait("dma", "poll")
        for blk in ("ra", "rb", "wr", "acc"):
            p.start(blk)
        p.config("dma", store(wc, l2c + off))
        p.wait("wr", "poll")
        p.start("dma")
        p.wait("dma", "poll")
    sc = Scenario(
        name="vecadd_tiled",
        cluster=cl,
        cluster_ref="../clusters/alu4.json",
        memory=[MemInit("l2", l2a, npy="a.npy"), MemInit("l2", l2b, npy="b.npy")],
        program=p.cmds,
        max_cycles=5000,
    )
    return sc, {"a.npy": a, "b.npy": b}


# fmul: FMUL_TILES tiles of FMUL_TILE elements, double buffered over two tile sets.
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


def fmul() -> tuple[Scenario, dict[str, np.ndarray]]:
    """c = a * b over FMUL_TILES tiles on mul1, the DMA hidden behind the multiplier.

    A buffer lives in one superbank: FMUL_TILE / 8 rows of 8 words, one
    wide beat per row, so the DMA pattern steps one L1 row (MUL_BANKS words)
    per beat and the streamers walk 8 words, then the next row. Tiles
    alternate between the two tile sets. While tile k runs in its set, the
    DMA stores c of tile k - 1 and loads a and b of tile k + 1 in the other
    set, and the controller programs the streamers for tile k + 1 (their
    registers are buffered, D36). A wide DMA grant only blocks its own
    superbank (D33), so the streamers never wait for it. Integer data (D28).
    """
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


def reduce() -> tuple[Scenario, dict[str, np.ndarray]]:
    """64 elements at L1 word 0 summed in 4 groups of 16 (16 beats, T = 4) to word 128."""
    x = np.random.default_rng(5).integers(-1000, 1000, 64)
    nb, t = 64 // LANES, 4
    cl = red4()
    p = Program(cl)
    p.config("ra", unit(0, nb, LANES))
    p.config("wr", unit(128, nb // t, 1))
    p.config("acc", {"n": nb, "T": t})
    for blk in ("ra", "wr", "acc"):
        p.start(blk)
    p.wait("wr", "signal")
    p.wait("acc", "poll")
    p.read("acc.busy_cycles")
    sc = Scenario(
        name="reduce",
        cluster=cl,
        cluster_ref="../clusters/red4.json",
        memory=[MemInit("l1", 0, npy="x.npy")],
        program=p.cmds,
        max_cycles=1000,
    )
    return sc, {"x.npy": x}


# The 2D L1 pattern of the DMA scenario: 8 beats 1 KiB apart, twice, 64 B apart.
DMA_L1 = DmaPattern(0, (8, 2), (1024, BEAT))
DMA_BEATS = 16
DMA_BACK = 4096  # L2 byte address of the copy back


def dma() -> tuple[Scenario, dict[str, np.ndarray]]:
    """16 beats from L2 into L1 with DMA_L1, then back to L2 at DMA_BACK, on alu4."""
    src = np.random.default_rng(11).integers(-1000, 1000, DMA_BEATS * BEAT // WORD)
    cl = alu4()
    p = Program(cl)
    p.config("dma", DmaDescriptor("l2_to_l1", contiguous(0, DMA_BEATS), DMA_L1))
    p.start("dma")
    p.wait("dma", "signal")
    p.config("dma", DmaDescriptor("l1_to_l2", DMA_L1, contiguous(DMA_BACK, DMA_BEATS)))
    p.start("dma")
    p.wait("dma", "poll")
    sc = Scenario(
        name="dma",
        cluster=cl,
        cluster_ref="../clusters/alu4.json",
        memory=[MemInit("l2", 0, npy="src.npy")],
        program=p.cmds,
        max_cycles=2000,
    )
    return sc, {"src.npy": src}


# =============================================================================
# Files
# =============================================================================


def _npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr, dtype=np.int64), allow_pickle=False)
    return buf.getvalue()


def generate() -> dict[str, bytes]:
    """Every file, by path relative to scenarios/, with its exact contents."""
    files = {
        "clusters/alu4.json": to_json(alu4().to_dict()).encode(),
        "clusters/red4.json": to_json(red4().to_dict()).encode(),
        "clusters/mul1.json": to_json(mul1().to_dict()).encode(),
    }
    for make in (vecadd, vecadd_conflict, vecadd_tiled, fmul, reduce, dma):
        sc, arrays = make()
        files[f"{sc.name}/scenario.json"] = to_json(sc.to_dict()).encode()
        for name, arr in arrays.items():
            files[f"{sc.name}/{name}"] = _npy(arr)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="only compare, write nothing")
    args = ap.parse_args(argv)
    stale = []
    for rel, data in generate().items():
        path = ROOT / rel
        if path.is_file() and path.read_bytes() == data:
            continue
        stale.append(rel)
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    if args.check and stale:
        print("out of date: " + ", ".join(stale), file=sys.stderr)
        return 1
    print(("stale: " if args.check else "written: ") + (", ".join(stale) or "nothing"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
