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
    vecadd/              the MOD7 vecadd (test_profile.run_vecadd), the ANC2 basis
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

# Controller costs of the MOD7 vecadd (test_profile.VECADD_CFG). Placeholders
# until ANC2 calibrates them (open item 10).
CTL = ControllerConfig(write_cost=1, kind_write_cost={"dma": 2}, read_cost=2, poll_interval=4)


# =============================================================================
# Clusters
# =============================================================================


def _streamer(name: str, write: bool, lanes: int) -> ComponentSpec:
    cfg = StreamerConfig(write=write, n_ports=lanes, fifo_depth=2)
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
    """test_profile.run_vecadd(mode="poll"): same data, same program, same cluster."""
    n_elems = 64
    nb, n_dma = n_elems // LANES, n_elems // 8
    rng = np.random.default_rng(3)
    a, b = rng.integers(-1000, 1000, n_elems), rng.integers(-1000, 1000, n_elems)
    l2a, l2b, l2c = 0, 1024, 2048
    wa, wb, wc = 0, 72, 144  # L1 word addresses

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
        name="vecadd",
        cluster=cl,
        cluster_ref="../clusters/alu4.json",
        memory=[MemInit("l2", l2a, npy="a.npy"), MemInit("l2", l2b, npy="b.npy")],
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
    }
    for make in (vecadd, reduce, dma):
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
