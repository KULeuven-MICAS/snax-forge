"""Tests for the SNAX-MODEL L2 and DMA (MOD6, D33, D34).

MOD6 acceptance (docs/STATUS.md): DMA bandwidth test; DMA-streamer
contention shows up in the counters. Plus: address patterns against a NumPy
enumeration, overlap of DMA and compute, a vecadd from L2 to L2, and skip
on/off identical under random descriptors and streamer traffic.

The multi-width port tests of the xbar are in test_xbar.py (section 8).

Most tests use 16 banks: two superbanks of 8 banks (the 512-bit wide port).

Sections:
  1. helpers
  2. L2
  3. address patterns and descriptor checks
  4. bandwidth: the cycle formula in both directions
  5. start and done
  6. contention with streamers
  7. overlap with compute
  8. vecadd end to end, L2 -> L1 -> L2
  9. random: skip on/off identical
"""

import random

import numpy as np
import pytest

from snax_forge.snax_model import (
    Cluster,
    Component,
    Dma,
    DmaConfig,
    DmaDescriptor,
    DmaPattern,
    L1Config,
    L1Memory,
    L2AccessError,
    L2Config,
    L2Memory,
    Phase,
    SimulationError,
    Streamer,
    StreamerConfig,
    StreamerRegs,
    Xbar,
    check_descriptor,
    elementwise_stub,
)
from snax_forge.snax_model.accel import Accelerator

WORD = 8  # bytes per bank word
BEAT = 64  # bytes per wide beat (512 bits)
NB = 16  # banks: two superbanks


# =============================================================================
# 1. Helpers
# =============================================================================


def build(skip=True, n_banks=NB, rows=64, l1_lat=1, l2_lat=1, l2_size=1 << 15):
    """Cluster with L1, xbar and L2. Returns (cluster, l1, xbar, l2)."""
    cl = Cluster(skip_idle=skip)
    mem = L1Memory(cl, L1Config(n_banks=n_banks, rows=rows, read_latency=l1_lat))
    xb = cl.add(Xbar("xbar", mem))
    l2 = L2Memory(cl, L2Config(size_bytes=l2_size, read_latency=l2_lat))
    return cl, mem, xb, l2


class LogDma(Dma):
    """Dma that logs the cycles in which it read and wrote a beat."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.read_cycles, self.write_cycles = [], []

    def commit(self, cycle):
        if self._w.src_moved:
            self.read_cycles.append(cycle)
        if self._w.dst_moved:
            self.write_cycles.append(cycle)
        super().commit(cycle)


class LogStreamer(Streamer):
    """Streamer that logs its grant cycles."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.grants = []

    def commit(self, cycle):
        if self._fire.any():
            self.grants.append(cycle)
        super().commit(cycle)


class ScriptedStarts(Component):
    """Starts components in scripted cycles: {cycle: [(comp, arg), ...]}."""

    phases = (Phase.CONTROL,)

    def __init__(self, name, script):
        super().__init__(name)
        self.script = script

    def tick(self, cycle, phase):
        for comp, arg in self.script.get(cycle, ()):
            comp.start(arg, cycle)

    def next_wake(self, cycle):
        later = [c for c in self.script if c > cycle]
        return min(later) if later else None


class Chain(Component):
    """Starts descriptors one after another: each ``delay`` cycles after the last is done."""

    phases = (Phase.CONTROL,)

    def __init__(self, name, dma, items):
        super().__init__(name)
        self.dma, self.items, self.i = dma, items, 0  # items: [(delay, desc)]
        self.starts = []

    def _at(self):
        base = self.dma.done_cycle if self.i else 0
        return base + self.items[self.i][0]

    def tick(self, cycle, phase):
        if self.i < len(self.items) and not self.dma.busy and cycle >= self._at():
            self.dma.start(self.items[self.i][1], cycle)
            self.starts.append(cycle)
            self.i += 1

    def next_wake(self, cycle):
        if self.i >= len(self.items) or self.dma.busy or self.dma._start.pending:
            return None  # asked again after every cycle: sees the DMA finish
        return max(cycle + 1, self._at())


def contiguous(base, n):
    return DmaPattern(base, (n,), (BEAT,))


def expected_total(cfg, n, src_latency):
    """done_cycle - start cycle for N beats without contention (dma.py module doc)."""
    return cfg.startup + src_latency + cfg.beat_interval * (n - 1) + 2 + cfg.done_latency


def assert_cycles_add_up(comp, total):
    assert sum(comp.cycles.values()) == total, comp.cycles


# =============================================================================
# 2. L2
# =============================================================================


def test_l2_timing_and_backdoor():
    cl = Cluster()
    l2 = L2Memory(cl, L2Config(size_bytes=1024, read_latency=3))
    l2.load(64, np.arange(8) + 10)
    assert list(l2.dump(64, 8)[:, 0]) == list(range(10, 18))
    ready = l2.read(5, 64, tag="x")
    l2.write(5, 64, 0)  # same cycle: the read still sees the old data
    l2.commit()
    assert ready == 8 and l2.next_response(5) == 8
    r = l2.resp(8)
    assert r.tag == "x" and list(r.data[:, 0]) == list(range(10, 18))
    assert (l2.dump(64, 8) == 0).all()


def test_l2_one_read_and_one_write_per_cycle():
    cl = Cluster()
    l2 = L2Memory(cl, L2Config(size_bytes=1024))
    l2.read(0, 0)
    l2.write(0, 64, 1)  # a read and a write together are fine
    with pytest.raises(L2AccessError):
        l2.read(0, 128)
    with pytest.raises(L2AccessError):
        l2.write(0, 192, 1)


def test_l2_alignment_and_range():
    cl = Cluster()
    l2 = L2Memory(cl, L2Config(size_bytes=1024))
    for bad in (8, 1024, -64):
        with pytest.raises(SimulationError):
            l2.read(0, bad)
    with pytest.raises(ValueError):
        L2Config(size_bytes=100)


# =============================================================================
# 3. Address patterns and descriptor checks
# =============================================================================


def superbanks_of(mem, addrs):
    """Per beat: (superbank, row) from the L1's own group_of."""
    out = []
    for a in addrs:
        banks = mem.group_of(int(a), 512)
        out.append((banks[0] // 8, mem.locate(int(a))[1]))
    return out


PATTERNS = {
    # name: (pattern with 32 banks, NumPy enumeration)
    "contiguous": (
        DmaPattern(0, (6,), (BEAT,)),
        np.arange(6) * BEAT,
    ),
    "one_superbank_rows": (  # stride n_banks * 8 = 256 B: superbank 0, rows 0..5
        DmaPattern(0, (6,), (32 * WORD,)),
        np.arange(6) * 256,
    ),
    "mix_two_superbanks": (  # 2 beats across superbanks, then the next row, 3 rows
        DmaPattern(BEAT, (2, 3), (BEAT, 32 * WORD)),
        BEAT + (np.arange(2)[None, :] * BEAT + np.arange(3)[:, None] * 256).ravel(),
    ),
}


@pytest.mark.parametrize("name", list(PATTERNS))
@pytest.mark.parametrize("skip", [True, False])
def test_patterns_match_numpy_and_land_in_their_banks(name, skip):
    """Beat addresses equal NumPy; a transfer writes exactly those superbank rows."""
    pat, expect = PATTERNS[name]
    assert list(pat.addresses()) == list(expect)
    cl, mem, xb, l2 = build(skip, n_banks=32)
    n = pat.n_beats
    data = np.arange(n * 8) + 1000
    l2.load(0, data)
    dma = cl.add(Dma("dma", xb, l2))
    dma.start(DmaDescriptor("l2_to_l1", contiguous(0, n), pat))
    cl.run()

    where = superbanks_of(mem, expect)
    if name == "contiguous":
        assert where == [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1)]
    elif name == "one_superbank_rows":
        assert where == [(0, r) for r in range(6)]
    else:
        assert where == [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2)]
    for k, a in enumerate(expect):
        assert list(mem.dump(int(a), 8)[:, 0]) == list(data[8 * k : 8 * k + 8])
    # Only the banks of the superbanks used were written, 8 writes per beat.
    used = {sb for sb, _ in where}
    per_sb = mem.writes.reshape(4, 8)
    assert all((per_sb[sb] > 0).all() == (sb in used) for sb in range(4))
    assert mem.writes.sum() == 8 * n


def test_descriptor_checks():
    l1, l2 = L1Config(n_banks=NB, rows=8), L2Config(size_bytes=4096)  # L1: 1 KiB
    ok = DmaDescriptor("l2_to_l1", contiguous(0, 4), contiguous(0, 4))
    src, dst = check_descriptor(ok, l1, l2)
    assert list(src) == list(dst) == [0, 64, 128, 192]
    bad = [
        DmaDescriptor("l2_to_l1", contiguous(0, 4), contiguous(8, 4)),  # misaligned L1 base
        DmaDescriptor("l2_to_l1", contiguous(0, 2), DmaPattern(0, (2,), (72,))),  # stride
        DmaDescriptor("l2_to_l1", contiguous(0, 4), contiguous(1024 - 128, 4)),  # leaves L1
        DmaDescriptor("l2_to_l1", contiguous(0, 2), DmaPattern(64, (2,), (-128,))),  # below L1
        DmaDescriptor("l1_to_l2", contiguous(0, 2), contiguous(4096 - 64, 2)),  # leaves L2
    ]
    for d in bad:
        with pytest.raises(ValueError):
            check_descriptor(d, l1, l2)
    with pytest.raises(ValueError):
        DmaDescriptor("l2_to_l1", contiguous(0, 3), contiguous(0, 4))  # beat counts differ
    with pytest.raises(ValueError):
        DmaDescriptor("l1_to_l1", contiguous(0, 1), contiguous(0, 1))


def test_dma_needs_a_matching_wide_port():
    cl = Cluster()
    xb = cl.add(Xbar("xbar", L1Memory(cl, L1Config(n_banks=4))))
    with pytest.raises(ValueError):  # 512 bits need a multiple of 8 banks
        Dma("dma", xb, L2Memory(cl, L2Config()))
    cl, _, xb, _ = build()
    with pytest.raises(ValueError):  # L2 beat must equal the wide port
        Dma("dma", xb, L2Memory(cl, L2Config(beat_bits=256)))


# =============================================================================
# 4. Bandwidth: the cycle formula in both directions
# =============================================================================

TIMINGS = [
    # (n, startup, l1_lat, l2_lat, beat_interval, l1_read_extra, done_latency)
    (1, 2, 1, 1, 1, 0, 0),
    (20, 2, 1, 1, 1, 0, 0),
    (20, 1, 0, 0, 1, 0, 0),
    (13, 3, 2, 5, 1, 2, 3),
    (9, 2, 1, 4, 2, 1, 0),
    (7, 4, 1, 2, 3, 0, 2),
]


@pytest.mark.parametrize("direction", ["l2_to_l1", "l1_to_l2"])
@pytest.mark.parametrize("timing", TIMINGS)
@pytest.mark.parametrize("skip", [True, False])
def test_uncontended_transfer_hits_formula(direction, timing, skip):
    """done_cycle - s = startup + Ls + k (N - 1) + 2 + done_latency, one beat per k cycles."""
    n, startup, l1_lat, l2_lat, k, extra, done = timing
    cfg = DmaConfig(startup=startup, beat_interval=k, l1_read_extra=extra, done_latency=done)
    cl, mem, xb, l2 = build(skip, l1_lat=l1_lat, l2_lat=l2_lat)
    dma = cl.add(LogDma("dma", xb, l2, cfg))
    rng = np.random.default_rng(n + k)
    data = rng.integers(-1000, 1000, n * 8)
    if direction == "l2_to_l1":
        l2.load(512, data)
        desc = DmaDescriptor(direction, contiguous(512, n), contiguous(128, n))
        src_lat = l2_lat
    else:
        mem.load(128, data)
        desc = DmaDescriptor(direction, contiguous(128, n), contiguous(512, n))
        src_lat = l1_lat + extra
    s = 5
    cl.add(ScriptedStarts("ctl", {s: [(dma, desc)]}))
    total = cl.run()

    assert dma.done_cycle - s == expected_total(cfg, n, src_lat)
    assert dma.read_cycles == [s + startup + k * i for i in range(n)]
    first_write = s + startup + src_lat + 1
    assert dma.write_cycles == [first_write + k * i for i in range(n)]
    got = mem.dump(128, 8 * n) if direction == "l2_to_l1" else l2.dump(512, 8 * n)
    assert list(got[:, 0]) == list(data)
    assert total == dma.done_cycle  # nothing else runs after the DMA
    assert dma.cycles["stall_l1"] == 0
    assert dma.cycles["busy"] == len(set(dma.read_cycles) | set(dma.write_cycles))
    assert_cycles_add_up(dma, total)
    assert xb.port_stalls.sum() == 0


# =============================================================================
# 5. Start and done
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
def test_zero_beats_and_restart(skip):
    cl, _, xb, l2 = build(skip)
    dma = cl.add(Dma("dma", xb, l2))
    zero = DmaDescriptor("l2_to_l1", DmaPattern(0, (0,), (BEAT,)), DmaPattern(0, (0,), (BEAT,)))
    one = DmaDescriptor("l2_to_l1", contiguous(0, 2), contiguous(0, 2))
    cl.add(ScriptedStarts("ctl", {3: [(dma, zero)], 10: [(dma, one)], 30: [(dma, one)]}))
    probe = []

    class Probe(Component):
        phases = (Phase.CONTROL,)

        def tick(self, cycle, phase):
            probe.append((cycle, dma.busy))

        def next_wake(self, cycle):
            return cycle + 1 if cycle < 40 else None

    cl.add(Probe("probe"))
    total = cl.run()
    busy = [c for c, b in probe if b]
    t = expected_total(DmaConfig(), 2, 1)
    assert busy == [*range(11, 10 + t), *range(31, 30 + t)]  # zero-beat task: never busy
    assert dma.done_cycle == 30 + t
    assert_cycles_add_up(dma, total)


def test_start_while_busy_raises():
    cl, _, xb, l2 = build()
    dma = cl.add(Dma("dma", xb, l2))
    d = DmaDescriptor("l2_to_l1", contiguous(0, 2), contiguous(0, 2))
    dma.start(d)
    with pytest.raises(SimulationError):
        dma.start(d)


def test_dims_limits_the_loops():
    """DmaConfig.dims (the loops the registers hold, MOD7): more loops is rejected."""
    cl, _, xb, l2 = build()
    dma = cl.add(Dma("dma", xb, l2, DmaConfig(dims=1)))
    flat = contiguous(0, 4)
    nested = DmaPattern(0, (2, 2), (BEAT, 2 * BEAT))
    with pytest.raises(ValueError, match="dims"):
        dma.start(DmaDescriptor("l2_to_l1", flat, nested))
    dma.start(DmaDescriptor("l2_to_l1", flat, flat))
    with pytest.raises(ValueError):
        DmaConfig(dims=0)


# =============================================================================
# 6. Contention with streamers
# =============================================================================


def rows16(bank, row0, n):
    """Streamer regs: n beats of one lane on ``bank``, rows row0, row0+1, ..."""
    return StreamerRegs((row0 * NB + bank) * WORD, (n,), (NB * WORD,))


@pytest.mark.parametrize("skip", [True, False])
def test_reader_on_dma_superbank_stalls_reader_elsewhere_does_not(skip):
    """DMA writes superbank 0 in cycles 3..8; a reader on bank 1 waits, one on bank 9 does not.

    DMA started before the run (as in cycle -1): first L2 read in 1, data
    visible in 3, writes 3..8. Readers issue from cycle 1: the bank 1 reader
    is granted in 1, 2, then refused in 3..8 (wider grant), then 9..12.
    """
    cl, mem, xb, l2 = build(skip)
    mem.load(0, np.arange(NB * 64) + 5)  # every word = its index + 5
    n = 6
    r1 = cl.add(LogStreamer("r1", xb, StreamerConfig(fifo_depth=8)))
    r9 = cl.add(LogStreamer("r9", xb, StreamerConfig(fifo_depth=8)))
    dma = cl.add(LogDma("dma", xb, l2))
    l2.load(0, np.arange(n * 8) + 7000)
    # DMA: rows 0..5 of superbank 0 (stride 16 banks * 8 B); readers use rows 8..13.
    dma.start(DmaDescriptor("l2_to_l1", contiguous(0, n), DmaPattern(0, (n,), (NB * WORD,))))
    r1.start(rows16(1, 8, n))
    r9.start(rows16(9, 8, n))
    total = cl.run()

    assert dma.write_cycles == list(range(3, 9))
    assert r1.grants == [1, 2, 9, 10, 11, 12]
    assert r9.grants == list(range(1, 7))
    assert xb.port_stalls_wider[r1.ports[0]] == 6 == xb.port_stalls[r1.ports[0]]
    assert xb.port_stalls[r9.ports[0]] == 0
    assert dma.done_cycle == expected_total(DmaConfig(), n, 1) - 1  # started in -1: not slowed
    # Data on both sides.
    for bank, r in ((1, r1), (9, r9)):
        got = [int(np.asarray(x).ravel()[0]) for x in r.fifo._q[0]]
        assert got == [(row * NB + bank) + 5 for row in range(8, 8 + n)]
    for k in range(n):
        assert list(mem.dump(k * NB * WORD, 8)[:, 0]) == list(range(7000 + 8 * k, 7008 + 8 * k))
    assert_cycles_add_up(dma, total)


@pytest.mark.parametrize("skip", [True, False])
def test_two_wide_ports_on_one_superbank_stall_l1(skip):
    """Not in SNAX (one DMA), but the model allows it: two DMAs with their own L2.

    Both write superbank 0 from cycle 3 on; equal width, so D31 round-robin
    per group alternates them, and the refused one holds (``stall_l1``).
    """
    cl, mem, xb, l2a = build(skip)
    l2b = L2Memory(cl, L2Config(size_bytes=1 << 15))
    n = 4
    da = cl.add(LogDma("da", xb, l2a))
    db = cl.add(LogDma("db", xb, l2b))
    l2a.load(0, np.arange(8 * n) + 100)
    l2b.load(0, np.arange(8 * n) + 900)
    row = NB * WORD
    da.start(DmaDescriptor("l2_to_l1", contiguous(0, n), DmaPattern(0, (n,), (row,))))
    db.start(DmaDescriptor("l2_to_l1", contiguous(0, n), DmaPattern(n * row, (n,), (row,))))
    total = cl.run()
    assert da.write_cycles == [3, 5, 7, 9] and db.write_cycles == [4, 6, 8, 10]
    # Refused: da in 4, 6, 8; db in 3, 5, 7, 9. Cycles 3 and 4 still have an L2
    # read (reads in 1..4), so they count as busy.
    assert da.cycles["stall_l1"] == 2 and db.cycles["stall_l1"] == 3
    assert xb.port_stalls_wider.sum() == 0
    for k in range(n):
        assert mem.dump(k * row, 8)[0, 0] == 100 + 8 * k
        assert mem.dump((n + k) * row, 8)[0, 0] == 900 + 8 * k
    for d in (da, db):
        assert_cycles_add_up(d, total)


# =============================================================================
# 7. Overlap with compute
# =============================================================================


def compute_parts(cl, mem, xb, n_beats, bases, latency=0):
    """Two 4-lane readers + elementwise add + 4-lane writer. Returns (acc, (ra, rb, wr))."""
    lanes = 4

    def regs(base):
        return StreamerRegs(base, (n_beats,), (NB * WORD,), (lanes,), (WORD,))  # row by row

    cfg = StreamerConfig(n_ports=lanes, fifo_depth=2)
    ra = cl.add(LogStreamer("ra", xb, cfg))
    rb = cl.add(LogStreamer("rb", xb, cfg))
    wr = cl.add(LogStreamer("wr", xb, StreamerConfig(write=True, n_ports=lanes, fifo_depth=2)))
    acc = cl.add(Accelerator("acc", cl, elementwise_stub(lanes=lanes, latency=latency)))
    acc.attach("a", ra.fifo)
    acc.attach("b", rb.fifo)
    acc.attach("out", wr.fifo)
    a, b, c = bases
    return acc, (ra, rb, wr), [(ra, regs(a)), (rb, regs(b)), (wr, regs(c)), (acc, {"n": n_beats})]


def run_overlap(skip, with_dma, with_compute, dma_desc, dma_start, n_beats, bases, latency):
    cl, mem, xb, l2 = build(skip)
    rng = np.random.default_rng(1)
    mem.load(0, rng.integers(-100, 100, NB * 64))
    l2.load(0, rng.integers(-100, 100, 4096))
    script = {}
    parts = None
    if with_compute:
        acc, streamers, starts = compute_parts(cl, mem, xb, n_beats, bases, latency)
        script[0] = starts
        parts = (acc, streamers)
    if with_dma:
        dma = cl.add(Dma("dma", xb, l2))
        script.setdefault(dma_start, []).append((dma, dma_desc))
    cl.add(ScriptedStarts("ctl", script))
    total = cl.run(max_cycles=5000)
    return total, mem, parts


@pytest.mark.parametrize("skip", [True, False])
def test_dma_in_one_superbank_overlaps_compute_in_the_other(skip):
    """DMA row by row in superbank 0; vecadd on banks 8..15. Compute is not slowed at all."""
    n_dma, n_beats = 40, 12
    desc = DmaDescriptor("l2_to_l1", contiguous(0, n_dma), DmaPattern(0, (n_dma,), (NB * WORD,)))
    bases = (8 * WORD, 12 * WORD, (20 * NB + 8) * WORD)  # banks 8-11, 12-15, 8-11 row 20+
    args = (desc, 0, n_beats, bases, 0)
    t_dma, _, _ = run_overlap(skip, True, False, *args)
    t_comp, _, (_, s0) = run_overlap(skip, False, True, *args)
    t_both, _, (_, s1) = run_overlap(skip, True, True, *args)
    assert t_both < t_dma + t_comp
    assert t_both == max(t_dma, t_comp)
    assert [s.grants for s in s0] == [s.grants for s in s1]  # identical compute timing


@pytest.mark.parametrize("skip", [True, False])
def test_dma_overlaps_a_long_pipeline_on_the_same_superbank(skip):
    """Readers finish early; during the 80-cycle pipeline the DMA uses the same superbank."""
    n_dma, n_beats, latency = 40, 4, 80
    desc = DmaDescriptor("l2_to_l1", contiguous(0, n_dma), DmaPattern(20 * NB * WORD, (n_dma,),
                                                                      (NB * WORD,)))  # fmt: skip
    bases = (0, 4 * WORD, (10 * NB + 4) * WORD)  # banks 0-3, 4-7, 4-7: all superbank 0
    args = (desc, 12, n_beats, bases, latency)  # DMA starts once the readers are done
    t_dma, _, _ = run_overlap(skip, True, False, *args)
    t_comp, _, (_, (ra, rb, wr)) = run_overlap(skip, False, True, *args)
    t_both, _, (_, (_, _, wr1)) = run_overlap(skip, True, True, *args)
    assert max(ra.grants + rb.grants) < 12  # the readers really are done by then
    assert t_both < (t_dma - 12) + t_comp  # shorter than DMA then compute
    assert t_both == t_comp  # the DMA is hidden completely in the pipeline
    assert wr1.grants == wr.grants


# =============================================================================
# 8. vecadd end to end, L2 -> L1 -> L2
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
def test_vecadd_from_l2_to_l2(skip):
    """DMA a and b into L1, vecadd with streamers, DMA c back; c in L2 equals a + b.

    Started by hand in cycles (no controller yet), from the formula:
    DMA a 0..12, DMA b 12..24, compute from 24 (writer done at 24 + 5 + beats),
    DMA c from there.
    """
    n_elems, lanes = 64, 4
    nb, n_dma = n_elems // lanes, n_elems // 8
    cl, _, xb, l2 = build(skip)
    rng = np.random.default_rng(3)
    a = rng.integers(-1000, 1000, n_elems)
    b = rng.integers(-1000, 1000, n_elems)
    l2a, l2b, l2c = 0, 1024, 2048
    l2.load(l2a, a)
    l2.load(l2b, b)
    # L1 layout: aligned to the wide beat; a, b and c on different banks in
    # steady state (a at 4j, b at 8 + 4j, writer at 4k while the readers are
    # on beat k + 3).
    wa, wb, wc = 0, 72, 144
    cfg = StreamerConfig(n_ports=lanes, fifo_depth=2)
    ra = cl.add(Streamer("ra", xb, cfg))
    rb = cl.add(Streamer("rb", xb, cfg))
    wr = cl.add(Streamer("wr", xb, StreamerConfig(write=True, n_ports=lanes, fifo_depth=2)))
    acc = cl.add(Accelerator("acc", cl, elementwise_stub(lanes=lanes)))
    acc.attach("a", ra.fifo)
    acc.attach("b", rb.fifo)
    acc.attach("out", wr.fifo)
    dma = cl.add(Dma("dma", xb, l2))

    def unit(w):
        return StreamerRegs(w * WORD, (nb,), (lanes * WORD,), (lanes,), (WORD,))

    t = expected_total(DmaConfig(), n_dma, 1)
    t_b, t_comp = t, 2 * t
    t_c = t_comp + 5 + nb
    cl.add(ScriptedStarts("ctl", {
        0: [(dma, DmaDescriptor("l2_to_l1", contiguous(l2a, n_dma), contiguous(wa * WORD, n_dma)))],
        t_b: [(dma, DmaDescriptor("l2_to_l1", contiguous(l2b, n_dma), contiguous(wb * WORD, n_dma)))],
        t_comp: [(ra, unit(wa)), (rb, unit(wb)), (wr, unit(wc)), (acc, {"n": nb})],
        t_c: [(dma, DmaDescriptor("l1_to_l2", contiguous(wc * WORD, n_dma), contiguous(l2c, n_dma)))],
    }))  # fmt: skip
    total = cl.run(max_cycles=2000)

    assert wr.done_cycle == t_c  # conflict-free compute, as planned
    assert dma.done_cycle == t_c + expected_total(DmaConfig(), n_dma, 1) == total
    assert np.array_equal(l2.dump(l2c, n_elems)[:, 0], a + b)
    for comp in (dma, ra, rb, wr, acc):
        assert_cycles_add_up(comp, total)


# =============================================================================
# 9. Random: skip on/off identical
# =============================================================================


def random_l1_pattern(rng, n_banks, rows, n):
    """An aligned pattern of n beats inside L1, as nested loops of random strides."""
    size = n_banks * rows * WORD
    while True:
        if n > 1 and rng.random() < 0.5:
            inner = rng.choice([d for d in range(1, n + 1) if n % d == 0])
            bounds = (inner, n // inner)
        else:
            bounds = (n,)
        strides = tuple(int(rng.choice([-2, -1, 1, 2, n_banks // 8, n_banks // 4])) * BEAT
                        for _ in bounds)  # fmt: skip
        base = int(rng.integers(0, size // BEAT)) * BEAT
        p = DmaPattern(base, bounds, strides)
        a = p.addresses()
        if a.min() >= 0 and a.max() + BEAT <= size:
            return p


def random_case(seed):
    rng = np.random.default_rng(seed)
    r = random.Random(seed)
    timing = {
        "startup": r.choice([1, 2, 4]),
        "beat_interval": r.choice([1, 1, 2, 3]),
        "l1_read_extra": r.choice([0, 0, 2]),
        "done_latency": r.choice([0, 0, 3]),
    }
    lat = {"l1_lat": r.choice([0, 1, 2]), "l2_lat": r.choice([0, 1, 5])}
    descs = []
    for _ in range(r.choice([2, 3, 4])):
        n = r.randint(1, 12)
        l1p = random_l1_pattern(rng, NB, 32, n)
        l2p = contiguous(r.randrange(0, 64) * BEAT, n)
        d = r.choice(["l2_to_l1", "l1_to_l2"])
        desc = DmaDescriptor(d, l2p, l1p) if d == "l2_to_l1" else DmaDescriptor(d, l1p, l2p)
        descs.append((r.choice([0, 0, 1, 7]), desc))
    readers = []
    for _ in range(r.choice([1, 2])):
        ports = r.choice([1, 2])
        nbeats = r.randint(3, 10)
        stride = r.choice([1, 3, 16]) * WORD
        base = r.randrange(0, NB * 8) * WORD
        readers.append((ports, StreamerRegs(base, (nbeats,), (stride,), (ports,), (WORD,)),
                        r.choice([0, 3, 15])))  # fmt: skip
    return timing, lat, descs, readers


XBAR_STATS = (
    "bank_grants",
    "bank_conflicts",
    "bank_stalls",
    "bank_blocked",
    "port_grants",
    "port_stalls",
    "port_stalls_wider",
)


def run_case(case, skip):
    timing, lat, descs, readers = case
    cl, mem, xb, l2 = build(skip, rows=32, l2_size=1 << 13, **lat)
    rng = np.random.default_rng(99)
    mem.load(0, rng.integers(-500, 500, NB * 32))
    l2.load(0, rng.integers(-500, 500, (1 << 13) // WORD))
    rs = [cl.add(LogStreamer(f"r{i}", xb, StreamerConfig(n_ports=p, fifo_depth=16)))
          for i, (p, _, _) in enumerate(readers)]  # fmt: skip
    dma = cl.add(LogDma("dma", xb, l2, DmaConfig(**timing)))
    chain = cl.add(Chain("chain", dma, descs))
    script = {}
    for s, (_, regs, at) in zip(rs, readers):
        script.setdefault(at, []).append((s, regs))
    cl.add(ScriptedStarts("ctl", script))
    total = cl.run(max_cycles=5000)
    fifos = [[[np.asarray(x).ravel()[0].item() for x in q] for q in s.fifo._q] for s in rs]
    return {
        "total": total,
        "starts": chain.starts,
        "done": dma.done_cycle,
        "cycles": dict(dma.cycles),
        "reads": dma.read_cycles,
        "writes": dma.write_cycles,
        "grants": [s.grants for s in rs],
        "fifos": fifos,
        "l1": mem.data.copy(),
        "l2": l2.data.copy(),
        "xb": {k: getattr(xb, k).copy() for k in XBAR_STATS},
        "sum_ok": sum(dma.cycles.values()) == total,
    }


@pytest.mark.parametrize("seed", range(16))
def test_random_skip_on_off_identical(seed):
    """Random descriptors, latencies and reader traffic: skipping changes nothing.

    Also checks the data: the DMA's destination equals its source as it was
    when the transfer started, replayed in order on a copy of the memories.
    """
    case = random_case(seed)
    a, b = run_case(case, True), run_case(case, False)
    for k in a:
        if isinstance(a[k], np.ndarray):
            assert np.array_equal(a[k], b[k]), k
        elif k == "xb":
            assert all(np.array_equal(a[k][x], b[k][x]) for x in a[k]), k
        else:
            assert a[k] == b[k], k
    assert a["sum_ok"]
    assert len(a["starts"]) == len(case[2])

    # Data: replay the transfers on copies (streamers only read, so the DMA
    # is the only writer and transfers do not overlap in time).
    _, mem0, _, l20 = build(True, rows=32, l2_size=1 << 13)
    rng = np.random.default_rng(99)
    mem0.load(0, rng.integers(-500, 500, NB * 32))
    l20.load(0, rng.integers(-500, 500, (1 << 13) // WORD))
    for _, d in case[2]:
        src = mem0 if d.direction == "l1_to_l2" else l20
        dst = l20 if d.direction == "l1_to_l2" else mem0
        beats = [src.dump(int(x), 8) for x in d.src.addresses()]
        for x, beat in zip(d.dst.addresses(), beats):
            dst.load(int(x), beat)
    assert np.array_equal(a["l1"], mem0.data)
    assert np.array_equal(a["l2"], l20.data)
