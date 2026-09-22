"""Shared helpers for the SNAX-MODEL tests (MOD10, housekeeping).

Every test file used to carry its own copy of these; they are one thing now,
so a change to the cluster shape or to a timing formula happens once. Only
helpers used by more than one file live here. Anything specific to one file
(the accelerator's Producer / Consumer toys, the DMA's Chain, the random
case generators) stays with its test.

Not imported by the model or by scenarios/make.py: this is test code.
"""

from __future__ import annotations

from snax_forge.snax_model import (
    Accelerator,
    Cluster,
    Component,
    Dma,
    DmaPattern,
    L1Config,
    L1Memory,
    L2Config,
    L2Memory,
    Phase,
    Streamer,
    StreamerConfig,
    StreamerRegs,
    Xbar,
    elementwise_stub,
)

WORD = 8  # bytes per bank word (64-bit banks)
BEAT = 64  # bytes per wide beat (512 bits)
NB = 16  # banks: two superbanks


# =============================================================================
# Clusters
# =============================================================================


def build(skip=True, trace=None, n_banks=NB, rows=64, l1_lat=1, l2_lat=1, l2_size=1 << 15):
    """Cluster with L1, xbar and L2. Returns (cluster, l1, xbar, l2)."""
    cl = Cluster(skip_idle=skip, trace=trace)
    mem = L1Memory(cl, L1Config(n_banks=n_banks, rows=rows, read_latency=l1_lat))
    xb = cl.add(Xbar("xbar", mem))
    l2 = L2Memory(cl, L2Config(size_bytes=l2_size, read_latency=l2_lat))
    return cl, mem, xb, l2


def build_l1(skip=True, n_banks=NB, rows=64, read_latency=1):
    """Cluster with L1 and xbar, no L2. Returns (cluster, l1, xbar)."""
    cl = Cluster(skip_idle=skip)
    mem = L1Memory(cl, L1Config(n_banks=n_banks, rows=rows, read_latency=read_latency))
    xb = cl.add(Xbar("xbar", mem))
    return cl, mem, xb


def compute_blocks(cl, xb, lanes=4, fifo_depth=2, acc_cfg=None, temporal_dims=1):
    """Readers ra, rb, writer wr and an accelerator (elementwise add by default).

    ``rb`` is None for a single-input accelerator (the reduce stub).
    """
    cfg = StreamerConfig(n_ports=lanes, fifo_depth=fifo_depth, temporal_dims=temporal_dims)
    wcfg = StreamerConfig(write=True, n_ports=lanes, fifo_depth=fifo_depth,
                          temporal_dims=temporal_dims)  # fmt: skip
    acc_cfg = acc_cfg or elementwise_stub(lanes=lanes)
    ra = cl.add(Streamer("ra", xb, cfg))
    rb = cl.add(Streamer("rb", xb, cfg)) if len(acc_cfg.inputs) > 1 else None
    wr = cl.add(Streamer("wr", xb, wcfg))
    acc = cl.add(Accelerator("acc", cl, acc_cfg))
    ins = [p.name for p in acc_cfg.inputs]
    acc.attach(ins[0], ra.fifo)
    if rb is not None:
        acc.attach(ins[1], rb.fifo)
    acc.attach(acc_cfg.outputs[0].name, wr.fifo)
    return ra, rb, wr, acc


# =============================================================================
# Block arguments
# =============================================================================


def contiguous(base, n):
    """n consecutive wide beats from byte ``base``."""
    return DmaPattern(base, (n,), (BEAT,))


def unit(word, n_beats, lanes):
    """Streamer task over n_beats contiguous beats of ``lanes`` words from word ``word``."""
    return StreamerRegs(word * WORD, (n_beats,), (lanes * WORD,), (lanes,), (WORD,))


# =============================================================================
# Timing formulas (the module docs of dma.py and ctrl.py)
# =============================================================================


def dma_total(cfg, n, src_latency):
    """done_cycle - start cycle for N beats without contention (D34)."""
    return cfg.startup + src_latency + cfg.beat_interval * (n - 1) + 2 + cfg.done_latency


def w_poll(t, d, c_r, p):
    """Length of a poll wait beginning in ``t`` for a block done in ``d`` (D37)."""
    i = max(0, -(-(d - t - c_r + 1) // p))
    return i * p + c_r


def w_signal(t, d, s):
    """Length of a signal wait beginning in ``t`` for a block done in ``d`` (D37)."""
    return max(t, d) - t + s


def assert_cycles_add_up(comp, total):
    assert sum(comp.cycles.values()) == total, comp.cycles


# =============================================================================
# Toy components and logging subclasses
# =============================================================================


class ScriptedStarts(Component):
    """Starts components in scripted cycles: {cycle: (comp, arg)} or {cycle: [...]}."""

    phases = (Phase.CONTROL,)

    def __init__(self, name, script):
        super().__init__(name)
        self.script = script

    def tick(self, cycle, phase):
        items = self.script.get(cycle)
        if items is None:
            return
        for comp, arg in [items] if isinstance(items, tuple) else items:
            comp.start(arg, cycle)

    def next_wake(self, cycle):
        later = [c for c in self.script if c > cycle]
        return min(later) if later else None


class LogStreamer(Streamer):
    """Streamer that logs its grant cycles."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.grants = []

    def commit(self, cycle):
        if any(self._fire):
            self.grants.append(cycle)
        super().commit(cycle)


class LogAccel(Accelerator):
    """Accelerator that logs firing and push cycles."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fire_cycles = []
        self.push_cycles = []

    def commit(self, cycle):
        if self._w.fired:
            self.fire_cycles.append(cycle)
        if self._w.pushed:
            self.push_cycles.append(cycle)
        super().commit(cycle)


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
