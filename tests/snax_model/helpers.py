"""Shared helpers for the SNAX-MODEL tests (MOD10, housekeeping).

Every test file used to carry its own copy of these; they are one thing now,
so a change to the cluster shape or to a timing formula happens once. Only
helpers used by more than one file live here. Anything specific to one file
(the accelerator's Producer / Consumer toys, the DMA's Chain, the random
case generators) stays with its test.

Not imported by the model or by scenarios/make.py: this is test code.
"""

from __future__ import annotations

import numpy as np

from snax_forge.snax_model import (
    Accelerator,
    Cluster,
    Component,
    Controller,
    Dma,
    DmaDescriptor,
    DmaPattern,
    L1Config,
    L1Memory,
    L2Config,
    L2Memory,
    Phase,
    RegisterMap,
    Streamer,
    StreamerConfig,
    StreamerRegs,
    Wait,
    Xbar,
    elementwise_stub,
)
from snax_forge.snax_model.trace import Trace

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


def compute_blocks(cl, xb, lanes=4, fifo_depth=2, acc_cfg=None, temporal_dims=1, names=None):
    """Readers ra, rb, writer wr and an accelerator (elementwise add by default).

    ``rb`` is None for a single-input accelerator (the reduce stub). ``names``
    renames the three streamers, e.g. to alu4's ``acc_a``, ``acc_b``, ``acc_out``.
    """
    cfg = StreamerConfig(n_ports=lanes, fifo_depth=fifo_depth, temporal_dims=temporal_dims)
    wcfg = StreamerConfig(write=True, n_ports=lanes, fifo_depth=fifo_depth,
                          temporal_dims=temporal_dims)  # fmt: skip
    acc_cfg = acc_cfg or elementwise_stub(lanes=lanes)
    na, nb, nw = names or ("ra", "rb", "wr")
    ra = cl.add(Streamer(na, xb, cfg))
    rb = cl.add(Streamer(nb, xb, cfg)) if len(acc_cfg.inputs) > 1 else None
    wr = cl.add(Streamer(nw, xb, wcfg))
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
# The MOD7 vecadd
# =============================================================================


def mod7_vecadd(mode="poll", skip=True, cfg=None, level=None):
    """c = a + b over 64 elements, L2 -> L1 -> L2, from a hand-written control program.

    Transfer b and the compute blocks are programmed while the previous
    transfer runs; each start comes after a wait on the block it needs. Named
    and placed as scenarios/vecadd (NAME1, D83): streamers acc_a, acc_b,
    acc_out, and a, b, c packed in L2 at 0, 512 and 1024. Written without
    SNAX-LOWER, so it is an independent reference for that scenario
    (test_scenario, D92). ``cfg`` sets the controller costs (the model's
    defaults when None), ``level`` the trace level (no trace when None).
    """
    n_elems, lanes = 64, 4
    nb, n_dma = n_elems // lanes, n_elems // 8
    trace = None if level is None else Trace(level)
    cl, mem, xb, l2 = build(skip, trace)
    rng = np.random.default_rng(3)
    a, b = rng.integers(-1000, 1000, n_elems), rng.integers(-1000, 1000, n_elems)
    l2a, l2b, l2c = 0, 512, 1024
    l2.load(l2a, a)
    l2.load(l2b, b)
    wa, wb, wc = 0, 72, 144  # L1 words: a and b in different banks
    dma = cl.add(Dma("dma", xb, l2))
    ra, rb, wr, acc = compute_blocks(cl, xb, lanes=lanes, names=("acc_a", "acc_b", "acc_out"))
    m = RegisterMap([("dma", dma), ("acc_a", ra), ("acc_b", rb), ("acc_out", wr), ("acc", acc)])

    def to_l1(src, word):
        return DmaDescriptor("l2_to_l1", contiguous(src, n_dma), contiguous(word * WORD, n_dma))

    def to_l2(word, dst):
        return DmaDescriptor("l1_to_l2", contiguous(word * WORD, n_dma), contiguous(dst, n_dma))

    prog = [
        *m.start_writes("dma", to_l1(l2a, wa)),
        *m.config_writes("dma", to_l1(l2b, wb)),
        Wait("dma", mode),
        m.start_write("dma"),
        *m.config_writes("acc_a", unit(wa, nb, lanes)),
        *m.config_writes("acc_b", unit(wb, nb, lanes)),
        *m.config_writes("acc_out", unit(wc, nb, lanes)),
        *m.config_writes("acc", {"n": nb}),
        Wait("dma", mode),
        *(m.start_write(x) for x in ("acc_a", "acc_b", "acc_out", "acc")),
        *m.config_writes("dma", to_l2(wc, l2c)),
        Wait("acc_out", mode),
        m.start_write("dma"),
        Wait("dma", mode),
    ]
    ctl = cl.add(Controller("ctl", m, prog, cfg))
    total = cl.run(max_cycles=5000)
    c = l2.dump(l2c, n_elems)[:, 0]
    assert np.array_equal(c, a + b)
    return {"cl": cl, "total": total, "trace": trace, "ctl": ctl, "mem": mem, "l2": l2,
            "a": a, "b": b, "c": c, "comps": (dma, ra, rb, wr, acc)}  # fmt: skip


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
