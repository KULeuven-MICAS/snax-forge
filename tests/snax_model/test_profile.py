"""Tests for the SNAX-MODEL profile and trace (MOD8, D38, D39, D40).

MOD8 acceptance (docs/STATUS.md): per accelerator, busy + idle + stalled =
total; per-bank access counts equal trace event counts. Plus: the other
components' classes add up, the trace is cross-checked against every
counter it can see, class intervals cover the run, FIFO occupancy on a
hand-worked case, control overhead in the MOD7 vecadd, round trips, and
recording does not change results (tracing off vs on, skipping off vs on).

Sections:
  1. helpers (a copy of the MOD7 vecadd and random programs, with a trace)
  2. ClassLog
  3. classes add up; control overhead
  4. trace vs counters
  5. class intervals
  6. FIFO occupancy, hand-worked
  7. levels, order inside a cycle, round trips
  8. recording does not change results
"""

import json
import random
from collections import Counter
from itertools import pairwise

import numpy as np
import pytest

from snax_forge.snax_model import (
    Accelerator,
    Cluster,
    Controller,
    ControllerConfig,
    CsrRead,
    CsrWrite,
    Dma,
    DmaConfig,
    DmaDescriptor,
    DmaPattern,
    L1Config,
    L1Memory,
    L2Config,
    L2Memory,
    RegisterMap,
    SimulationError,
    Streamer,
    StreamerConfig,
    StreamerRegs,
    Wait,
    Xbar,
    elementwise_stub,
    reduce_stub,
)
from snax_forge.snax_model.profile import Profile, build_profile
from snax_forge.snax_model.sched import ClassLog
from snax_forge.snax_model.trace import EVENT_KINDS, Trace

WORD = 8  # bytes per bank word
BEAT = 64  # bytes per wide beat (512 bits)
NB = 16  # banks: two superbanks
VECADD_CFG = ControllerConfig(
    write_cost=1, kind_write_cost={"dma": 2}, read_cost=2, poll_interval=4
)


# =============================================================================
# 1. Helpers
# =============================================================================


def build(skip=True, trace=None, rows=64, l1_lat=1, l2_size=1 << 15):
    """Cluster with L1, xbar and L2. Returns (cluster, l1, xbar, l2)."""
    cl = Cluster(skip_idle=skip, trace=trace)
    mem = L1Memory(cl, L1Config(n_banks=NB, rows=rows, read_latency=l1_lat))
    xb = cl.add(Xbar("xbar", mem))
    l2 = L2Memory(cl, L2Config(size_bytes=l2_size, read_latency=1))
    return cl, mem, xb, l2


def compute_blocks(cl, xb, lanes=4, fifo_depth=2):
    """Readers ra, rb, writer wr and an elementwise-add accelerator."""
    cfg = StreamerConfig(n_ports=lanes, fifo_depth=fifo_depth)
    wcfg = StreamerConfig(write=True, n_ports=lanes, fifo_depth=fifo_depth)
    ra = cl.add(Streamer("ra", xb, cfg))
    rb = cl.add(Streamer("rb", xb, cfg))
    wr = cl.add(Streamer("wr", xb, wcfg))
    acc = cl.add(Accelerator("acc", cl, elementwise_stub(lanes=lanes)))
    acc.attach("a", ra.fifo)
    acc.attach("b", rb.fifo)
    acc.attach("out", wr.fifo)
    return ra, rb, wr, acc


def contiguous(base, n):
    return DmaPattern(base, (n,), (BEAT,))


def unit(word, n_beats, lanes):
    """Streamer task over n_beats contiguous beats of `lanes` words from word `word`."""
    return StreamerRegs(word * WORD, (n_beats,), (lanes * WORD,), (lanes,), (WORD,))


def run_vecadd(mode="poll", skip=True, level=None, cfg=VECADD_CFG):
    """The MOD7 vecadd (test_ctrl section 7), optionally traced."""
    n_elems, lanes = 64, 4
    nb, n_dma = n_elems // lanes, n_elems // 8
    trace = None if level is None else Trace(level)
    cl, mem, xb, l2 = build(skip, trace)
    rng = np.random.default_rng(3)
    a, b = rng.integers(-1000, 1000, n_elems), rng.integers(-1000, 1000, n_elems)
    l2a, l2b, l2c = 0, 1024, 2048
    l2.load(l2a, a)
    l2.load(l2b, b)
    wa, wb, wc = 0, 72, 144
    dma = cl.add(Dma("dma", xb, l2))
    ra, rb, wr, acc = compute_blocks(cl, xb, lanes=lanes)
    m = RegisterMap([("dma", dma), ("ra", ra), ("rb", rb), ("wr", wr), ("acc", acc)])

    def to_l1(src, word):
        return DmaDescriptor("l2_to_l1", contiguous(src, n_dma), contiguous(word * WORD, n_dma))

    def to_l2(word, dst):
        return DmaDescriptor("l1_to_l2", contiguous(word * WORD, n_dma), contiguous(dst, n_dma))

    prog = [
        *m.start_writes("dma", to_l1(l2a, wa)),
        *m.config_writes("dma", to_l1(l2b, wb)),
        Wait("dma", mode),
        m.start_write("dma"),
        *m.config_writes("ra", unit(wa, nb, lanes)),
        *m.config_writes("rb", unit(wb, nb, lanes)),
        *m.config_writes("wr", unit(wc, nb, lanes)),
        *m.config_writes("acc", {"n": nb}),
        Wait("dma", mode),
        *(m.start_write(x) for x in ("ra", "rb", "wr", "acc")),
        *m.config_writes("dma", to_l2(wc, l2c)),
        Wait("wr", mode),
        m.start_write("dma"),
        Wait("dma", mode),
    ]
    ctl = cl.add(Controller("ctl", m, prog, cfg))
    total = cl.run(max_cycles=5000)
    assert np.array_equal(l2.dump(l2c, n_elems)[:, 0], a + b)
    return {"cl": cl, "total": total, "trace": trace, "ctl": ctl, "mem": mem, "l2": l2}


# --- Random programs (as test_ctrl section 9), traced -------------------------

L1_ROWS = 64
L1_WORDS = NB * L1_ROWS
L2_SIZE = 1 << 13


def random_l1_dma(rng, n):
    while True:
        if n > 1 and rng.random() < 0.5:
            inner = int(rng.choice([d for d in range(1, n + 1) if n % d == 0]))
            bounds = (inner, n // inner)
        else:
            bounds = (n,)
        strides = tuple(int(rng.choice([1, 2, NB // 8])) * BEAT for _ in bounds)
        p = DmaPattern(int(rng.integers(0, L1_WORDS * WORD // BEAT)) * BEAT, bounds, strides)
        if p.addresses().max() + BEAT <= L1_WORDS * WORD:
            return p


def random_program(m, r, rng, lanes):
    prog, live = [], set()

    def wait(block):
        prog.append(Wait(block, r.choice(["poll", "signal"])))
        live.discard(block)

    def reads():
        for _ in range(r.choice([0, 0, 1, 2])):
            blk = r.choice(list(m.blocks))
            reg = r.choice(["busy", "busy_cycles", *m[blk].config_names])
            prog.append(CsrRead(m.addr(f"{blk}.{reg}")))

    for _ in range(r.randint(1, 3)):
        if r.random() < 0.8:
            n = r.randint(1, 8)
            l1p = random_l1_dma(rng, n)
            l2p = contiguous(r.randrange(0, L2_SIZE // BEAT - n) * BEAT, n)
            d = r.choice(["l2_to_l1", "l1_to_l2"])
            desc = DmaDescriptor(d, l2p, l1p) if d == "l2_to_l1" else DmaDescriptor(d, l1p, l2p)
            prog.extend(m.config_writes("dma", desc))
            reads()
            if "dma" in live:
                wait("dma")
            prog.append(m.start_write("dma"))
            live.add("dma")
        nb = r.randint(1, 10)
        for s in ("ra", "rb", "wr"):
            base = r.randrange(0, L1_WORDS - nb * lanes)
            prog.extend(m.config_writes(s, unit(base, nb, lanes)))
        prog.extend(m.config_writes("acc", {"n": nb}))
        reads()
        compute = ["ra", "rb", "wr", "acc"]
        for blk in r.sample(compute, len(compute)):
            if blk in live:
                wait(blk)
        for blk in r.sample(compute, len(compute)):
            prog.append(m.start_write(blk))
            live.add(blk)
        reads()
    for blk in r.sample(sorted(live), len(live)):
        wait(blk)
    reads()
    return prog


def run_random(seed, skip, level="beat"):
    r = random.Random(seed)
    lanes, fifo = r.choice([1, 2, 4]), r.choice([1, 2, 4])
    dcfg = DmaConfig(startup=r.choice([1, 2, 4]), beat_interval=r.choice([1, 1, 2]),
                     done_latency=r.choice([0, 0, 3]))  # fmt: skip
    rc = {k: r.choice([1, 2, 3]) for k in ("streamer", "accel", "dma")}
    ccfg = ControllerConfig(
        write_cost=r.choice([1, 2]),
        read_cost=r.choice([1, 2]),
        kind_write_cost={k: r.choice([1, 2, 4]) for k in ("dma", "streamer") if r.random() < 0.5},
        kind_read_cost=rc,
        poll_interval=max(rc.values()) + r.choice([0, 1, 4]),
        signal_latency=r.choice([1, 1, 3]),
    )
    trace = None if level is None else Trace(level)
    cl, mem, xb, l2 = build(skip, trace, rows=L1_ROWS, l1_lat=r.choice([0, 1, 2]), l2_size=L2_SIZE)
    rng = np.random.default_rng(seed)
    mem.load(0, rng.integers(-500, 500, L1_WORDS))
    l2.load(0, rng.integers(-500, 500, L2_SIZE // WORD))
    dma = cl.add(Dma("dma", xb, l2, dcfg))
    ra, rb, wr, acc = compute_blocks(cl, xb, lanes=lanes, fifo_depth=fifo)
    m = RegisterMap([("dma", dma), ("ra", ra), ("rb", rb), ("wr", wr), ("acc", acc)])
    prog = random_program(m, random.Random(seed + 1000), np.random.default_rng(seed + 1000), lanes)
    ctl = cl.add(Controller("ctl", m, prog, ccfg))
    total = cl.run(max_cycles=20000)
    assert ctl.finished
    return {"cl": cl, "total": total, "trace": trace, "ctl": ctl, "mem": mem, "l2": l2}


def classified(cl):
    """Components that keep cycle classes."""
    return [c for c in cl if isinstance(getattr(c, "cycles", None), ClassLog)]


def cover(events, n_banks):
    """Per bank: number of events whose `banks` include it."""
    out = np.zeros(n_banks, dtype=np.int64)
    for e in events:
        out[list(e.banks)] += 1
    return out


# =============================================================================
# 2. ClassLog
# =============================================================================


def test_classlog_totals_and_runs():
    log = ClassLog(("busy", "idle"))
    log.add("idle", 0, 3)  # not recording yet: totals only
    assert dict(log) == {"busy": 0, "idle": 3} and log.runs is None
    log = ClassLog(("busy", "idle"))
    log.record(0)
    for cls, a, b in [("idle", 0, 2), ("idle", 2, 3), ("busy", 3, 4), ("busy", 4, 4)]:
        log.add(cls, a, b)
    assert log.runs == [["idle", 0, 3], ["busy", 3, 4]]  # merged; empty range ignored
    with pytest.raises(SimulationError, match="not contiguous"):
        log.add("idle", 5, 6)
    with pytest.raises(SimulationError, match="unknown cycle class"):
        log.add("stall", 4, 5)


# =============================================================================
# 3. Classes add up; control overhead
# =============================================================================


@pytest.mark.parametrize("mode", ["poll", "signal"])
@pytest.mark.parametrize("skip", [True, False])
def test_classes_add_up_to_total(mode, skip):
    r = run_vecadd(mode, skip)
    p = build_profile(r["cl"])
    assert p.total_cycles == r["total"]
    parts = [p.controller.cycles]
    parts += [a.cycles for a in p.accelerators.values()]
    parts += [s.cycles for s in p.streamers.values()]
    parts += [d.cycles for d in p.dmas.values()]
    assert len(parts) == 6
    for cycles in parts:
        assert sum(cycles.values()) == r["total"], cycles


@pytest.mark.parametrize("mode", ["poll", "signal"])
def test_control_overhead_in_vecadd(mode):
    """Command cycles = sum of command costs; wait cycles = sum of wait lengths."""
    r = run_vecadd(mode)
    ctl, p = r["ctl"], build_profile(r["cl"]).controller
    kind = {b: ctl.map[b].kind for b in ctl.map.blocks}
    costs = 0
    for cmd in ctl.program:
        if isinstance(cmd, CsrWrite):
            costs += ctl.cfg.write_cost_of(kind[ctl.map.register(cmd.addr).block])
        elif isinstance(cmd, CsrRead):
            costs += ctl.cfg.read_cost_of(kind[ctl.map.register(cmd.addr).block])
    assert p.cycles["command"] == costs == 89
    assert p.cycles["wait"] == sum(w.last - w.first + 1 for w in p.waits)
    assert p.commands == sum(not isinstance(c, Wait) for c in ctl.program)
    assert [w.mode for w in p.waits] == [mode] * 4
    if mode == "poll":
        assert p.cycles["wait"] == 20 and p.polls == 7


def test_vecadd_profile_numbers():
    """The example of the MOD8 proposal: a fixed point for later changes."""
    p = build_profile(run_vecadd("poll")["cl"])
    assert p.total_cycles == 109
    assert p.accelerators["acc"].cycles == {"busy": 16, "stall_out": 0, "stall_in": 1, "idle": 92}
    assert p.dmas["dma"].bytes_read == p.dmas["dma"].bytes_written == 24 * BEAT
    assert p.banks.reads == [12] * NB and p.banks.writes == [12] * NB
    assert p.ports["dma.wide"].grants == 24 and p.ports["ra.0"].grants == 16
    assert p.l2["l2"].reads == 16 and p.l2["l2"].writes == 8
    assert p.streamers["rb"].fifo.hist == [[93, 16, 0]] * 4
    assert p.functional_check is None


# =============================================================================
# 4. Trace vs counters
# =============================================================================


def check_trace_against_counters(r):
    """Every count the trace can see equals the component's own counter (D38)."""
    cl, tr, p = r["cl"], r["trace"], build_profile(r["cl"])
    grants, stalls = tr.of_kind("grant"), tr.of_kind("stall")
    # L1 accesses: per bank, split by reads and writes (L1Memory counters).
    assert np.array_equal(cover([e for e in grants if not e.w], NB), p.banks.reads)
    assert np.array_equal(cover([e for e in grants if e.w], NB), p.banks.writes)
    assert np.array_equal(cover(grants, NB), p.banks.grants)
    assert np.array_equal(cover(stalls, NB), p.banks.stalls)
    # blocked counts once per refused (width, group) arbitration, not per port.
    wider = {(e.t, e.banks) for e in stalls if e.wider}
    blocked = np.zeros(NB, dtype=np.int64)
    for _, banks in wider:
        blocked[list(banks)] += 1
    assert np.array_equal(blocked, p.banks.blocked)
    # conflicts: cycles with >= 2 requests covering the bank.
    per_cycle: dict[int, list] = {}
    for e in grants + stalls:
        per_cycle.setdefault(e.t, []).append(e)
    conflicts = sum((cover(es, NB) >= 2).astype(np.int64) for es in per_cycle.values())
    assert np.array_equal(np.asarray(conflicts) if per_cycle else np.zeros(NB), p.banks.conflicts)
    # Ports.
    g, s = Counter(e.port for e in grants), Counter(e.port for e in stalls)
    sw = Counter(e.port for e in stalls if e.wider)
    for name, pp in p.ports.items():
        assert (g[name], s[name], sw[name]) == (pp.grants, pp.stalls, pp.stalls_wider), name
    # Rows and addresses are those of the L1 address map.
    mem = r["mem"]
    for e in grants[:50]:
        assert (e.banks[0], e.row) == mem.locate(e.addr)
    # DMA and L2.
    beats = tr.of_kind("dma_beat")
    for name, d in p.dmas.items():
        mine = [e for e in beats if e.src == name]
        assert sum(e.side == "src" for e in mine) == d.beats_read
        assert sum(e.side == "dst" for e in mine) == d.beats_written
    assert sum(e.mem == "l2" and e.side == "src" for e in beats) == p.l2["l2"].reads
    assert sum(e.mem == "l2" and e.side == "dst" for e in beats) == p.l2["l2"].writes
    # Accelerator firings: a busy cycle is a firing cycle.
    for name, a in p.accelerators.items():
        assert sum(e.src == name for e in tr.of_kind("fire")) == a.cycles["busy"]
    # Controller: one cmd per span, polls, and the waits.
    ctl = r["ctl"]
    cmds = tr.of_kind("cmd")
    assert [(e.pc, e.t, e.last) for e in cmds] == ctl.spans
    assert len(tr.of_kind("poll")) == ctl.polls
    assert [(e.pc, e.block, e.t, e.done, e.last) for e in cmds if e.op == "wait"] == ctl.waits
    assert [e.value for e in cmds if e.op == "csr_read"] == [v for _, _, v in ctl.reads]
    # Starts land where the controller's start writes end; dones match done_cycle.
    for c in classified(cl):
        if c is ctl:
            continue
        starts = [e.t for e in tr.of_kind("start") if e.src == c.name]
        writes = [e.last for e in cmds if e.op == "csr_write" and e.reg == f"{c.name}.start"]
        assert starts == writes, c.name
        dones = [e.t for e in tr.of_kind("done") if e.src == c.name]
        assert len(dones) == len(starts) and (dones[-1] if dones else None) == c.done_cycle
    # FIFO events rebuild the occupancy histogram.
    for name, sp in p.streamers.items():
        f = sp.fifo
        lanes = len(f.hist)
        since, count = [0] * lanes, [0] * lanes
        hist = np.zeros((lanes, f.depth + 1), dtype=np.int64)
        for e in tr.of_kind("fifo"):
            if e.src == f.name:
                hist[e.lane, count[e.lane]] += e.t - since[e.lane]
                since[e.lane], count[e.lane] = e.t, e.count
        for j in range(lanes):
            hist[j, count[j]] += r["total"] - since[j]
        assert hist.tolist() == f.hist, name


def test_vecadd_trace_against_counters():
    check_trace_against_counters(run_vecadd("poll", level="beat"))


def test_random_trace_against_counters():
    """Random programs have contention, so stalls and wider stalls are checked too."""
    seen = Counter()
    for seed in range(20):
        r = run_random(seed, True)
        check_trace_against_counters(r)
        seen["stall"] += len(r["trace"].of_kind("stall"))
        seen["wider"] += sum(e.wider for e in r["trace"].of_kind("stall"))
        seen["poll"] += len(r["trace"].of_kind("poll"))
    assert seen["stall"] and seen["wider"] and seen["poll"], seen


# =============================================================================
# 5. Class intervals
# =============================================================================


def check_intervals(r):
    tr, total = r["trace"], r["total"]
    comps = classified(r["cl"])
    assert set(tr.intervals) == {c.name for c in comps}
    for c in comps:
        runs = tr.intervals[c.name]
        assert runs[0][1] == 0 and runs[-1][2] == total, c.name
        for (c0, _, b0), (c1, a1, _) in pairwise(runs):
            assert b0 == a1, c.name  # contiguous, no overlap
            assert c0 != c1, c.name  # merged: neighbours differ
        assert all(a < b for _, a, b in runs), c.name
        sums = Counter()
        for cls, a, b in runs:
            sums[cls] += b - a
        assert {k: v for k, v in c.cycles.items() if v} == dict(sums), c.name


@pytest.mark.parametrize("skip", [True, False])
def test_intervals_vecadd(skip):
    check_intervals(run_vecadd("signal", skip, level="task"))


@pytest.mark.parametrize("seed", range(10))
def test_intervals_random(seed):
    check_intervals(run_random(seed, True, level="task"))


# =============================================================================
# 6. FIFO occupancy, hand-worked
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
def test_fifo_occupancy_slow_accelerator(skip):
    """Reader (1 lane, depth 2, L1 latency 1) feeding an accelerator with II = 4.

    Started before the run: requests in 1, 2 (credit 2), then one per pop.
    Beat 0 is pushed in 2, visible in 3. The accelerator fires (pops) in
    3, 7, 11, 15. A pop frees credit in the same cycle, so the next request
    goes out then and its data arrives one cycle later:

        cycle   0-2  3  4  5-7  8  9-11  12-15  16
        count    0   1  1   2   1   2      1     0

    So 0 for 3 + (total - 16) cycles, 1 for 7, 2 for 6.
    """
    tr = Trace("beat")
    cl = Cluster(skip, trace=tr)
    mem = L1Memory(cl, L1Config(n_banks=8, rows=16, read_latency=1))
    xb = cl.add(Xbar("xbar", mem))
    rd = cl.add(Streamer("rd", xb, StreamerConfig(n_ports=1, fifo_depth=2)))
    wr = cl.add(Streamer("wr", xb, StreamerConfig(write=True, n_ports=1, fifo_depth=2)))
    acc = cl.add(Accelerator("acc", cl, reduce_stub(lanes=1, latency=0, ii=4)))
    acc.attach("in", rd.fifo)
    acc.attach("out", wr.fifo)
    mem.load(0, [1, 2, 3, 4])
    rd.start(StreamerRegs(0, (4,), (WORD,), (1,), (WORD,)))
    wr.start(StreamerRegs(64, (1,), (WORD,), (1,), (WORD,)))
    acc.start({"n": 4, "T": 4})
    total = cl.run()
    assert [e.t for e in tr.of_kind("fire")] == [3, 7, 11, 15]
    f = build_profile(cl).streamers["rd"].fifo
    assert f.hist == [[3 + total - 16, 7, 6]]
    assert f.max == [2] and f.mean == [pytest.approx((7 + 2 * 6) / total)]
    assert int(mem.peek(64)[0]) == 10


# =============================================================================
# 7. Levels, order inside a cycle, round trips
# =============================================================================

BEAT_KINDS = {k for k, c in EVENT_KINDS.items() if c.level == "beat"}


def test_levels():
    off, task, beat = (run_vecadd("poll", level=lv)["trace"] for lv in ("off", "task", "beat"))
    assert off.events == [] and off.intervals == {}
    assert {e.kind for e in task.events} == {"cmd", "start", "done"}
    # No contention in vecadd, so no stall events (they are checked in section 4).
    assert {e.kind for e in beat.events} == {"cmd", "start", "done"} | BEAT_KINDS - {"stall"}
    assert task.intervals == beat.intervals
    assert [e for e in beat.events if e.kind not in BEAT_KINDS] == task.events
    with pytest.raises(ValueError):
        Trace("verbose")


def test_order_inside_a_cycle():
    """Sorted by (t, group, source); several events share a cycle."""
    tr = run_vecadd("poll", level="beat")["trace"]
    rank = {s: i for i, s in enumerate(tr.sources)}
    keys = [(e.t, e.group, rank[e.src]) for e in tr.events]
    assert keys == sorted(keys)
    per_cycle = Counter(e.t for e in tr.events)
    assert max(per_cycle.values()) >= 12  # a streaming cycle: 12 streamer grants and more
    assert tr.sources == ["xbar", "dma", "ra", "ra.fifo", "rb", "rb.fifo", "wr", "wr.fifo",
                          "acc", "ctl"]  # fmt: skip
    # Grants of one cycle come in port order.
    ports = [p.name for p in run_vecadd("poll")["cl"]["xbar"].ports]
    for t in per_cycle:
        mine = [ports.index(e.port) for e in tr.events if e.t == t and e.kind == "grant"]
        assert mine == sorted(mine)


def test_round_trips():
    r = run_vecadd("poll", level="beat")
    p, tr = build_profile(r["cl"]), r["trace"]
    pd, td = json.loads(json.dumps(p.to_dict())), json.loads(json.dumps(tr.to_dict()))
    assert Profile.from_dict(pd) == p and Profile.from_dict(pd).to_dict() == p.to_dict()
    assert Trace.from_dict(td) == tr and Trace.from_dict(td).to_dict() == tr.to_dict()
    ev = td["events"]
    assert all(list(e)[:3] == ["t", "k", "src"] for e in ev)  # fixed key order
    grant = next(e for e in ev if e["k"] == "grant" and e["port"] == "dma.wide")
    assert grant == {"t": 27, "k": "grant", "src": "xbar", "port": "dma.wide", "mem": "l1",
                     "w": True, "addr": 0, "banks": list(range(8)), "row": 0}  # fmt: skip


# =============================================================================
# 8. Recording does not change results
# =============================================================================


def results(r):
    """Everything the model produces, independent of the trace."""
    return {
        "total": r["total"],
        "l1": r["mem"].data.copy(),
        "l2": r["l2"].data.copy(),
        "spans": list(r["ctl"].spans),
        "waits": list(r["ctl"].waits),
        "profile": build_profile(r["cl"]).to_dict(),
    }


def same(a, b):
    for k in a:
        if isinstance(a[k], np.ndarray):
            assert np.array_equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


@pytest.mark.parametrize("mode", ["poll", "signal"])
def test_tracing_off_vs_on(mode):
    base = results(run_vecadd(mode, level=None))
    for level in ("off", "task", "beat"):
        same(base, results(run_vecadd(mode, level=level)))


@pytest.mark.parametrize("seed", range(20))
def test_random_skip_on_off_identical(seed):
    """Profile and beat-level trace identical with skipping on and off."""
    on, off = run_random(seed, True), run_random(seed, False)
    same(results(on), results(off))
    assert on["trace"].to_dict() == off["trace"].to_dict()
    same(results(on), results(run_random(seed, True, level=None)))  # and tracing off
