"""Tests for the SNAX-MODEL register interface and controller (MOD7, D36, D37).

MOD7 acceptance (docs/STATUS.md): poll and signal give the same output data;
cycle counts differ only by the expected control overhead. Plus: the
register map, the adapters per block kind, busy and busy_cycles at the right
cycles, programming the next task while a block runs, command costs, the
MOD6 vecadd driven by a control program, the error cases, and skip on/off
identical under random programs.

Sections:
  1. helpers
  2. register map and commands
  3. per block kind: registers -> start argument, busy, busy_cycles
  4. programming the next task while the block runs
  5. command costs
  6. poll vs signal: same data, cycle difference from the formula
  7. vecadd end to end, L2 -> L1 -> L2, from a control program
  8. errors
  9. random: skip on/off identical
"""

import random

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
    address_stream,
    command_from_dict,
    elementwise_stub,
    program_from_dicts,
    program_to_dicts,
    reduce_stub,
)

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


def compute_blocks(cl, xb, lanes=4, fifo_depth=2, acc_cfg=None, temporal_dims=1):
    """Readers ra, rb, writer wr and an accelerator (elementwise add by default)."""
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


def contiguous(base, n):
    return DmaPattern(base, (n,), (BEAT,))


def unit(word, n_beats, lanes):
    """Streamer task over n_beats contiguous beats of `lanes` words from word `word`."""
    return StreamerRegs(word * WORD, (n_beats,), (lanes * WORD,), (lanes,), (WORD,))


def dma_total(cfg, n, src_latency):
    """done_cycle - start cycle for N beats without contention (D34)."""
    return cfg.startup + src_latency + cfg.beat_interval * (n - 1) + 2 + cfg.done_latency


def w_poll(t, d, c_r, p):
    """Length of a poll wait (ctrl.py module doc)."""
    i = max(0, -(-(d - t - c_r + 1) // p))
    return i * p + c_r


def w_signal(t, d, s):
    return max(t, d) - t + s


def check_waits(ctl, mode):
    """Every wait lasted exactly as the formula says. Returns the wait lengths."""
    c, lens = ctl.cfg, []
    for pc, block, t, d, last in ctl.waits:
        n = last - t + 1
        if mode == "poll":
            assert n == w_poll(t, d, c.read_cost_of(ctl.map[block].kind), c.poll_interval), pc
        else:
            assert n == w_signal(t, d, c.signal_latency), pc
        lens.append(n)
    return lens


def check_status_reads(ctl, block, s, d):
    """Reads of block.busy / block.busy_cycles after its start in s (done in d)."""
    m = ctl.map
    busy, cnt = m.addr(f"{block}.busy"), m.addr(f"{block}.busy_cycles")
    seen = 0
    for r, addr, v in ctl.reads:
        if r <= s:
            continue
        if addr == busy:
            assert v == int(r < d), (r, v)
            seen += 1
        elif addr == cnt:
            assert v == min(r, d) - s - 1, (r, v)
            seen += 1
    return seen


def start_cycle(ctl, block):
    """Cycle in which the (last) start write of `block` landed."""
    addr = ctl.map.addr(f"{block}.start")
    return [last for pc, _, last in ctl.spans if ctl.program[pc] == CsrWrite(addr, 1)][-1]


def assert_cycles_add_up(comp, total):
    assert sum(comp.cycles.values()) == total, comp.cycles


# =============================================================================
# 2. Register map and commands
# =============================================================================


def vecadd_map(temporal_dims=2):
    cl, _, xb, l2 = build()
    dma = cl.add(Dma("dma", xb, l2))
    ra, rb, wr, acc = compute_blocks(cl, xb, temporal_dims=temporal_dims)
    m = RegisterMap([("dma", dma), ("ra", ra), ("rb", rb), ("wr", wr), ("acc", acc)])
    return m


def test_layout_of_the_vecadd_cluster():
    """Status registers at offsets 0-2 of each 32-register window, configuration after."""
    m = vecadd_map()
    d = m.to_dict()
    assert d["window"] == 32
    assert {b: v["base"] for b, v in d["blocks"].items()} == {
        "dma": 0, "ra": 32, "rb": 64, "wr": 96, "acc": 128}  # fmt: skip
    assert {b: v["kind"] for b, v in d["blocks"].items()} == {
        "dma": "dma", "ra": "streamer", "rb": "streamer", "wr": "streamer", "acc": "accel"}  # fmt: skip
    dma = d["blocks"]["dma"]["registers"]
    assert list(dma) == ["start", "busy", "busy_cycles", "direction", "src_base", "src_bound[0]",
                         "src_bound[1]", "src_stride[0]", "src_stride[1]", "dst_base",
                         "dst_bound[0]", "dst_bound[1]", "dst_stride[0]", "dst_stride[1]"]  # fmt: skip
    assert list(dma.values()) == list(range(14))
    ra = d["blocks"]["ra"]["registers"]
    assert ra == {"start": 32, "busy": 33, "busy_cycles": 34, "base": 35, "tbound[0]": 36,
                  "tbound[1]": 37, "tstride[0]": 38, "tstride[1]": 39, "sstride[0]": 40}  # fmt: skip
    assert d["blocks"]["acc"]["registers"] == {"start": 128, "busy": 129, "busy_cycles": 130,
                                               "n": 131}  # fmt: skip
    assert m.addr("dma.src_base") == 4 and m.describe(35) == "ra.base"
    assert m.describe(20) == "<unmapped 20>"


def test_a_block_does_not_move_when_another_changes():
    """Windows are fixed: rb, wr and acc keep their addresses when ra's config grows."""
    a, b = vecadd_map(temporal_dims=1), vecadd_map(temporal_dims=4)
    for name in ("rb.start", "wr.busy", "acc.n", "dma.dst_base"):
        assert a.addr(name) == b.addr(name)
    assert b.addr("ra.sstride[0]") == 32 + 3 + 1 + 4 + 4


def test_reduce_rates_and_spatial_bounds():
    cl, _, xb, _ = build()
    ra, _, wr, acc = compute_blocks(cl, xb, lanes=4, acc_cfg=reduce_stub(lanes=4))
    m = RegisterMap([("ra", ra), ("wr", wr), ("acc", acc)], spatial_bounds={"ra": (2, 2)})
    assert m["acc"].config_names == ["n", "T"]
    assert m["ra"].config_names == ["base", "tbound[0]", "tstride[0]", "sstride[0]", "sstride[1]"]
    assert m["wr"].config_names == ["base", "tbound[0]", "tstride[0]", "sstride[0]"]
    regs = StreamerRegs(0, (3,), (32,), (2, 2), (8, 64))
    vals = {w.addr: w.value for w in m.config_writes("ra", regs)}
    assert vals[m.addr("ra.sstride[1]")] == 64


def test_explicit_bases_and_window():
    cl, _, xb, l2 = build()
    dma = cl.add(Dma("dma", xb, l2, DmaConfig(dims=1)))
    m = RegisterMap([("dma", dma)], window=16, bases={"dma": 48})
    assert m.addr("dma.start") == 48 and m.addr("dma.dst_stride[0]") == 48 + 3 + 6


def test_command_dict_round_trip():
    prog = [CsrWrite(4, -64), CsrWrite(0, 1), CsrRead(1), Wait("dma", "poll"), Wait("ra")]
    ds = program_to_dicts(prog)
    assert ds[0] == {"op": "csr_write", "addr": 4, "value": -64}
    assert ds[4] == {"op": "wait", "block": "ra", "mode": "signal"}
    assert program_from_dicts(ds) == prog
    with pytest.raises(ValueError):
        command_from_dict({"op": "dma"})
    with pytest.raises(ValueError):
        Wait("dma", "sleep")
    c = ControllerConfig(kind_write_cost={"dma": 3})
    assert ControllerConfig.from_dict(c.to_dict()) == c


# =============================================================================
# 3. Per block kind: registers -> start argument, busy, busy_cycles
# =============================================================================


def status_reads(m, block, k):
    return [CsrRead(m.addr(f"{block}.{r}")) for _ in range(k) for r in ("busy", "busy_cycles")]


@pytest.mark.parametrize("skip", [True, False])
def test_streamer_from_registers(skip):
    """A reader alone (3 beats fit its FIFO): start in s, done in s + 2 + 3."""
    cl, mem, xb, _ = build(skip)
    mem.load(0, np.arange(64))
    rd = cl.add(Streamer("rd", xb, StreamerConfig(n_ports=2, temporal_dims=2, fifo_depth=8)))
    m = RegisterMap([("rd", rd)])
    regs = StreamerRegs(16, (3,), (16,), (2,), (8,))
    prog = [*m.start_writes("rd", regs), *status_reads(m, "rd", 4), Wait("rd"),
            *status_reads(m, "rd", 1)]  # fmt: skip
    ctl = cl.add(Controller("ctl", m, prog))
    total = cl.run(max_cycles=500)

    assert ctl.finished
    s = start_cycle(ctl, "rd")
    assert s == 6  # 6 configuration writes + start, 1 cycle each
    assert rd.regs == StreamerRegs(16, (3, 1), (16, 0), (2,), (8,))  # padded loop
    assert np.array_equal(address_stream(rd.regs), address_stream(regs))
    assert rd.done_cycle == s + 2 + 3
    assert check_status_reads(ctl, "rd", s, rd.done_cycle) == 10
    assert ctl.reads[-1][2] == 2 + 3 - 1  # busy_cycles after done = done - s - 1
    lanes = [[int(np.asarray(x).ravel()[0]) for x in q] for q in rd.fifo._q]
    assert lanes == [[2, 4, 6], [3, 5, 7]]
    assert_cycles_add_up(ctl, total)


@pytest.mark.parametrize("skip", [True, False])
def test_accelerator_from_registers(skip):
    """Reduce stub with n and T from registers: 8 firings, T = 4 -> 2 output beats."""
    cl, mem, xb, _ = build(skip)
    x = np.arange(16) + 1
    mem.load(0, x)
    ra, _, wr, acc = compute_blocks(cl, xb, lanes=2, acc_cfg=reduce_stub(lanes=2))
    m = RegisterMap([("ra", ra), ("wr", wr), ("acc", acc)])
    prog = [*m.start_writes("ra", unit(0, 8, 2)), *m.start_writes("wr", unit(32, 2, 2)),
            *m.start_writes("acc", {"n": 8, "T": 4}), *status_reads(m, "acc", 6),
            Wait("wr", "poll"), Wait("acc"), *status_reads(m, "acc", 1)]  # fmt: skip
    ctl = cl.add(Controller("ctl", m, prog))
    total = cl.run(max_cycles=500)

    assert ctl.finished
    assert acc.params == {"n": 8, "T": 4}
    s = start_cycle(ctl, "acc")
    assert check_status_reads(ctl, "acc", s, acc.done_cycle) == 14
    sums = x.reshape(2, 4, 2).sum(axis=1)  # per lane over T = 4 beats
    assert np.array_equal(mem.dump(32 * WORD, 4)[:, 0], sums.ravel())
    for comp in (ctl, ra, wr, acc):
        assert_cycles_add_up(comp, total)


@pytest.mark.parametrize("skip", [True, False])
def test_dma_from_registers(skip):
    """2-loop destination: done = start + the D34 formula."""
    cl, mem, xb, l2 = build(skip, l2_lat=3)
    l2.load(0, np.arange(64) + 100)
    dma = cl.add(Dma("dma", xb, l2))
    m = RegisterMap([("dma", dma)])
    desc = DmaDescriptor("l2_to_l1", contiguous(0, 4), DmaPattern(0, (2, 2), (BEAT, NB * 16)))
    prog = [*m.start_writes("dma", desc), *status_reads(m, "dma", 3), Wait("dma"),
            *status_reads(m, "dma", 1)]  # fmt: skip
    ctl = cl.add(Controller("ctl", m, prog))
    total = cl.run(max_cycles=500)

    assert ctl.finished
    s = start_cycle(ctl, "dma")
    assert s == 11  # 11 configuration writes + start
    assert dma.desc.direction == "l2_to_l1"
    assert np.array_equal(dma.desc.src.addresses(), desc.src.addresses())
    assert np.array_equal(dma.desc.dst.addresses(), desc.dst.addresses())
    assert dma.done_cycle == s + dma_total(DmaConfig(), 4, 3)
    assert check_status_reads(ctl, "dma", s, dma.done_cycle) == 8
    for i, a in enumerate(desc.dst.addresses()):
        assert np.array_equal(mem.dump(int(a), 8)[:, 0], np.arange(8) + 100 + 8 * i)
    assert_cycles_add_up(ctl, total)


# =============================================================================
# 4. Programming the next task while the block runs
# =============================================================================


@pytest.mark.parametrize("mode", ["poll", "signal"])
@pytest.mark.parametrize("skip", [True, False])
def test_program_next_task_while_running(skip, mode):
    """Configuration of transfer b is written while a runs; start after the wait."""
    cl, mem, xb, l2 = build(skip)
    l2.load(0, np.arange(256))
    dma = cl.add(Dma("dma", xb, l2))
    m = RegisterMap([("dma", dma)])
    a = DmaDescriptor("l2_to_l1", contiguous(0, 16), contiguous(0, 16))
    b = DmaDescriptor("l2_to_l1", contiguous(16 * BEAT, 6), contiguous(32 * BEAT, 6))
    prog = [*m.start_writes("dma", a), *m.config_writes("dma", b),
            CsrRead(m.addr("dma.busy")), CsrRead(m.addr("dma.src_base")),
            Wait("dma", mode), m.start_write("dma"), Wait("dma", mode)]  # fmt: skip
    ctl = cl.add(Controller("ctl", m, prog))
    cl.run(max_cycles=500)

    assert ctl.finished
    first_done = ctl.waits[0][3]
    assert ctl.spans[22][2] < first_done  # b fully programmed while a still ran
    assert [v for _, _, v in ctl.reads] == [1, 16 * BEAT]  # busy, and the shadow value
    assert np.array_equal(mem.dump(0, 128)[:, 0], np.arange(128))
    assert np.array_equal(mem.dump(32 * BEAT, 48)[:, 0], np.arange(128, 176))
    check_waits(ctl, mode)


# =============================================================================
# 5. Command costs
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
def test_command_costs_per_kind(skip):
    cl, _, xb, l2 = build(skip)
    dma = cl.add(Dma("dma", xb, l2))
    ra, rb, wr, acc = compute_blocks(cl, xb)
    m = RegisterMap([("dma", dma), ("ra", ra), ("rb", rb), ("wr", wr), ("acc", acc)])
    cfg = ControllerConfig(write_cost=2, read_cost=3, kind_write_cost={"dma": 5},
                           kind_read_cost={"streamer": 4})  # fmt: skip
    prog = [CsrWrite(m.addr("dma.src_base"), 64), CsrWrite(m.addr("ra.base"), 8),
            CsrRead(m.addr("dma.busy")), CsrRead(m.addr("ra.base")),
            CsrWrite(m.addr("acc.n"), 7), CsrRead(m.addr("acc.n"))]  # fmt: skip
    ctl = cl.add(Controller("ctl", m, prog, cfg))
    total = cl.run()

    assert [last - first + 1 for _, first, last in ctl.spans] == [5, 2, 3, 4, 2, 3]
    assert [(r, v) for r, _, v in ctl.reads] == [(9, 0), (13, 8), (18, 7)]
    assert total == 19
    assert ctl.cycles == {"command": 19, "wait": 0, "idle": 0}


# =============================================================================
# 6. Poll vs signal: same data, cycle difference from the formula
# =============================================================================

WAIT_CASES = [  # (write_cost, dma read cost, poll_interval, signal_latency)
    (1, 1, 1, 1),
    (2, 1, 3, 1),
    (1, 3, 3, 2),
    (1, 2, 7, 4),
    (3, 4, 5, 1),
]


def run_one_dma(case, mode, skip):
    wc, rc, p, s = case
    cl, mem, xb, l2 = build(skip)
    l2.load(0, np.arange(80) * 3)
    dma = cl.add(Dma("dma", xb, l2))
    m = RegisterMap([("dma", dma)])
    cfg = ControllerConfig(write_cost=wc, kind_read_cost={"dma": rc}, poll_interval=p,
                           signal_latency=s)  # fmt: skip
    desc = DmaDescriptor("l2_to_l1", contiguous(0, 10), contiguous(0, 10))
    prog = [*m.start_writes("dma", desc), Wait("dma", mode), CsrRead(m.addr("dma.busy_cycles"))]
    ctl = cl.add(Controller("ctl", m, prog, cfg))
    total = cl.run(max_cycles=1000)
    return ctl, mem.data.copy(), total


@pytest.mark.parametrize("case", WAIT_CASES)
@pytest.mark.parametrize("skip", [True, False])
def test_poll_vs_signal_one_task(case, skip):
    """T_ctrl = (n+1) c_w + W + c_r; W_poll - W_signal follows from the busy cycles B."""
    wc, rc, p, s = case
    runs = {mode: run_one_dma(case, mode, skip) for mode in ("poll", "signal")}
    assert np.array_equal(runs["poll"][1], runs["signal"][1])
    n_cfg = 11
    b = runs["signal"][0].reads[-1][2]  # busy_cycles after done
    assert b == runs["poll"][0].reads[-1][2] == dma_total(DmaConfig(), 10, 1) - 1
    w = {"signal": b + s, "poll": max(0, -(-(b - rc + 1) // p)) * p + rc}
    for mode, (ctl, _, total) in runs.items():
        assert check_waits(ctl, mode) == [w[mode]]
        assert total == (n_cfg + 1) * wc + w[mode] + rc
        assert ctl.cycles == {"command": (n_cfg + 1) * wc + rc, "wait": w[mode], "idle": 0}
    assert runs["poll"][2] - runs["signal"][2] == w["poll"] - w["signal"]


def test_poll_counts_its_reads():
    ctl, _, _ = run_one_dma((1, 2, 3, 1), "poll", True)
    _, _, t, d, _ = ctl.waits[0]
    assert ctl.polls == max(0, -(-(d - t - 2 + 1) // 3)) + 1
    assert len(ctl.reads) == 1  # polls are not in the read log


# =============================================================================
# 7. Vecadd end to end, L2 -> L1 -> L2, from a control program
# =============================================================================


def run_vecadd(mode, skip, cfg=None):
    """The MOD6 vecadd, now driven by a control program instead of hand-picked cycles.

    Transfer b and the compute blocks are programmed while the previous
    transfer runs; each start comes after a wait on the block it needs.
    """
    n_elems, lanes = 64, 4
    nb, n_dma = n_elems // lanes, n_elems // 8
    cl, mem, xb, l2 = build(skip)
    rng = np.random.default_rng(3)
    a = rng.integers(-1000, 1000, n_elems)
    b = rng.integers(-1000, 1000, n_elems)
    l2a, l2b, l2c = 0, 1024, 2048
    l2.load(l2a, a)
    l2.load(l2b, b)
    wa, wb, wc = 0, 72, 144  # same L1 layout as test_dma
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
    return {"ctl": ctl, "total": total, "c": l2.dump(l2c, n_elems)[:, 0], "a": a, "b": b,
            "l1": mem.data.copy(), "l2": l2.data.copy(), "comps": (dma, ra, rb, wr, acc)}  # fmt: skip


@pytest.mark.parametrize("mode", ["poll", "signal"])
@pytest.mark.parametrize("skip", [True, False])
def test_vecadd_from_a_control_program(mode, skip):
    cfg = ControllerConfig(write_cost=1, kind_write_cost={"dma": 2}, read_cost=2, poll_interval=4)
    r = run_vecadd(mode, skip, cfg)
    ctl, total = r["ctl"], r["total"]
    assert ctl.finished
    assert np.array_equal(r["c"], r["a"] + r["b"])
    # The first transfer runs exactly as D34 says from its start write.
    dma = r["comps"][0]
    s = ctl.spans[11][2]
    assert s == 12 * 2 - 1
    assert ctl.waits[0][3] == s + dma_total(DmaConfig(), 8, 1)
    # Controller time = commands + waits, each wait from the formula.
    waits = check_waits(ctl, mode)
    commands = sum(last - first + 1 for pc, first, last in ctl.spans
                   if not isinstance(ctl.program[pc], Wait))  # fmt: skip
    assert ctl.cycles["command"] == commands and ctl.cycles["wait"] == sum(waits)
    assert ctl.spans[-1][2] + 1 == commands + sum(waits)
    # The run ends with the last wait (the DMA is done before it resumes).
    assert total == ctl.spans[-1][2] + 1 and dma.done_cycle <= total
    for comp in (ctl, *r["comps"]):
        assert_cycles_add_up(comp, total)


def test_vecadd_poll_and_signal_same_data():
    """Identical L1 and L2; the difference is the sum of the per-wait differences."""
    cfg = ControllerConfig(read_cost=2, poll_interval=5, signal_latency=2)
    p, s = run_vecadd("poll", True, cfg), run_vecadd("signal", True, cfg)
    assert np.array_equal(p["l1"], s["l1"]) and np.array_equal(p["l2"], s["l2"])
    diff = sum(check_waits(p["ctl"], "poll")) - sum(check_waits(s["ctl"], "signal"))
    assert p["total"] - s["total"] == diff


# =============================================================================
# 8. Errors
# =============================================================================


def small_setup():
    cl, _, xb, l2 = build()
    dma = cl.add(Dma("dma", xb, l2))
    ra, _, wr, acc = compute_blocks(cl, xb, lanes=2, acc_cfg=reduce_stub(lanes=2))
    m = RegisterMap([("dma", dma), ("ra", ra), ("wr", wr), ("acc", acc)])
    return cl, m


STATIC_ERRORS = {
    "unknown address write": lambda m: [CsrWrite(20, 1)],
    "unknown address read": lambda m: [CsrRead(999)],
    "write busy": lambda m: [CsrWrite(m.addr("dma.busy"), 0)],
    "write busy_cycles": lambda m: [CsrWrite(m.addr("ra.busy_cycles"), 0)],
    "read start": lambda m: [CsrRead(m.addr("acc.start"))],
    "start value 0": lambda m: [CsrWrite(m.addr("dma.start"), 0)],
    "wait unknown block": lambda m: [Wait("ghost")],
    "wait never started": lambda m: [CsrWrite(m.addr("ra.base"), 0), Wait("ra")],
    "wait before start": lambda m: [Wait("acc"), CsrWrite(m.addr("acc.start"), 1)],
    "poll interval below read cost": lambda m: [m.start_write("dma"), Wait("dma", "poll")],
}


@pytest.mark.parametrize("name", list(STATIC_ERRORS))
def test_static_errors(name):
    _, m = small_setup()
    cfg = ControllerConfig(read_cost=3, poll_interval=2)
    with pytest.raises(ValueError):
        Controller("ctl", m, STATIC_ERRORS[name](m), cfg)


def good_dma():
    return DmaDescriptor("l2_to_l1", contiguous(0, 2), contiguous(0, 2))


RUN_ERRORS = {
    "start while busy": lambda m: [*m.start_writes("dma", good_dma()), m.start_write("dma")],
    "negative bound": lambda m: [*m.start_writes("ra", unit(0, 4, 2)),
                                 CsrWrite(m.addr("ra.tbound[0]"), -1), m.start_write("ra")],
    "bad direction": lambda m: [*m.config_writes("dma", good_dma()),
                                CsrWrite(m.addr("dma.direction"), 7), m.start_write("dma")],
    "unaligned dma base": lambda m: [*m.config_writes("dma", good_dma()),
                                     CsrWrite(m.addr("dma.dst_base"), 8), m.start_write("dma")],
    "n not a multiple of T": lambda m: [*m.start_writes("acc", {"n": 6, "T": 4})],
    # Not modelled yet (D32, open item 5): the streamer's own NotImplementedError.
    "reader temporal stride 0": lambda m: [*m.config_writes("ra", unit(0, 4, 2)),
                                           CsrWrite(m.addr("ra.tstride[0]"), 0),
                                           m.start_write("ra")],
}  # fmt: skip


@pytest.mark.parametrize("name", list(RUN_ERRORS))
def test_run_errors(name):
    cl, m = small_setup()
    cl.add(Controller("ctl", m, RUN_ERRORS[name](m)))
    with pytest.raises((SimulationError, NotImplementedError)):
        cl.run(max_cycles=500)


def test_start_while_busy_names_the_block_and_cycle():
    cl, m = small_setup()
    cl.add(Controller("ctl", m, RUN_ERRORS["start while busy"](m)))
    with pytest.raises(SimulationError, match=r"cycle 12: dma.start: start while busy"):
        cl.run()


def test_zero_beat_start_is_not_busy():
    """All registers at reset (0): the DMA starts a zero-beat transfer and is done at once."""
    cl, m = small_setup()
    ctl = cl.add(Controller("ctl", m, [m.start_write("dma"), Wait("dma", "poll"),
                                       CsrRead(m.addr("dma.busy_cycles"))]))  # fmt: skip
    cl.run()
    assert ctl.finished and ctl.reads[-1][2] == 0 and ctl.waits[0][3] == 1


def test_unfinished_signal_wait():
    """A writer that never gets data: the run ends with the wait pending."""
    cl, m = small_setup()
    ctl = cl.add(Controller("ctl", m, [*m.start_writes("wr", unit(0, 2, 2)), Wait("wr")]))
    total = cl.run(max_cycles=500)
    assert not ctl.finished
    assert_cycles_add_up(ctl, total)


MAP_ERRORS = {
    "window not a power of two": lambda b: RegisterMap(b, window=24),
    "block does not fit": lambda b: RegisterMap(b, window=8),
    "unaligned base": lambda b: RegisterMap(b, bases={"dma": 40}),
    "shared window": lambda b: RegisterMap(b, bases={"ra": 0}),
    "duplicate name": lambda b: RegisterMap([b[0], b[0]]),
    "spatial bounds wrong lanes": lambda b: RegisterMap(b, spatial_bounds={"ra": (3,)}),
    "spatial bounds on a non-streamer": lambda b: RegisterMap(b, spatial_bounds={"dma": (2,)}),
    "unknown block option": lambda b: RegisterMap(b, bases={"ghost": 0}),
    "no adapter": lambda b: RegisterMap([("x", b[0][1].xbar)]),
}


@pytest.mark.parametrize("name", list(MAP_ERRORS))
def test_map_errors(name):
    cl, _, xb, l2 = build()
    dma = cl.add(Dma("dma", xb, l2))
    ra = cl.add(Streamer("ra", xb, StreamerConfig(n_ports=2)))
    with pytest.raises(ValueError):
        MAP_ERRORS[name]([("dma", dma), ("ra", ra)])


def test_encode_errors():
    cl, _, xb, l2 = build()
    dma = cl.add(Dma("dma", xb, l2, DmaConfig(dims=1)))
    ra, _, _, acc = compute_blocks(cl, xb, lanes=2, acc_cfg=reduce_stub(lanes=2))
    m = RegisterMap([("dma", dma), ("ra", ra), ("acc", acc)])
    with pytest.raises(ValueError):  # more loops than DmaConfig.dims
        m.config_writes("dma", DmaDescriptor("l2_to_l1", contiguous(0, 4),
                                             DmaPattern(0, (2, 2), (BEAT, 2 * BEAT))))  # fmt: skip
    with pytest.raises(ValueError):  # more loops than temporal_dims
        m.config_writes("ra", StreamerRegs(0, (2, 2), (16, 32), (2,), (8,)))
    with pytest.raises(ValueError):  # spatial bounds are design-time
        m.config_writes("ra", StreamerRegs(0, (2,), (16,), (1, 2), (8, 8)))
    with pytest.raises(ValueError):  # T missing
        m.config_writes("acc", {"n": 4})


# =============================================================================
# 9. Random: skip on/off identical
# =============================================================================

L1_ROWS = 64
L1_WORDS = NB * L1_ROWS
L2_SIZE = 1 << 13


def random_l1_dma(rng, n):
    """n aligned beats inside L1: contiguous or 2 loops."""
    while True:
        if n > 1 and rng.random() < 0.5:
            inner = int(rng.choice([d for d in range(1, n + 1) if n % d == 0]))
            bounds = (inner, n // inner)
        else:
            bounds = (n,)
        strides = tuple(int(rng.choice([1, 2, NB // 8])) * BEAT for _ in bounds)
        p = DmaPattern(int(rng.integers(0, L1_WORDS * WORD // BEAT)) * BEAT, bounds, strides)
        a = p.addresses()
        if a.max() + BEAT <= L1_WORDS * WORD:
            return p


def random_program(m, r, rng, lanes):
    """Rounds of DMA transfers and compute tasks, with waits before every restart.

    Configuration writes come before the wait on the same block (buffered
    registers), csr_reads and status reads are sprinkled in, modes random.
    """
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
        compute = ["ra", "rb", "wr", "acc"]
        for s in ("ra", "rb", "wr"):
            base = r.randrange(0, L1_WORDS - nb * lanes)
            prog.extend(m.config_writes(s, unit(base, nb, lanes)))
        prog.extend(m.config_writes("acc", {"n": nb}))
        reads()
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


def random_case(seed):
    r = random.Random(seed)
    lanes = r.choice([1, 2, 4])
    fifo = r.choice([1, 2, 4])
    dma = DmaConfig(startup=r.choice([1, 2, 4]), beat_interval=r.choice([1, 1, 2]),
                    done_latency=r.choice([0, 0, 3]))  # fmt: skip
    rc = {k: r.choice([1, 2, 3]) for k in ("streamer", "accel", "dma")}
    ctl = ControllerConfig(
        write_cost=r.choice([1, 2]),
        read_cost=r.choice([1, 2]),
        kind_write_cost={k: r.choice([1, 2, 4]) for k in ("dma", "streamer") if r.random() < 0.5},
        kind_read_cost=rc,
        poll_interval=max(rc.values()) + r.choice([0, 1, 4]),
        signal_latency=r.choice([1, 1, 3]),
    )
    return seed, lanes, fifo, dma, ctl, r.choice([0, 1, 2])


def run_case(case, skip):
    seed, lanes, fifo, dcfg, ccfg, l1_lat = case
    cl, mem, xb, l2 = build(skip, rows=L1_ROWS, l1_lat=l1_lat, l2_size=L2_SIZE)
    rng = np.random.default_rng(seed)
    mem.load(0, rng.integers(-500, 500, L1_WORDS))
    l2.load(0, rng.integers(-500, 500, L2_SIZE // WORD))
    dma = cl.add(Dma("dma", xb, l2, dcfg))
    ra, rb, wr, acc = compute_blocks(cl, xb, lanes=lanes, fifo_depth=fifo)
    m = RegisterMap([("dma", dma), ("ra", ra), ("rb", rb), ("wr", wr), ("acc", acc)])
    prog = random_program(m, random.Random(seed + 1000), np.random.default_rng(seed + 1000), lanes)
    ctl = cl.add(Controller("ctl", m, prog, ccfg))
    total = cl.run(max_cycles=20000)
    comps = (ctl, dma, ra, rb, wr, acc)
    return {
        "total": total,
        "finished": ctl.finished,
        "reads": ctl.reads,
        "spans": ctl.spans,
        "waits": ctl.waits,
        "polls": ctl.polls,
        "cycles": [dict(c.cycles) for c in comps],
        "done": [c.done_cycle for c in comps[1:]],
        "l1": mem.data.copy(),
        "l2": l2.data.copy(),
        "grants": xb.port_grants.copy(),
        "stalls": xb.port_stalls.copy(),
        "sums_ok": all(sum(c.cycles.values()) == total for c in comps),
        "prog": prog,
        "kinds": {b: m[b].kind for b in m.blocks},
    }


@pytest.mark.parametrize("seed", range(20))
def test_random_skip_on_off_identical(seed):
    case = random_case(seed)
    a, b = run_case(case, True), run_case(case, False)
    for k in a:
        if isinstance(a[k], np.ndarray):
            assert np.array_equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k
    assert a["finished"] and a["sums_ok"]
    # Every wait lasted as the formula says, in both modes.
    c = case[4]
    assert a["waits"]
    for pc, block, t, d, last in a["waits"]:
        if a["prog"][pc].mode == "poll":
            want = w_poll(t, d, c.read_cost_of(a["kinds"][block]), c.poll_interval)
        else:
            want = w_signal(t, d, c.signal_latency)
        assert last - t + 1 == want, pc
