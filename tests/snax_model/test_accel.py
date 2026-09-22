"""Tests for the SNAX-MODEL accelerator interface and stubs (MOD5).

MOD5 acceptance (docs/STATUS.md):
  * with ideal streams, both stubs hit their cycle formulas, and
  * the reduce stub proves unequal port rates work.
Plus: a slow consumer freezes the pipeline and the freeze reaches the
inputs (toy producers and real reader streamers), vecadd end to end with
real streamers, and skip on/off identical under random configs and timing.

How the tests are built
-----------------------
"Ideal streams" are toy components in COMPUTE, registered after the
accelerator: ``Producer`` pushes beats into an input FIFO (pipe = true,
like a reader's) and ``Consumer`` pops beats from an output FIFO
(pipe = false, like a writer's). Each is willing in a given set of cycles
(every cycle by default).

With a producer pushing beat i in cycle i, beat 0 is visible in cycle 1.
An accelerator started before the run then fires at

    f_k = 1 + k * II

and a result fired at f is pushed at f + L. For a port of rate r, output
beat j comes from firing (j+1)r - 1. ``done_cycle`` is the cycle after the
last push.

Sections:
  1. helpers and toy components
  2. interface and wiring checks
  3. acceptance: elementwise stub cycle formula
  4. acceptance: reduce stub, unequal port rates
  5. freeze on a full output reaches the inputs
  6. start and done
  7. vecadd end to end with streamers
  8. random configs: skip on/off
"""

import bisect
import random

import numpy as np
import pytest

from snax_forge.snax_model import (
    AccelConfig,
    Accelerator,
    AccelPort,
    Cluster,
    Component,
    Fifo,
    L1Config,
    L1Memory,
    Phase,
    SimulationError,
    Streamer,
    StreamerConfig,
    StreamerRegs,
    Xbar,
    elementwise_stub,
    reduce_stub,
)

WORD = 8  # bytes per word with the default 64-bit banks


# =============================================================================
# 1. Helpers and toy components
# =============================================================================


class Willing:
    """A set of cycles; None means every cycle. ``after``: every cycle from then on."""

    def __init__(self, cycles=None, after=None):
        self.every = cycles is None and after is None
        self.cycles = sorted(set(cycles or ()))
        self.after = after

    def __contains__(self, c):
        return self.every or (self.after is not None and c >= self.after) or c in self.cycles

    def next_after(self, c):
        if self.every:
            return c + 1
        i = bisect.bisect_right(self.cycles, c)
        cands = [self.cycles[i]] if i < len(self.cycles) else []
        if self.after is not None:
            cands.append(max(c + 1, self.after))
        return min(cands) if cands else None


class Producer(Component):
    """Pushes ``beats`` into ``fifo`` in COMPUTE, in willing cycles."""

    phases = (Phase.COMPUTE,)

    def __init__(self, name, fifo, beats, willing=None):
        super().__init__(name)
        self.fifo, self.beats = fifo, [list(b) for b in beats]
        self.willing = willing or Willing()
        fifo.pusher = self
        self.i = 0
        self.push_cycles = []

    def tick(self, cycle, phase):
        if self.i < len(self.beats) and cycle in self.willing and self.fifo.can_push():
            self.fifo.push(cycle, self.beats[self.i])
            self.push_cycles.append(cycle)
            self.i += 1

    def next_wake(self, cycle):
        return None if self.i >= len(self.beats) else self.willing.next_after(cycle)


class Consumer(Component):
    """Pops ``total`` beats from ``fifo`` in COMPUTE, in willing cycles."""

    phases = (Phase.COMPUTE,)

    def __init__(self, name, fifo, total, willing=None):
        super().__init__(name)
        self.fifo, self.total = fifo, total
        self.willing = willing or Willing()
        fifo.popper = self
        self.beats = []  # (cycle, beat)

    def tick(self, cycle, phase):
        if len(self.beats) < self.total and cycle in self.willing and self.fifo.can_pop():
            beat = self.fifo.pop(cycle)
            self.beats.append((cycle, [int(np.asarray(x).ravel()[0]) for x in beat]))

    def next_wake(self, cycle):
        return None if len(self.beats) >= self.total else self.willing.next_after(cycle)

    @property
    def values(self):
        return np.array([v for _, v in self.beats], dtype=np.int64).reshape(-1, self.fifo.lanes)

    @property
    def pop_cycles(self):
        return [c for c, _ in self.beats]


class ScriptedStarts(Component):
    """Starts components in scripted cycles: {cycle: (comp, arg)}."""

    phases = (Phase.CONTROL,)

    def __init__(self, name, script):
        super().__init__(name)
        self.script = script

    def tick(self, cycle, phase):
        if cycle in self.script:
            comp, arg = self.script[cycle]
            comp.start(arg, cycle)

    def next_wake(self, cycle):
        later = [c for c in self.script if c > cycle]
        return min(later) if later else None


class Probe(Component):
    """Records ``comp.busy`` in every cycle of ``window``."""

    phases = (Phase.CONTROL,)

    def __init__(self, name, comp, window):
        super().__init__(name)
        self.c, self.window = comp, window
        self.busy = {}

    def tick(self, cycle, phase):
        if cycle in self.window:
            self.busy[cycle] = self.c.busy

    def next_wake(self, cycle):
        later = [c for c in self.window if c > cycle]
        return min(later) if later else None


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


class LogStreamer(Streamer):
    """Streamer that logs grant cycles."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.grants = []

    def commit(self, cycle):
        if self._fire.any():
            self.grants.append(cycle)
        super().commit(cycle)


def toy_setup(cfg, in_beats, n_out, skip=True, depth_in=2, depth_out=2, willing_in=None,
              willing_out=None):  # fmt: skip
    """Accelerator with a Producer per input and a Consumer per output.

    ``in_beats``: {port: beats}; ``n_out``: {port: beats to pop}.
    Returns (cluster, acc, producers, consumers).
    """
    cl = Cluster(skip_idle=skip)
    acc = cl.add(LogAccel("acc", cl, cfg))
    prods, cons = {}, {}
    for p in cfg.inputs:
        f = Fifo(cl, p.lanes, depth_in, pipe=True, name=f"{p.name}.fifo")
        acc.attach(p.name, f)
        w = (willing_in or {}).get(p.name)
        prods[p.name] = cl.add(Producer(f"prod_{p.name}", f, in_beats[p.name], w))
    for p in cfg.outputs:
        f = Fifo(cl, p.lanes, depth_out, pipe=False, name=f"{p.name}.fifo")
        acc.attach(p.name, f)
        w = (willing_out or {}).get(p.name)
        cons[p.name] = cl.add(Consumer(f"cons_{p.name}", f, n_out[p.name], w))
    return cl, acc, prods, cons


def assert_cycles_add_up(comp, total):
    assert sum(comp.cycles.values()) == total, comp.cycles


def rand_beats(rng, n, lanes, lo=-50, hi=50):
    return rng.integers(lo, hi, size=(n, lanes))


# =============================================================================
# 2. Interface and wiring checks
# =============================================================================


def test_config_validation():
    fn = elementwise_stub().fn
    with pytest.raises(ValueError):
        AccelPort("a", "sideways")
    with pytest.raises(ValueError):
        AccelPort("a", "in", lanes=0)
    with pytest.raises(ValueError):
        AccelPort("a", "in", rate=0)
    with pytest.raises(ValueError):  # duplicate names
        AccelConfig((AccelPort("a", "in"), AccelPort("a", "out")), fn)
    with pytest.raises(ValueError):  # no output
        AccelConfig((AccelPort("a", "in"),), fn)
    with pytest.raises(ValueError):
        AccelConfig((AccelPort("a", "in"), AccelPort("c", "out")), fn, latency=-1)
    with pytest.raises(ValueError):
        AccelConfig((AccelPort("a", "in"), AccelPort("c", "out")), fn, ii=0)
    with pytest.raises(ValueError):
        reduce_stub(lanes=4, lanes_out=2)


def test_stubs_are_configs():
    ew, rd = elementwise_stub(lanes=4), reduce_stub(lanes=2)
    assert [(p.name, p.direction, p.lanes, p.rate) for p in ew.ports] == [
        ("a", "in", 4, 1), ("b", "in", 4, 1), ("out", "out", 4, 1)]  # fmt: skip
    assert [(p.name, p.direction, p.lanes, p.rate) for p in rd.ports] == [
        ("in", "in", 2, 1), ("out", "out", 2, "T")]  # fmt: skip
    assert (ew.latency, ew.ii, rd.latency, rd.ii) == (0, 1, 1, 1)
    assert type(Accelerator("x", Cluster(), ew)) is Accelerator


def test_attach_checks():
    cl = Cluster()
    acc = Accelerator("acc", cl, elementwise_stub(lanes=2))
    with pytest.raises(ValueError):  # lane mismatch
        acc.attach("a", Fifo(cl, 3, 2))
    with pytest.raises(KeyError):
        acc.attach("zz", Fifo(cl, 2, 2))
    f = Fifo(cl, 2, 2)
    acc.attach("a", f)
    assert f.popper is acc
    with pytest.raises(ValueError):  # already attached
        acc.attach("a", Fifo(cl, 2, 2))
    with pytest.raises(ValueError):  # start with ports missing
        acc.start({"n": 1})
    acc.attach("b", Fifo(cl, 2, 2))
    # A FIFO whose push side is taken cannot be an output.
    g = Fifo(cl, 2, 2)
    g.pusher = Component("someone")
    with pytest.raises(ValueError):
        acc.attach("out", g)
    out = Fifo(cl, 2, 2)
    acc.attach("out", out)
    assert out.pusher is acc
    with pytest.raises(ValueError):
        acc.start({})  # no n


def test_attach_to_streamer_fifos():
    """Reader FIFO -> input, writer FIFO -> output; the wrong way round fails."""
    cl = Cluster()
    mem = L1Memory(cl, L1Config(n_banks=4, rows=16))
    xb = cl.add(Xbar("xbar", mem))
    rd = Streamer("rd", xb, StreamerConfig(write=False, n_ports=2))
    wr = Streamer("wr", xb, StreamerConfig(write=True, n_ports=2))
    acc = Accelerator("acc", cl, elementwise_stub(lanes=2, n_inputs=1))
    with pytest.raises(ValueError):
        acc.attach("a", wr.fifo)
    with pytest.raises(ValueError):
        acc.attach("out", rd.fifo)
    acc.attach("a", rd.fifo)
    acc.attach("out", wr.fifo)


def test_rate_must_divide_n():
    _, acc, _, _ = toy_setup(reduce_stub(), {"in": []}, {"out": 0})
    with pytest.raises(ValueError):
        acc.start({"n": 10, "T": 4})
    with pytest.raises(ValueError):
        acc.start({"n": 8})  # T missing


def test_fn_output_is_checked():
    def bad(k, ins, state, params):
        return {}  # output due but not returned

    cfg = AccelConfig((AccelPort("a", "in"), AccelPort("c", "out")), bad)
    cl, acc, _, _ = toy_setup(cfg, {"a": [[1]]}, {"c": 1})
    acc.start({"n": 1})
    with pytest.raises(SimulationError):
        cl.run(max_cycles=50)


# =============================================================================
# 3. Acceptance: elementwise stub cycle formula
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
@pytest.mark.parametrize("ii", [1, 2, 3])
@pytest.mark.parametrize("latency", [0, 1, 3])
@pytest.mark.parametrize("lanes", [1, 4])
def test_elementwise_ideal(lanes, latency, ii, skip):
    n = 12
    rng = np.random.default_rng(lanes * 100 + latency * 10 + ii)
    a, b = rand_beats(rng, n, lanes), rand_beats(rng, n, lanes)
    cfg = elementwise_stub(lanes=lanes, latency=latency, ii=ii)
    cl, acc, _, cons = toy_setup(cfg, {"a": a, "b": b}, {"out": n}, skip)
    acc.start({"n": n})
    total = cl.run(max_cycles=500)

    fires = [1 + k * ii for k in range(n)]
    assert acc.fire_cycles == fires
    assert acc.push_cycles == [f + latency for f in fires]
    assert cons["out"].pop_cycles == [f + latency + 1 for f in fires]
    assert acc.done_cycle == fires[-1] + latency + 1
    assert np.array_equal(cons["out"].values, a + b)
    # Classes: n firings, cycle 0 waits for the first beat, the rest idle
    # (II gaps with inputs present, the L-cycle drain, the last pop).
    assert acc.cycles["busy"] == n and acc.cycles["stall_out"] == 0
    assert acc.cycles["stall_in"] == 1
    assert_cycles_add_up(acc, total)
    assert not acc.busy


def test_elementwise_three_inputs_and_op():
    n, lanes = 5, 3
    rng = np.random.default_rng(7)
    ins = {c: rand_beats(rng, n, lanes) for c in "abc"}
    cfg = elementwise_stub(lanes=lanes, n_inputs=3, op=np.multiply)
    cl, acc, _, cons = toy_setup(cfg, ins, {"out": n})
    acc.start({"n": n})
    cl.run(max_cycles=200)
    assert np.array_equal(cons["out"].values, ins["a"] * ins["b"] * ins["c"])


# =============================================================================
# 4. Acceptance: reduce stub, unequal port rates
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
@pytest.mark.parametrize("lanes_out", [None, 1])
@pytest.mark.parametrize("ii", [1, 2])
@pytest.mark.parametrize("latency", [1, 2])
@pytest.mark.parametrize("T", [1, 4])
def test_reduce_ideal(T, latency, ii, lanes_out, skip):
    lanes, n_out = 2, 3
    n = T * n_out
    x = rand_beats(np.random.default_rng(T + 10 * latency + 100 * ii), n, lanes)
    cfg = reduce_stub(lanes=lanes, lanes_out=lanes_out, latency=latency, ii=ii)
    cl, acc, _, cons = toy_setup(cfg, {"in": x}, {"out": n_out}, skip)
    acc.start({"n": n, "T": T})
    total = cl.run(max_cycles=500)

    fires = [1 + k * ii for k in range(n)]
    last = [fires[(j + 1) * T - 1] for j in range(n_out)]  # firing that completes out j
    assert acc.fire_cycles == fires
    assert acc.push_cycles == [f + latency for f in last]
    assert acc.done_cycle == fires[-1] + latency + 1
    # Unequal rates: T beats in per beat out.
    assert acc.beats == {"in": n, "out": n_out}
    expect = x.reshape(n_out, T, lanes).sum(axis=1)
    if lanes_out == 1:
        expect = expect.sum(axis=1, keepdims=True)
    assert np.array_equal(cons["out"].values, expect)
    assert acc.cycles["busy"] == n
    assert_cycles_add_up(acc, total)


@pytest.mark.parametrize("skip", [True, False])
def test_input_rate_above_one(skip):
    """A port with rate T on the input side: one coefficient per T beats of x."""

    def fn(k, ins, state, params):
        if "c" in ins:
            state["c"] = ins["c"]
        return {"y": ins["x"] * state["c"]}

    T, groups = 3, 4
    cfg = AccelConfig(
        (AccelPort("x", "in", 2), AccelPort("c", "in", 2, "T"), AccelPort("y", "out", 2)),
        fn, latency=1)  # fmt: skip
    rng = np.random.default_rng(3)
    x, c = rand_beats(rng, T * groups, 2), rand_beats(rng, groups, 2)
    cl, acc, _, cons = toy_setup(cfg, {"x": x, "c": c}, {"y": T * groups}, skip)
    acc.start({"n": T * groups, "T": T})
    cl.run(max_cycles=500)
    assert np.array_equal(cons["y"].values, x * np.repeat(c, T, axis=0))
    assert acc.beats == {"x": T * groups, "c": groups, "y": T * groups}


# =============================================================================
# 5. Freeze on a full output reaches the inputs
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
def test_freeze_holds_pipeline_and_inputs(skip):
    """L = 2, output FIFO of depth 1, consumer only willing from cycle 30.

    Fire at 1 is pushed at 3 and fills the output FIFO (visible from 4).
    Fires at 2 and 3 sit in slots 1 and 0. In cycle 4 the head cannot be
    pushed: frozen. Nothing fires, the input FIFO (depth 2) fills and the
    producer stops. The consumer pops at 30, room is visible at 31, and the
    pipeline moves on: push at 31, fire at 31. After that a depth-1 output
    FIFO without pipe takes one beat every 2 cycles (as the MOD4 writer).
    """
    n, L, wake = 10, 2, 30
    rng = np.random.default_rng(1)
    a, b = rand_beats(rng, n, 2), rand_beats(rng, n, 2)
    cfg = elementwise_stub(lanes=2, latency=L)
    cl, acc, prods, cons = toy_setup(
        cfg, {"a": a, "b": b}, {"out": n}, skip, depth_out=1,
        willing_out={"out": Willing(after=wake)})  # fmt: skip
    acc.start({"n": n})
    total = cl.run(max_cycles=500)

    assert acc.fire_cycles == [1, 2, 3, *range(31, 45, 2)]
    assert acc.push_cycles == [3, *range(31, 49, 2)]
    # Frozen 4..30, then every other cycle while the output FIFO drains.
    assert acc.cycles["stall_out"] == 27 + 8
    # Back-pressure reached the producer: the input FIFO (2 beats) is full
    # from cycle 5, and the next push is the pipe push in the thaw cycle.
    assert prods["a"].push_cycles == [0, 1, 2, 3, 4, *range(31, 41, 2)]
    assert np.array_equal(cons["out"].values, a + b)
    assert_cycles_add_up(acc, total)


@pytest.mark.parametrize("skip", [True, False])
def test_freeze_l0(skip):
    """L = 0: a firing needs room for its output; while full, inputs are not popped."""
    n = 6
    a, b = np.arange(n).reshape(-1, 1), np.arange(n).reshape(-1, 1) * 10
    cl, acc, _, cons = toy_setup(
        elementwise_stub(), {"a": a, "b": b}, {"out": n}, skip, depth_out=1,
        willing_out={"out": Willing([10, 20], after=40)})  # fmt: skip
    acc.start({"n": n})
    total = cl.run(max_cycles=500)
    # Fire at 1 fills the FIFO; next fire only when a pop's room is visible.
    assert acc.fire_cycles[:3] == [1, 11, 21]
    assert np.array_equal(cons["out"].values, a + b)
    assert_cycles_add_up(acc, total)


@pytest.mark.parametrize("skip", [True, False])
def test_reduce_freeze_keeps_partial_state(skip):
    T, n_out = 4, 3
    x = np.arange(T * n_out * 2).reshape(-1, 2)
    cl, acc, _, cons = toy_setup(
        reduce_stub(lanes=2, latency=2, ii=2), {"in": x}, {"out": n_out}, skip,
        depth_out=1, willing_out={"out": Willing([25], after=60)})  # fmt: skip
    acc.start({"n": T * n_out, "T": T})
    total = cl.run(max_cycles=500)
    assert np.array_equal(cons["out"].values, x.reshape(n_out, T, 2).sum(axis=1))
    assert acc.cycles["stall_out"] > 0
    assert_cycles_add_up(acc, total)


def build_l1(skip, n_banks=16, rows=64, read_latency=1):
    cl = Cluster(skip_idle=skip)
    mem = L1Memory(cl, L1Config(n_banks=n_banks, rows=rows, read_latency=read_latency))
    xb = cl.add(Xbar("xbar", mem))
    return cl, mem, xb


def unit_regs(n_beats, lanes, base):
    return StreamerRegs(base, (n_beats,), (lanes * WORD,), (lanes,), (WORD,))


@pytest.mark.parametrize("skip", [True, False])
def test_freeze_reaches_reader_streamers(skip):
    """Readers stop issuing while the accelerator is frozen (credit runs out)."""
    lanes, n, wake = 2, 12, 40
    cl, mem, xb = build_l1(skip)
    rng = np.random.default_rng(5)
    a, b = rand_beats(rng, n, lanes), rand_beats(rng, n, lanes)
    mem.load(0, a.ravel())
    mem.load(256 * WORD, b.ravel())
    ra = cl.add(LogStreamer("ra", xb, StreamerConfig(n_ports=lanes, fifo_depth=2)))
    rb = cl.add(LogStreamer("rb", xb, StreamerConfig(n_ports=lanes, fifo_depth=2)))
    acc = cl.add(LogAccel("acc", cl, elementwise_stub(lanes=lanes, latency=1)))
    out = Fifo(cl, lanes, 1, name="out.fifo")
    acc.attach("a", ra.fifo)
    acc.attach("b", rb.fifo)
    acc.attach("out", out)
    con = cl.add(Consumer("con", out, n, Willing(after=wake)))
    ra.start(unit_regs(n, lanes, 0))
    rb.start(unit_regs(n, lanes, 256 * WORD))
    acc.start({"n": n})
    total = cl.run(max_cycles=1000)

    frozen_from = acc.push_cycles[0] + 1
    for s in (ra, rb):
        assert s.cycles["stall_fifo"] > 0
        idle_grants = [c for c in s.grants if frozen_from + 3 <= c <= wake]
        assert not idle_grants, s.grants
        assert_cycles_add_up(s, total)
    assert np.array_equal(con.values, a + b)
    assert_cycles_add_up(acc, total)


# =============================================================================
# 6. Start and done
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
def test_start_timing_and_restart(skip):
    """Start at 5: busy from 6, first firing at 6; restart after done."""
    n = 3
    a = np.arange(2 * n).reshape(-1, 1)
    cl = Cluster(skip_idle=skip)
    acc = cl.add(LogAccel("acc", cl, elementwise_stub(n_inputs=1, latency=1)))
    fin = Fifo(cl, 1, 8, pipe=True)
    fout = Fifo(cl, 1, 8)
    acc.attach("a", fin)
    acc.attach("out", fout)
    cl.add(ScriptedStarts("ctl", {5: (acc, {"n": n}), 20: (acc, {"n": n})}))
    probe = cl.add(Probe("probe", acc, range(30)))
    cl.add(Producer("prod", fin, a))
    con = cl.add(Consumer("con", fout, 2 * n))
    cl.run(max_cycles=200)

    assert acc.fire_cycles == [6, 7, 8, 21, 22, 23]
    assert acc.push_cycles == [7, 8, 9, 22, 23, 24]
    assert [c for c in range(30) if probe.busy[c]] == [*range(6, 10), *range(21, 25)]
    assert acc.done_cycle == 25
    assert np.array_equal(con.values.ravel(), a.ravel())


def test_start_while_busy_and_zero_firings():
    cl, acc, _, _ = toy_setup(elementwise_stub(), {"a": [[1]], "b": [[2]]}, {"out": 1})
    acc.start({"n": 1})
    with pytest.raises(SimulationError):
        acc.start({"n": 1})

    cl, acc, _, _ = toy_setup(elementwise_stub(), {"a": [], "b": []}, {"out": 0})
    acc.start({"n": 0})
    assert not acc.busy and acc.done_cycle == 0
    cl.run(max_cycles=10)
    assert acc.fire_cycles == []


# =============================================================================
# 7. vecadd end to end with streamers
# =============================================================================


def run_vecadd(n_elems, lanes, skip, n_banks=16, fifo_depth=2, read_latency=1,
               latency=0, ii=1, bases=None):  # fmt: skip
    """Two readers + elementwise stub + one writer. Returns (total, c, a + b, acc, streamers)."""
    cl, mem, xb = build_l1(skip, n_banks=n_banks, read_latency=read_latency)
    rng = np.random.default_rng(n_elems + lanes)
    a = rng.integers(-1000, 1000, n_elems)
    b = rng.integers(-1000, 1000, n_elems)
    nb = n_elems // lanes
    # Default layout: b and c shifted by 4 words so a, b and the writer
    # hit different banks in steady state (with 16 banks and 4 lanes).
    wa, wb, wc = bases or (0, n_elems + 4, 2 * n_elems + 4)
    mem.load(wa * WORD, a)
    mem.load(wb * WORD, b)
    cfg = StreamerConfig(n_ports=lanes, fifo_depth=fifo_depth)
    ra = cl.add(LogStreamer("ra", xb, cfg))
    rb = cl.add(LogStreamer("rb", xb, cfg))
    wr = cl.add(LogStreamer("wr", xb, StreamerConfig(write=True, n_ports=lanes,
                                                     fifo_depth=fifo_depth)))  # fmt: skip
    acc = cl.add(LogAccel("acc", cl, elementwise_stub(lanes=lanes, latency=latency, ii=ii)))
    acc.attach("a", ra.fifo)
    acc.attach("b", rb.fifo)
    acc.attach("out", wr.fifo)
    ra.start(unit_regs(nb, lanes, wa * WORD))
    rb.start(unit_regs(nb, lanes, wb * WORD))
    wr.start(unit_regs(nb, lanes, wc * WORD))
    acc.start({"n": nb})
    total = cl.run(max_cycles=5000)
    c = mem.dump(wc * WORD, n_elems)[:, 0]
    return total, c, a + b, acc, (ra, rb, wr)


@pytest.mark.parametrize("skip", [True, False])
def test_vecadd_end_to_end(skip):
    """Output equals NumPy; conflict-free layout fires every cycle.

    Streamers started before the run: first read request at 1, data at 2,
    visible in the FIFO at 3, so the accelerator fires at 3, 4, ... and the
    writer writes beat k at 4 + k.
    """
    n_elems, lanes = 32, 4
    nb = n_elems // lanes
    total, c, expect, acc, (ra, rb, wr) = run_vecadd(n_elems, lanes, skip)
    assert np.array_equal(c, expect)
    assert acc.fire_cycles == list(range(3, 3 + nb))
    assert wr.grants == list(range(4, 4 + nb))
    assert wr.done_cycle == 4 + nb and acc.done_cycle == 3 + nb
    for comp in (acc, ra, rb, wr):
        assert_cycles_add_up(comp, total)
        assert not comp.busy


def test_vecadd_with_bank_conflicts_still_correct():
    """a and b in the same banks: slower, same data."""
    n_elems, lanes = 32, 4
    for skip in (True, False):
        _, c, expect, acc, _ = run_vecadd(n_elems, lanes, skip, bases=(0, 32, 64))
        assert np.array_equal(c, expect)
        assert len(acc.fire_cycles) == n_elems // lanes


# =============================================================================
# 8. Random configs: skip on/off
# =============================================================================


def random_toy_case(seed):
    rng = random.Random(seed)
    kind = rng.choice(["ew", "red"])
    lanes = rng.randint(1, 3)
    latency, ii = rng.randint(0, 3), rng.randint(1, 3)
    if kind == "ew":
        cfg = elementwise_stub(lanes=lanes, n_inputs=rng.randint(1, 3), latency=latency, ii=ii)
        n = rng.randint(1, 12)
        params = {"n": n}
        n_out = n
    else:
        lo = rng.choice([lanes, 1])
        cfg = reduce_stub(lanes=lanes, lanes_out=lo, latency=latency, ii=ii)
        T, n_out = rng.randint(1, 4), rng.randint(1, 5)
        n = T * n_out
        params = {"n": n, "T": T}
    nrng = np.random.default_rng(seed)
    in_beats = {p.name: rand_beats(nrng, n, p.lanes) for p in cfg.inputs}
    win = {p.name: Willing([c for c in range(150) if rng.random() < 0.5], 150) for p in cfg.inputs}
    wout = {p.name: Willing([c for c in range(150) if rng.random() < 0.4], 150)
            for p in cfg.outputs}  # fmt: skip
    depths = (rng.randint(1, 3), rng.randint(1, 3))
    return cfg, params, in_beats, n_out, win, wout, depths


def expected_toy(cfg, params, in_beats):
    if cfg.kind == "elementwise":
        return sum(in_beats[p.name] for p in cfg.inputs)
    T = params["T"]
    x = in_beats["in"]
    s = x.reshape(-1, T, x.shape[1]).sum(axis=1)
    return s if cfg.port("out").lanes == x.shape[1] else s.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("seed", range(40))
def test_random_toy_skip_on_off(seed):
    cfg, params, in_beats, n_out, win, wout, (di, do) = random_toy_case(seed)
    results = []
    for skip in (True, False):
        cl, acc, prods, cons = toy_setup(cfg, in_beats, {"out": n_out}, skip, di, do, win, wout)
        acc.start(params)
        total = cl.run(max_cycles=5000)
        assert_cycles_add_up(acc, total)
        assert not acc.busy
        assert np.array_equal(cons["out"].values, expected_toy(cfg, params, in_beats))
        results.append((total, acc.fire_cycles, acc.push_cycles, dict(acc.cycles),
                        [p.push_cycles for p in prods.values()], cons["out"].beats,
                        acc.done_cycle))  # fmt: skip
    assert results[0] == results[1]


@pytest.mark.parametrize("seed", range(20))
def test_random_vecadd_skip_on_off(seed):
    rng = random.Random(1000 + seed)
    lanes = rng.randint(1, 4)
    nb = rng.randint(1, 10)
    kw = {
        "n_banks": rng.choice([4, 8, 16]),
        "fifo_depth": rng.randint(1, 3),
        "read_latency": rng.randint(0, 2),
        "latency": rng.randint(0, 3),
        "ii": rng.randint(1, 3),
    }
    n = nb * lanes
    offs = (0, n + rng.randint(0, 5), 2 * n + rng.randint(5, 10))
    results = []
    for skip in (True, False):
        total, c, expect, acc, ss = run_vecadd(n, lanes, skip, bases=offs, **kw)
        assert np.array_equal(c, expect)
        for comp in (acc, *ss):
            assert_cycles_add_up(comp, total)
        results.append((total, acc.fire_cycles, dict(acc.cycles),
                        [s.grants for s in ss], [dict(s.cycles) for s in ss]))  # fmt: skip
    assert results[0] == results[1]
