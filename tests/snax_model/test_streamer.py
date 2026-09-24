"""Tests for the SNAX-MODEL streamers (MOD4).

MOD4 acceptance (docs/STATUS.md):
  * address streams equal a NumPy enumeration for 1D, 2D and strided nests,
  * a full FIFO causes stalls, and
  * conflict-free throughput equals the port count per cycle.
Plus: skip on/off identical under random configs and consumer timing.

How the tests are built
-----------------------
There is no accelerator yet (MOD5). ``Consumer`` pops beats from a reader's
FIFO and ``Producer`` pushes beats into a writer's FIFO, both in COMPUTE,
the phase the accelerator will use. Each is "willing" in a given set of
cycles (every cycle by default) and acts when willing and the FIFO allows.

``LogStreamer`` is a Streamer that also logs grants as (cycle, port) and
the cycles in which it was ticked.

The L1 is loaded with each word's own index (address // 8), so the data a
reader returns is its address stream divided by 8.

All runs use ``check_hold=True`` on the xbar, so any streamer that drops or
changes a refused request fails the test.

Sections:
  1. helpers and toy components
  2. acceptance: address streams vs NumPy
  3. registers and FIFO rules
  4. reader and writer data through the xbar
  5. acceptance: conflict-free throughput
  6. acceptance: full FIFO stalls (and wake on pop)
  7. independent ports under conflicts
  8. start and done
  9. random configs: skip on/off
"""

import bisect
import random
from math import prod

import numpy as np
import pytest

from snax_forge.snax_model import (
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
    address_stream,
)

WORD = 8  # bytes per word with the default 64-bit banks
NB = 8  # banks in most tests


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
        """Earliest willing cycle > c, or None."""
        if self.every:
            return c + 1
        i = bisect.bisect_right(self.cycles, c)
        cands = [self.cycles[i]] if i < len(self.cycles) else []
        if self.after is not None:
            cands.append(max(c + 1, self.after))
        return min(cands) if cands else None


class Consumer(Component):
    """Pops ``total`` beats from ``fifo`` in COMPUTE, in willing cycles."""

    phases = (Phase.COMPUTE,)

    def __init__(self, name, fifo, total, willing=None):
        super().__init__(name)
        self.fifo, self.total = fifo, total
        self.willing = willing or Willing()
        fifo.popper = self
        self.beats = []  # (cycle, [first element of each lane])

    def tick(self, cycle, phase):
        if len(self.beats) < self.total and cycle in self.willing and self.fifo.can_pop():
            beat = self.fifo.pop(cycle)
            self.beats.append((cycle, [int(np.asarray(x).ravel()[0]) for x in beat]))

    def next_wake(self, cycle):
        if len(self.beats) >= self.total:
            return None
        return self.willing.next_after(cycle)

    @property
    def values(self):
        return np.array([v for _, v in self.beats], dtype=np.int64).reshape(-1, self.fifo.lanes)

    @property
    def pop_cycles(self):
        return [c for c, _ in self.beats]


class Producer(Component):
    """Pushes ``beats`` (one value per lane) into ``fifo`` in COMPUTE, in willing cycles."""

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
        if self.i >= len(self.beats):
            return None
        return self.willing.next_after(cycle)


class ScriptedStarts(Component):
    """Starts streamers in scripted cycles: {cycle: (streamer, regs)}."""

    phases = (Phase.CONTROL,)

    def __init__(self, name, script):
        super().__init__(name)
        self.script = script

    def tick(self, cycle, phase):
        if cycle in self.script:
            s, regs = self.script[cycle]
            s.start(regs, cycle)

    def next_wake(self, cycle):
        later = [c for c in self.script if c > cycle]
        return min(later) if later else None


class Probe(Component):
    """Records ``streamer.busy`` in every cycle of ``window``."""

    phases = (Phase.CONTROL,)

    def __init__(self, name, streamer, window):
        super().__init__(name)
        self.s, self.window = streamer, window
        self.busy = {}

    def tick(self, cycle, phase):
        if cycle in self.window:
            self.busy[cycle] = self.s.busy

    def next_wake(self, cycle):
        later = [c for c in self.window if c > cycle]
        return min(later) if later else None


class LogStreamer(Streamer):
    """Streamer that logs grants (cycle, port) and tick cycles."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fires = []
        self.ticks = []

    def tick(self, cycle, phase):
        if phase == Phase.REQUEST:
            self.ticks.append(cycle)
        super().tick(cycle, phase)

    def commit(self, cycle):
        self.fires += [(cycle, int(j)) for j in np.flatnonzero(self._fire)]
        super().commit(cycle)

    def grant_cycles(self, port=None):
        return sorted({c for c, j in self.fires if port is None or j == port})


def build(skip=True, n_banks=NB, rows=64, read_latency=1):
    """Cluster, L1 loaded with word indices, and an Xbar. Returns (cl, mem, xb)."""
    cl = Cluster(skip_idle=skip)
    mem = L1Memory(cl, L1Config(n_banks=n_banks, rows=rows, read_latency=read_latency))
    mem.load(0, np.arange(n_banks * rows))
    xb = cl.add(Xbar("xbar", mem))
    return cl, mem, xb


def reader(cl, xb, name, n_ports=1, **cfg):
    return cl.add(LogStreamer(name, xb, StreamerConfig(write=False, n_ports=n_ports, **cfg)))


def writer(cl, xb, name, n_ports=1, **cfg):
    return cl.add(LogStreamer(name, xb, StreamerConfig(write=True, n_ports=n_ports, **cfg)))


def unit_regs(n_beats, n_ports, base=0):
    """Contiguous stream: lanes on consecutive words, beats one after another."""
    return StreamerRegs(base, (n_beats,), (n_ports * WORD,), (n_ports,), (WORD,))


def assert_cycles_add_up(s, total):
    assert sum(s.cycles.values()) == total, s.cycles


# =============================================================================
# 2. Acceptance: address streams vs NumPy
# =============================================================================


def numpy_stream(regs):
    """Direct enumeration: loop 0 innermost/fastest, one row per beat."""
    tb, ts = regs.temporal_bounds, regs.temporal_strides
    sb, ss = regs.spatial_bounds, regs.spatial_strides
    n, lanes = prod(tb), prod(sb)
    if n == 0:
        return np.empty((0, lanes), dtype=np.int64)
    i = np.array(np.unravel_index(np.arange(n), tb, order="F"))  # [dims, n]
    j = np.array(np.unravel_index(np.arange(lanes), sb, order="F"))  # [dims, lanes]
    t_off = (i * np.array(ts)[:, None]).sum(0)
    s_off = (j * np.array(ss)[:, None]).sum(0)
    return regs.base + t_off[:, None] + s_off[None, :]


CASES = {
    "1d": StreamerRegs(64, (10,), (8,)),
    "1d_4lanes": StreamerRegs(0, (6,), (32,), (4,), (8,)),
    "2d": StreamerRegs(16, (3, 5), (8, 64), (2,), (24,)),
    "3d": StreamerRegs(0, (2, 3, 4), (8, 40, 200), (2,), (16,)),
    "strided": StreamerRegs(0, (7,), (24,), (3,), (40,)),
    "2d_spatial": StreamerRegs(8, (4, 2), (16, 128), (2, 3), (8, 256)),
    "negative": StreamerRegs(4096, (5, 3), (-8, -200), (2,), (-24,)),
    "zero_stride": StreamerRegs(0, (3, 4), (8, 0)),
    "bound_1": StreamerRegs(0, (1, 5, 1), (99, 8, 77), (1, 2), (55, 8)),
    "bound_0": StreamerRegs(0, (3, 0), (8, 8)),
    "repeat": StreamerRegs(8, (3, 4), (0, 32), (4,), (8,)),  # reader repeat (D69)
    "repeat_3d": StreamerRegs(0, (2, 3, 2), (0, 16, 0), (2,), (8,)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_address_stream_matches_numpy(name):
    regs = CASES[name]
    got = address_stream(regs)
    assert got.shape == (regs.n_beats, regs.n_lanes)
    assert np.array_equal(got, numpy_stream(regs))


def test_loop_order_by_hand():
    """Temporal loop 0 innermost, spatial dim 0 fastest in the lane index.

    base 100, temporal (2, 3) with strides (10, 100), spatial (2,) stride 1.
    """
    regs = StreamerRegs(100, (2, 3), (10, 100), (2,), (1,))
    assert address_stream(regs).tolist() == [
        [100, 101], [110, 111], [200, 201], [210, 211], [300, 301], [310, 311],
    ]  # fmt: skip
    regs = StreamerRegs(0, (1,), (0,), (2, 2), (1, 10))
    assert address_stream(regs).tolist() == [[0, 1, 10, 11]]


@pytest.mark.parametrize("seed", range(20))
def test_address_stream_random(seed):
    rng = random.Random(seed)
    dt, ds = rng.randint(1, 4), rng.randint(1, 3)
    regs = StreamerRegs(
        rng.randrange(0, 1 << 16, 8),
        [rng.randint(0, 5) for _ in range(dt)],
        [rng.randint(-8, 8) * 8 for _ in range(dt)],
        [rng.randint(1, 3) for _ in range(ds)],
        [rng.randint(-8, 8) * 8 for _ in range(ds)],
    )
    assert np.array_equal(address_stream(regs), numpy_stream(regs))


# =============================================================================
# 3. Registers and FIFO rules
# =============================================================================


def test_regs_validation():
    with pytest.raises(ValueError):
        StreamerRegs(0, (2, 3), (8,))  # unequal temporal lists
    with pytest.raises(ValueError):
        StreamerRegs(0, (2,), (8,), (2, 2), (8,))  # unequal spatial lists
    with pytest.raises(ValueError):
        StreamerRegs(0, (-1,), (8,))
    with pytest.raises(ValueError):
        StreamerRegs(0, (2,), (8,), (0,), (8,))
    with pytest.raises(ValueError):
        StreamerRegs(0, (), ())
    r = StreamerRegs(0, [2, 3], [8, 16])  # lists become tuples
    assert r.temporal_bounds == (2, 3) and r.n_beats == 6 and r.n_lanes == 1


def test_start_checks_against_hardware():
    cl, _, xb = build()
    rd = reader(cl, xb, "rd", n_ports=2, temporal_dims=2)
    with pytest.raises(ValueError):
        rd.start(StreamerRegs(0, (4,), (16,), (3,), (8,)))  # 3 lanes on 2 ports
    with pytest.raises(ValueError):
        rd.start(StreamerRegs(0, (2, 2, 2), (16, 32, 64), (2,), (8,)))  # 3 loops > 2
    wr = writer(cl, xb, "wr", n_ports=2, temporal_dims=2)
    wr.start(StreamerRegs(0, (4, 2), (0, 16), (2,), (8,)))  # fine for a writer
    rd.start(StreamerRegs(0, (4,), (16,), (2,), (8,)))  # fewer loops than counters: fine


class _NoSched:
    def touch(self, elem):
        pass


def test_fifo_push_visible_next_cycle():
    f = Fifo(_NoSched(), lanes=2, depth=2)
    f.push(0, [1, 2])
    assert not f.can_pop()  # flow = false
    f.commit()
    assert f.can_pop() and f.peek() == [1, 2]
    assert f.pop(1) == [1, 2]
    assert not f.can_pop()  # one pop per cycle
    f.commit()
    assert f.empty


@pytest.mark.parametrize("pipe", [True, False])
def test_fifo_pipe(pipe):
    """Full lane: a pop earlier in the cycle frees the slot only with pipe."""
    f = Fifo(_NoSched(), lanes=1, depth=1, pipe=pipe)
    f.push_lane(0, 0, "a")
    f.commit()
    assert not f.can_push_lane(0)
    f.pop_lane(1, 0)
    assert f.can_push_lane(0) == pipe
    if pipe:
        f.push_lane(1, 0, "b")
    f.commit()
    assert f.count(0) == (1 if pipe else 0)


def test_fifo_wide_pop_needs_every_lane():
    f = Fifo(_NoSched(), lanes=3, depth=2)
    f.push_lane(0, 0, 10)
    f.push_lane(0, 2, 12)
    f.commit()
    assert not f.can_pop()
    with pytest.raises(SimulationError):
        f.pop(1)
    f.push_lane(1, 1, 11)
    f.commit()
    assert f.pop(2) == [10, 11, 12]


# =============================================================================
# 4. Reader and writer data through the xbar
# =============================================================================


@pytest.mark.parametrize("latency", [0, 1, 3])
@pytest.mark.parametrize(
    "name", ["1d_4lanes", "2d", "3d", "strided", "2d_spatial", "repeat", "repeat_3d"]
)
@pytest.mark.parametrize("skip", [True, False])
def test_reader_data(name, latency, skip):
    """Every beat holds exactly the words at the NumPy-enumerated addresses."""
    regs = CASES[name]
    cl, _, xb = build(skip, read_latency=latency)
    rd = reader(cl, xb, "rd", regs.n_lanes, temporal_dims=3, fifo_depth=2)
    con = cl.add(Consumer("acc", rd.fifo, regs.n_beats))
    rd.start(regs)
    total = cl.run(max_cycles=2000)
    assert np.array_equal(con.values, numpy_stream(regs) // WORD)
    assert not rd.busy and rd.fifo.empty
    assert_cycles_add_up(rd, total)


@pytest.mark.parametrize("skip", [True, False])
def test_writer_data(skip):
    """2D writer: each word lands at its enumerated address; nothing else changes."""
    regs = StreamerRegs(8, (3, 4), (16, 128), (2,), (48,))
    cl, mem, xb = build(skip)
    before = mem.data.copy()
    wr = writer(cl, xb, "wr", 2, temporal_dims=2)
    values = np.arange(regs.n_beats * 2).reshape(-1, 2) + 5000
    cl.add(Producer("acc", wr.fifo, values))
    wr.start(regs)
    total = cl.run(max_cycles=2000)
    addrs = numpy_stream(regs)
    assert len(set(addrs.ravel())) == addrs.size  # test layout: no duplicates
    expect = before.copy()
    for a, v in zip(addrs.ravel(), values.ravel()):
        b, r = mem.locate(int(a))
        expect[b, r] = v
    assert np.array_equal(mem.data, expect)
    assert_cycles_add_up(wr, total)


# =============================================================================
# 5. Acceptance: conflict-free throughput
# =============================================================================


@pytest.mark.parametrize("n_ports", [1, 2, 4, 8])
@pytest.mark.parametrize("skip", [True, False])
def test_reader_throughput(n_ports, skip):
    """Unit-stride stream on 8 banks: every port is granted every cycle.

    Started before the run (as if in cycle -1):
        cycle 0      AGU pushes beat 0's addresses
        cycle 1+k    beat k requested and granted on all ports
        cycle 2+k    read data arrives, pushed into the FIFO
        cycle 3+k    beat k visible and popped
    so N beats take N + 3 cycles, with N busy cycles.
    """
    n = 20
    cl, _, xb = build(skip)
    rd = reader(cl, xb, "rd", n_ports, fifo_depth=2)
    con = cl.add(Consumer("acc", rd.fifo, n))
    rd.start(unit_regs(n, n_ports))
    total = cl.run()
    assert total == n + 3
    assert rd.grant_cycles() == list(range(1, n + 1))
    assert all(int(xb.port_grants[p]) == n for p in rd.ports)
    assert xb.port_stalls.sum() == 0
    assert con.pop_cycles == list(range(3, n + 3))
    assert rd.cycles == {"busy": n, "stall_xbar": 0, "stall_fifo": 0, "idle": 3}


@pytest.mark.parametrize("skip", [True, False])
def test_reader_depth_1_halves_throughput(skip):
    """Depth 1: the next read waits for the pop, which frees credit in the same cycle.

    Beat k is requested in 1+2k and popped in 3+2k, so N beats take 2N + 2.
    """
    n = 10
    cl, _, xb = build(skip)
    rd = reader(cl, xb, "rd", 4, fifo_depth=1)
    con = cl.add(Consumer("acc", rd.fifo, n))
    rd.start(unit_regs(n, 4))
    assert cl.run() == 2 * n + 2
    assert rd.grant_cycles() == [1 + 2 * k for k in range(n)]
    assert con.pop_cycles == [3 + 2 * k for k in range(n)]


@pytest.mark.parametrize("n_ports", [1, 4])
@pytest.mark.parametrize("skip", [True, False])
def test_writer_throughput(n_ports, skip):
    """Producer pushes from cycle 0; beat k is written in 1+k: N + 1 cycles."""
    n = 20
    cl, mem, xb = build(skip)
    wr = writer(cl, xb, "wr", n_ports, fifo_depth=2)
    values = np.arange(n * n_ports).reshape(n, n_ports) + 7000
    cl.add(Producer("acc", wr.fifo, values))
    wr.start(unit_regs(n, n_ports, base=64 * WORD))
    total = cl.run()
    assert total == n + 1
    assert wr.grant_cycles() == list(range(1, n + 1))
    assert wr.done_cycle == n + 1
    assert np.array_equal(mem.dump(64 * WORD, n * n_ports).ravel(), values.ravel())
    assert wr.cycles == {"busy": n, "stall_xbar": 0, "stall_fifo": 0, "idle": 1}


@pytest.mark.parametrize("skip", [True, False])
def test_writer_depth_1_no_pipe(skip):
    """Writer FIFO has no pipe: a pop does not free the slot for that cycle's push.

    Producer pushes in 0, 2, 4, ...; writes happen in 1, 3, 5, ...
    """
    n = 6
    cl, _, xb = build(skip)
    wr = writer(cl, xb, "wr", 2, fifo_depth=1)
    prod_ = cl.add(Producer("acc", wr.fifo, [[k, k] for k in range(n)]))
    wr.start(unit_regs(n, 2, base=64 * WORD))
    assert cl.run() == 2 * n
    assert prod_.push_cycles == [2 * k for k in range(n)]
    assert wr.grant_cycles() == [1 + 2 * k for k in range(n)]


# =============================================================================
# 6. Acceptance: full FIFO stalls (and wake on pop)
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
def test_full_fifo_stalls_reader(skip):
    """Depth 3, consumer asleep until cycle 40.

        cycles 1..3    beats 0..2 read; credit is now used up
        cycles 4..39   stall_fifo; AGU still fills the address queues until 9
        cycle 40       consumer pops beat 0 and, in the same cycle, beat 3 is
                       requested (the pop frees credit combinationally)
        cycles 40..46  beats 3..9 requested
        cycles 40..49  beats 0..9 popped
    With skipping on, the reader must sleep in 10..39 and still be awake in 40:
    it wakes with its consumer.
    """
    n, d = 10, 3
    cl, _, xb = build(skip)
    rd = reader(cl, xb, "rd", 4, fifo_depth=d)
    con = cl.add(Consumer("acc", rd.fifo, n, Willing(after=40)))
    rd.start(unit_regs(n, 4))
    total = cl.run()
    assert total == 50
    assert rd.grant_cycles() == [1, 2, 3] + list(range(40, 47))
    assert con.pop_cycles == list(range(40, 50))
    assert np.array_equal(con.values, address_stream(unit_regs(n, 4)) // WORD)
    assert rd.cycles == {"busy": 10, "stall_xbar": 0, "stall_fifo": 36, "idle": 4}
    if skip:
        assert not any(10 <= c < 40 for c in rd.ticks)
        assert 40 in rd.ticks


@pytest.mark.parametrize("skip", [True, False])
def test_empty_fifo_stalls_writer(skip):
    """Producer starts in cycle 20: writer has addresses from cycle 1 but no data.

    cycle 0 idle, 1..20 stall_fifo, beat k written in 21+k.
    """
    n = 6
    cl, _, xb = build(skip)
    wr = writer(cl, xb, "wr", 2, fifo_depth=2)
    cl.add(Producer("acc", wr.fifo, [[k, k] for k in range(n)], Willing(after=20)))
    wr.start(unit_regs(n, 2, base=64 * WORD))
    assert cl.run() == 21 + n
    assert wr.grant_cycles() == list(range(21, 21 + n))
    assert wr.cycles == {"busy": n, "stall_xbar": 0, "stall_fifo": 20, "idle": 1}
    if skip:
        assert not any(n <= c < 20 for c in wr.ticks)


@pytest.mark.parametrize("skip", [True, False])
def test_slow_consumer_sets_the_rate(skip):
    """Consumer pops every 3rd cycle: after the FIFO fills, one read per 3 cycles."""
    n = 12
    cl, _, xb = build(skip)
    rd = reader(cl, xb, "rd", 2, fifo_depth=2)
    con = cl.add(Consumer("acc", rd.fifo, n, Willing(range(0, 200, 3))))
    rd.start(unit_regs(n, 2))
    total = cl.run()
    assert np.diff(con.pop_cycles).tolist() == [3] * (n - 1)
    assert rd.cycles["stall_fifo"] > 0
    assert_cycles_add_up(rd, total)


# =============================================================================
# 7. Independent ports under conflicts
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
def test_ports_advance_independently(skip):
    """Reader A: port 0 always on bank 0, port 1 always on bank 1.

    Reader B keeps bank 0 busy, so A's port 0 wins only every other cycle.
    A's port 1 runs ahead, limited by its credit (fifo_depth) since beats
    only leave the FIFO when port 0's data is there too.
    """
    n, d = 16, 4
    cl, _, xb = build(skip)
    a = reader(cl, xb, "a", 2, fifo_depth=d)
    b = reader(cl, xb, "b", 1, fifo_depth=d)
    ca = cl.add(Consumer("acc_a", a.fifo, n))
    cl.add(Consumer("acc_b", b.fifo, n))
    a.start(StreamerRegs(0, (n,), (NB * WORD,), (2,), (WORD,)))  # banks 0 and 1
    b.start(StreamerRegs(NB * WORD * 32, (n,), (NB * WORD,)))  # bank 0
    total = cl.run()

    g0, g1 = a.grant_cycles(0), a.grant_cycles(1)
    assert xb.port_stalls[a.ports[1]] == 0 and xb.port_stalls[a.ports[0]] > 0
    skew = [sum(c <= t for c in g1) - sum(c <= t for c in g0) for t in range(total)]
    assert 1 < max(skew) <= d
    assert g1[-1] < g0[-1]
    # Once port 1 is credit-blocked or done, port 0's refused cycles are xbar stalls.
    assert a.cycles["stall_xbar"] > 0 and b.cycles["stall_xbar"] > 0
    assert np.array_equal(ca.values, address_stream(a.regs) // WORD)
    assert_cycles_add_up(a, total)
    assert_cycles_add_up(b, total)


# =============================================================================
# 8. Start and done
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
def test_start_timing_and_restart(skip):
    """start in cycle 5: busy from 6, AGU pushes in 6, first grant in 7.

    Task 1 (4 beats) is granted in 7..10, so busy drops in 11. Task 2 starts
    in 30 and is granted in 32..34.
    """
    cl, _, xb = build(skip)
    rd = reader(cl, xb, "rd", 2, fifo_depth=2)
    t1, t2 = unit_regs(4, 2), unit_regs(3, 2, base=128)
    cl.add(ScriptedStarts("ctl", {5: (rd, t1), 30: (rd, t2)}))
    probe = cl.add(Probe("probe", rd, [4, 5, 6, 10, 11, 12]))
    con = cl.add(Consumer("acc", rd.fifo, 7))
    total = cl.run()
    assert probe.busy == {4: False, 5: False, 6: True, 10: True, 11: False, 12: False}
    assert rd.grant_cycles() == [7, 8, 9, 10, 32, 33, 34]
    assert rd.done_cycle == 35
    assert con.values[:, 0].tolist() == [0, 2, 4, 6, 16, 18, 20]
    assert_cycles_add_up(rd, total)


def test_start_while_busy_raises():
    cl, _, xb = build()
    rd = reader(cl, xb, "rd", 1)
    rd.start(StreamerRegs(0, (4,), (8,)))
    with pytest.raises(SimulationError):
        rd.start(StreamerRegs(0, (4,), (8,)))
    cl.add(Consumer("acc", rd.fifo, 4))
    cl.add(ScriptedStarts("ctl", {2: (rd, StreamerRegs(0, (4,), (8,)))}))
    with pytest.raises(SimulationError):
        cl.run()


def test_zero_bound_does_nothing():
    """A temporal bound of 0: the RTL AGU ignores start; never busy, no requests."""
    cl, _, xb = build()
    rd = reader(cl, xb, "rd", 1, temporal_dims=2)
    rd.start(StreamerRegs(0, (4, 0), (8, 8)))
    assert not rd.busy and rd.done_cycle == 0
    assert cl.run() == 0
    assert rd.grant_cycles() == []


# =============================================================================
# 9. Random configs: skip on/off
# =============================================================================


def random_regs(rng, n_ports, write, lo, hi):
    """Regs with non-negative strides whose addresses stay in [lo, hi) bytes.

    A reader's loop-0 stride may be 0, which exercises the repeat (D69).
    """
    sb = (2, 2) if n_ports == 4 and rng.random() < 0.5 else (n_ports,)
    for _ in range(1000):
        dt = rng.randint(1, 3)
        tb = [rng.choice([0, 1, 2, 3, 4, 5]) if rng.random() < 0.1 else rng.randint(1, 5)
              for _ in range(dt)]  # fmt: skip
        ts = [rng.randint(0, 6) * WORD for _ in range(dt)]
        ss = [rng.randint(0, 3) * WORD for _ in sb]
        span = sum((b - 1) * s for b, s in zip(tb, ts) if b) + sum(
            (b - 1) * s for b, s in zip(sb, ss)
        )
        if span < hi - lo:
            base = rng.randrange(lo, hi - span, WORD)
            return StreamerRegs(base, tb, ts, sb, ss)
    raise AssertionError("no regs found")


def random_case(seed):
    """Two readers and a writer on 4 banks with random depths and timing.

    Readers read the lower half of L1, the writer writes the upper half, so
    read data never depends on write timing. Consumers and producers are
    willing on a random ~half of the first 300 cycles, then always.
    """
    rng = random.Random(seed)
    specs = []
    for kind in ("r", "r", "w"):
        n_ports = rng.choice([1, 2, 3, 4])
        cfg = {
            "n_ports": n_ports,
            "temporal_dims": 3,
            "fifo_depth": rng.randint(1, 4),
            "addr_depth": rng.randint(1, 4),
            "prio": rng.choice([0, 0, 1]),
        }
        half = 4 * 64 * WORD // 2
        lo, hi = (0, half) if kind == "r" else (half, 2 * half)
        regs = random_regs(rng, n_ports, kind == "w", lo, hi)
        willing = [c for c in range(300) if rng.random() < 0.5]
        specs.append((kind, cfg, regs, willing))
    latency = rng.choice([0, 1, 2])
    return specs, latency


def run_case(specs, latency, skip):
    cl, mem, xb = build(skip, n_banks=4, read_latency=latency)
    ss, toys = [], []
    for i, (kind, cfg, regs, willing) in enumerate(specs):
        if kind == "r":
            s = reader(cl, xb, f"s{i}", **cfg)
            toys.append(cl.add(Consumer(f"t{i}", s.fifo, regs.n_beats, Willing(willing, 300))))
        else:
            s = writer(cl, xb, f"s{i}", **cfg)
            vals = np.arange(regs.n_beats * regs.n_lanes).reshape(-1, regs.n_lanes) + 10000
            toys.append(cl.add(Producer(f"t{i}", s.fifo, vals, Willing(willing, 300))))
        s.start(regs)
        ss.append(s)
    total = cl.run(max_cycles=20000)
    return total, mem, xb, ss, toys


@pytest.mark.parametrize("seed", range(30))
def test_random_skip_on_off(seed):
    specs, latency = random_case(seed)
    results = []
    for skip in (True, False):
        total, mem, xb, ss, toys = run_case(specs, latency, skip)
        results.append(
            (
                total,
                [s.fires for s in ss],
                [dict(s.cycles) for s in ss],
                [t.beats if isinstance(t, Consumer) else t.push_cycles for t in toys],
                mem.data.copy(),
                xb.port_grants.copy(),
                xb.port_stalls.copy(),
                xb.bank_conflicts.copy(),
            )
        )
        for s in ss:
            assert_cycles_add_up(s, total)
            assert not s.busy

        # Data: readers return their NumPy stream; words the writer wrote
        # exactly once hold the written value.
        for (kind, _, regs, _), s, t in zip(specs, ss, toys):
            if kind == "r":
                assert np.array_equal(t.values, numpy_stream(regs) // WORD)
            else:
                addrs = numpy_stream(regs).ravel()
                vals = np.arange(addrs.size) + 10000
                uniq, cnt = np.unique(addrs, return_counts=True)
                once = set(uniq[cnt == 1].tolist())
                for a, v in zip(addrs, vals):
                    if int(a) in once:
                        assert mem.peek(int(a))[0] == v

    a, b = results
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]
    assert all(np.array_equal(x, y) for x, y in zip(a[4:], b[4:]))


# =============================================================================
# 10. Reader repeat on temporal stride 0 (D69)
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
def test_reader_repeat_reads_each_group_once(skip):
    """tstride[0] = 0, tbound[0] = 3: 4 reads, 12 beats handed out back to back.

    Credit comes back only when a group's last hand-out pops the FIFO, so
    with depth 2 the first two groups are read in cycles 1 and 2, and each
    later group in the cycle the group two before it is popped (5, 8).
    """
    regs = StreamerRegs(0, (3, 4), (0, 32), (4,), (8,))
    cl, _, xb = build(skip)
    rd = reader(cl, xb, "rd", 4, temporal_dims=2, fifo_depth=2)
    con = cl.add(Consumer("acc", rd.fifo, regs.n_beats))
    rd.start(regs)
    total = cl.run()
    assert total == regs.n_beats + 3
    assert con.pop_cycles == list(range(3, regs.n_beats + 3))
    assert np.array_equal(con.values, numpy_stream(regs) // WORD)
    assert rd.grant_cycles() == [1, 2, 5, 8]
    assert all(int(xb.port_grants[p]) == 4 for p in rd.ports)
    assert rd.fifo.popped.tolist() == [4] * 4 and rd.fifo.empty
    assert rd.cycles["busy"] == 4
    assert_cycles_add_up(rd, total)


def test_writer_stride_0_does_not_repeat():
    """Only readers repeat: a writer with tstride[0] = 0 writes every beat."""
    cl, _, xb = build()
    wr = writer(cl, xb, "wr", 1, temporal_dims=2)
    cl.add(Producer("acc", wr.fifo, [[1], [2], [3], [4]]))
    wr.start(StreamerRegs(0, (2, 2), (0, 8), (1,), (0,)))
    cl.run()
    assert len(wr.fires) == 4 and wr.fifo.repeat == 1


def test_fifo_repeat_hands_out_then_pops():
    f = Fifo(_NoSched(), lanes=1, depth=2)
    f.push(0, [7])
    f.commit()
    f.push(1, [8])
    f.commit()
    f.restart_repeat(3)
    for t in (2, 3, 4):
        assert f.pop(t) == [7]
        assert not f.can_pop()  # one hand-out per cycle
        assert f.popped_now(t, 0) == (t == 4)  # only the last one pops
        f.commit()
        assert f.count(0) == (2 if t < 4 else 1)
    assert f.popped.tolist() == [1]


def test_fifo_restart_resets_the_count():
    """A restart in the cycle of a hand-out wins over it, as the RTL counter's reset."""
    f = Fifo(_NoSched(), lanes=1, depth=2)
    f.push(0, [7])
    f.commit()
    f.restart_repeat(3)
    assert f.pop(1) == [7]
    f.restart_repeat(2)  # the start lands in the same cycle
    f.commit()
    assert f.pop(2) == [7] and not f.popped_now(2, 0)  # count restarted: 1 of 2
    f.commit()
    assert f.pop(3) == [7] and f.popped_now(3, 0)
    f.commit()
    assert f.empty
    with pytest.raises(ValueError):
        f.restart_repeat(0)
