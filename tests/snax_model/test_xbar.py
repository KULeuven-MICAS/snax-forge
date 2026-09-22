"""Tests for the SNAX-MODEL TCDM interconnect (MOD3, MOD6 multi-width ports).

MOD3 acceptance (docs/STATUS.md):
  * grant sequences for 2-3 masters on one bank match hand-worked tables, and
  * distinct banks proceed in parallel.
MOD6 (D33): ports wider than a bank; wider wins per cycle and per group.
Plus: skip on/off identical under random traffic, checked against a
reference that follows the Chisel RoundRobinArbiter line by line.

How the tests are built
-----------------------
``ToyMaster`` owns one port. It works through a script of (earliest cycle,
BankReq): it drives the current entry from its earliest cycle on, every
cycle, until it is granted (valid/ready hold), then moves to the next one.
It logs grants as (cycle, entry index) and read data as (cycle, entry index,
value).

``FixedMaster`` drives a request in exactly the scripted cycles, granted or
not. With ``check_hold=False`` it is the only way to make a master drop a
refused request, the RTL case the lock rule has to handle.

``Blocker`` owns one wide port and reads its bank group in scripted
cycles. It always wins (it is the widest port on its group), so the narrow
ports of that group see "not ready", as with the DMA in SNAX
(mem_wide_narrow_mux). It replaces the MOD3 ``block_bank`` hook (D33).

In the hand-worked tables, ``prev`` is the per-bank pointer at the start of
the cycle and N-1 is its reset value ("start from port 0").

Sections:
  1. helpers and toy components
  2. acceptance: grant tables on one bank
  3. pointer, priority and lock rules
  4. acceptance: distinct banks in parallel
  5. data routing and read latency
  6. hold rule and misuse
  7. random traffic: skip on/off and a reference arbiter
  8. multi-width ports (MOD6)
"""

import random

import numpy as np
import pytest

from snax_forge.snax_model import (
    BankReq,
    Cluster,
    Component,
    HoldViolation,
    L1Config,
    L1Memory,
    Phase,
    SimulationError,
    Xbar,
)

WORD = 8  # bytes per word with the default 64-bit banks
NB = 4  # banks in most tests: small, so hand-worked tables stay readable


def addr(bank, row=0, n_banks=NB):
    """Byte address of (bank, row) under the word-interleaved map."""
    return (row * n_banks + bank) * WORD


# =============================================================================
# 1. Helpers and toy components
# =============================================================================


class ToyMaster(Component):
    """One port; issues its script in order, holding each request until granted."""

    phases = (Phase.REQUEST, Phase.RESPONSE)

    def __init__(self, name, xb, script, prio=0, width_bits=None):
        super().__init__(name)
        self.xb = xb
        self.port = xb.add_port(self, width_bits=width_bits)
        self.script = script
        for k, (_, req) in enumerate(script):
            req.tag = k
        self.prio = prio
        self.i = 0  # committed: next script entry to issue
        self._took = False  # wire: granted this cycle
        self.grants = []  # (cycle, k)
        self.rdata = []  # (cycle, k, first element)

    def tick(self, cycle, phase):
        if phase == Phase.REQUEST:
            if self.i < len(self.script) and cycle >= self.script[self.i][0]:
                self.xb.request(cycle, self.port, self.script[self.i][1], self.prio)
        elif phase == Phase.RESPONSE:
            if self.xb.granted(cycle, self.port):
                self.grants.append((cycle, self.i))
                self._took = True
            r = self.xb.rdata(cycle, self.port)
            if r is not None:
                self.rdata.append((cycle, r.tag, r.data.ravel()[0].item()))
                self.last_rdata = r

    def commit(self, cycle):
        if self._took:
            self.i += 1
        self._took = False

    def next_wake(self, cycle):
        # Next request (now+1 while holding one) or the next read data.
        cands = []
        if self.i < len(self.script):
            cands.append(max(cycle + 1, self.script[self.i][0]))
        r = self.xb.next_rdata(cycle, self.port)
        if r is not None:
            cands.append(r)
        return min(cands) if cands else None

    @property
    def grant_cycles(self):
        return [c for c, _ in self.grants]


class FixedMaster(Component):
    """Drives a read of ``script[cycle]`` (a bank) in exactly those cycles."""

    phases = (Phase.REQUEST, Phase.RESPONSE)

    def __init__(self, name, xb, script):
        super().__init__(name)
        self.xb = xb
        self.port = xb.add_port(self)
        self.script = script  # cycle -> bank, or cycle -> BankReq
        self.grants = []

    def _req(self, cycle):
        v = self.script[cycle]
        return v if isinstance(v, BankReq) else BankReq(addr(v))

    def tick(self, cycle, phase):
        if phase == Phase.REQUEST and cycle in self.script:
            self.xb.request(cycle, self.port, self._req(cycle))
        elif phase == Phase.RESPONSE and self.xb.granted(cycle, self.port):
            self.grants.append(cycle)

    def next_wake(self, cycle):
        later = [c for c in self.script if c > cycle]
        return min(later) if later else None


class Blocker(Component):
    """Wide port that reads the group of ``bank`` in scripted cycles: {cycle, ...}."""

    phases = (Phase.REQUEST,)

    def __init__(self, name, xb, cycles, width_bits=None, bank=0):
        super().__init__(name)
        self.xb = xb
        self.cycles = set(cycles)
        cfg = xb.mem.cfg
        width_bits = width_bits or cfg.width_bits * cfg.n_banks  # default: every bank
        self.port = xb.add_port(self, name, width_bits=width_bits)
        g = cfg.group_banks(width_bits)
        self.addr = (bank // g) * g * cfg.word_bytes  # row 0 of the group

    def tick(self, cycle, phase):
        if cycle in self.cycles:
            self.xb.request(cycle, self.port, BankReq(self.addr))

    def next_wake(self, cycle):
        later = [c for c in self.cycles if c > cycle]
        return min(later) if later else None


def build(skip=True, check_hold=True, **cfg):
    """Cluster with an L1 of NB banks and an Xbar. Returns (cluster, mem, xbar)."""
    cfg.setdefault("n_banks", NB)
    cfg.setdefault("rows", 64)
    cl = Cluster(skip_idle=skip)
    mem = L1Memory(cl, L1Config(**cfg))
    xb = cl.add(Xbar("xbar", mem, check_hold=check_hold))
    return cl, mem, xb


def reads(bank, n, start=0, row0=0):
    """Script of ``n`` reads to ``bank`` (rows row0, row0+1, ...), all from ``start``."""
    return [(start, BankReq(addr(bank, row0 + k))) for k in range(n)]


def masters(cl, xb, scripts, prios=None):
    prios = prios or [0] * len(scripts)
    return [cl.add(ToyMaster(f"m{i}", xb, s, p)) for i, (s, p) in enumerate(zip(scripts, prios))]


# =============================================================================
# 2. Acceptance: grant tables on one bank
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
def test_two_masters_one_bank(skip):
    """Two masters, three reads each, all to bank 0 from cycle 0.

        cycle  requesters  prev  selected
          0      0 1        1     0        (nothing > 1: wrap to 0)
          1      0 1        0     1
          2      0 1        1     0
          3      0 1        0     1
          4      0 1        1     0        (m0 done)
          5        1        0     1        (m1 done)

    Read data returns one cycle after each grant, to the right master.
    """
    cl, mem, xb = build(skip)
    m0, m1 = masters(cl, xb, [reads(0, 3, row0=0), reads(0, 3, row0=10)])
    for r in (0, 1, 2, 10, 11, 12):
        mem.poke(addr(0, r), 100 + r)
    total = cl.run()

    assert m0.grants == [(0, 0), (2, 1), (4, 2)]
    assert m1.grants == [(1, 0), (3, 1), (5, 2)]
    assert m0.rdata == [(1, 0, 100), (3, 1, 101), (5, 2, 102)]
    assert m1.rdata == [(2, 0, 110), (4, 1, 111), (6, 2, 112)]
    assert total == 7  # last read data in cycle 6
    assert xb.bank_conflicts[0] == 5  # cycles 0..4
    assert xb.bank_grants[0] == 6
    assert list(xb.port_stalls) == [2, 3]  # m0 waits in 1, 3; m1 in 0, 2, 4
    assert xb.bank_stalls[0] == 5


@pytest.mark.parametrize("skip", [True, False])
def test_three_masters_rotate(skip):
    """Three masters, two reads each to bank 1: strict rotation 0 1 2 0 1 2."""
    cl, _, xb = build(skip)
    ms = masters(cl, xb, [reads(1, 2), reads(1, 2), reads(1, 2)])
    cl.run()
    assert [m.grant_cycles for m in ms] == [[0, 3], [1, 4], [2, 5]]


@pytest.mark.parametrize("skip", [True, False])
def test_three_masters_late_joiner(skip):
    """m0: 3 reads, m2: 2 reads from cycle 0; m1: 1 read from cycle 1. Bank 0.

        cycle  requesters  prev  selected
          0      0   2      2     0        (reset pointer: lowest wins)
          1      0 1 2      0     1        (m1 joins, next after 0)
          2      0   2      1     2
          3      0   2      2     0
          4      0   2      0     2        (m2 done)
          5      0          2     0        (m0 done)

    The winner is always skipped over next cycle while others wait.
    """
    cl, _, xb = build(skip)
    ms = masters(cl, xb, [reads(0, 3), reads(0, 1, start=1), reads(0, 2)])
    cl.run()
    assert [m.grant_cycles for m in ms] == [[0, 3, 5], [1], [2, 4]]


# =============================================================================
# 3. Pointer, priority and lock rules
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
def test_pointer_follows_last_selection(skip):
    """m0 alone in cycle 0 moves the pointer to 0, so m1 wins cycle 1.

    cycle  requesters  prev  selected
      0      0          1     0
      1      0 1        0     1
      2      0          1     0
    """
    cl, _, xb = build(skip)
    m0, m1 = masters(cl, xb, [reads(0, 2), reads(0, 1, start=1)])
    cl.run()
    assert (m0.grant_cycles, m1.grant_cycles) == ([0, 2], [1])


@pytest.mark.parametrize("gap", [2, 12])
@pytest.mark.parametrize("skip", [True, False])
def test_idle_cycle_resets_pointer(gap, skip):
    """Same as above, but bank 0 is idle between: the pointer resets to N-1.

        cycle   requesters  prev  selected
          0       0          1     0
          1..       (idle: selection N-1 = 1)
          gap     0 1        1     0        (without reset m1 would win)
          gap+1     1        0     1

    The first access is a write, so no read data wakes anyone in cycle 1:
    with skipping on, the idle cycles really are skipped and the reset comes
    from on_gap; with skipping off it comes from the idle ticks.
    """
    cl, _, xb = build(skip)
    first = BankReq(addr(0), write=True, wdata=1)
    m0, m1 = masters(cl, xb, [[(0, first), (gap, BankReq(addr(0, 1)))], reads(0, 1, start=gap)])
    cl.run()
    assert (m0.grant_cycles, m1.grant_cycles) == ([0, gap], [gap + 1])


@pytest.mark.parametrize("skip", [True, False])
def test_pointer_is_per_bank(skip):
    """Bank 0 idles in cycle 1 while bank 1 is busy: bank 0 still resets.

    m2 on bank 1 keeps the xbar awake in cycle 1 even with skipping on, so
    this checks the reset comes from bank 0 having no request, not from
    the xbar sleeping.
    """
    cl, _, xb = build(skip)
    m0, m1, m2 = masters(
        cl,
        xb,
        [
            [(0, BankReq(addr(0))), (2, BankReq(addr(0, 1)))],
            reads(0, 1, start=2),
            reads(1, 1, start=1),
        ],
    )
    cl.run()
    assert (m0.grant_cycles, m1.grant_cycles, m2.grant_cycles) == ([0, 2], [3], [1])


@pytest.mark.parametrize("skip", [True, False])
def test_priority_wins(skip):
    """m1 has priority 1: it takes both its grants before m0 gets any."""
    cl, _, xb = build(skip)
    m0, m1 = masters(cl, xb, [reads(0, 2), reads(0, 2)], prios=[0, 1])
    cl.run()
    assert (m0.grant_cycles, m1.grant_cycles) == ([2, 3], [0, 1])


@pytest.mark.parametrize("skip", [True, False])
def test_lock_keeps_refused_winner(skip):
    """Bank 0 not ready in cycles 0 and 1 (a wide grant): the selection is locked.

        cycle  requesters  prev  lock  selected  ready
          0      0 1        1     0     0         no   -> lock
          1      0 1        0     1     0         no   -> lock (not 1)
          2      0 1        0     1     0         yes
          3        1        0     0     1         yes

    Without the lock, round-robin from prev = 0 would give cycle 1 to m1.
    """
    cl, _, xb = build(skip)
    m0, m1 = masters(cl, xb, [reads(0, 1), reads(0, 1)])
    cl.add(Blocker("dma", xb, {0, 1}))
    cl.run()
    assert (m0.grant_cycles, m1.grant_cycles) == ([2], [3])
    assert xb.bank_blocked[0] == 2
    assert list(xb.port_stalls[:2]) == [2, 3]
    assert list(xb.port_stalls_wider[:2]) == [2, 2]  # m1's cycle 2 is a narrow conflict


@pytest.mark.parametrize("skip", [True, False])
def test_lock_released_when_winner_drops(skip):
    """RTL case: the locked winner drops its request (hold check off).

    cycle  requesters  prev  lock  selected  ready
      0      0 1 2      2     0     0         no   -> lock
      1        1 2      0     1     1         yes  (0 gone: normal RR)
      2          2      1     0     2         yes
    """
    cl, _, xb = build(skip, check_hold=False)
    m0 = cl.add(FixedMaster("m0", xb, {0: 0}))
    m1 = cl.add(FixedMaster("m1", xb, {0: 0, 1: 0}))
    m2 = cl.add(FixedMaster("m2", xb, {0: 0, 1: 0, 2: 0}))
    cl.add(Blocker("dma", xb, {0}))
    cl.run()
    assert (m0.grants, m1.grants, m2.grants) == ([], [1], [2])


# =============================================================================
# 4. Acceptance: distinct banks in parallel
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
def test_distinct_banks_in_parallel(skip):
    """Four masters, each walking the banks with a different offset.

    In every cycle they hit four different banks, so all are granted every
    cycle: no conflicts, no stalls, done in 4 cycles.
    """
    cl, _, xb = build(skip)
    scripts = [[(0, BankReq(addr((i + k) % NB, k))) for k in range(4)] for i in range(4)]
    ms = masters(cl, xb, scripts)
    cl.run()
    assert all(m.grant_cycles == [0, 1, 2, 3] for m in ms)
    assert xb.bank_conflicts.sum() == 0 and xb.port_stalls.sum() == 0
    assert list(xb.bank_grants) == [4, 4, 4, 4]


@pytest.mark.parametrize("skip", [True, False])
def test_conflict_does_not_slow_other_banks(skip):
    """m0 and m1 fight over bank 0; m2 and m3 on banks 1 and 2 are unaffected."""
    cl, _, xb = build(skip)
    ms = masters(cl, xb, [reads(0, 2), reads(0, 2), reads(1, 2), reads(2, 2)])
    cl.run()
    assert [m.grant_cycles for m in ms] == [[0, 2], [1, 3], [0, 1], [0, 1]]


# =============================================================================
# 5. Data routing and read latency
# =============================================================================


@pytest.mark.parametrize("latency", [0, 1, 2])
@pytest.mark.parametrize("skip", [True, False])
def test_write_then_read_through_xbar(latency, skip):
    """m0 writes 7 to (bank 2, row 3) and m1 reads it, both in cycle 0.

    m0 wins cycle 0 (write lands at its end); m1 reads in cycle 1 and so
    sees 7, with the data arriving in cycle 1 + latency, tagged with m1's
    own script index.
    """
    cl, _, xb = build(skip, read_latency=latency)
    m0, m1 = masters(
        cl, xb, [[(0, BankReq(addr(2, 3), write=True, wdata=7))], [(0, BankReq(addr(2, 3)))]]
    )
    cl.run()
    assert m0.grant_cycles == [0] and m0.rdata == []
    assert m1.rdata == [(1 + latency, 0, 7)]


# =============================================================================
# 6. Hold rule and misuse
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
def test_dropping_refused_request_raises(skip):
    """m1 loses cycle 0 and does not come back in cycle 1."""
    cl, _, xb = build(skip)
    cl.add(FixedMaster("m0", xb, {0: 0}))
    cl.add(FixedMaster("m1", xb, {0: 0, 5: 0}))
    with pytest.raises(HoldViolation):
        cl.run()


def test_changing_refused_request_raises():
    """m1 loses cycle 0 and retries with a different address."""
    cl, _, xb = build()
    cl.add(FixedMaster("m0", xb, {0: 0}))
    cl.add(FixedMaster("m1", xb, {0: 0, 1: BankReq(addr(0, 1))}))
    with pytest.raises(HoldViolation):
        cl.run()


def test_two_requests_on_one_port_raise():
    class Twice(FixedMaster):
        def tick(self, cycle, phase):
            if phase == Phase.REQUEST and cycle in self.script:
                self.xb.request(cycle, self.port, BankReq(addr(0)))
                self.xb.request(cycle, self.port, BankReq(addr(1)))

    cl, _, xb = build()
    cl.add(Twice("m", xb, {0: 0}))
    with pytest.raises(SimulationError):
        cl.run()


def test_no_ports_after_run():
    cl, _, xb = build()
    m = cl.add(FixedMaster("m", xb, {0: 0}))
    cl.run()
    with pytest.raises(SimulationError):
        xb.add_port(m)


# =============================================================================
# 7. Random traffic: skip on/off and a reference arbiter
# =============================================================================


def reference_grants(scripts, prios, blocks, n_banks):
    """Grant cycles per master, following the Chisel code literally.

    Every cycle, per bank: mask by priority, then RoundRobinArbiter with
    ``previous = RegNext(selection, N-1)``, ``PriorityEncoder(all zero) =
    N-1`` and ``lock := valid && !ready``. Written independently of Xbar
    (full per-input bit vectors, every cycle ticked).
    """
    n = len(scripts)
    idx = [0] * n
    prev = [n - 1] * n_banks
    lock = [False] * n_banks
    grants = [[] for _ in range(n)]
    t = 0
    while any(idx[p] < len(scripts[p]) for p in range(n)):
        want = {}
        for p in range(n):
            if idx[p] < len(scripts[p]) and t >= scripts[p][idx[p]][0]:
                want[p] = scripts[p][idx[p]][1]
        new_prev, new_lock, won = [n - 1] * n_banks, [False] * n_banks, []
        for b in range(n_banks):
            valid = [want.get(p) == b for p in range(n)]
            eff = [prios[p] if valid[p] else 0 for p in range(n)]
            top = max(eff)
            req = [valid[p] and prios[p] == top for p in range(n)]
            nxt = [req[p] and p > prev[b] for p in range(n)]
            if lock[b] and req[prev[b]]:
                sel = prev[b]
            elif any(nxt):
                sel = nxt.index(True)
            else:
                sel = req.index(True) if any(req) else n - 1
            ready = (t, b) not in blocks
            if any(req) and ready:
                won.append(sel)
            new_lock[b] = any(req) and not ready
            new_prev[b] = sel
        for p in won:
            grants[p].append(t)
            idx[p] += 1
        prev, lock = new_prev, new_lock
        t += 1
    return grants


def random_case(seed):
    """3-4 masters, 25 requests each on 4 banks, gaps of 0..30 cycles.

    Most traffic goes to banks 0-1 so conflicts are common; long gaps give
    skipping something to skip and exercise the idle reset. About a third
    are writes with unique values, so reads really depend on grant order.
    """
    rng = random.Random(seed)
    n = rng.choice([3, 4])
    prios = [rng.choice([0, 0, 1]) for _ in range(n)] if seed % 2 else [0] * n
    specs = []  # per master: list of (earliest cycle, bank, row, write, value)
    for p in range(n):
        t, s = 0, []
        for k in range(25):
            t += rng.choice([0, 0, 0, 1, 2, 8, 30])
            bank = rng.choice([0, 0, 1, 1, 2, 3])
            write = rng.random() < 0.35
            s.append((t, bank, rng.randrange(4), write, 1000 * (p + 1) + k))
        specs.append(s)
    # Wide blocks: (cycle, pair) with a 128-bit port per bank pair {0, 1}, {2, 3}.
    blocks = {(rng.randrange(400), rng.randrange(NB // 2)) for _ in range(40)}
    return specs, prios, blocks


def run_case(specs, prios, blocks, skip):
    cl, mem, xb = build(skip)
    scripts = [
        [(t, BankReq(addr(b, r), write=w, wdata=v if w else None)) for t, b, r, w, v in s]
        for s in specs
    ]
    ms = masters(cl, xb, scripts, prios)
    for pair in range(NB // 2):
        cycles = {t for t, q in blocks if q == pair}
        cl.add(Blocker(f"wide{pair}", xb, cycles, width_bits=2 * 64, bank=2 * pair))
    total = cl.run()
    return cl, mem, xb, ms, total


@pytest.mark.parametrize("seed", range(12))
def test_random_traffic(seed):
    """Skipping on and off give identical results, which match the reference.

    Narrow masters plus two 128-bit blockers, one per bank pair.

    Also replays every granted access in grant order and checks that each
    read returned the last value written to its address before it.
    """
    specs, prios, blocks = random_case(seed)
    results = []
    for skip in (True, False):
        _, mem, xb, ms, total = run_case(specs, prios, blocks, skip)
        results.append(
            (
                total,
                [m.grants for m in ms],
                [m.rdata for m in ms],
                mem.data.copy(),
                {k: getattr(xb, k).copy() for k in ("bank_grants", "bank_conflicts",
                                                    "bank_stalls", "bank_blocked",
                                                    "port_grants", "port_stalls",
                                                    "port_stalls_wider")},
            )
        )  # fmt: skip
    (t0, g0, r0, d0, s0), (t1, g1, r1, d1, s1) = results
    assert t0 == t1 and g0 == g1 and r0 == r1
    assert np.array_equal(d0, d1)
    assert all(np.array_equal(s0[k], s1[k]) for k in s0)

    # Grant cycles equal the literal Chisel arbiter; a wide block makes both
    # banks of its pair not ready.
    not_ready = {(t, 2 * q + i) for t, q in blocks for i in range(2)}
    ref = reference_grants([[(t, b) for t, b, *_ in s] for s in specs], prios, not_ready, NB)
    assert [[c for c, _ in g] for g in g0] == ref

    # Data: replay accesses in grant order. Within a cycle every access is
    # to a different bank, so the order inside a cycle does not matter.
    events = sorted((c, p, k) for p, g in enumerate(g0) for c, k in g)
    shadow, expect = {}, {}
    for c, p, k in events:
        _, b, r, w, v = specs[p][k]
        if w:
            shadow[(b, r)] = v
        else:
            expect[(p, k)] = shadow.get((b, r), 0)
    got = {(p, k): v for p, log in enumerate(r0) for _, k, v in log}
    assert got == expect
    # Every wide block is granted and counts as an access on both banks.
    assert s0["bank_grants"].sum() == sum(len(s) for s in specs) + 2 * len(blocks)


# =============================================================================
# 8. Multi-width ports (MOD6, D33)
# =============================================================================

NB16 = 16  # two superbanks of 8 banks with the default 512-bit wide port


def addr16(bank, row=0):
    return addr(bank, row, n_banks=NB16)


def build16(skip=True, **cfg):
    return build(skip, n_banks=NB16, **cfg)


def reads16(bank, n, start=0, row0=0):
    return [(start, BankReq(addr16(bank, row0 + k))) for k in range(n)]


def wide_master(cl, xb, script, width_bits=512, name="wide"):
    """A ToyMaster on a wide port (script entries are aligned BankReqs)."""
    return cl.add(ToyMaster(name, xb, script, width_bits=width_bits))


@pytest.mark.parametrize("skip", [True, False])
def test_wide_write_touches_exactly_its_superbank(skip):
    """One 512-bit write to superbank 1 row 0: banks 8..15 get one access each."""
    cl, mem, xb = build16(skip)
    wide_master(cl, xb, [(0, BankReq(addr16(8), write=True, wdata=np.arange(8) + 50))])
    cl.run()
    assert list(mem.writes) == [0] * 8 + [1] * 8
    assert list(xb.bank_grants) == [0] * 8 + [1] * 8
    assert list(mem.dump(addr16(8), 8)[:, 0]) == list(range(50, 58))


@pytest.mark.parametrize("latency", [0, 1, 2])
@pytest.mark.parametrize("skip", [True, False])
def test_wide_read_returns_the_group(latency, skip):
    """A 512-bit read returns [8, epw], lane i from bank 8 + i, after the read latency."""
    cl, mem, xb = build16(skip, read_latency=latency)
    mem.load(addr16(0, 3), np.arange(16) + 100)  # row 3 of every bank
    m = wide_master(cl, xb, [(0, BankReq(addr16(8, 3)))])
    cl.run()
    ((cyc, _, first),) = m.rdata
    r = m.last_rdata
    assert cyc == latency and r.data.shape == (8, 1) and r.bank == 8 and first == 108
    assert list(r.data[:, 0]) == list(range(108, 116))


@pytest.mark.parametrize("skip", [True, False])
def test_wide_blocks_narrow_on_its_superbank_only(skip):
    """Wide writes superbank 0 in cycles 0 and 1 (rows 0, 1).

        cycle   wide      narrow bank 1 (sb 0)   narrow bank 9 (sb 1)
          0     grant     refused (wider)        grant
          1     grant     refused (wider)        grant
          2     -         grant                  grant

    The narrow master on superbank 0 is granted in the first cycle the wide
    port does not use it; the one on superbank 1 is never slowed.
    """
    cl, _, xb = build16(skip)
    w = wide_master(cl, xb, [(0, BankReq(addr16(0, r), write=True, wdata=r)) for r in (0, 1)])
    n1, n9 = masters(cl, xb, [reads16(1, 1, row0=5), reads16(9, 3, row0=5)])
    cl.run()
    assert w.grant_cycles == [0, 1]
    assert n1.grant_cycles == [2]
    assert n9.grant_cycles == [0, 1, 2]
    assert xb.port_stalls_wider[n1.port] == 2 and xb.port_stalls[n1.port] == 2
    assert xb.port_stalls[n9.port] == 0
    assert xb.bank_blocked[1] == 2 and xb.bank_blocked[9] == 0


@pytest.mark.parametrize("skip", [True, False])
def test_narrow_gets_through_between_wide_beats(skip):
    """Wide uses superbank 0 in cycles 0 and 2 only: narrow bank 3 wins cycle 1."""
    cl, _, xb = build16(skip)
    w = wide_master(cl, xb, [(0, BankReq(addr16(0, 0))), (2, BankReq(addr16(0, 1)))])
    (n,) = masters(cl, xb, [reads16(3, 2, row0=4)])
    cl.run()
    assert w.grant_cycles == [0, 2]
    assert n.grant_cycles == [1, 3]


@pytest.mark.parametrize("skip", [True, False])
def test_widths_128_and_256(skip):
    """128 and 256 are accepted; the wider wins on overlapping banks; disjoint pairs run.

    cycle 0: 256 on banks 0-3 (grant), 128 on banks 2-3 (refused, wider),
             128 on banks 4-5 (grant), narrow on bank 0 (refused, wider)
    cycle 1: 128 on 2-3 (grant), narrow on 0 (grant)
    """
    cl, _, xb = build16(skip)
    w256 = wide_master(cl, xb, [(0, BankReq(addr16(0)))], width_bits=256, name="w256")
    a128 = wide_master(cl, xb, [(0, BankReq(addr16(2)))], width_bits=128, name="a128")
    b128 = wide_master(cl, xb, [(0, BankReq(addr16(4)))], width_bits=128, name="b128")
    (n,) = masters(cl, xb, [reads16(0, 1, row0=1)])
    cl.run()
    assert (w256.grant_cycles, a128.grant_cycles, b128.grant_cycles) == ([0], [1], [0])
    assert n.grant_cycles == [1]
    assert xb.port_stalls_wider[a128.port] == 1


@pytest.mark.parametrize("skip", [True, False])
def test_equal_width_ports_round_robin_and_lock(skip):
    """Two 512-bit ports on one superbank: D31 round-robin per group; a refused one holds."""
    cl, _, xb = build16(skip)
    a = wide_master(cl, xb, [(0, BankReq(addr16(0, r))) for r in (0, 1)], name="a")
    b = wide_master(cl, xb, [(0, BankReq(addr16(0, r))) for r in (2, 3)], name="b")
    cl.run()
    assert (a.grant_cycles, b.grant_cycles) == ([0, 2], [1, 3])
    assert xb.port_stalls_wider.sum() == 0  # same width: a normal conflict


def test_misaligned_wide_request_raises():
    cl, _, xb = build16()
    wide_master(cl, xb, [(0, BankReq(addr16(1)))])  # 512 bits must start at a superbank
    with pytest.raises(SimulationError):
        cl.run()


def test_wide_write_needs_one_value_per_lane():
    cl, _, xb = build16()
    wide_master(cl, xb, [(0, BankReq(addr16(0), write=True, wdata=[1, 2, 3]))])
    with pytest.raises(SimulationError):
        cl.run()


@pytest.mark.parametrize("width", [96, 192, 320, 32, 1024])
def test_bad_port_widths_raise(width):
    """Not a power-of-two multiple of 64 (or narrower), or wider than wide_bits."""
    cl, _, xb = build16()
    m = cl.add(FixedMaster("m", xb, {}))
    with pytest.raises(ValueError):
        xb.add_port(m, "bad", width_bits=width)


@pytest.mark.parametrize(("n_banks", "width"), [(4, 512), (12, 512), (6, 256), (3, 128)])
def test_bank_count_not_multiple_of_group_raises(n_banks, width):
    cl, _, xb = build(n_banks=n_banks)  # the L1 itself is fine
    m = cl.add(FixedMaster("m", xb, {}))
    with pytest.raises(ValueError):
        xb.add_port(m, "wide", width_bits=width)


def test_wide_bits_is_checked():
    for bad in (192, 32, 600):
        with pytest.raises(ValueError):
            L1Config(wide_bits=bad)
    assert L1Config(wide_bits=256).group_banks(256) == 4
    assert L1Config().group_banks(64) == 1 and L1Config().group_banks(512) == 8


def test_scattering_address_map_is_refused():
    """A custom map that does not put a wide beat into one aligned group."""

    class Reversed:
        def __init__(self, n):
            self.n = n

        def decode(self, word):
            return self.n - 1 - word % self.n, word // self.n

        def encode(self, bank, row):
            return row * self.n + self.n - 1 - bank

    cl = Cluster()
    mem = L1Memory(cl, L1Config(n_banks=16, rows=8), amap=Reversed(16))
    with pytest.raises(SimulationError):
        mem.group_of(0, 512)
    assert mem.group_of(0, 64) == (15,)
