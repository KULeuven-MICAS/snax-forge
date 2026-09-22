"""Tests for the SNAX-MODEL L1 banks (MOD2).

MOD2 acceptance (docs/STATUS.md):
  * read latency is exact, and
  * a second access to the same bank in the same cycle is refused.

How the tests are built
-----------------------
There is no interconnect yet (MOD3), so ``ToyMaster`` stands in for it. It
calls ``L1Memory.request`` in its MEMORY tick and collects read data in its
RESPONSE tick, the same phases the interconnect will use. Its requests come
from a fixed script, so every test knows exactly which access happens in
which cycle.

Sections:
  1. helpers and the toy master
  2. acceptance: exact read latency
  3. acceptance: one access per bank per cycle
  4. write/read timing rules (D29)
  5. random traffic: skip on/off and a reference model
  6. counters, addressing, configuration, packing
"""

import random

import numpy as np
import pytest

from snax_forge.snax_model import (
    BankConflictError,
    BankReq,
    Cluster,
    Component,
    L1Config,
    L1Memory,
    Phase,
    SimulationError,
)

BASE = 0  # default L1Config.base_addr
WORD = 8  # bytes per word with the default 64-bit banks


def addr(bank, row=0, n_banks=32):
    """Byte address of (bank, row) under the default word-interleaved map.

    Lets a test say "bank 5, row 7" instead of working out the address.
    Pass ``n_banks`` when the test uses a config with fewer banks.
    """
    return BASE + (row * n_banks + bank) * WORD


# =============================================================================
# 1. Toy master
# =============================================================================


class ToyMaster(Component):
    """Issues scripted requests and logs the read data it gets back.

    ``script`` is a list of (cycle, BankReq). All entries for a cycle are
    issued in that cycle, in list order. The tag of the k-th entry is set to
    k, so a response can be traced back to its script entry.

    ``log`` collects (cycle, tag, first element of data) for every read
    response, where ``cycle`` is when the data left the bank.

    The master has no state besides the log, so its next_wake depends only
    on the script and on the L1's committed responses. That keeps it
    correct with skipping on and off.
    """

    # Same phases as the interconnect will use: serve in MEMORY, collect in
    # RESPONSE.
    phases = (Phase.MEMORY, Phase.RESPONSE)

    def __init__(self, name, mem, script):
        super().__init__(name)
        self.mem = mem
        self.script = script
        for k, (_, req) in enumerate(script):
            req.tag = k
        self.log = []

    def tick(self, cycle, phase):
        if phase == Phase.MEMORY:
            # Send everything scheduled for this cycle. A same-bank pair
            # here raises BankConflictError from inside the run.
            for c, req in self.script:
                if c == cycle:
                    self.mem.request(cycle, req)
        elif phase == Phase.RESPONSE:
            # Pick up whatever read data leaves the banks this cycle.
            for r in self.mem.responses(cycle):
                self.log.append((cycle, r.tag, r.data[0].item()))

    def next_wake(self, cycle):
        # Wake for the next scripted request or the next read data,
        # whichever comes first. None when both are done: the run ends.
        issues = [c for c, _ in self.script if c > cycle]
        resp = self.mem.next_response(cycle)
        cands = issues + ([resp] if resp is not None else [])
        return min(cands) if cands else None


def run(script, skip=True, **cfg):
    """Build a cluster with one L1 and one ToyMaster. Returns (cluster, mem, master).

    It does not start the run, so a test can first fill memory with
    ``mem.poke`` and then call ``cluster.run()``. Extra keyword arguments go
    to ``L1Config``.
    """
    cl = Cluster(skip_idle=skip)
    mem = L1Memory(cl, L1Config(**cfg))
    m = cl.add(ToyMaster("m", mem, script))
    return cl, mem, m


# =============================================================================
# 2. Acceptance: exact read latency
# =============================================================================


@pytest.mark.parametrize("latency", [0, 1, 2, 4])
@pytest.mark.parametrize("skip", [True, False])
def test_read_latency_exact(latency, skip):
    """A read issued in cycle 3 returns its data in cycle 3 + latency, no earlier.

    The log holds every response the master saw, so a single entry at the
    expected cycle also shows nothing arrived early or twice. Latency 0 is
    the combinational case: data in the same cycle's RESPONSE phase.
    """
    cl, mem, m = run([(3, BankReq(addr(5, 7)))], skip=skip, read_latency=latency)
    mem.poke(addr(5, 7), 42)  # value the read should return
    total = cl.run()
    assert m.log == [(3 + latency, 0, 42)]
    # The master's last wake-up is the response cycle, so the run ends
    # right after it. Checks next_response does not keep anyone awake.
    assert total == 3 + latency + 1


@pytest.mark.parametrize("latency", [1, 3])
def test_pipelined_reads(latency):
    """Back-to-back reads to one bank return one per cycle, each after the latency.

    Reads in cycles 0..3 to rows 0..3 of bank 2. With latency 3, up to three
    reads are in flight at once; each must come back with its own data.
    """
    script = [(c, BankReq(addr(2, c))) for c in range(4)]
    cl, mem, m = run(script, read_latency=latency)
    for c in range(4):
        mem.poke(addr(2, c), 100 + c)
    cl.run()
    assert m.log == [(c + latency, c, 100 + c) for c in range(4)]


# =============================================================================
# 3. Acceptance: one access per bank per cycle
# =============================================================================


@pytest.mark.parametrize(
    "second",
    [BankReq(addr(4, 1)), BankReq(addr(4, 9), write=True, wdata=1)],
    ids=["read", "write"],
)
def test_same_bank_same_cycle_refused(second):
    """A second access to bank 4 in cycle 2 raises, whether read or write, any row.

    The first access is a read of row 0. The second hits the same bank but
    a different row, so the check is on the bank, not the address.
    """
    cl, _, _ = run([(2, BankReq(addr(4, 0))), (2, second)])
    with pytest.raises(BankConflictError):
        cl.run()


def test_distinct_banks_in_parallel():
    """All 32 banks can be read in the same cycle; data returns together.

    The other half of the rule: different banks never conflict.
    """
    script = [(1, BankReq(addr(b, 3))) for b in range(32)]
    cl, mem, m = run(script)
    for b in range(32):
        mem.poke(addr(b, 3), b)
    cl.run()
    assert m.log == [(2, b, b) for b in range(32)]


def test_same_bank_next_cycle_ok():
    """The busy mark clears at commit: the same bank is free again next cycle.

    Would catch commit() forgetting to clear _busy.
    """
    cl, _, m = run([(1, BankReq(addr(0, 0))), (2, BankReq(addr(0, 1)))])
    cl.run()
    assert [t for t, _, _ in m.log] == [2, 3]


# =============================================================================
# 4. Write and read timing (D29)
# =============================================================================


def test_write_visible_next_cycle():
    """A write in cycle 1 is seen by a read in cycle 2.

    The write lands in commit() at the end of cycle 1; the read in cycle 2
    samples storage after that.
    """
    a = addr(6, 2)
    cl, _, m = run([(1, BankReq(a, write=True, wdata=7)), (2, BankReq(a))])
    cl.run()
    assert m.log == [(3, 1, 7)]


def test_read_samples_at_request():
    """A read returns the value from when it was issued, not when its data leaves.

    Read in cycle 1 with latency 3, so data leaves in cycle 4. A write to the
    same address in cycle 2 commits before that. The read must still return
    the old value 5, like an SRAM that reads its array in the request cycle.
    """
    a = addr(6, 2)
    cl, mem, m = run([(1, BankReq(a)), (2, BankReq(a, write=True, wdata=9))], read_latency=3)
    mem.poke(a, 5)
    cl.run()
    assert m.log == [(4, 0, 5)]
    assert mem.peek(a)[0] == 9  # the write did land


def test_writes_do_not_touch_other_banks_this_cycle():
    """A write and a read to different banks in one cycle do not interact.

    Write bank 0 and read bank 1 in cycle 1. The read gets bank 1's value;
    the write lands in bank 0 only.
    """
    cl, mem, m = run([(1, BankReq(addr(0), write=True, wdata=3)), (1, BankReq(addr(1)))])
    mem.poke(addr(1), 11)
    cl.run()
    assert m.log == [(2, 1, 11)]
    assert mem.peek(addr(0))[0] == 3


# =============================================================================
# 5. Random traffic: skip on/off and a reference model
# =============================================================================


def random_script(seed, n=200, n_banks=8, rows=4):
    """Random reads and writes, at most one per bank per cycle, with idle gaps.

    Few banks and rows, so the same addresses are hit often and reads
    really depend on earlier writes. Gaps of 5 and 20 cycles give skipping
    something to skip. The same seed always gives the same script.
    """
    rng = random.Random(seed)
    script = []
    cycle = 0
    while len(script) < n:
        cycle += rng.choice([1, 1, 1, 5, 20])
        # rng.sample picks distinct banks, so no conflicts in one cycle.
        for b in rng.sample(range(n_banks), rng.randint(1, n_banks)):
            a = addr(b, rng.randrange(rows), n_banks)
            if rng.random() < 0.5:
                script.append((cycle, BankReq(a, write=True, wdata=rng.randint(-99, 99))))
            else:
                script.append((cycle, BankReq(a)))
    return script


def reference(script, latency):
    """Expected read log, worked out without the scheduler or L1Memory.

    Walk the cycles in order. In each cycle, first answer every read from
    the current state, then apply that cycle's writes. That is the D29 rule:
    reads see all earlier cycles' writes and none from their own cycle.
    Unwritten words read as 0, like the zero-filled storage.
    """
    state = {}
    log = []
    for c in sorted({c for c, _ in script}):
        now = [(k, r) for k, (cc, r) in enumerate(script) if cc == c]
        for k, r in now:
            if not r.write:
                log.append((c + latency, k, state.get(r.addr, 0)))
        for _, r in now:
            if r.write:
                state[r.addr] = r.wdata
    return sorted(log)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("latency", [1, 2])
def test_random_traffic(seed, latency):
    """Skipping on and off give the same results, and both match the reference.

    Compared between the two runs: total cycles, the read log, the final
    storage and the per-bank counters. The read log is also compared with
    ``reference``. Each run builds its own script, since ToyMaster writes
    tags into the requests.
    """
    cfg = {"n_banks": 8, "rows": 4, "read_latency": latency}
    results = []
    for skip in (True, False):
        script = random_script(seed)
        cl, mem, m = run(script, skip=skip, **cfg)
        total = cl.run()
        results.append((total, sorted(m.log), mem.data.copy(), mem.reads.copy(), mem.writes.copy()))
    a, b = results
    assert a[0] == b[0]  # same total cycles
    assert a[1] == b[1]  # same read data at the same cycles
    for x, y in zip(a[2:], b[2:]):  # same final storage and counters
        assert np.array_equal(x, y)
    assert a[1] == reference(random_script(seed), latency)


# =============================================================================
# 6. Counters, addressing, configuration, packing
# =============================================================================


def test_access_counts():
    """Per-bank read and write counts equal the accesses in the script.

    Bank 0: two reads. Bank 3: two writes. Bank 31: one read. MOD8 builds
    its per-bank access numbers on these counters.
    """
    script = [
        (1, BankReq(addr(0))),
        (2, BankReq(addr(0, 1))),
        (2, BankReq(addr(3), write=True, wdata=1)),
        (3, BankReq(addr(3, 2), write=True, wdata=2)),
        (3, BankReq(addr(31))),
    ]
    cl, mem, _ = run(script)
    cl.run()
    assert mem.reads[0] == 2 and mem.reads[31] == 1 and mem.reads.sum() == 3
    assert mem.writes[3] == 2 and mem.writes.sum() == 2


def test_default_config_matches_snax_alu_cluster():
    """128 KiB, 32 banks of 64 bits, base 0, word-interleaved.

    Also spot-checks the map: word 1 is bank 1 row 0, word 32 wraps to
    bank 0 row 1, and addr_of inverts locate.
    """
    mem = L1Memory(Cluster())
    assert mem.cfg.size_bytes == 128 * 1024
    assert mem.data.shape == (32, 512, 1)
    assert mem.locate(BASE) == (0, 0)
    assert mem.locate(BASE + 8) == (1, 0)
    assert mem.locate(BASE + 32 * 8) == (0, 1)
    assert mem.addr_of(1, 1) == BASE + 33 * 8


@pytest.mark.parametrize("bad", [BASE + 4, BASE - 8, BASE + 128 * 1024])
def test_bad_address_rejected(bad):
    """Misaligned, below-base and past-the-end addresses raise."""
    with pytest.raises(SimulationError):
        L1Memory(Cluster()).locate(bad)


def test_custom_address_map():
    """The address map is replaceable: here each bank holds one contiguous block.

    Words 0..511 are bank 0, words 512..1023 bank 1, and so on. Any object
    with decode/encode works; no subclassing needed.
    """

    class Blocked:
        def decode(self, word):
            return word // 512, word % 512

        def encode(self, bank, row):
            return bank * 512 + row

    mem = L1Memory(Cluster(), amap=Blocked())
    assert mem.locate(BASE + 8) == (0, 1)
    assert mem.locate(BASE + 512 * 8) == (1, 0)


def test_element_must_fit_word():
    """Element size times elements per word may not exceed the bank width.

    3 x 32 = 96 bits does not fit in 64; 8 x 8 = 64 bits does.
    """
    with pytest.raises(ValueError):
        L1Config(width_bits=64, dtype="int32", elems_per_word=3)
    L1Config(width_bits=64, dtype="int8", elems_per_word=8)  # packing fits


def test_strobed_write_packed_word():
    """With 2 x int32 per word, a strobed write changes only the selected element.

    Not used in v1 (one element per word), but shows the data model is
    ready for packing (D13).
    """
    cl = Cluster()
    mem = L1Memory(cl, L1Config(dtype="int32", elems_per_word=2))
    mem.poke(BASE, [1, 2])
    req = BankReq(BASE, write=True, wdata=[9, 9], strb=[False, True])
    cl.add(ToyMaster("m", mem, [(0, req)]))
    cl.run()
    assert mem.peek(BASE).tolist() == [1, 9]


def test_bank_groups_for_wide_ports():
    """MOD6 (D33): group sizes per width, and the banks of an aligned wide access.

    With 16 banks of 64 bits, a 512-bit access at byte 64 is row 0 of banks
    8..15; at byte 128 it is row 1 of banks 0..7.
    """
    cfg = L1Config(n_banks=16, rows=8)
    assert [cfg.group_banks(w) for w in (64, 128, 256, 512)] == [1, 2, 4, 8]
    for bad in (32, 96, 192, 1024):
        with pytest.raises(ValueError):
            cfg.group_banks(bad)
    mem = L1Memory(Cluster(), cfg)
    assert mem.group_of(64, 512) == tuple(range(8, 16))
    assert mem.group_of(128, 512) == tuple(range(8))
    assert mem.group_of(16, 128) == (2, 3)
    for bad_addr in (8, 32, 16 * 8 * 8):  # misaligned twice, then outside L1
        with pytest.raises(SimulationError):
            mem.group_of(bad_addr, 512)
    # A 4-bank L1 is fine by itself; only a 512-bit port on it is refused (Xbar.add_port).
    L1Config(n_banks=4)
