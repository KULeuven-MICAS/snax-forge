"""Tests for the SNAX-MODEL scheduler (MOD1).

MOD1 acceptance (docs/STATUS.md):
  * toy components give identical results with skipping on and off, and
  * two runs of the same setup give identical results.

The tests use small toy components instead of real hardware models, so that
only the scheduler itself is under test. The toys are:

  ToyFifo   a registered FIFO shared between a producer and a consumer
  Producer  pushes items into the FIFO at given release cycles
  Consumer  pops items from the FIFO and is then busy for a few cycles
  Sleeper   wakes up at random cycles, to create long idle stretches

``build()`` combines them into one setup that the main tests reuse.
"""

import random

import pytest

from snax_forge.snax_model import (
    Cluster,
    Component,
    Phase,
    SimulationError,
    SimulationTimeout,
)


# =============================================================================
# Toy parts
# =============================================================================


class ToyFifo:
    """A registered FIFO: pushes and pops only become visible next cycle.

    ``push`` and ``pop`` do not change the contents directly. They record what
    should happen and call ``cluster.touch(self)``, so the scheduler calls
    ``commit`` at the end of the cycle. ``full()`` and ``empty()`` read only
    the committed contents, which hold the values from the previous cycle.

    This is a shared state element (``Stateful`` in sched.py), not a
    Component: it has no tick, no phases and no next_wake.
    """

    def __init__(self, cluster, depth):
        self.cluster = cluster
        self.depth = depth
        self.q = []        # committed contents, oldest item first
        self._push = []    # items pushed this cycle
        self._pops = 0     # number of items popped this cycle

    def full(self):
        return len(self.q) >= self.depth

    def empty(self):
        return not self.q

    def push(self, x):
        self._push.append(x)
        self.cluster.touch(self)

    def pop(self):
        # Return the oldest committed item that was not yet popped this cycle.
        x = self.q[self._pops]
        self._pops += 1
        self.cluster.touch(self)
        return x

    def commit(self):
        # Remove this cycle's pops from the front, add this cycle's pushes at
        # the back. Because full() looks at committed contents only, a push
        # and a pop in the same cycle can never overflow the FIFO.
        self.q = self.q[self._pops:] + self._push
        assert len(self.q) <= self.depth
        self._push = []
        self._pops = 0


class Counted(Component):
    """Base class for the toys: records ticked cycles and gap cycles.

    ``ticked`` is the set of cycles in which the component was ticked.
    ``gap`` is the total number of cycles reported through ``on_gap``.
    Together they must cover the whole run (see
    test_ticks_plus_gaps_cover_run).
    """

    def __init__(self, name):
        super().__init__(name)
        self.ticked = set()
        self.gap = 0

    def tick(self, cycle, phase):
        self.ticked.add(cycle)

    def on_gap(self, start, stop):
        # Cycles start .. stop-1 passed without a tick.
        self.gap += stop - start


class Producer(Counted):
    """Pushes item k into the FIFO no earlier than cycle release[k].

    If the FIFO is full, the producer stalls and retries every cycle.
    The items are just the numbers 0, 1, 2, ... in order.
    """

    phases = (Phase.COMPUTE,)

    def __init__(self, name, fifo, release):
        super().__init__(name)
        self.fifo = fifo
        self.release = release  # release[k] = earliest cycle for item k
        self.k = 0              # committed: index of the next item to push
        self._k_next = 0        # next state of k, applied in commit

    def tick(self, cycle, phase):
        super().tick(cycle, phase)
        have_item = self.k < len(self.release)
        if have_item and self.release[self.k] <= cycle and not self.fifo.full():
            self.fifo.push(self.k)
            self._k_next = self.k + 1

    def commit(self, cycle):
        self.k = self._k_next

    def next_wake(self, cycle):
        if self.k >= len(self.release):
            return None           # everything pushed: nothing left to do
        if self.fifo.full():
            return cycle + 1      # stalled: retry next cycle
        # Wait until the next item is released (or go now if it already is).
        return max(cycle + 1, self.release[self.k])


class Consumer(Counted):
    """Pops one item when free, then stays busy for ``service`` cycles.

    Every pop is logged as (cycle, item). The tests compare these logs
    between runs to check that timing is identical.
    """

    phases = (Phase.COMPUTE,)

    def __init__(self, name, fifo, service):
        super().__init__(name)
        self.fifo = fifo
        self.service = service
        self.free_at = 0        # committed: first cycle it may pop again
        self._free_at_next = 0  # next state of free_at, applied in commit
        self.log = []

    def tick(self, cycle, phase):
        super().tick(cycle, phase)
        if cycle >= self.free_at and not self.fifo.empty():
            self.log.append((cycle, self.fifo.pop()))
            self._free_at_next = cycle + self.service

    def commit(self, cycle):
        self.free_at = self._free_at_next

    def next_wake(self, cycle):
        # Busy: wake when free. The answer must not depend on when it is
        # asked, so "<=" (not "<") keeps returning free_at up to and
        # including the cycle just before it. With "<", a run with skipping
        # off would get None one cycle earlier and end one cycle sooner.
        if cycle + 1 <= self.free_at:
            return self.free_at
        if not self.fifo.empty():
            return cycle + 1      # free and data waiting: pop next cycle
        return None               # free and FIFO empty: nothing to do


class Sleeper(Counted):
    """Wakes at ``times`` random cycles (fixed seed) and logs them.

    It is not connected to anything. Its gaps of up to 50 cycles are what
    skipping should jump over.
    """

    phases = (Phase.CONTROL,)

    def __init__(self, name, seed, times):
        super().__init__(name)
        rng = random.Random(seed)
        t = 0
        self.wakes = []
        for _ in range(times):
            t += rng.randint(1, 50)
            self.wakes.append(t)
        self.log = []

    def tick(self, cycle, phase):
        super().tick(cycle, phase)
        # With skipping off it is ticked every cycle, so only log the cycles
        # it actually asked for.
        if cycle in self.wakes:
            self.log.append(cycle)

    def next_wake(self, cycle):
        # The first planned wake-up after this cycle, or None when done.
        return next((w for w in self.wakes if w > cycle), None)


def build(skip, seed=1):
    """Build the standard toy setup.

    A producer with 40 items released at random cycles in 0..300 feeds a
    consumer through a FIFO of depth 2. The consumer needs 7 cycles per item,
    so the FIFO sometimes fills up and the producer really stalls. A separate
    sleeper adds idle stretches. The same seed always gives the same setup.

    Returns (cluster, producer, consumer, sleeper).
    """
    cl = Cluster(skip_idle=skip)
    fifo = ToyFifo(cl, depth=2)
    rng = random.Random(seed)
    release = sorted(rng.randint(0, 300) for _ in range(40))
    prod = cl.add(Producer("prod", fifo, release))
    cons = cl.add(Consumer("cons", fifo, service=7))
    slp = cl.add(Sleeper("sleep", seed, times=10))
    return cl, prod, cons, slp


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.parametrize("seed", range(5))
def test_skip_on_off_identical(seed):
    """Main MOD1 acceptance test: skipping must not change any result.

    The same setup runs once with skipping and once without. Total cycles,
    every pop with its cycle, and every sleeper wake-up must be equal. Five
    seeds cover different release and stall patterns.

    A failure means some component's next_wake is wrong: it slept through a
    cycle where it should have acted, or its answer depends on when it is
    asked. Every future component should pass this same kind of test.
    """
    a, _, cons_a, slp_a = build(skip=True, seed=seed)
    b, _, cons_b, slp_b = build(skip=False, seed=seed)

    assert a.run() == b.run()          # same total cycle count
    assert cons_a.log == cons_b.log    # same pops at the same cycles
    assert slp_a.log == slp_b.log      # same wake-ups
    assert len(cons_a.log) == 40       # all items arrived


def test_repeatable():
    """Two runs of the same setup give identical results.

    Would catch anything that makes a run depend on more than its input,
    e.g. iterating over a set with varying order or an unseeded random
    generator.
    """
    runs = []
    for _ in range(2):
        cl, _, cons, slp = build(skip=True)
        runs.append((cl.run(), cons.log, slp.log))
    assert runs[0] == runs[1]


def test_skipping_skips():
    """Skipping really saves ticks.

    test_skip_on_off_identical only shows skipping is correct. This shows it
    does something: without it, a scheduler that ticks everyone every cycle
    would pass all other tests.
    """
    a, _, _, slp_a = build(skip=True)
    b, _, _, slp_b = build(skip=False)
    a.run()
    b.run()
    assert len(slp_a.ticked) < len(slp_b.ticked)


@pytest.mark.parametrize("skip", [True, False])
def test_ticks_plus_gaps_cover_run(skip):
    """For every component, ticked cycles + gap cycles = total cycles.

    Would catch a missing on_gap call, a gap of the wrong length, or a gap
    overlapping a tick. MOD8's check "busy + idle + stalled = total" relies
    on this.
    """
    cl, *comps = build(skip=skip)
    total = cl.run()
    for c in comps:
        assert len(c.ticked) + c.gap == total, c.name


def test_fifo_is_registered():
    """An item pushed in cycle t is first seen by the consumer in cycle t+1.

    The producer pushes item 0 in cycle 5. The FIFO commits at the end of
    cycle 5, so the consumer can only pop it in cycle 6. Would catch a break
    in the RTL-like semantics, e.g. shared elements committing immediately.
    """
    cl = Cluster()
    fifo = ToyFifo(cl, depth=4)
    cl.add(Producer("p", fifo, release=[5]))
    cons = cl.add(Consumer("c", fifo, service=1))
    cl.run()
    assert cons.log == [(6, 0)]  # item 0 popped in cycle 6


def test_bad_wake_rejected():
    """A next_wake that is not in the future raises SimulationError.

    Returning the current cycle would make the loop simulate the same cycle
    forever; the scheduler must refuse it.
    """

    class Bad(Component):
        phases = (Phase.COMPUTE,)

        def tick(self, cycle, phase):
            pass

        def next_wake(self, cycle):
            return cycle  # wrong: must be > cycle

    cl = Cluster()
    cl.add(Bad("bad"))
    with pytest.raises(SimulationError):
        cl.run()


def test_timeout():
    """A component that never goes idle hits max_cycles and raises.

    Protects against runs that would otherwise never end.
    """

    class Forever(Component):
        phases = (Phase.COMPUTE,)

        def tick(self, cycle, phase):
            pass

        def next_wake(self, cycle):
            return cycle + 1  # always active

    cl = Cluster()
    cl.add(Forever("f"))
    with pytest.raises(SimulationTimeout):
        cl.run(max_cycles=100)