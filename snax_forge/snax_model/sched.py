"""Cycle-level, event-driven scheduler for SNAX-MODEL (MOD1, D10, D21).

Cycle semantics (RTL-like):

* A cycle runs in the fixed phases of ``Phase``. A component takes part in one
  or more phases and is ticked once per phase it declares.
* During ``tick`` a component may read:
    - registers (committed state) of any component or shared element, which
      hold the values from the end of the previous cycle, and
    - wires driven earlier in this cycle by components in earlier phases
      (combinational paths such as request -> grant).
  It writes only its own next state, its own wires, and shared elements
  through their own write methods (e.g. ``Fifo.push``).
* After all phases, every ticked component's ``commit`` runs, then every
  shared element touched this cycle is committed. ``commit`` makes next state
  visible and clears wires. It must not read other components' wires.

Skipping: after each simulated cycle the scheduler asks every component for
its next wake-up cycle and jumps to the earliest one. Cycles in which a
component is not ticked are reported to it through ``on_gap`` so statistics
still add up to the total cycle count. With ``skip_idle=False`` every
component is ticked every cycle; results must be identical either way.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional, Protocol, Sequence


class Phase(IntEnum):
    """Order of work inside one cycle."""

    CONTROL = 0    # controller issues commands
    COMPUTE = 1    # accelerators take from / push to FIFOs
    REQUEST = 2    # streamers and DMA drive memory requests
    ARBITRATE = 3  # interconnect grants one master per bank
    MEMORY = 4     # banks serve granted requests
    RESPONSE = 5   # masters see grants, read data returns


class SimulationError(RuntimeError):
    """A component broke the scheduler contract."""


class SimulationTimeout(RuntimeError):
    """The run did not finish within ``max_cycles``."""


class Component:
    """Base class for every sub-model of the cluster.

    Subclasses set ``phases`` and override ``tick``. The remaining methods have
    safe defaults.
    """

    phases: Sequence[Phase] = ()

    def __init__(self, name: str) -> None:
        self.name = name

    def tick(self, cycle: int, phase: Phase) -> None:
        """Advance by one step in ``phase``. Must be a no-op if there is no work."""
        raise NotImplementedError

    def commit(self, cycle: int) -> None:
        """Make next state visible and clear wires."""

    def next_wake(self, cycle: int) -> Optional[int]:
        """Earliest cycle > ``cycle`` at which a tick is needed, or None if idle.

        Called after ``cycle`` has been committed (with ``cycle = -1`` before the
        first cycle). Returning ``cycle + 1`` means active, including stalled.

        The answer must be consistent over time: if nothing the component
        reads has changed, asking again at any cycle before the returned one
        must give the same answer. Otherwise runs with and without skipping
        diverge. The run ends after the last cycle in which any component was
        woken.
        """
        return None

    def on_gap(self, start: int, stop: int) -> None:
        """Cycles ``[start, stop)`` passed without a tick (idle for this component)."""


class Stateful(Protocol):
    """A shared state element (e.g. a FIFO between a streamer and an accelerator)."""

    def commit(self) -> None: ...


class Scheduler:
    def __init__(self, skip_idle: bool = True) -> None:
        self.skip_idle = skip_idle
        self.cycle = 0  # next cycle to simulate; total cycle count after run()
        self.components: list[Component] = []
        self._touched: list[Stateful] = []
        self._touched_ids: set[int] = set()
        self._running = False

    # -- setup --------------------------------------------------------------

    def add(self, comp: Component) -> Component:
        if self._running:
            raise SimulationError("components cannot be added during a run")
        if not comp.phases:
            raise SimulationError(f"{comp.name}: no phases declared")
        self.components.append(comp)
        return comp

    def touch(self, elem: Stateful) -> None:
        """Schedule ``elem.commit()`` at the end of the current cycle (once)."""
        if id(elem) not in self._touched_ids:
            self._touched_ids.add(id(elem))
            self._touched.append(elem)

    # -- run ----------------------------------------------------------------

    def run(self, max_cycles: int = 10**9) -> int:
        """Simulate until no component has pending work. Returns total cycles."""
        comps = self.components
        n = len(comps)
        # Tick order: by phase, then by registration order (deterministic).
        order = sorted(
            ((p, i) for i, c in enumerate(comps) for p in c.phases),
            key=lambda pi: (pi[0], pi[1]),
        )
        last = [self.cycle - 1] * n  # last cycle each component was ticked
        wake = [self._checked_wake(c, self.cycle - 1) for c in comps]
        now = self.cycle

        self._running = True
        try:
            while True:
                pending = [w for w in wake if w is not None]
                if not pending:
                    break
                t = min(pending) if self.skip_idle else now
                if t >= max_cycles:
                    raise SimulationTimeout(f"still active at cycle {t}")

                if self.skip_idle:
                    active = [i for i in range(n) if wake[i] == t]
                else:
                    active = list(range(n))
                is_active = [False] * n
                for i in active:
                    is_active[i] = True
                    if last[i] < t - 1:
                        comps[i].on_gap(last[i] + 1, t)

                for phase, i in order:
                    if is_active[i]:
                        comps[i].tick(t, phase)
                for i in active:
                    comps[i].commit(t)
                    last[i] = t
                for elem in self._touched:
                    elem.commit()
                self._touched.clear()
                self._touched_ids.clear()

                wake = [self._checked_wake(c, t) for c in comps]
                now = t + 1
        finally:
            self._running = False

        for i, c in enumerate(comps):
            if last[i] < now - 1:
                c.on_gap(last[i] + 1, now)
        self.cycle = now
        return now

    @staticmethod
    def _checked_wake(comp: Component, cycle: int) -> Optional[int]:
        w = comp.next_wake(cycle)
        if w is not None and w <= cycle:
            raise SimulationError(
                f"{comp.name}: next_wake({cycle}) returned {w}, must be > {cycle}"
            )
        return w