"""Where a task list waits (D64).

``placed_waits(tasks, point)`` walks the steps in order and places the waits
correctness needs, for the backend (``c.py``):

    configure  no wait
    start      the waits its tasks need, before their starts
    sync       one wait on the task's component, always placed
    read       no wait

The waits a start needs: one for every ``after`` task not yet waited for,
and one for every listed component whose latest task has not been waited
for. There is at most one wait per component, and a wait that another of
these waits covers is left out; waits are ordered by when the task they end
on was started, and each uses that task's ``wait_mode``. A wait is on a
component, so it ends on the component's latest task and covers every
earlier one on it.

A wait on a writer streamer also covers the unit it serves (a linked
group's: the group) and its reader streamers, for the tasks the same start launched: their data flows
into the writer, so the writer finishes last (``upstream``, from the design
point's streamers: their ``instance`` and ``write``).

An idle mover's task (every temporal bound 0: another group's mover in an
accelerator of several, D148) never needs a wait: it never starts and moves
nothing.

Nothing here decides anything (principle 4): where configures, starts and
syncs go, and every dependency, are the task list's; only the waits
correctness requires are placed. Each wait comes with the step it belongs to
(the start it comes before, or the sync it is), its component and mode, and
what it covers. The values are checked apart (``check_values``, tasks.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .tasks import Configure, Start, Sync, TaskList

if TYPE_CHECKING:
    from snax_forge.design.point import DesignPoint


def upstream(point: DesignPoint) -> dict[str, list[str]]:
    """Per component, the components a wait on it also covers (module doc).

    A writer streamer -> the unit it serves (a linked group's: the group); a unit or group ->
    its reader streamers, in port order.
    """
    from snax_forge.design.streamers import groups

    owner = {u: g for g, units in groups(point.graph).items() if len(units) > 1 for u in units}
    up: dict[str, list[str]] = {}
    for s in point.streamers.values():
        comp = owner.get(s.instance, s.instance)
        if s.write:
            up.setdefault(s.name, []).append(comp)
        else:
            up.setdefault(comp, []).append(s.name)
    return up


def idle(c: Configure) -> bool:
    """An idle mover's task: a streamer task with every temporal bound 0 (D148)."""
    return c.type == "streamer" and not any(c.values["temporal_bounds"])


@dataclass(frozen=True)
class PlacedWait:
    """One wait: before the start at ``step``, or the sync at ``step``."""

    step: int
    component: str
    mode: str
    covers: dict[str, str]  # component -> the latest task the wait ends on


class _Waits:
    def __init__(self, tasks: TaskList, point: DesignPoint) -> None:
        tasks.check()
        self.tasks = tasks
        self.up = upstream(point)
        self.conf: dict[str, Configure] = tasks.configured()
        self.started_at: dict[str, tuple[int, int]] = {}  # task -> (step, place in the start)
        self.latest: dict[str, str] = {}  # component -> its latest started task
        self.covered: dict[str, str] = {}  # component -> latest task a wait has covered
        self.placed: list[PlacedWait] = []

    def is_covered(self, task: str) -> bool:
        if idle(self.conf[task]):
            return True  # it never started
        c = self.covered.get(self.conf[task].component)
        return c is not None and self.started_at[c] >= self.started_at[task]

    def covers(self, comp: str) -> dict[str, str]:
        """What one wait on ``comp`` covers: component -> its task (module doc)."""
        step = self.started_at[self.latest[comp]][0]
        out: dict[str, str] = {}
        stack = [comp]
        while stack:
            x = stack.pop()
            t = self.latest.get(x)
            if x in out or t is None or (x != comp and self.started_at[t][0] != step):
                continue
            out[x] = t
            stack += self.up.get(x, [])
        return out

    def wait(self, i: int, comp: str, mode: str) -> None:
        """One wait on ``comp``: covers its latest task and, from a writer, its group."""
        covers = self.covers(comp)
        self.placed.append(PlacedWait(i, comp, mode, covers))
        self.covered.update(covers)

    def start(self, s: Start, i: int) -> None:
        need: dict[str, str] = {}  # component -> the task a wait on it ends on
        for t in s.tasks:
            c = self.conf[t].component
            if c in self.latest and not self.is_covered(self.latest[c]):
                need[c] = self.latest[c]
            for a in self.conf[t].after:
                if not self.is_covered(a):
                    ac = self.conf[a].component
                    need[ac] = self.latest[ac]
        # A wait another needed wait covers is left out (a writer's covers its group).
        covered_by_others = {x for c in need for x in self.covers(c) if x != c}
        for c in sorted(need, key=lambda c: self.started_at[need[c]]):
            if c not in covered_by_others and not self.is_covered(need[c]):
                self.wait(i, c, self.conf[need[c]].wait_mode)
        for j, t in enumerate(s.tasks):
            c = self.conf[t].component
            self.latest[c] = t
            self.started_at[t] = (i, j)

    def run(self) -> list[PlacedWait]:
        for i, s in enumerate(self.tasks.steps):
            if isinstance(s, Start):
                self.start(s, i)
            elif isinstance(s, Sync):
                self.wait(i, self.conf[s.task].component, s.mode)
        return self.placed


def placed_waits(tasks: TaskList, point: DesignPoint) -> list[PlacedWait]:
    """The waits of ``tasks`` on ``point``'s components, in program order (module doc)."""
    return _Waits(tasks, point).run()
