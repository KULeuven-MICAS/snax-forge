"""Where a task list waits.

``placed_waits(tasks, point)`` walks the steps in order and places the waits correctness needs,
as CONTRACTS, "Task list" gives them (its "Waits" part), for the backend (c.py): none at a
configure, those its tasks need before a start, one on the task's component at a sync. Each
``PlacedWait`` holds its step, component, mode (its task's ``wait_mode``) and what it covers;
waits are ordered by when the task they end on was started.

A wait on a writer streamer covers its unit and readers of the same start (``upstream``). A wait
on an accelerator task (a unit or a linked group) covers its writer streamers, and so their
readers, since the program polls the group's STATUS, then its set's STREAMER_BUSY: so the serial
sync on the previous accelerator leaves out the data wait on its writer. An idle mover's task
(``idle``) needs no wait. Nothing is decided here; the values are checked apart (tasks.py).
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
    its reader streamers, in port order, then its writer streamers.
    """
    from snax_forge.design.streamers import groups

    owner = {u: g for g, units in groups(point.graph).items() if len(units) > 1 for u in units}
    up: dict[str, list[str]] = {}
    writers: dict[str, list[str]] = {}
    for s in point.streamers.values():
        comp = owner.get(s.instance, s.instance)
        if s.write:
            up.setdefault(s.name, []).append(comp)
            writers.setdefault(comp, []).append(s.name)
        else:
            up.setdefault(comp, []).append(s.name)
    for comp, ws in writers.items():
        up.setdefault(comp, []).extend(ws)
    return up


def idle(c: Configure) -> bool:
    """An idle mover's task: a streamer task with every temporal bound 0."""
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
        # A wait another needed wait covers is left out (a writer's covers its group, a group's
        # its writers): the widest first, ties to the one started first, then in start order.
        chosen: list[str] = []
        taken: set[str] = set()
        for c in sorted(need, key=lambda c: (-len(self.covers(c)), self.started_at[need[c]])):
            if c not in taken and not self.is_covered(need[c]):
                chosen.append(c)
                taken |= set(self.covers(c))
        for c in sorted(chosen, key=lambda c: self.started_at[need[c]]):
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
