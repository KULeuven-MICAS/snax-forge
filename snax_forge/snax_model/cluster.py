"""The simulated SNAX cluster: owns the components and runs them.

In MOD1 it is only a named registry around the scheduler. Later tasks add a
``ClusterConfig`` and ``Cluster.from_config`` that builds banks (MOD2), the
interconnect (MOD3), streamers (MOD4), accelerators (MOD5), DMA/L2 (MOD6) and
the controller (MOD7), and wires them together.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVar

from .sched import Component, Scheduler, Stateful

C = TypeVar("C", bound=Component)


class Cluster:
    def __init__(self, skip_idle: bool = True) -> None:
        self.scheduler = Scheduler(skip_idle=skip_idle)
        self._by_name: dict[str, Component] = {}

    def add(self, comp: C) -> C:
        if comp.name in self._by_name:
            raise ValueError(f"duplicate component name: {comp.name}")
        self.scheduler.add(comp)
        self._by_name[comp.name] = comp
        return comp

    def touch(self, elem: Stateful) -> None:
        self.scheduler.touch(elem)

    def __getitem__(self, name: str) -> Component:
        return self._by_name[name]

    def __iter__(self) -> Iterator[Component]:
        return iter(self._by_name.values())

    @property
    def cycle(self) -> int:
        return self.scheduler.cycle

    def run(self, max_cycles: int = 10**9) -> int:
        return self.scheduler.run(max_cycles)
