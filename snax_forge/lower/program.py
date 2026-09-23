"""Command builder in name form (moved from scenarios/make.py, open item 24).

``Program`` programs one block at a time through the model's register map:
``config`` writes every configuration register of a block for a start
argument (``RegisterMap.config_writes``, D36), ``start`` writes its start
register, ``wait`` and ``read`` add the other two commands. The commands are
in name form (``"reg": "dma.src_base"``), as the scenario files hold them
(D42). ``lower_program`` builds its output with it; scenarios that are
scheduled by hand (``scenarios/fmul``) use it directly.
"""

from __future__ import annotations

from typing import Any

from snax_forge.snax_model import RegisterMap, Wait
from snax_forge.snax_model.scenario import (
    ClusterConfig,
    NamedRead,
    ScenarioCommand,
    named,
    register_map_of,
)


class Program:
    """Collects commands in name form for one cluster's register map."""

    def __init__(self, cluster: ClusterConfig) -> None:
        self.map: RegisterMap = register_map_of(cluster)
        self.cmds: list[ScenarioCommand] = []

    def config(self, block: str, arg: Any) -> None:
        self.cmds += [named(c, self.map) for c in self.map.config_writes(block, arg)]

    def start(self, block: str) -> None:
        self.cmds.append(named(self.map.start_write(block), self.map))

    def wait(self, block: str, mode: str) -> None:
        self.cmds.append(Wait(block, mode))

    def read(self, reg: str) -> None:
        self.cmds.append(NamedRead(reg))
