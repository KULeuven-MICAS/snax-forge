"""The design point: one bound graph on one platform, with its memory plan (DP1b, D74, D87).

What SNAX-LOWER reads to make the cluster file (LOW1c) and the task list
(LOW1a). The design step writes it once every check has passed; it is not
edited by hand.

    {
     "name":       "vecadd",
     "graph_from": "out/sandbox/vecadd/2_bind.snaxdfg",
     "graph":      {...},     the bound graph as it was read: instances, code,
                              replaced, loop.split (D77, D82)
     "platform":   {...},     the platform working copy: values, base, changes
     "streamers":  {...},     the resolved shell, <instance>_<port> -> instance,
                              port, write, n_ports, spatial_bounds and options
     "memory":     {...}      the memory plan: passes, changes (pins), layouts
    }

There is no recipe in it: the graph holds every instance, and the sandbox
folder the graph came from keeps the recipe that made it. Loading a design
point runs every check again on what it holds, and also requires its
``streamers`` to be the ones the platform's shell gives for its graph, so
an edited or stale file is caught rather than lowered.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snax_forge.dfg import DfgError, Graph
from snax_forge.snax_model.config import check_keys, to_json

from .check import Design, run_checks
from .memory import MemoryPlan
from .platform import Platform, StreamerOptions
from .problems import DesignError, Problem
from .streamers import Streamer, resolve

KEYS = ("name", "graph_from", "graph", "platform", "streamers", "memory")


@dataclass(frozen=True)
class DesignPoint:
    """A checked design; see the module doc."""

    name: str
    graph_from: str
    graph: Graph
    platform: Platform
    streamers: dict[str, Streamer]
    memory: MemoryPlan

    @classmethod
    def of(cls, design: Design, name: str) -> DesignPoint:
        """The design point of a Design whose checks have all passed."""
        if design.memory is None or run_checks(design):
            raise ValueError("a design point needs a design whose checks passed")
        return cls(
            name,
            design.graph_path,
            design.graph,
            design.platform,
            resolve(design.platform, design.instances),
            design.memory,
        )

    def design(self) -> Design:
        """The Design it holds, for the checks."""
        return Design(
            self.platform, self.graph, graph_path=self.graph_from, memory=self.memory,
            memory_spec=self.memory.spec,
        )  # fmt: skip

    def check(self) -> None:
        """Every check, and the streamers equal to the platform's shell for the graph."""
        design = self.design()
        problems = run_checks(design)
        if not problems:
            want = resolve(self.platform, design.instances)
            if want != self.streamers:
                got = {k: s.to_dict() for k, s in self.streamers.items()}
                problems.append(
                    Problem(
                        "point.streamers", "streamers",
                        f"are not the platform's shell for the graph: have {got}, the shell "
                        f"gives { {k: s.to_dict() for k, s in want.items()} }",
                        "write the design point again with pixi run design",
                    )
                )  # fmt: skip
        if problems:
            raise DesignError(problems, f"design point {self.name}")

    # --- files ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "graph_from": self.graph_from,
            "graph": self.graph.to_dict(),
            "platform": self.platform.to_dict(),
            "streamers": {k: s.to_dict() for k, s in self.streamers.items()},
            "memory": self.memory.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Any) -> DesignPoint:
        """Parse and check a design point; raises DesignError."""
        if not isinstance(d, Mapping):
            raise DesignError([Problem("point.keys", "design point", "must be an object")])
        try:
            check_keys(d, KEYS, "design point")
        except ValueError as e:
            raise DesignError([Problem("point.keys", "design point", str(e))]) from None
        missing = [k for k in KEYS if k not in d]
        if missing:
            raise DesignError([Problem("point.keys", "design point", f"missing keys {missing}")])
        try:
            graph = Graph.from_dict(d["graph"])
        except DfgError as e:
            raise DesignError([Problem("graph.load", "design point graph", str(e))]) from None
        streamers = {k: _streamer(k, s) for k, s in d["streamers"].items()}
        point = cls(
            str(d["name"]),
            str(d["graph_from"]),
            graph,
            Platform.from_dict(d["platform"]),
            streamers,
            MemoryPlan.from_dict(d["memory"]),
        )
        point.check()
        return point

    def to_json(self) -> str:
        return to_json(self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> DesignPoint:
        try:
            d = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise DesignError([Problem("point.keys", str(path), str(e))]) from None
        return cls.from_dict(d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())


def _streamer(name: str, d: Any) -> Streamer:
    what = f"streamers.{name}"
    keys = ("instance", "port", "write", "n_ports", "spatial_bounds", *StreamerOptions().to_dict())
    try:
        check_keys(d, keys, what)
        opts = StreamerOptions.from_dict({k: d[k] for k in StreamerOptions().to_dict()})
        s = Streamer(
            name, str(d["instance"]), str(d["port"]), bool(d["write"]),
            tuple(int(x) for x in d["spatial_bounds"]), opts,
        )  # fmt: skip
    except (KeyError, TypeError, ValueError) as e:
        raise DesignError([Problem("point.keys", what, f"{type(e).__name__}: {e}")]) from None
    if s.n_ports != d["n_ports"]:
        raise DesignError(
            [Problem("point.keys", what, f"n_ports {d['n_ports']} is not {list(s.spatial_bounds)}")]
        )
    return s
