"""The design point: one bound graph on one platform, with its memory plan.

``DesignPoint`` is what SNAX-LOWER reads to make the task list (lower/derive.py) and the
generator the cluster file (gen/gvsoc.py); its fields are CONTRACTS, "Memory plan and design
point". The design step writes it once every check has passed; it is not edited by hand.
``design_of`` gives the design's name. Loading runs every check again and requires
``streamers`` to be the platform's shell for the graph, so an edited or stale file is caught
rather than lowered.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snax_forge.common import JsonFile, check_keys
from snax_forge.dfg import DfgError, Graph

from .check import Design, run_checks
from .memory import MemoryPlan
from .platform import OPTIONS, Platform, StreamerOptions
from .problems import DesignError, Problem, read_json
from .streamers import Streamer, resolve

KEYS = ("name", "graph_from", "graph", "platform", "streamers", "memory")
STEP = re.compile(r"\d+_\w+\.snaxdfg")


def design_name(graph: Path) -> str:
    """The default name of a graph's design folder: the sandbox folder of a step's graph
    (``out/sandbox/vecadd_w8/2_bind.snaxdfg`` -> ``vecadd_w8``), else the file's stem."""
    return graph.parent.name if STEP.fullmatch(graph.name) else graph.stem


def design_of(graph: Path) -> str:
    """The design's name, which the design point holds and the generated entry is named after
    (``<accelerator_slot>_<name>``): the ``name`` of the recipe the sandbox (or Save)
    kept beside a step's graph, ``recipe.json``, which no ``--set`` changes
    (``out/flow/vecmul_W32/sandbox/2_bind.snaxdfg`` -> ``vecmul``); else ``design_name``."""
    recipe = Path(graph).parent / "recipe.json"
    if STEP.fullmatch(Path(graph).name) and recipe.is_file():
        try:
            name = json.loads(recipe.read_text()).get("name")
        except (OSError, ValueError, AttributeError):
            name = None
        if isinstance(name, str) and name:
            return name
    return design_name(Path(graph))


@dataclass(frozen=True)
class DesignPoint(JsonFile):
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
            resolve(design.platform, design.graph, design.instances),
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
            want = resolve(self.platform, self.graph, design.instances)
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
        try:
            check_keys(d, KEYS, "design point", KEYS)
        except (TypeError, ValueError) as e:
            raise DesignError([Problem("point.keys", "design point", str(e))]) from None
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

    @classmethod
    def load(cls, path: str | Path) -> DesignPoint:
        return cls.from_dict(read_json(path, "point.keys"))


def _streamer(name: str, d: Any) -> Streamer:
    what = f"streamers.{name}"
    keys = ("instance", "port", "write", "n_ports", "spatial_bounds", "temporal_dims", "dtype",
            "num_slc", "num_ch", *OPTIONS)  # fmt: skip
    try:
        check_keys(d, keys, what)
        opts = StreamerOptions.from_dict({k: d[k] for k in OPTIONS})
        s = Streamer(
            name, str(d["instance"]), str(d["port"]), bool(d["write"]),
            tuple(int(x) for x in d["spatial_bounds"]), int(d["temporal_dims"]), opts,
            str(d["dtype"]), int(d["num_slc"]), int(d["num_ch"]),
        )  # fmt: skip
    except (KeyError, TypeError, ValueError) as e:
        raise DesignError([Problem("point.keys", what, f"{type(e).__name__}: {e}")]) from None
    if s.n_ports != d["n_ports"]:
        raise DesignError(
            [Problem("point.keys", what, f"n_ports {d['n_ports']} is not {list(s.spatial_bounds)}")]
        )
    return s
