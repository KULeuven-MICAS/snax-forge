"""The accelerators of the ideal estimate: one streamer set each, what it holds.

    accelerators(graph, platform=None) -> list[AcceleratorEstimate]

Per accelerator, in CSR order (design/streamers.py's ``accelerators``, the generator's order):
its ``name``, its ``groups``, its CSRs (``csrs``: its first and last, its streamer's then its
own, packed from 960 as gen/csrs.py's ``csr_layout`` numbers them) and its streamer
(``streamer``: ``movers``, its outer ports; ``channels``, their num_ch added up;
``temporal_dims``, each mover's in the streamer's order, readers first; ``csrs``, the
streamer's own, gen/csrs.py's ``streamer_csrs``). A group's own CSRs are START, JOBS, STATUS,
BUSY_COUNTER and per fold
of a T the design params do not fix a JOB_BEATS (gen/accelerator.py's ``_unit``). The
streamer's CSRs depend on the platform (its streamer entries' spatial bounds): without
one, ``csrs`` is null in both places. Its cost, per port's streamer (``cost.streamers``) and its
words through the crossbar and the banks (``cost.l1``), is the estimate's (cost.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from snax_forge.design.streamers import (
    accelerators as graph_accelerators,
)
from snax_forge.design.streamers import (
    groups as graph_groups,
)
from snax_forge.design.streamers import (
    instance_nodes,
    instances,
    resolve,
)

if TYPE_CHECKING:
    from snax_forge.design.platform import Platform
    from snax_forge.dfg import Graph

# The first SNAX CSR (gen/csrs.py's SNAX_CSR_FIRST)
CSR_FIRST = 0x3C0


@dataclass
class StreamerEstimate:
    """An accelerator's streamer set's streamer (module doc)."""

    movers: int
    channels: int
    temporal_dims: list[int]
    csrs: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"movers": self.movers, "channels": self.channels,
                "temporal_dims": list(self.temporal_dims), "csrs": self.csrs}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StreamerEstimate:
        return cls(d["movers"], d["channels"], list(d["temporal_dims"]), d["csrs"])


@dataclass
class AcceleratorEstimate:
    """One accelerator, its streamer set (module doc); ``csrs`` its first and last CSR, null
    without a platform."""

    name: str
    groups: list[str]
    own_csrs: int
    streamer: StreamerEstimate
    csrs: tuple[int, int] | None = None

    @property
    def n_csrs(self) -> int | None:
        """Its CSRs, its streamer's and its own; null without a platform."""
        return None if self.csrs is None else self.csrs[1] - self.csrs[0] + 1

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "groups": list(self.groups), "own_csrs": self.own_csrs,
                "csrs": None if self.csrs is None else list(self.csrs),
                "streamer": self.streamer.to_dict()}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AcceleratorEstimate:
        csrs = d["csrs"]
        return cls(d["name"], list(d["groups"]), d["own_csrs"],
                   StreamerEstimate.from_dict(d["streamer"]),
                   None if csrs is None else (csrs[0], csrs[1]))  # fmt: skip


def _own_csrs(graph: Graph, members: list[str], insts: dict[str, Any]) -> int:
    """A group's own CSRs: START, JOBS, STATUS, BUSY_COUNTER, and a JOB_BEATS per fold whose T the
    design params do not fix (gen/accelerator.py's ``_unit``)."""
    from snax_forge.gen.accelerator import fixed_fold
    from snax_forge.lower.c import fold_rate

    nodes = instance_nodes(graph)
    folds = sum(fold_rate(insts[u]) is not None and fixed_fold(graph, nodes[u], insts[u]) is None
                for u in members)  # fmt: skip
    return 4 + folds


def accelerators(graph: Graph, platform: Platform | None = None) -> list[AcceleratorEstimate]:
    """The graph's accelerators (module doc)."""
    insts = instances(graph)
    by_group = graph_groups(graph)
    shell = resolve(platform, graph, insts) if platform is not None else None
    out, first = [], CSR_FIRST
    for name, units in graph_accelerators(graph).items():
        mine_groups = [g for g, members in by_group.items() if members[0] in units]
        own = sum(_own_csrs(graph, by_group[g], insts) for g in mine_groups)
        if shell is None:  # the movers from the graph's outer ports, no CSRs
            from .data import ports

            outer = ports(graph)
            mine = [(u, p) for u in units for p in insts[u].ports
                    if f"{u}.{p.name}" in outer]  # fmt: skip
            mine = ([x for x in mine if x[1].direction == "in"]
                    + [x for x in mine if x[1].direction == "out"])  # fmt: skip
            got = [outer[f"{u}.{p.name}"] for u, p in mine]
            streamer = StreamerEstimate(len(got), sum(x.num_ch for x in got),
                                        [max(1, len(x.loops)) for x in got])  # fmt: skip
            out.append(AcceleratorEstimate(name, mine_groups, own, streamer))
            continue
        # The streamer's movers, readers first, each set's units in order (gen/gvsoc.py)
        movers = [s for u in units for s in shell.values() if s.instance == u]
        movers = [s for s in movers if not s.write] + [s for s in movers if s.write]
        n = sum(2 + len(s.spatial_bounds) + 2 * s.temporal_dims for s in movers) + 3
        streamer = StreamerEstimate(len(movers), sum(s.num_ch for s in movers),
                                    [s.temporal_dims for s in movers], n)  # fmt: skip
        last = first + n + own - 1
        out.append(AcceleratorEstimate(name, mine_groups, own, streamer, (first, last)))
        first = last + 1
    return out
