"""SNAX-ESTIMATE: the ideal, engine-alone estimate of a bound graph (D151, C§19).

    estimate(graph, platform=None) -> Estimate

What a mapping buys before any run: per group the exact cycles of snax-gvsoc's group rule on
ideal streams, split into bottleneck, ramp and stall (cycles.py), and the data side, per
container its placement, bytes and L1 traffic, per outer port its streamer's loops, beats and
channels, per tailed split its lane use (data.py). No bank conflicts, no streamer fill, no DMA,
no core control: a run's COUNTER is a group's cycles plus the streamers' fill (C§17), and its
window is longer. Work no unit runs is a not-mapped count, never cycles. The viewers and
``design.md`` read it (report/design.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from snax_forge.common import JsonFile

from .cycles import EstimateError, GroupEstimate, group_cycles, groups, not_mapped
from .data import ContainerData, PortData, SplitData, containers, outer_ports, ports, splits

if TYPE_CHECKING:
    from snax_forge.design.platform import Platform
    from snax_forge.dfg import Graph


@dataclass
class Estimate(JsonFile):
    """C§19's artefact; see the module doc."""

    groups: list[GroupEstimate]
    not_mapped: int
    containers: dict[str, ContainerData]
    l1_size: int | None = None
    ports: dict[str, PortData] = field(default_factory=dict)
    splits: dict[str, SplitData] = field(default_factory=dict)

    @property
    def accelerator_cycles(self) -> int:
        """The groups' cycles added up: they run one after another through L1 (D105)."""
        return sum(g.cycles for g in self.groups)

    @property
    def l1_footprint(self) -> int:
        """The bytes of the containers placed in L1."""
        return sum(c.bytes for c in self.containers.values() if c.placement == "l1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": [g.to_dict() for g in self.groups],
            "accelerator_cycles": self.accelerator_cycles,
            "not_mapped": self.not_mapped,
            "containers": {k: c.to_dict() for k, c in self.containers.items()},
            "l1_footprint": self.l1_footprint,
            "l1_size": self.l1_size,
            "ports": {k: p.to_dict() for k, p in self.ports.items()},
            "splits": {k: s.to_dict() for k, s in self.splits.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Estimate:
        return cls(
            [GroupEstimate.from_dict(g) for g in d["groups"]],
            d["not_mapped"],
            {k: ContainerData.from_dict(c) for k, c in d["containers"].items()},
            d["l1_size"],
            {k: PortData.from_dict(p) for k, p in d["ports"].items()},
            {k: SplitData.from_dict(s) for k, s in d["splits"].items()},
        )


def estimate(graph: Graph, platform: Platform | None = None) -> Estimate:
    """The ideal estimate of a bound graph (its symbols bound), on ``platform``'s L1 if given."""
    found = groups(graph)
    unit_group = {u: g for g in found for u in g.units}
    words = dict.fromkeys((g.name for g in found), 0)
    for n, _, port in outer_ports(graph):
        words[unit_group[n.attrs["instance"]].name] += port.beats * port.num_ch
    for g in found:
        g.words_per_cycle = words[g.name] / g.cycles if g.cycles else 0.0
    for k, port in ports(graph).items():
        unit_group[k.split(".")[0]].channels += port.num_ch
    l1 = platform.l1 if platform is not None else None
    size = None if l1 is None else l1.n_banks * l1.rows * l1.width_bits // 8
    return Estimate(found, not_mapped(graph), containers(graph), size, ports(graph), splits(graph))


__all__ = [
    "ContainerData",
    "Estimate",
    "EstimateError",
    "GroupEstimate",
    "PortData",
    "SplitData",
    "estimate",
    "group_cycles",
]
