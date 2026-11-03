"""SNAX-ESTIMATE: the ideal, engine-alone estimate of a bound graph (D155, C§19).

    estimate(graph, platform=None, tech="nangate45") -> Estimate

What a mapping buys before any run: per group the exact cycles of snax-gvsoc's group rule on
ideal streams, split into bottleneck, ramp and stall, and the register bits of its datapaths
and of its control unit, CHAR1's cost model's counts (cycles.py, bits.py, D154); on a
technology (``tech``, snax_forge/char/; None skips it) the cost, per unit, buffer link, control
unit, group and accelerator its area and energy (cost.py, C§20); and the data side, per
container its placement, bytes and L1 traffic, per outer port its streamer's loops, beats and
channels, per tailed split its lane use (data.py). No bank conflicts, no streamer fill, no DMA,
no core control: a run's COUNTER is a group's cycles plus the streamers' fill (C§17), and its
window is longer. Beside the ideal cycles, on a platform, each group's ``bank_bound``:
ceil(words its outer ports move / l1.n_banks), the fewest cycles L1's single-port banks allow,
a note where it exceeds the cycles (vecmul at W = 16: 192 against 128, COUNTER 194). Work no unit runs is a not-mapped count, never cycles. The viewers and
``design.md`` read it (report/design.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from snax_forge.char import DEFAULT_TECH, load
from snax_forge.common import JsonFile

from .bits import ctrl_bits, ctrl_registers
from .cost import BlockCost, GroupCost, group_cost
from .cycles import EstimateError, GroupEstimate, group_blocks, group_cycles, not_mapped
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
    tech: str | None = None
    char_source: dict[str, str] | None = None

    @property
    def area(self) -> float | None:
        """µm²: the groups' areas added up; None with no technology."""
        return None if self.tech is None else sum(g.cost.area for g in self.groups if g.cost)

    @property
    def energy(self) -> float | None:
        """nJ per run: the groups' energies added up; None with no technology."""
        return None if self.tech is None else sum(g.cost.energy for g in self.groups if g.cost)

    @property
    def missing(self) -> list[str]:
        """The units and buffer links the technology's table does not characterize."""
        return [k for g in self.groups if g.cost for k in g.cost.missing]

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
            "cost": None if self.tech is None else {
                "tech": self.tech, "source": dict(self.char_source or {}), "area": self.area,
                "energy": self.energy, "missing": self.missing},
        }  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Estimate:
        return cls(
            [GroupEstimate.from_dict(g) for g in d["groups"]],
            d["not_mapped"],
            {k: ContainerData.from_dict(c) for k, c in d["containers"].items()},
            d["l1_size"],
            {k: PortData.from_dict(p) for k, p in d["ports"].items()},
            {k: SplitData.from_dict(s) for k, s in d["splits"].items()},
            (d.get("cost") or {}).get("tech"),
            (d.get("cost") or {}).get("source"),
        )


def estimate(graph: Graph, platform: Platform | None = None,
             tech: str | None = DEFAULT_TECH) -> Estimate:  # fmt: skip
    """The ideal estimate of a bound graph (its symbols bound), on ``platform``'s L1 if given,
    costed on ``tech``'s table unless None."""
    blocks = group_blocks(graph, platform.acc_csr.job_width if platform is not None else 32)
    found = [g for g, _, _ in blocks]
    table = None if tech is None else load(tech)
    if table is not None:
        c_acc = sum(g.cycles for g in found)
        for g, units, bufs in blocks:
            g.cost = group_cost(table, units, bufs, g.bits, g.ctrl_bits, c_acc)
    unit_group = {u: g for g in found for u in g.units}
    words = dict.fromkeys((g.name for g in found), 0)
    for n, _, port in outer_ports(graph):
        words[unit_group[n.attrs["instance"]].name] += port.beats * port.num_ch
    for g in found:
        g.words_per_cycle = words[g.name] / g.cycles if g.cycles else 0.0
        if platform is not None:  # the fewest cycles L1's banks allow (D155)
            g.n_banks = platform.l1.n_banks
            g.bank_bound = -(-words[g.name] // g.n_banks)
    for k, port in ports(graph).items():
        unit_group[k.split(".")[0]].channels += port.num_ch
    l1 = platform.l1 if platform is not None else None
    size = None if l1 is None else l1.n_banks * l1.rows * l1.width_bits // 8
    return Estimate(found, not_mapped(graph), containers(graph), size, ports(graph), splits(graph),
                    tech, None if table is None else dict(table.source))  # fmt: skip


__all__ = [
    "BlockCost",
    "ContainerData",
    "Estimate",
    "EstimateError",
    "GroupCost",
    "GroupEstimate",
    "PortData",
    "SplitData",
    "ctrl_bits",
    "ctrl_registers",
    "estimate",
    "group_cycles",
]
