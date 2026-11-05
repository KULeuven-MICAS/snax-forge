"""SNAX-ESTIMATE: the ideal, engine-alone estimate of a bound graph (D155, C§19).

    estimate(graph, platform=None, tech="nangate45") -> Estimate

What a mapping buys before any run: per group the exact cycles of snax-gvsoc's group rule on
ideal streams, split into bottleneck, ramp and stall, and the register bits of its datapaths
and of its control unit, CHAR1's cost model's counts (cycles.py, bits.py, D154); on a
technology (``tech``, snax_forge/char/; None skips it) the cost, per unit, buffer link, control
unit, group and accelerator its area and energy, and per accelerator its L1 accesses, the words
its streamers read and write in L1 at 4 × e_load per bit, a placeholder added to its energy
(``l1``, cost.py, C§20, D154); per accelerator its streamer set, its groups, CSRs and
streamer (``accelerators``, accelerators.py, D157), its streamer not costed (open item 110);
and the data side, per
container its placement, bytes and L1 traffic, per outer port its streamer's loops, beats and
channels, per tailed split its lane use (data.py). No bank conflicts, no streamer fill, no DMA,
no core control. A group's expected COUNTER, its cycles plus the streamers' fill (C§17), is a
guide, not a check: a run's COUNTER counts from the group's START until busy drops, so it holds
bank conflicts and is shortened by the streamers' head start (D155). Beside the ideal cycles, on
a platform, each group's ``bank_bound``: ceil(words its streams move / l1.n_banks), the fewest
cycles of L1 time its streams take, a bound on the streams and not on the COUNTER, a note where
it exceeds the cycles (dot 1 at W_mul = 16: mul's 192 against 128). Work no unit runs is a
not-mapped count of iterations (spatial maps
skipped, C§19's ``not_mapped``) and, over every map, spatial ones included, since a tag alone
runs nothing (``unmapped_iterations``), at ``UNMAPPED_CYCLES_PER_ITERATION`` (1) cycle each,
``unmapped_cycles``: an assumption (a core
takes several), held in this one constant so a measured value can replace it (D155); the
``total_cycles`` are the accelerator's plus the unmapped ones. A group's COUNTER never holds
them. The viewers and
``design.md`` read it (report/design.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from snax_forge.char import DEFAULT_TECH, load
from snax_forge.common import JsonFile
from snax_forge.design.streamers import accelerator_of

from .accelerators import STREAMER_MISSING, AcceleratorEstimate, StreamerEstimate, accelerators
from .bits import ctrl_bits, ctrl_registers
from .cost import BlockCost, GroupCost, L1Cost, group_cost, l1_cost
from .cycles import EstimateError, GroupEstimate, group_blocks, group_cycles, not_mapped
from .data import ContainerData, PortData, SplitData, containers, outer_ports, ports, splits

if TYPE_CHECKING:
    from snax_forge.design.platform import Platform
    from snax_forge.dfg import Graph


# The cycles an iteration no unit runs is counted at (D155): an assumption, not a measurement (a
# core takes several cycles per iteration); one constant, so a measured value can replace it.
UNMAPPED_CYCLES_PER_ITERATION = 1


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
    unmapped_iterations: int = 0  # not_mapped over every map, spatial ones included (D155)
    l1_words: dict[str, int] = field(default_factory=dict)  # per accelerator, its L1 accesses
    accelerators: list[AcceleratorEstimate] = field(default_factory=list)  # in CSR order

    @property
    def streamers(self) -> int:
        """The streamers, one per accelerator's set, none of them costed (open item 110)."""
        return len(self.accelerators)

    @property
    def csrs(self) -> int | None:
        """Every CSR of the entry, the streamers' and the accelerators' own; None without a
        platform."""
        n = [a.n_csrs for a in self.accelerators]
        return None if not n or None in n else sum(n)  # type: ignore[arg-type]

    @property
    def area(self) -> float | None:
        """µm²: the groups' areas added up; None with no technology."""
        return None if self.tech is None else sum(g.cost.area for g in self.groups if g.cost)

    @property
    def l1(self) -> dict[str, L1Cost]:
        """Per accelerator its L1 accesses on the technology (a placeholder, D154); empty with
        none."""
        if self.tech is None:
            return {}
        table = load(self.tech)
        return {a: l1_cost(table, w) for a, w in self.l1_words.items()}

    @property
    def kernel_energy(self) -> float | None:
        """nJ per run: the groups' energies (units, buffer links, control units) added up, the
        experiments' kernel energy; None with no technology."""
        return None if self.tech is None else sum(g.cost.energy for g in self.groups if g.cost)

    @property
    def energy(self) -> float | None:
        """nJ per run: the kernel's energy plus the L1 accesses'; None with no technology."""
        if self.tech is None:
            return None
        return self.kernel_energy + sum(x.energy_nj for x in self.l1.values())

    @property
    def missing(self) -> list[str]:
        """The units and buffer links the technology's table does not characterize."""
        return [k for g in self.groups if g.cost for k in g.cost.missing]

    @property
    def accelerator_cycles(self) -> int:
        """The groups' cycles added up: they run one after another through L1 (D105)."""
        return sum(g.cycles for g in self.groups)

    @property
    def unmapped_cycles(self) -> int:
        """The work no unit runs, as cycles: ``unmapped_iterations`` (over every map, a tag alone
        runs nothing) at ``UNMAPPED_CYCLES_PER_ITERATION`` (D155, an assumption)."""
        return self.unmapped_iterations * UNMAPPED_CYCLES_PER_ITERATION

    @property
    def total_cycles(self) -> int:
        """The accelerator's cycles plus the unmapped ones (D155)."""
        return self.accelerator_cycles + self.unmapped_cycles

    @property
    def l1_footprint(self) -> int:
        """The bytes of the containers placed in L1."""
        return sum(c.bytes for c in self.containers.values() if c.placement == "l1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": [g.to_dict() for g in self.groups],
            "accelerator_cycles": self.accelerator_cycles,
            "not_mapped": self.not_mapped,
            "unmapped_iterations": self.unmapped_iterations,
            "unmapped_cycles": self.unmapped_cycles,
            "total_cycles": self.total_cycles,
            "containers": {k: c.to_dict() for k, c in self.containers.items()},
            "l1_footprint": self.l1_footprint,
            "l1_size": self.l1_size,
            "ports": {k: p.to_dict() for k, p in self.ports.items()},
            "splits": {k: s.to_dict() for k, s in self.splits.items()},
            "accelerators": [a.to_dict() | {"cost": None if self.tech is None else {
                "l1": self.l1[a.name].to_dict() if a.name in self.l1 else None}}
                for a in self.accelerators],
            "cost": None if self.tech is None else {
                "tech": self.tech, "source": dict(self.char_source or {}), "area": self.area,
                "energy": self.energy, "kernel_energy": self.kernel_energy,
                "missing": self.missing},
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
            d["unmapped_iterations"],
            {a["name"]: a["cost"]["l1"]["words"] for a in d.get("accelerators", [])
             if (a.get("cost") or {}).get("l1")},
            [AcceleratorEstimate.from_dict(a) for a in d.get("accelerators", [])],
        )  # fmt: skip


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
    data = containers(graph)
    l1_words: dict[str, int] = {}
    for n, _, port in outer_ports(graph):
        words[unit_group[n.attrs["instance"]].name] += port.beats * port.num_ch
        acc = l1_words.setdefault(accelerator_of(n), 0)
        if data[port.container].placement == "l1":  # a link's is not in L1 (D154)
            l1_words[accelerator_of(n)] = acc + port.beats * port.num_ch
    for g in found:
        g.words_per_cycle = words[g.name] / g.cycles if g.cycles else 0.0
        if platform is not None:  # the fewest cycles L1's banks allow (D155)
            g.n_banks = platform.l1.n_banks
            g.bank_bound = -(-words[g.name] // g.n_banks)
    for k, port in ports(graph).items():
        unit_group[k.split(".")[0]].channels += port.num_ch
    l1 = platform.l1 if platform is not None else None
    size = None if l1 is None else l1.n_banks * l1.rows * l1.width_bits // 8
    return Estimate(found, not_mapped(graph), data, size, ports(graph), splits(graph),
                    tech, None if table is None else dict(table.source),
                    not_mapped(graph, spatial=True), l1_words,
                    accelerators(graph, platform))  # fmt: skip


__all__ = [
    "STREAMER_MISSING",
    "UNMAPPED_CYCLES_PER_ITERATION",
    "AcceleratorEstimate",
    "StreamerEstimate",
    "BlockCost",
    "ContainerData",
    "Estimate",
    "EstimateError",
    "GroupCost",
    "GroupEstimate",
    "L1Cost",
    "PortData",
    "SplitData",
    "ctrl_bits",
    "ctrl_registers",
    "estimate",
    "group_cycles",
]
