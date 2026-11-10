"""SNAX-ESTIMATE: the ideal, engine-alone estimate of a bound graph (D155, C§19).

    estimate(graph, platform=None, tech="nangate45") -> Estimate

What a mapping buys before any run: per group the exact cycles of snax-gvsoc's group rule on
ideal streams, split into bottleneck, ramp and stall, and the register bits of its datapaths
and of its control unit, CHAR1's cost model's counts (cycles.py, bits.py, D154); on a
technology (``tech``, snax_forge/char/; None skips it) the cost, per unit, buffer link, control
unit, group and accelerator its area and energy, and on a platform HWCOST's: per port its
streamer, the crossbar and the L1 banks, per accelerator its words through the crossbar and the
banks (``l1``), and the design's cost in five parts (``parts``: datapath, control, streamers,
crossbar, memory), cost.py, C§20, D154, D167; per accelerator its streamer set, its groups, CSRs
and streamer (``accelerators``, accelerators.py, D157); and the data side, per
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
from functools import cached_property
from typing import TYPE_CHECKING, Any

from snax_forge.char import DEFAULT_TECH, load
from snax_forge.char.tables import Costs, load_costs
from snax_forge.common import JsonFile
from snax_forge.design.streamers import accelerator_of, instances, resolve
from snax_forge.design.streamers import accelerators as graph_accelerators

from .accelerators import AcceleratorEstimate, StreamerEstimate, accelerators
from .bits import ctrl_bits, ctrl_registers
from .cost import (
    PARTS,
    BlockCost,
    GroupCost,
    L1Cost,
    MemoryCost,
    Part,
    StreamerCost,
    StreamerCounts,
    XbarCost,
    group_cost,
    l1_cost,
    memory_cost,
    streamer_cost,
    xbar_cost,
)
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
    l1_words: dict[str, int] = field(default_factory=dict)  # per accelerator, its L1 words
    accelerators: list[AcceleratorEstimate] = field(default_factory=list)  # in CSR order
    # HWCOST's counts on a platform (D167): per port's streamer, the crossbar's inputs (its
    # channels and the cores' and AXI's ports), the banks (count, depth, width)
    streamer_counts: list[StreamerCounts] = field(default_factory=list)
    xbar_inputs: tuple[int, int] | None = None
    bank_shape: tuple[int, int, int] | None = None

    @property
    def streamers(self) -> int:
        """The streamer sets, one per accelerator."""
        return len(self.accelerators)

    @property
    def csrs(self) -> int | None:
        """Every CSR of the entry, the streamers' and the accelerators' own; None without a
        platform."""
        n = [a.n_csrs for a in self.accelerators]
        return None if not n or None in n else sum(n)  # type: ignore[arg-type]

    @property
    def costs(self) -> Costs | None:
        """HWCOST's tables of the technology (snax_forge/char/<tech>/), None without them or a
        platform's counts."""
        if self.tech is None or self.bank_shape is None:
            return None
        return load_costs(self.tech)

    @cached_property
    def streamer_costs(self) -> list[StreamerCost]:
        """Per port its streamer priced (cost.py); empty when not costed."""
        c = self.costs
        return [] if c is None else [streamer_cost(c, x) for x in self.streamer_counts]

    def _rw(self, accelerator: str | None = None) -> tuple[int, int]:
        mine = [x for x in self.streamer_counts if accelerator in (None, x.accelerator)]
        return (sum(x.words for x in mine if not x.write), sum(x.words for x in mine if x.write))

    @cached_property
    def xbar(self) -> XbarCost | None:
        """The crossbar priced at its inputs (check.py's ``crossbar_inputs``); None when not
        costed."""
        c = self.costs
        if c is None or self.xbar_inputs is None or self.bank_shape is None:
            return None
        return xbar_cost(c, sum(self.xbar_inputs), self.bank_shape[0], sum(self._rw()))

    @cached_property
    def memory(self) -> MemoryCost | None:
        """The L1 banks priced; None when not costed."""
        c = self.costs
        if c is None or self.bank_shape is None:
            return None
        return memory_cost(c, *self.bank_shape, *self._rw())

    @property
    def l1(self) -> dict[str, L1Cost]:
        """Per accelerator its words through the crossbar and the banks; empty when not
        costed."""
        x, m = self.xbar, self.memory
        if x is None or m is None:
            return {}
        return {a.name: l1_cost(x, m, *self._rw(a.name)) for a in self.accelerators
                if any(s.accelerator == a.name for s in self.streamer_counts)}  # fmt: skip

    @property
    def parts(self) -> dict[str, Part] | None:
        """The design's cost in its five parts (``PARTS``), None with no technology; a part not
        costed has None in it."""
        if self.tech is None:
            return None
        cost = [g.cost for g in self.groups if g.cost]
        blocks = [b for c in cost for b in (*c.units.values(), *c.links.values())]
        s, x, m = self.streamer_costs, self.xbar, self.memory
        return {
            "datapath": Part(sum(b.total_um2 for b in blocks), sum(b.task_nj for b in blocks)),
            "control": Part(sum(c.ctrl_um2 for c in cost), sum(c.ctrl_nj for c in cost)),
            "streamers": Part(sum(y.area_um2 for y in s), sum(y.energy_nj for y in s))
            if self.costs is not None else Part(None, None),
            "crossbar": Part(x.area_um2, x.energy_nj) if x else Part(None, None),
            "memory": Part(m.area_um2, m.energy_nj) if m else Part(None, None),
        }  # fmt: skip

    @property
    def area(self) -> float | None:
        """µm²: every part costed added up, the banks included; None with no technology."""
        p = self.parts
        return None if p is None else sum(x.area_um2 or 0.0 for x in p.values())

    @property
    def area_design(self) -> float | None:
        """µm²: ``area`` without the banks, the part every variant shares."""
        p = self.parts
        return None if p is None else self.area - (p["memory"].area_um2 or 0.0)

    @property
    def kernel_energy(self) -> float | None:
        """nJ per run: the groups' energies (units, buffer links, control units) added up, the
        experiments' kernel energy; None with no technology."""
        return None if self.tech is None else sum(g.cost.energy for g in self.groups if g.cost)

    @property
    def energy(self) -> float | None:
        """nJ per run: every part costed added up; None with no technology."""
        p = self.parts
        return None if p is None else sum(x.energy_nj or 0.0 for x in p.values())

    @property
    def missing(self) -> list[str]:
        """The units and buffer links the technology's table does not characterize."""
        return [k for g in self.groups if g.cost for k in g.cost.missing]

    @property
    def not_costed(self) -> list[str]:
        """The parts not costed, each with why (``PARTS``)."""
        if self.tech is None:
            return []
        if self.bank_shape is None:
            return ["streamers, crossbar and memory: no platform"]
        if self.costs is None:
            return [f"streamers, crossbar and memory: no tables for {self.tech}"]
        out = []
        if self.xbar_inputs is None:
            out.append("crossbar: its inputs do not resolve")
        m = self.memory
        if m is not None and not m.found:
            out.append(f"memory: no {m.depth} × {m.width} bank in the table")
        return out

    @property
    def extrapolated(self) -> list[str]:
        """What was priced outside a table's grid: a streamer by name, ``crossbar``."""
        out = [s.counts.name for s in self.streamer_costs if s.extrapolated]
        x = self.xbar
        return out + (["crossbar"] if x is not None and x.extrapolated else [])

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
        l1, streamers, xbar, memory = self.l1, self.streamer_costs, self.xbar, self.memory
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
                "l1": l1[a.name].to_dict() if a.name in l1 else None,
                "streamers": [x.to_dict() for x in streamers
                              if x.counts.accelerator == a.name]}}
                for a in self.accelerators],
            "cost": None if self.tech is None else {
                "tech": self.tech, "source": dict(self.char_source or {}), "area": self.area,
                "area_design": self.area_design, "energy": self.energy,
                "kernel_energy": self.kernel_energy, "missing": self.missing,
                "parts": {k: v.to_dict() for k, v in (self.parts or {}).items()},
                "xbar": None if xbar is None else xbar.to_dict(),
                "memory": None if memory is None else memory.to_dict(),
                "not_costed": self.not_costed, "extrapolated": self.extrapolated},
            "hw_counts": {
                "streamers": [x.to_dict() for x in self.streamer_counts],
                "xbar_inputs": None if self.xbar_inputs is None else list(self.xbar_inputs),
                "bank_shape": None if self.bank_shape is None else list(self.bank_shape)},
        }  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Estimate:
        hw = d.get("hw_counts") or {}
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
            [StreamerCounts.from_dict(x) for x in hw.get("streamers", [])],
            None if hw.get("xbar_inputs") is None else tuple(hw["xbar_inputs"]),
            None if hw.get("bank_shape") is None else tuple(hw["bank_shape"]),
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
    counts, xin, shape = (
        _hw_counts(graph, platform, data) if platform is not None else ([], None, None)
    )
    return Estimate(found, not_mapped(graph), data, size, ports(graph), splits(graph),
                    tech, None if table is None else dict(table.source),
                    not_mapped(graph, spatial=True), l1_words,
                    accelerators(graph, platform), counts, xin, shape)  # fmt: skip


def _hw_counts(graph: Graph, platform: Platform, data: dict[str, ContainerData]
               ) -> tuple[list[StreamerCounts], tuple[int, int] | None, tuple[int, int, int]]:  # fmt: skip
    """HWCOST's counts (D167): per streamer of the shell its FIFO depth, loops, channels and the
    words and beats it moves in L1 over every node of its instance (a shared instance runs once
    per node, D165); the crossbar's inputs (check.py's ``crossbar_inputs``, the one count PORT1
    changes); the banks' count and shape from the platform's L1."""
    from snax_forge.design.check import crossbar_inputs

    insts = instances(graph)
    shell = resolve(platform, graph, insts)
    unit_acc = {u: a for a, units in graph_accelerators(graph).items() for u in units}
    moved: dict[tuple[str, str], list[int]] = {}
    for n, name, port in outer_ports(graph):
        if data[port.container].placement != "l1":  # a link's is not in L1 (D154)
            continue
        w = moved.setdefault((n.attrs["instance"], name), [0, 0])
        w[0] += port.beats * port.num_ch
        w[1] += port.beats
    counts = [StreamerCounts(s.name, unit_acc[s.instance], s.write, s.options.fifo_depth,
                             s.temporal_dims, s.num_ch, *moved.get((s.instance, s.port), (0, 0)))
              for s in shell.values()]  # fmt: skip
    l1 = platform.l1
    return counts, crossbar_inputs(platform, graph, insts), (l1.n_banks, l1.rows, l1.width_bits)


__all__ = [
    "PARTS",
    "UNMAPPED_CYCLES_PER_ITERATION",
    "AcceleratorEstimate",
    "BlockCost",
    "ContainerData",
    "Estimate",
    "EstimateError",
    "GroupCost",
    "GroupEstimate",
    "L1Cost",
    "MemoryCost",
    "Part",
    "PortData",
    "SplitData",
    "StreamerCost",
    "StreamerCounts",
    "StreamerEstimate",
    "XbarCost",
    "ctrl_bits",
    "ctrl_registers",
    "estimate",
    "group_cycles",
]
