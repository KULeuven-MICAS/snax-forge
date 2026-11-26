"""SNAX-ESTIMATE: the ideal, engine-alone estimate of a bound graph (CONTRACTS, "Estimate").

    estimate(graph, platform=None, tech="nangate45") -> Estimate

What a mapping buys before any run, on ideal streams (no bank conflicts, no streamer fill, no
DMA, no core control): per group its cycles, their split and its register bits (cycles.py,
bits.py), per accelerator its streamer set and CSRs (accelerators.py), the data side (data.py)
and, on a technology, the cost (cost.py; ``tech=None`` leaves it out): the dynamic energy in
``PARTS``, the cluster's ``FIXED`` parts in area, every part's leakage (``leakage``, apart from
the energy). The work no unit runs is
``unmapped_iterations`` at ``UNMAPPED_CYCLES_PER_ITERATION`` cycles each, an assumption held in
this one constant so a measured value can replace it; a copy (data_transpose's) at the DMA's
measured rate (copies.py); ``total_cycles`` adds both to the accelerator's. Under a pipelined
``schedule`` the accelerator's cycles are the longest path through the runs (schedule.py),
not their sum; ``serial`` and ``sequential`` add them. The viewers and ``design.md`` read
it (report/design.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

from snax_forge.char import DEFAULT_TECH, load
from snax_forge.char.tables import Costs, load_costs
from snax_forge.common import JsonFile
from snax_forge.design.check import IN_SERIES
from snax_forge.design.streamers import accelerator_of, instances, resolve, shared_movers
from snax_forge.design.streamers import accelerators as graph_accelerators

from .accelerators import AcceleratorEstimate, StreamerEstimate, accelerators
from .bits import ctrl_bits, ctrl_registers
from .copies import DMA_CYCLES_PER_BURST, CopyEstimate, copies
from .cost import (
    FIXED,
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
    leakage_nj,
    memory_cost,
    streamer_cost,
    xbar_cost,
    xbar_kinds,
)
from .cycles import EstimateError, GroupEstimate, group_blocks, group_cycles, not_mapped
from .data import ContainerData, PortData, SplitData, containers, outer_ports, ports, splits
from .schedule import RunEstimate, longest_path, overlap
from .schedule import runs as group_runs

if TYPE_CHECKING:
    from snax_forge.design.platform import Platform
    from snax_forge.dfg import Graph


# The cycles an iteration no unit runs is counted at: an assumption, not a measurement (a
# core takes several cycles per iteration); one constant, so a measured value can replace it.
# NB (D155): an assumption, not a measurement: a core takes several cycles per iteration.
UNMAPPED_CYCLES_PER_ITERATION = 1


@dataclass
class Estimate(JsonFile):
    """The artefact (CONTRACTS, "Estimate")."""

    groups: list[GroupEstimate]
    not_mapped: int
    containers: dict[str, ContainerData]
    l1_size: int | None = None
    ports: dict[str, PortData] = field(default_factory=dict)
    splits: dict[str, SplitData] = field(default_factory=dict)
    tech: str | None = None
    char_source: dict[str, str] | None = None
    unmapped_iterations: int = 0  # not_mapped over every map, spatial ones included
    l1_words: dict[str, int] = field(default_factory=dict)  # per accelerator, its L1 words
    accelerators: list[AcceleratorEstimate] = field(default_factory=list)  # in CSR order
    # The cost model's counts on a platform: per port's streamer, the crossbar's inputs (its
    # channels and the cores' and AXI's ports), the banks (count, depth, width)
    streamer_counts: list[StreamerCounts] = field(default_factory=list)
    xbar_inputs: tuple[int, int] | None = None
    bank_shape: tuple[int, int, int] | None = None
    schedule: str = "serial"  # the design point's
    runs: list[RunEstimate] = field(default_factory=list)  # in start order (schedule.py)
    copies: list[CopyEstimate] = field(default_factory=list)  # data_transpose's (copies.py)
    # The banks a DMA beat on L1 reads or writes: the DMA's beat over the bank's word, the
    # platform's wide_bits / width_bits (a superbank); None without a platform
    superbank_banks: int | None = None

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
        """The cost tables of the technology (snax_forge/char/<tech>/), None without them or a
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
        """The crossbar priced at its inputs (check.py's ``crossbar_inputs``), its readers' and
        writers' channels as read-only and write-only inputs, in the factor form
        (CONTRACTS, "Cost model"); None when not costed."""
        return self.xbar_in("factor")

    def xbar_in(self, form: str) -> XbarCost | None:
        """The crossbar priced as ``xbar`` in ``form`` (``"factor"``, the default, or
        ``"crosspoint"``, cost.py); None when not costed."""
        c = self.costs
        if c is None or self.xbar_inputs is None or self.bank_shape is None:
            return None
        ro, wo = xbar_kinds(self.streamer_counts)
        if ro + wo != self.xbar_inputs[0]:
            raise ValueError(f"the streamers' {ro} read and {wo} write channels are not the "
                             f"crossbar's {self.xbar_inputs[0]} (crossbar_inputs)")  # fmt: skip
        reads, writes = self._rw()
        k = self.copy_bursts
        return xbar_cost(c, sum(self.xbar_inputs), self.bank_shape[0], reads + k, writes + k, ro, wo,
                         form)  # fmt: skip

    @cached_property
    def memory(self) -> MemoryCost | None:
        """The L1 banks priced, the streamers' words and the copies' accesses (copies.py);
        None when not costed."""
        c = self.costs
        if c is None or self.bank_shape is None:
            return None
        reads, writes = self._rw()
        k = self.copy_bursts
        if k and self.superbank_banks is None:
            raise ValueError("the estimate has copies and no superbank_banks: estimate it again "
                             "on its platform")  # fmt: skip
        k *= self.superbank_banks or 0
        return memory_cost(c, *self.bank_shape, reads + k, writes + k)

    @property
    def copy_bursts(self) -> int:
        """The copies' bursts, each one beat: one read and one write on the superbank port, each a
        crossbar access, and a read and a write on every bank of the superbank; and the DMA
        engine's energy per burst (``dma_copy.json``)."""
        return sum(x.bursts for x in self.copies)

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
        """The design's cost in its six parts (``PARTS``), each with its leakage, None with no
        technology; a part not costed has None in it. The DMA's area and leakage are the
        cluster's DMA, its energy the copies' bursts."""
        if self.tech is None:
            return None
        cost = [g.cost for g in self.groups if g.cost]
        blocks = [b for c in cost for b in (*c.units.values(), *c.links.values())]
        s, x, m, c = self.streamer_costs, self.xbar, self.memory, self.costs
        return {
            "datapath": Part(sum(b.total_um2 for b in blocks), sum(b.task_nj for b in blocks),
                             sum(b.leak_uw for b in blocks)),
            "control": Part(sum(g.ctrl_um2 for g in cost), sum(g.ctrl_nj for g in cost),
                            sum(g.ctrl_leak_uw for g in cost)),
            "streamers": Part(sum(y.area_um2 for y in s), sum(y.energy_nj for y in s),
                              sum(y.leak_uw for y in s))
            if c is not None else Part(None, None),
            "crossbar": Part(x.area_um2, x.energy_nj, x.leak_uw) if x else Part(None, None),
            "memory": Part(m.area_um2, m.energy_nj, m.leak_uw) if m else Part(None, None),
            "dma": Part(c.fixed["dma"].area_um2, self.copy_bursts * c.dma_burst_pj / 1000,
                        c.fixed["dma"].leak_uw) if c is not None else Part(None, None),
        }  # fmt: skip

    @property
    def fixed(self) -> dict[str, Part] | None:
        """The cluster's parts every design pays (``FIXED``: the core, the rest), in area and
        leakage only, from ``cluster_fixed.json``; None without the tables."""
        c = self.costs
        if c is None:
            return None
        return {k: Part(c.fixed[k].area_um2, None, c.fixed[k].leak_uw) for k in FIXED}

    @property
    def area(self) -> float | None:
        """µm²: every part costed added up, the banks and the cluster's fixed parts included;
        None with no technology."""
        p = self.parts
        if p is None:
            return None
        return sum(x.area_um2 or 0.0 for x in (*p.values(), *(self.fixed or {}).values()))

    @property
    def area_design(self) -> float | None:
        """µm²: ``area`` without what every variant shares, the banks, the DMA and the
        cluster's fixed parts: the datapath, control, streamers and crossbar."""
        p = self.parts
        if p is None:
            return None
        shared = (p["memory"], p["dma"], *(self.fixed or {}).values())
        return self.area - sum(x.area_um2 or 0.0 for x in shared)

    @property
    def leakage(self) -> dict[str, float] | None:
        """The parts' leakage over ``total_cycles`` at the table's clock: ``power`` (µW,
        the six parts'), ``cycles``, ``clock_ns``, ``energy`` (nJ); apart from ``energy``, which
        stays dynamic. None when not costed."""
        p = self.parts
        if p is None or self.costs is None:
            return None
        power = sum(x.leak_uw or 0.0 for x in p.values())
        clock = load(self.tech or DEFAULT_TECH).clock_ns
        return {"power": power, "cycles": self.total_cycles, "clock_ns": clock,
                "energy": leakage_nj(power, self.total_cycles, clock)}  # fmt: skip

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
            return ["streamers, crossbar, memory, DMA and the cluster: no platform"]
        if self.costs is None:
            return [f"streamers, crossbar, memory, DMA and the cluster: no tables for {self.tech}"]
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
        """The groups' cycles added up, serial or sequential: they run one after another through
        L1; under a pipelined schedule, the longest path through the runs (schedule.py)."""
        if self.schedule in IN_SERIES or not self.runs:
            return sum(g.cycles for g in self.groups)
        return longest_path(self.runs)

    @property
    def overlap_note(self) -> str | None:
        """One line when, pipelined, runs of different accelerators at once move more L1 words a
        cycle than there are banks (design.md, run.md): a bound on the streams, not a cycle
        count."""
        if self.schedule in IN_SERIES or self.bank_shape is None:
            return None
        rate, accs = overlap(self.runs)
        if rate <= self.bank_shape[0]:
            return None
        return (f"{' and '.join(accs)} run at once under {self.schedule}: their streams move "
                f"{rate:.1f} words a cycle over {self.bank_shape[0]} banks, so bank conflicts "
                "stretch them beyond the ideal")  # fmt: skip

    @property
    def unmapped_cycles(self) -> int:
        """The work no unit runs, as cycles: ``unmapped_iterations`` (over every map, a tag alone
        runs nothing) at ``UNMAPPED_CYCLES_PER_ITERATION`` (an assumption)."""
        return self.unmapped_iterations * UNMAPPED_CYCLES_PER_ITERATION

    @property
    def copy_cycles(self) -> int:
        """The copies' cycles (copies.py), added up and rounded; under a pipelined schedule
        added as well, for now."""
        return round(sum(x.cycles for x in self.copies))

    @property
    def total_cycles(self) -> int:
        """The accelerator's cycles plus the unmapped ones and the copies'."""
        return self.accelerator_cycles + self.unmapped_cycles + self.copy_cycles

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
            "copies": [x.to_dict() for x in self.copies],
            "copy_cycles": self.copy_cycles,
            "total_cycles": self.total_cycles,
            "schedule": self.schedule,
            "runs": [r.to_dict() for r in self.runs],
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
                "fixed": {k: v.to_dict() for k, v in (self.fixed or {}).items()},
                "leakage": self.leakage,
                "xbar": None if xbar is None else xbar.to_dict(),
                "memory": None if memory is None else memory.to_dict(),
                "not_costed": self.not_costed, "extrapolated": self.extrapolated},
            "hw_counts": {
                "streamers": [x.to_dict() for x in self.streamer_counts],
                "xbar_inputs": None if self.xbar_inputs is None else list(self.xbar_inputs),
                "bank_shape": None if self.bank_shape is None else list(self.bank_shape),
                "superbank_banks": self.superbank_banks},
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
            d.get("schedule", "serial"),
            [RunEstimate.from_dict(r) for r in d.get("runs", [])],
            [CopyEstimate.from_dict(x) for x in d.get("copies", [])],
            hw.get("superbank_banks"),
        )  # fmt: skip


def estimate(graph: Graph, platform: Platform | None = None,
             tech: str | None = DEFAULT_TECH, schedule: str = "serial") -> Estimate:  # fmt: skip
    """The ideal estimate of a bound graph (its symbols bound), on ``platform``'s L1 if given,
    costed on ``tech``'s table unless None, its accelerators ordered by ``schedule`` (a design
    point's)."""
    blocks = group_blocks(graph, platform.acc_csr.job_width if platform is not None else 32)
    found = [g for g, _, _ in blocks]
    data = containers(graph)
    node_words: dict[str, int] = {}
    for n, _, port in outer_ports(graph):
        if data[port.container].placement == "l1":
            node_words[n.id] = node_words.get(n.id, 0) + port.beats * port.num_ch
    in_l1 = {c for c, x in data.items() if x.placement == "l1"}
    run_list = group_runs(graph, found, in_l1, node_words)
    table = None if tech is None else load(tech)
    if table is not None:  # each control unit clocked over its own group's cycles
        for g, units, bufs in blocks:
            g.cost = group_cost(table, units, bufs, g.bits, g.ctrl_bits, g.cycles)
    unit_group = {u: g for g in found for u in g.units}
    words = dict.fromkeys((g.name for g in found), 0)
    l1_words: dict[str, int] = {}
    for n, _, port in outer_ports(graph):
        words[unit_group[n.attrs["instance"]].name] += port.beats * port.num_ch
        acc = l1_words.setdefault(accelerator_of(n), 0)
        if data[port.container].placement == "l1":  # a link's is not in L1
            l1_words[accelerator_of(n)] = acc + port.beats * port.num_ch
    for g in found:
        g.words_per_cycle = words[g.name] / g.cycles if g.cycles else 0.0
        if platform is not None:  # the fewest cycles L1's banks allow
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
                    accelerators(graph, platform), counts, xin, shape, schedule, run_list,
                    copies(graph), None if l1 is None else l1.wide_bits // l1.width_bits)  # fmt: skip


def _hw_counts(graph: Graph, platform: Platform, data: dict[str, ContainerData]
               ) -> tuple[list[StreamerCounts], tuple[int, int] | None, tuple[int, int, int]]:  # fmt: skip
    """The cost model's counts: per streamer of the shell its FIFO depth, loops, channels and the
    words and beats it moves in L1 over every node of its instance (a shared instance runs once
    per node); the crossbar's inputs (check.py's ``crossbar_inputs``, the one count the port sharing
    changes); the banks' count and shape from the platform's L1."""
    from snax_forge.design.check import crossbar_inputs

    insts = instances(graph)
    shell = resolve(platform, graph, insts)
    unit_acc = {u: a for a, units in graph_accelerators(graph).items() for u in units}
    moved: dict[tuple[str, str], list[int]] = {}
    for n, name, port in outer_ports(graph):
        if data[port.container].placement != "l1":  # a link's is not in L1
            continue
        w = moved.setdefault((n.attrs["instance"], name), [0, 0])
        w[0] += port.beats * port.num_ch
        w[1] += port.beats
    shared = {u.streamer: (m, i == 0) for movers in shared_movers(graph, shell).values()
              for m in movers for i, u in enumerate(m.users)}  # fmt: skip
    counts = []
    for s in shell.values():
        got = moved.get((s.instance, s.port), (0, 0))
        if s.name in shared:  # on a shared mover: its depth, loops and channels
            m, first = shared[s.name]
            counts.append(StreamerCounts(s.name, unit_acc[s.instance], s.write, m.fifo_depth,
                                         m.temporal_dims, s.num_ch, *got, m.name, m.num_ch,
                                         first))  # fmt: skip
        else:
            counts.append(StreamerCounts(s.name, unit_acc[s.instance], s.write,
                                         s.options.fifo_depth, s.temporal_dims, s.num_ch, *got))  # fmt: skip
    l1 = platform.l1
    return counts, crossbar_inputs(platform, graph, insts), (l1.n_banks, l1.rows, l1.width_bits)


__all__ = [
    "DMA_CYCLES_PER_BURST",
    "PARTS",
    "UNMAPPED_CYCLES_PER_ITERATION",
    "AcceleratorEstimate",
    "BlockCost",
    "ContainerData",
    "CopyEstimate",
    "Estimate",
    "EstimateError",
    "GroupCost",
    "GroupEstimate",
    "L1Cost",
    "MemoryCost",
    "Part",
    "PortData",
    "RunEstimate",
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
