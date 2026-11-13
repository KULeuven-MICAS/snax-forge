"""The design report, ``design.md``: what was built, before any cycle is counted.

Everything here comes from the design point, its recipe and the cluster as a
run names it (run/cluster.py: the generator's description files and the design
point, D137), its regions as run/regions.py gives them (L2's from the ELF's symbols in
run.json); SNAX-GVSoC is not run. A flow from a bound graph has no recipe:
its header names the graph the design point was made from instead, and takes
the kernel from the run's functional check.

    header       kernel, recipe and its params (or the graph), symbols,
                 platform (base and changes), memory plan passes and pins
    components   the accelerator, its movers, the DMA and the cores
    accelerators per accelerator, one streamer set each (D157): its groups, its
                 streamer's movers, channels (num_ch added up) and temporal
                 dims, its CSRs (its streamer's, then its own, first to last)
                 and each group's own CSRs; then per unit its instance, unit,
                 BRM and implementation, params, latency, target II (the
                 datapath file's), ports
    streamers    one per mover: instance port, direction, the container's
                 dtype, num_slc (its slices, elements, per 64-bit word),
                 num_ch (the channels the port uses, the mover's
                 num_channel: words per beat, D142), FIFO depth,
                 temporal loops and the container it streams (from the
                 graph's memlets)
    memories     per memory the design point's layouts: base, end (the extent
                 the memory checks use, padding included: run.padding_of),
                 bytes, share, padding (a tail's pad elements and the bytes
                 to a whole DMA beat, D144), banks (L1) or words of a beat
                 (L2), rows or beats
    estimate     the ideal estimate (D155, C§19): per group its cycles, bottleneck,
                 ramp, stall and its expected BUSY_COUNTER (the cycles plus the streamers'
                 fill), a guide and not a check, a bank-bound note per group whose
                 bank_bound (its streams' L1 time) exceeds its cycles and, on the
                 estimate's technology, its area and
                 energy; then the accelerator's cycles, the work no unit runs as
                 cycles (one per iteration, an assumption, D155), their total and the
                 L1 footprint against the L1; then the cost (C§20, D167): per unit,
                 buffer link and control unit how it is costed, per port its
                 streamer, per accelerator its words through the crossbar and the
                 banks, the cost by part (datapath, control, streamers, crossbar,
                 memory) with each part's share, the total and the total without
                 the banks, the technology and its tables' sources, the blocks the
                 table does not characterize, the parts not costed and the points
                 extrapolated, and the streamer sets and CSRs
    notes        the design notes first (check.py's ``run_notes``: a design that
                 passes but is worth a second look, with its fix); then
                 facts about addresses: movers of one group whose
                 containers start in the same banks, or lie a whole number of
                 bank rows apart, so that element i of both is in one bank
                 (two groups run one after the other, C§14);
                 and containers the DMA moves as padded beats
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from snax_forge.char import DEFAULT_TECH
from snax_forge.estimate.cost import L1_LABEL, PARTS, caveat, dec3, heading, shares, sig3
from snax_forge.lower.layout import Layout, round_up
from snax_forge.run import Cluster, Region, padding_of, regions_of

from .markdown import grouped, kib, kv, pct, span_text, spans, table
from .record import Record


@dataclass
class Header(Record):
    kernel: str
    recipe: str
    params: dict[str, Any]
    symbols: dict[str, Any]
    platform: str
    platform_base: str | None
    platform_changes: dict[str, Any]
    passes: dict[str, str]
    pins: dict[str, Any]
    graph_from: str = ""  # a flow from a bound graph: the graph, in place of a recipe


@dataclass
class ComponentRow(Record):
    name: str
    kind: str
    detail: str


@dataclass
class AccelRow(Record):
    instance: str
    kind: str
    brm: str | None
    implementation: str | None
    params: dict[str, Any]
    latency: int
    target_ii: int
    ports: list[str]
    group: str | None = None


@dataclass
class SetRow(Record):
    """One accelerator, its streamer set (D157): its groups, its streamer's movers, channels and
    temporal dims, its CSRs first to last and each group's own CSRs, first to last."""

    name: str
    groups: list[str]
    movers: int
    channels: int
    temporal_dims: list[int]
    first: int
    last: int
    group_csrs: dict[str, list[int]] = field(default_factory=dict)  # [first, last] per group
    shared: bool = False  # its groups share its movers (share_ports, D169)


@dataclass
class StreamerRow(Record):
    name: str
    instance: str | None
    port: str | None
    direction: str
    dtype: str
    num_slc: int
    num_ch: int
    fifo_depth: int
    temporal_dims: int
    container: str | None


@dataclass
class RegionRow(Record):
    name: str
    base: int
    end: int  # the extent the memory checks use, padding included
    bytes: int
    share: float
    banks: str  # columns touched, as spans ("0–15")
    rows: str
    data_end: int  # the padded block's end
    pad_elements: int  # a tail's pads
    beat_pad_bytes: int  # up to a whole DMA beat

    @property
    def padding(self) -> str:
        parts = [f"{self.pad_elements} pad element{'s' if self.pad_elements != 1 else ''}"
                 if self.pad_elements else "",
                 f"{self.beat_pad_bytes} B to a whole beat" if self.beat_pad_bytes else ""]  # fmt: skip
        return ", ".join(p for p in parts if p) or "none"


@dataclass
class MemoryTable(Record):
    mem: str
    columns: int
    rows: int
    word_bytes: int
    size_bytes: int
    used_bytes: int
    regions: list[RegionRow] = field(default_factory=list)


@dataclass
class EstimateRow(Record):
    group: str
    cycles: int
    bottleneck: int
    ramp: int
    stall: int
    counter: int
    area_um2: float | None = None
    energy_nj: float | None = None


@dataclass
class CostRow(Record):
    """One unit, buffer link or control unit of the cost (C§20); ``found`` None for a block
    the table does not hold; ``e_fire_pj`` and ``e_hold_pj`` None for a control unit (its comb
    area equal to its flops', its comb energy negligible: a placeholder, D154), ``e_hold_pj``
    None for a block with no flops."""

    group: str
    block: str
    datapath: str | None
    found: str | None
    fires: int | None
    bits: int
    comb_um2: float | None
    flops_um2: float
    e_fire_pj: float | None
    e_hold_pj: float | None
    energy_nj: float


@dataclass
class L1Row(Record):
    """One accelerator's words in L1 (C§20, D167): through the crossbar and the banks."""

    accelerator: str
    words: int
    bits: int
    energy_nj: float
    reads: int = 0
    writes: int = 0
    xbar_nj: float = 0.0
    banks_nj: float = 0.0


@dataclass
class StreamerCostRow(Record):
    """One port's streamer priced (C§20, D167); ``extrapolated`` the axes off the grid."""

    accelerator: str
    streamer: str
    direction: str
    depth: int
    loops: int
    channels: int
    beats: int
    area_um2: float
    e_beat_pj: float
    energy_nj: float
    extrapolated: list[str] = field(default_factory=list)


@dataclass
class PartRow(Record):
    """One part of the design's cost (D167) and its share of the totals; None where not
    costed."""

    part: str
    area_um2: float | None
    energy_nj: float | None
    area_share: float | None
    energy_share: float | None


@dataclass
class EstimateTable(Record):
    groups: list[EstimateRow]
    accelerator_cycles: int
    not_mapped: int
    l1_footprint: int
    l1_size: int | None
    tech: str | None = None
    char_source: dict[str, str] = field(default_factory=dict)
    area_um2: float | None = None
    energy_nj: float | None = None
    missing: list[str] = field(default_factory=list)
    cost: list[CostRow] = field(default_factory=list)
    bank_notes: list[str] = field(default_factory=list)  # a group's bank bound over its cycles
    unmapped_cycles: int = 0  # unmapped_iterations at UNMAPPED_CYCLES_PER_ITERATION (D155)
    unmapped_iterations: int = 0  # over every map, spatial ones included
    total_cycles: int = 0  # accelerator_cycles + unmapped_cycles
    l1: list[L1Row] = field(default_factory=list)  # per accelerator, its words in L1
    streamers: int = 1  # the streamer sets, one per accelerator
    csrs: int | None = None  # every CSR of the entry; None without a platform
    parts: list[PartRow] = field(default_factory=list)  # the cost by part (D167)
    area_design_um2: float | None = None  # the area without the banks
    streamer_cost: list[StreamerCostRow] = field(default_factory=list)
    xbar: dict[str, Any] | None = None  # its inputs, banks, accesses, area and energy
    memory: dict[str, Any] | None = None  # its banks, shape, reads, writes, area and energy
    not_costed: list[str] = field(default_factory=list)
    extrapolated: list[str] = field(default_factory=list)


@dataclass
class DesignReport(Record):
    name: str
    header: Header | None
    components: list[ComponentRow]
    accelerators: list[AccelRow]
    streamers: list[StreamerRow]
    memories: list[MemoryTable]
    notes: list[str]
    groups: int = 1  # the accelerator's; with more than one, the accelerators' table names each's
    estimate: EstimateTable | None = None
    sets: list[SetRow] = field(default_factory=list)  # per accelerator (streamer set)


# =============================================================================
# Building
# =============================================================================


def _words(lay: Layout, base: int, word: int) -> np.ndarray:
    """The word index from ``base`` of every element of a layout (several narrow elements
    share a word)."""
    idx = np.indices(lay.shape).reshape(len(lay.shape), -1).T
    return (lay.address(idx) - base) // word


def _memory_tables(cluster: Cluster, regions: list[Region], point: Any) -> list[MemoryTable]:
    """L1 as banks by rows, L2 as the words of a beat by beats; the run's regions."""
    word = cluster.bank_bytes
    l2 = point.platform.l2
    geo = {
        "l1": (cluster.l1_base, cluster.banks, cluster.l1_size // (word * cluster.banks)),
        "l2": (l2.base_addr, l2.beat_bits // 8 // word, l2.size_bytes // (l2.beat_bits // 8)),
    }
    out = []
    for mem, (base, columns, rows) in geo.items():
        size = columns * rows * word
        region_rows = []
        for region in regions:
            if region.memory != mem:
                continue
            name, lay = region.container, region.layout
            w = _words(lay, base, word)
            n = int(w.size) * region.itemsize
            lo, _ = lay.extent(region.itemsize)
            pad = padding_of(point, region, cluster.dma_beat_bytes)
            beat_pad = pad.beat_pad[1] - pad.beat_pad[0] if pad.beat_pad else 0
            region_rows.append(RegionRow(name, lo, pad.end, n, n / size,
                                     spans(sorted({int(x) % columns for x in w})),
                                     span_text([int(w.min()) // columns, int(w.max()) // columns]),
                                     pad.data_end, len(pad.pads), beat_pad))  # fmt: skip
        region_rows.sort(key=lambda r: r.base)
        out.append(MemoryTable(mem, columns, rows, word, size, sum(r.bytes for r in region_rows),
                               region_rows))  # fmt: skip
    return out


def _notes(cluster: Cluster, regions: list[Region], point: Any,
           streamers: list[StreamerRow]) -> list[str]:  # fmt: skip
    """Address facts about the movers of the accelerator (module doc), the bank notes over the
    streamers of one group only: an accelerator's groups run one after the other (D148, C§14)."""
    word, banks = cluster.bank_bytes, cluster.banks
    group_of = cluster.accelerator.unit_of_mover
    row_bytes = word * banks
    l1 = {r.container: r.layout for r in regions if r.memory == "l1"}
    elem = {r.container: r.itemsize for r in regions if r.memory == "l1"}
    lays = {s.container: l1[s.container] for s in streamers if s.container in l1}
    ss = [s for s in streamers if s.container in lays]
    notes = []
    for i, a in enumerate(ss):
        for b in ss[i + 1 :]:
            if group_of(a.name) != group_of(b.name):
                continue  # two groups never stream at once
            la, lb = lays[a.container], lays[b.container]  # type: ignore[index]

            def first_banks(lay: Layout, num_ch: int) -> list[int]:
                """The banks of a mover's first beat: its first ``num_ch`` words (several
                narrow elements share one)."""
                w = list(dict.fromkeys(int(x) for x in _words(lay, cluster.l1_base, word)))
                return sorted({x % banks for x in w[:num_ch]})

            both = sorted(set(first_banks(la, a.num_ch)) & set(first_banks(lb, b.num_ch)))
            if (
                la.strides == lb.strides
                and la.shape == lb.shape
                and (lb.base - la.base) % row_bytes == 0
            ):
                notes.append(
                    f"{a.name} and {b.name}: {a.container} and {b.container} lie "
                    f"{abs(lb.base - la.base)} B apart, a whole number of {row_bytes}-byte bank "
                    f"rows, so {a.container}[i] and {b.container}[i] are in the same bank for every i."
                )
            elif both:
                notes.append(f"{a.name} ({a.container}) and {b.name} ({b.container}) start in "
                             f"the same banks, {spans(both)}.")  # fmt: skip
    if not notes and ss:
        several = len(cluster.accelerator.units) > 1
        notes.append(f"No two streamers of {'one group' if several else 'the accelerator'} start "
                     "in the same banks.")  # fmt: skip
    beat = point.platform.l2.beat_bits // 8
    l2 = {r.container for r in regions if r.memory == "l2"}
    for name, lay in l1.items():
        if name not in l2:
            continue
        size = int(np.prod(lay.shape)) * elem[name]
        if size % beat:
            moved = round_up(size, beat)
            base = lay.base
            notes.append(
                f"{name}: {size} B, moved by the DMA as {moved // beat} whole {beat}-byte "
                f"beat{'s' if moved > beat else ''}; bytes {base + size}–{base + moved} are "
                "kept free in L1 and L2."
            )
    return notes


def file_groups(file: dict[str, Any]) -> list[dict[str, Any]]:
    """accelerator.json's groups, accelerator by accelerator (C§17, D157)."""
    return [g for a in file["accelerators"] for g in a["groups"]]


def members(file: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """The slot's accelerator.json units the run's unit ``name`` stands for: a linked group's
    units (SNAX-GVSoC's one component, named after the group), else the unit itself."""
    groups = file_groups(file)
    for g in groups:
        if g["name"] == name and len(g["units"]) > 1:
            return list(g["units"])
    return [next(u for g in groups for u in g["units"] if u["name"] == name)]


def unit_view(file: dict[str, Any], instance: str) -> tuple[str, str, int, int]:
    """The kind, the control, the latency and the II of ``instance`` in the slot's
    accelerator.json: its unit's datapath under the nest controller, or a linked group's
    datapaths in chain order, each link a wire or a buffer of its depth, under the group's
    control unit (the latency the chain's)."""
    units = members(file, instance)
    if len(units) == 1:
        (u,) = units
        return u["datapath"], "nest controller", u["timing"]["latency"], u["timing"]["ii"]
    (g,) = (g for g in file_groups(file) if g["name"] == instance)
    kind = units[0]["datapath"]
    for link, u in zip(g["links"], units[1:], strict=True):
        kind += f" → {u['datapath']}" if not link["depth"] else (
            f" →[{link['depth']}] {u['datapath']}")  # fmt: skip
    return kind, "group control unit", sum(u["timing"]["latency"] for u in units), 1


def build_design(
    name: str,
    cluster: Cluster,
    point: Any,
    recipe: Any = None,
    kernel: str = "",
    elf_symbols: dict[str, Any] | None = None,
    tech: str | None = DEFAULT_TECH,
) -> DesignReport:
    """The design report of a flow (``point`` its DesignPoint, ``recipe`` its recipe, or for a
    flow from a bound graph its ``kernel``; ``elf_symbols`` its run.json's; ``tech`` the
    estimate's technology, None for no cost)."""
    from snax_forge.design import note_text, run_checks, run_notes
    from snax_forge.design.streamers import accelerated, streamer_name

    regions = regions_of(point, elf_symbols)
    design = point.design()
    run_checks(design)  # resolves the instances (BRM, implementation, params)
    instances = dict(design.instances)
    design_notes = [note_text(n).replace("\n    fix: ", " Fix: ") for n in run_notes(design)]
    header = Header(
        kernel=recipe.kernel if recipe is not None else kernel or point.graph.name,
        recipe=recipe.name if recipe is not None else point.name,
        params=dict(recipe.params) if recipe is not None else {},
        symbols=point.graph.bound_symbols,
        platform=point.platform.name,
        platform_base=point.platform.base,
        platform_changes=dict(point.platform.changes),
        passes=dict(point.memory.passes),
        pins=dict(point.memory.changes),
        graph_from="" if recipe is not None else point.graph_from,
    )
    containers: dict[str, str] = {}
    for n in accelerated(point.graph):
        for side in (n.inputs, n.outputs):
            for port, m in side.items():
                containers[streamer_name(n.attrs["instance"], port)] = m.data

    a = cluster.accelerator
    several = len(a.units) > 1
    views = {u.name: unit_view(a.description, u.name) for u in a.units}
    components = [ComponentRow(u.name, "accelerator",
                               f"{views[u.name][0]} on core{a.core}, {views[u.name][1]}, slot "
                               f"{a.slot}" + (f", group {u.group} of {a.name}" if several else ""))
                  for u in a.units]  # fmt: skip
    components += [ComponentRow(m.name, "streamer", f"{'writer' if m.write else 'reader'} "
                                f"{m.gvsoc}, num_ch {m.lanes}, FIFO depth {m.fifo_depth}")
                   for m in a.movers]  # fmt: skip
    components.append(ComponentRow("dma", "dma", f"the iDMA on core{cluster.dma_core}"))
    components += [ComponentRow(c, "core", "the accelerator's" if i == a.core
                                else "the DMA's" if i == cluster.dma_core else "")
                   for i, c in enumerate(cluster.cores)]  # fmt: skip

    accels = []
    for u in a.units:
        for m in members(a.description, u.name):  # a linked group: a row per unit, in chain order
            inst = instances.get(m["name"])
            accels.append(AccelRow(m["name"], m["datapath"], inst.brm.name if inst else None,
                                   inst.implementation if inst else None,
                                   dict(inst.params) if inst else {}, m["timing"]["latency"],
                                   m["timing"]["ii"],
                                   [f"{x.port} ← {x.name}" for x in a.movers
                                    if (m["name"] in x.user_units if x.users
                                        else point.streamers[x.name].instance == m["name"])],
                                   u.group))  # fmt: skip
    streamers = []
    for m in a.movers:
        if m.users:  # a shared mover (D169): its first user's dtype, no one port or container
            ref = point.streamers[m.users[0]]
            streamers.append(StreamerRow(m.name, None, m.port, "write" if m.write else "read",
                                         ref.dtype, ref.num_slc, m.lanes, m.fifo_depth,
                                         m.temporal_dim, None))  # fmt: skip
            continue
        ref = point.streamers[m.name]
        streamers.append(StreamerRow(m.name, ref.instance, m.port, "write" if m.write else "read",
                                     ref.dtype, ref.num_slc, m.lanes, m.fifo_depth,
                                     ref.temporal_dims, containers.get(m.name)))  # fmt: skip
    sets = []
    for acc_name, groups in a.accelerators():
        movers = [m for m in a.movers if m.set == acc_name]
        first, last = a.window(acc_name)
        sets.append(SetRow(acc_name, groups, len(movers), sum(m.lanes for m in movers),
                           [m.temporal_dim for m in movers], first, last,
                           {u.group: [u.start, u.counter] for u in a.units if u.group in groups},
                           any(m.users for m in movers)))  # fmt: skip
    return DesignReport(name, header, components, accels, streamers,
                        _memory_tables(cluster, regions, point),
                        design_notes + _notes(cluster, regions, point, streamers), len(a.units),
                        estimate_table(point, tech), sets)  # fmt: skip


def estimate_table(point: Any, tech: str | None = DEFAULT_TECH) -> EstimateTable:
    """The design point's ideal estimate (D155) as the report shows it, costed on ``tech``."""
    from snax_forge.estimate import estimate

    e = estimate(point.graph, point.platform, tech)
    rows = [EstimateRow(g.name, g.cycles, g.bottleneck, g.ramp, g.stall, g.counter,
                        g.cost.area if g.cost else None, g.cost.energy if g.cost else None)
            for g in e.groups]  # fmt: skip
    cost = []
    for g in e.groups:
        if g.cost is None:
            continue
        for k, b in (*g.cost.units.items(), *g.cost.links.items()):
            cost.append(CostRow(g.name, k, b.datapath, b.found, b.fires, g.bits.get(k, 0),
                                b.comb_um2, b.flops_um2, b.e_fire_pj, b.e_hold_pj or None,
                                b.task_nj))  # fmt: skip
        c = g.cost
        cost.append(CostRow(g.name, "ctrl", "group control unit", None, None, c.ctrl_bits,
                            c.ctrl_comb_um2, c.ctrl_flops_um2, None, None, c.ctrl_nj))  # fmt: skip
    notes = [g.bank_bound_note for g in e.groups if g.bank_bound_note]
    got = e.parts
    share = shares(got) if got else {}
    parts = [PartRow(k, got[k].area_um2, got[k].energy_nj, *share[k]) for k in PARTS] if got else []
    return EstimateTable(rows, e.accelerator_cycles, e.not_mapped, e.l1_footprint, e.l1_size,
                         e.tech, dict(e.char_source or {}), e.area, e.energy, e.missing, cost,
                         notes, unmapped_cycles=e.unmapped_cycles,
                         unmapped_iterations=e.unmapped_iterations,
                         total_cycles=e.total_cycles,
                         l1=[L1Row(a, x.words, x.bits, x.energy_nj, x.reads, x.writes, x.xbar_nj,
                                   x.banks_nj) for a, x in e.l1.items()],
                         streamers=e.streamers, csrs=e.csrs, parts=parts,
                         area_design_um2=e.area_design,
                         streamer_cost=[StreamerCostRow(
                             c.counts.accelerator, c.counts.name,
                             "write" if c.counts.write else "read", c.counts.depth,
                             c.counts.loops, c.counts.channels, c.counts.beats, c.area_um2,
                             c.e_beat_pj, c.energy_nj, list(c.extrapolated))
                             for c in e.streamer_costs],
                         xbar=None if e.xbar is None else e.xbar.to_dict(),
                         memory=None if e.memory is None else e.memory.to_dict(),
                         not_costed=e.not_costed, extrapolated=e.extrapolated)  # fmt: skip


# =============================================================================
# Rendering
# =============================================================================


def render_design(r: DesignReport) -> str:
    """``design.md``: the report as Markdown, one section per part."""
    out = [f"# Design report: {r.name}", ""]
    if r.header is not None:
        h = r.header
        base = f", base {h.platform_base}" if h.platform_base else ""
        out += table(
            ["", ""],
            [
                ["Kernel", h.kernel],
                ["Graph", h.graph_from]
                if h.graph_from
                else ["Recipe", f"{h.recipe} ({kv(h.params)})"],
                ["Symbols", kv(h.symbols)],
                ["Platform", f"{h.platform}{base}; changes: {kv(h.platform_changes)}"],
                ["Memory plan", f"{kv(h.passes)}; pins: {kv(h.pins)}"],
            ],
        )
        out.append("")
    out += ["## Cluster", ""]
    out += table(
        ["Component", "Kind", "Details"], [[c.name, c.kind, c.detail] for c in r.components]
    )
    out += ["", "## Accelerators", ""]
    out += render_sets(r.sets)
    out += ["", "### Units", ""]
    out += grouped(r.groups, 1, table(
        [
            "Instance",
            "Kind",
            "BRM",
            "Implementation",
            "Params",
            "Latency",
            "Target II",
            "Ports",
        ],
        [
            [
                a.instance,
                a.kind,
                a.brm,
                a.implementation,
                kv(a.params),
                a.latency,
                a.target_ii,
                "; ".join(a.ports),
            ]
            for a in r.accelerators
        ],
        {5, 6},
    ), [a.group for a in r.accelerators])  # fmt: skip
    out += ["", "## Streamers", ""]
    out += table(
        [
            "Streamer",
            "Serves",
            "Direction",
            "dtype",
            "num_slc",
            "num_ch",
            "FIFO depth",
            "Temporal loops",
            "Container",
        ],
        [
            [
                s.name,
                f"{s.instance}.{s.port}" if s.instance else None,
                s.direction,
                s.dtype,
                s.num_slc,
                s.num_ch,
                s.fifo_depth,
                s.temporal_dims,
                s.container,
            ]
            for s in r.streamers
        ],
        {4, 5, 6, 7},
    )
    for m in r.memories:
        if m.mem == "l1":
            title = f"{m.columns} banks × {m.rows} rows of {m.word_bytes}-byte words"
        else:
            title = f"{m.columns * m.rows} words, as {m.rows} beats of {m.columns} words"
        out += ["", f"## {m.mem.upper()}: {title}, {kib(m.size_bytes)}", ""]
        if not m.regions:
            out.append("No region lives here.")
            continue
        out += table(
            [
                "Region",
                "Base",
                "End",
                "Bytes",
                "Share",
                "Padding",
                "Banks" if m.mem == "l1" else "Words",
                "Rows" if m.mem == "l1" else "Beats",
            ],
            [
                [x.name, x.base, x.end, x.bytes, pct(x.share), x.padding, x.banks, x.rows]
                for x in m.regions
            ],
            {1, 2, 3, 4},
        )
        out += [
            "",
            f"Used: {m.used_bytes} of {m.size_bytes} bytes ({pct(m.used_bytes / m.size_bytes)}).",
        ]
    if r.estimate is not None:
        out += ["", "## Estimate", ""]
        out += render_estimate(r.estimate)
    out += ["", "## Notes", ""]
    out += [f"- {n}" for n in r.notes] or ["- None."]
    return "\n".join(out) + "\n"


def cycles_line(e: EstimateTable) -> str:
    """The Estimate's cycles (D155): the accelerator's, the unmapped work's at one cycle per
    iteration (an assumption; left out at 0), the total."""
    from snax_forge.estimate import UNMAPPED_CYCLES_PER_ITERATION as PER

    unmapped = ""
    if e.unmapped_cycles:
        each = f"{PER} cycle{'' if PER == 1 else 's'} each"
        unmapped = (f" Unmapped: {e.unmapped_cycles} cycles ({e.unmapped_iterations} iterations at "
                    f"{each}, an assumption).")  # fmt: skip
    return f"Accelerator cycles: {e.accelerator_cycles}.{unmapped} Total: {e.total_cycles}."


def render_sets(rows: list[SetRow]) -> list[str]:
    """The accelerators, one streamer set each (D157)."""
    out = table(
        ["Accelerator", "Groups", "Movers", "Channels", "Temporal dims", "CSRs", "Groups' CSRs"],
        [[x.name, ", ".join(x.groups), x.movers, x.channels,
          ", ".join(map(str, x.temporal_dims)), f"{x.first}–{x.last}",
          "; ".join(f"{g} {a}–{b}" for g, (a, b) in x.group_csrs.items())] for x in rows],
        {2, 3},
    )  # fmt: skip
    out += ["", ("Each accelerator is one streamer set: its streamer's CSRs first, then its own, "
                 "packed from 960 (C§16); its movers are its groups' outer ports, readers "
                 "first.")]  # fmt: skip
    shared = [x.name for x in rows if x.shared]
    if shared:
        out += ["", (f"{', '.join(shared)} {'shares' if len(shared) == 1 else 'share'} its movers "
                     "among its groups (share_ports, D169): per role the k-th widest port of each "
                     "group on mover k, a narrower one on its first channels; a port mux picks the "
                     "group by PORT_MUX_SEL, which the program writes before each start. The mux "
                     "costs nothing here.")]  # fmt: skip
    return out


def render_estimate(e: EstimateTable) -> list[str]:
    """The estimate section: per group its cycles, then the accelerator's (D155)."""
    out = [("The ideal estimate (D155): the group rule on ideal streams, with no bank conflicts, "
            "streamer fill, DMA or core control. The expected BUSY_COUNTER, the cycles plus the "
            "streamers' fill (3, or 2 for a group with one outer input), is a guide, not a check: "
            "a run's BUSY_COUNTER counts from its group's START until busy drops, so it includes bank "
            "conflicts and is shortened by the streamers' head start; its window is longer."),
           ""]  # fmt: skip
    costed = e.tech is not None
    cols = ["Group", "Cycles", "Bottleneck", "Ramp", "Stall", "Expected BUSY_COUNTER"]
    cols += ["Area (µm²)", "Energy (nJ)"] if costed else []
    out += table(cols, [[g.group, g.cycles, g.bottleneck, g.ramp, g.stall, g.counter]
                        + ([_shown(g.area_um2), _energy(g.energy_nj)] if costed else [])
                        for g in e.groups], set(range(1, len(cols))))  # fmt: skip
    l1 = f"{e.l1_footprint} bytes"
    if e.l1_size is not None:
        l1 += f" of {e.l1_size} ({pct(e.l1_footprint / e.l1_size)})"
    out += ["", f"{cycles_line(e)} L1 footprint: {l1}."]
    if e.bank_notes:  # the bank bound over the ideal cycles (D155), a line per group
        out += ["", ("Bank bound: L1's single-port banks serve one word each per cycle, so these "
                     "groups' streams take more cycles of L1 time than the ideal cycles (a bound "
                     "on the streams, not on the BUSY_COUNTER):"), ""]  # fmt: skip
        out += [f"- {n}." for n in e.bank_notes]
    if costed:
        out += ["", *render_cost(e)]
    return out


def _shown(x: float | None, unit: str = "") -> str | None:
    """An area as shown (C§20): 3 significant figures."""
    v = sig3(x)
    return None if v is None else f"{v}{unit}"


def _energy(x: float | None, unit: str = "") -> str | None:
    """An energy as shown (C§20): 3 decimal places in its unit."""
    v = dec3(x)
    return None if v is None else f"{v}{unit}"


def render_cost(e: EstimateTable) -> list[str]:
    """The cost subsection (C§20, D167): how each block is costed, each port's streamer, each
    accelerator's words in L1, then the cost by part with its shares and the totals; an area
    to 3 significant figures, an energy to 3 decimals."""
    src = e.char_source.get("snax_forge_experiments", "")[:7]
    out = [f"### {heading(e.tech)}", "", caveat(), "",
           (f"CHAR1's model (D154, C§20) on `snax_forge/char/{e.tech}.json` (snax-forge-experiments "
            f"{src}), per run of the accelerator: a unit or buffer link pays its comb energy per "
            "fire, from the table as measured, copied (lanes × the lanes-1 point) or "
            "interpolated, its register bits at e_load in the cycles it fires and e_hold in every "
            "other cycle; a control unit its bits × e_hold × the counter factor every cycle, its "
            "area its flops' and an equal comb area (a rough estimate). HWCOST (D167) on "
            f"`snax_forge/char/{e.tech}/`: each port's streamer its area and its beats × its "
            "energy per beat at its FIFO depth, loops and channels; the crossbar its area and "
            "every word through it × its energy per access at its inputs and banks; the L1 banks "
            "their area and their reads and writes × a bank's energy per read and write (a "
            "link's container is not in L1, the DMA's traffic not counted)."),
           ""]  # fmt: skip
    out += table(["Group", "Block", "Datapath", "Found", "Fires", "Bits", "Comb (µm²)",
                  "Flops (µm²)", "e_fire (pJ)", "Hold (pJ/cycle)", "Energy (nJ)"],
                 [[c.group, c.block, c.datapath,
                   c.found or ("not characterized" if c.block != "ctrl" else None), c.fires,
                   c.bits, _shown(c.comb_um2), _shown(c.flops_um2), _energy(c.e_fire_pj),
                   _energy(c.e_hold_pj), _energy(c.energy_nj)] for c in e.cost],
                 {4, 5, 6, 7, 8, 9, 10})  # fmt: skip
    if e.streamer_cost:
        out += ["", *table(["Accelerator", "Streamer", "Direction", "FIFO depth", "Loops",
                            "Channels", "Beats", "Area (µm²)", "e_beat (pJ)", "Energy (nJ)"],
                           [[x.accelerator, x.streamer + (" (extrapolated)" if x.extrapolated
                                                          else ""),
                             x.direction, x.depth, x.loops, x.channels, x.beats,
                             _shown(x.area_um2), _energy(x.e_beat_pj), _energy(x.energy_nj)]
                            for x in e.streamer_cost], {3, 4, 5, 6, 7, 8, 9})]  # fmt: skip
    if e.l1:
        out += ["", *table(["Accelerator", "Block", "Reads", "Writes", "Crossbar (nJ)",
                            "Banks (nJ)", "Energy (nJ)"],
                           [[x.accelerator, L1_LABEL, x.reads, x.writes, _energy(x.xbar_nj),
                             _energy(x.banks_nj), _energy(x.energy_nj)] for x in e.l1],
                           {2, 3, 4, 5, 6})]  # fmt: skip
    if e.parts:
        out += ["", "Cost by part:", "", *table(
            ["Part", "Area (µm²)", "Area share", "Energy (nJ)", "Energy share"],
            [[p.part, _shown(p.area_um2) or "not costed", _share(p.area_share),
              _energy(p.energy_nj) or "not costed", _share(p.energy_share)] for p in e.parts]
            + [["total", _shown(e.area_um2), None, _energy(e.energy_nj), None],
               ["without the banks", _shown(e.area_design_um2), None, None, None]],
            {1, 2, 3, 4})]  # fmt: skip
    x, m = e.xbar, e.memory
    if x is not None and m is not None:
        out += ["", (f"The crossbar: {x['inputs']} inputs on {x['banks']} banks, "
                     f"{x['accesses']} accesses at {_energy(x['e_access'], ' pJ')} each. The L1: "
                     f"{m['banks']} banks of {m['depth']} × {m['width']} bits, {m['reads']} reads "
                     f"at {_energy(m['e_rd'], ' pJ')} and {m['writes']} writes at "
                     f"{_energy(m['e_wr'], ' pJ')}.")]  # fmt: skip
    out += ["", (f"Design on {e.tech}: about {_shown(e.area_um2, ' µm²')} "
                 f"({_shown(e.area_design_um2, ' µm²')} without the banks) and "
                 f"{_energy(e.energy_nj, ' nJ')} per run.")]  # fmt: skip
    if e.missing:
        out += ["", (f"Not characterized on {e.tech} (comb counted as 0, flops counted): "
                     f"{', '.join(e.missing)}; see the notes.")]  # fmt: skip
    if e.not_costed:
        out += ["", f"Not costed: {'; '.join(e.not_costed)}."]
    if e.extrapolated:
        out += ["", (f"Outside the tables' grid, extrapolated: {', '.join(e.extrapolated)}; see "
                     "the notes.")]  # fmt: skip
    out += ["", streamers_line(e.streamers, e.csrs)]
    return out


def _share(x: float | None) -> str | None:
    return None if x is None else f"{100 * x:.1f} %"


def streamers_line(streamers: int, csrs: int | None) -> str:
    """The streamer sets (one per accelerator) and the entry's CSRs, which are not costed
    (D152)."""
    n = f"{streamers} streamer set{'' if streamers == 1 else 's'}"
    c = (
        ""
        if csrs is None
        else f"; {csrs} CSRs, the streamers' and the accelerators' own, not costed"
    )
    return f"{n}, one per accelerator{c}."
