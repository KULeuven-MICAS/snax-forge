"""The design report, ``design.md``: what was built, before any cycle is counted.

Everything here comes from the design point, its recipe and the cluster as a
run names it (run/cluster.py: the generator's hjson files and the design
point, D137), its regions as run/regions.py gives them (L2's from the ELF's symbols in
run.json); SNAX-GVSoC is not run. A flow from a bound graph has no recipe:
its header names the graph the design point was made from instead, and takes
the kernel from the run's functional check.

    header       kernel, recipe and its params (or the graph), symbols,
                 platform (base and changes), memory plan passes and pins
    components   the accelerator, its movers, the DMA and the cores
    accelerators instance, unit, BRM and implementation, params, latency,
                 target II (the datapath file's), ports
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
    notes        facts about addresses only: movers of the accelerator whose
                 containers start in the same banks, or lie a whole number of
                 bank rows apart, so that element i of both is in one bank;
                 and containers the DMA moves as padded beats
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
    pad_elements: int  # a tail's pads (D144)
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
class DesignReport(Record):
    name: str
    header: Header | None
    components: list[ComponentRow]
    accelerators: list[AccelRow]
    streamers: list[StreamerRow]
    memories: list[MemoryTable]
    notes: list[str]
    groups: int = 1  # the accelerator's; with more than one, the accelerators' table names each's


# =============================================================================
# Building
# =============================================================================


def _words(lay: Layout, base: int, word: int) -> np.ndarray:
    """The word index from ``base`` of every element of a layout (several narrow elements
    share a word, D142)."""
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
    """Address facts about the movers of the accelerator (module doc)."""
    word, banks = cluster.bank_bytes, cluster.banks
    row_bytes = word * banks
    l1 = {r.container: r.layout for r in regions if r.memory == "l1"}
    elem = {r.container: r.itemsize for r in regions if r.memory == "l1"}
    lays = {s.container: l1[s.container] for s in streamers if s.container in l1}
    ss = [s for s in streamers if s.container in lays]
    notes = []
    for i, a in enumerate(ss):
        for b in ss[i + 1 :]:
            la, lb = lays[a.container], lays[b.container]  # type: ignore[index]

            def first_banks(lay: Layout, num_ch: int) -> list[int]:
                """The banks of a mover's first beat: its first ``num_ch`` words (several
                narrow elements share one, D142)."""
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
        notes.append("No two streamers of the accelerator start in the same banks.")
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


def unit_view(file: dict[str, Any], instance: str) -> tuple[str, str, int, int]:
    """The kind, the control, the latency and the II of ``instance`` in the slot's
    accelerator.json: its unit's datapath under the nest controller."""
    u = next(u for g in file["groups"] for u in g["units"] if u["name"] == instance)
    return u["datapath"], "nest controller", u["timing"]["latency"], u["timing"]["ii"]


def build_design(
    name: str,
    cluster: Cluster,
    point: Any,
    recipe: Any = None,
    kernel: str = "",
    elf_symbols: dict[str, Any] | None = None,
) -> DesignReport:
    """The design report of a flow (``point`` its DesignPoint, ``recipe`` its recipe, or for a
    flow from a bound graph its ``kernel``; ``elf_symbols`` its run.json's)."""
    from snax_forge.design import run_checks
    from snax_forge.design.streamers import accelerated, streamer_name

    regions = regions_of(point, elf_symbols)
    design = point.design()
    run_checks(design)  # resolves the instances (BRM, implementation, params)
    instances = dict(design.instances)
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
    views = {u.name: unit_view(a.unit, u.name) for u in a.units}
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
        inst = instances.get(u.name)
        kind, _, latency, ii = views[u.name]
        accels.append(AccelRow(u.name, kind, inst.brm.name if inst else None,
                               inst.implementation if inst else None,
                               dict(inst.params) if inst else {}, latency, ii,
                               [f"{m.port} ← {m.name}" for m in a.movers
                                if point.streamers[m.name].instance == u.name], u.group))  # fmt: skip
    streamers = [
        StreamerRow(m.name, point.streamers[m.name].instance, m.port,
                    "write" if m.write else "read",
                    point.streamers[m.name].dtype, point.streamers[m.name].num_slc, m.lanes,
                    m.fifo_depth, point.streamers[m.name].temporal_dims, containers.get(m.name))
        for m in a.movers
    ]  # fmt: skip
    return DesignReport(name, header, components, accels, streamers,
                        _memory_tables(cluster, regions, point),
                        _notes(cluster, regions, point, streamers), len(a.units))  # fmt: skip


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
    out += ["", "## Notes", ""]
    out += [f"- {n}" for n in r.notes] or ["- None."]
    return "\n".join(out) + "\n"
