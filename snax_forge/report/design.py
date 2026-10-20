"""The design report, ``design.md``: what was built, before any cycle is counted.

Everything here comes from the design point, its recipe and the cluster as a
run names it (run/cluster.py: the generator's hjson files and the design
point, D137); SNAX-GVSoC is not run. A flow from a bound graph has no recipe:
its header names the graph the design point was made from instead, and takes
the kernel from the run's functional check.

    header       kernel, recipe and its params (or the graph), symbols,
                 platform (base and changes), memory plan passes and pins
    components   the accelerator, its movers, the DMA and the cores
    accelerators instance, unit, BRM and implementation, params, latency,
                 target II (the datapath file's), ports
    streamers    one per mover: instance port, direction, lanes, FIFO depth,
                 temporal loops and the container it streams (from the
                 graph's memlets)
    memories     per memory the design point's layouts: base, end, bytes,
                 share, banks (L1) or words of a beat (L2), rows or beats
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
from snax_forge.run import Cluster

from .markdown import kib, kv, pct, span_text, spans, table
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


@dataclass
class StreamerRow(Record):
    name: str
    instance: str | None
    port: str | None
    direction: str
    lanes: int
    fifo_depth: int
    temporal_dims: int
    container: str | None


@dataclass
class RegionRow(Record):
    name: str
    base: int
    end: int
    bytes: int
    share: float
    banks: str  # columns touched, as spans ("0–15")
    rows: str


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


# =============================================================================
# Building
# =============================================================================


def _words(lay: Layout, base: int, word: int) -> np.ndarray:
    """The word index from ``base`` of every element of a layout, one element per word."""
    idx = np.indices(lay.shape).reshape(len(lay.shape), -1).T
    return (lay.address(idx) - base) // word


def _memory_tables(cluster: Cluster, point: Any) -> list[MemoryTable]:
    """L1 as banks by rows, L2 as the words of a beat by beats; the regions from the layouts."""
    word = cluster.bank_bytes
    l2 = point.platform.l2
    geo = {
        "l1": (cluster.l1_base, cluster.banks, cluster.l1_size // (word * cluster.banks)),
        "l2": (l2.base_addr, l2.beat_bits // 8 // word, l2.size_bytes // (l2.beat_bits // 8)),
    }
    out = []
    for mem, (base, columns, rows) in geo.items():
        size = columns * rows * word
        regions = []
        for name, lays in point.memory.layouts.items():
            lay = lays.get(mem)
            if lay is None:
                continue
            w = _words(lay, base, word)
            n = int(w.size) * word
            lo, hi = lay.span()
            regions.append(RegionRow(name, lo, hi + word, n, n / size,
                                     spans(sorted({int(x) % columns for x in w})),
                                     span_text([int(w.min()) // columns, int(w.max()) // columns])))  # fmt: skip
        regions.sort(key=lambda r: r.base)
        out.append(MemoryTable(mem, columns, rows, word, size, sum(r.bytes for r in regions),
                               regions))  # fmt: skip
    return out


def _notes(cluster: Cluster, point: Any, streamers: list[StreamerRow]) -> list[str]:
    """Address facts about the movers of the accelerator (module doc)."""
    word, banks = cluster.bank_bytes, cluster.banks
    row_bytes = word * banks
    lays = {s.container: point.memory.layouts[s.container]["l1"] for s in streamers
            if s.container in point.memory.layouts and "l1" in point.memory.layouts[s.container]}  # fmt: skip
    ss = [s for s in streamers if s.container in lays]
    notes = []
    for i, a in enumerate(ss):
        for b in ss[i + 1 :]:
            la, lb = lays[a.container], lays[b.container]  # type: ignore[index]

            def first_banks(lay: Layout, lanes: int) -> list[int]:
                w = _words(lay, cluster.l1_base, word)[:lanes]
                return sorted({int(x) % banks for x in w})

            both = sorted(set(first_banks(la, a.lanes)) & set(first_banks(lb, b.lanes)))
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
    for name, lay in point.memory.layouts.items():
        if "l1" not in lay or "l2" not in lay:
            continue
        size = int(np.prod(lay["l1"].shape)) * word
        if size % beat:
            moved = round_up(size, beat)
            base = lay["l1"].base
            notes.append(
                f"{name}: {size} B, moved by the DMA as {moved // beat} whole {beat}-byte "
                f"beat{'s' if moved > beat else ''}; bytes {base + size}–{base + moved} are "
                "kept free in L1 and L2."
            )
    return notes


def build_design(
    name: str, cluster: Cluster, point: Any, recipe: Any = None, kernel: str = ""
) -> DesignReport:
    """The design report of a flow (``point`` its DesignPoint, ``recipe`` its recipe, or for a
    flow from a bound graph its ``kernel``)."""
    from snax_forge.design import run_checks
    from snax_forge.design.streamers import accelerated, streamer_name

    design = point.design()
    run_checks(design)  # resolves the instances (BRM, implementation, params)
    instances = dict(design.instances)
    header = Header(
        kernel=recipe.kernel if recipe is not None else kernel or point.graph.name,
        recipe=recipe.name if recipe is not None else point.name,
        params=dict(recipe.params) if recipe is not None else {},
        symbols={k: v for k, v in point.graph.symbols.items() if v is not None},
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
    u = a.unit
    components = [ComponentRow(a.name, "accelerator", f"{u.get('unit', '')} on core{a.core}, "
                               f"{u.get('wait', '')}, slot {a.slot}")]  # fmt: skip
    components += [ComponentRow(m.name, "streamer", f"{'writer' if m.write else 'reader'} "
                                f"{m.gvsoc}, {m.lanes} lanes, FIFO depth {m.fifo_depth}")
                   for m in a.movers]  # fmt: skip
    components.append(ComponentRow("dma", "dma", f"the iDMA on core{cluster.dma_core}"))
    components += [ComponentRow(c, "core", "the accelerator's" if i == a.core
                                else "the DMA's" if i == cluster.dma_core else "")
                   for i, c in enumerate(cluster.cores)]  # fmt: skip

    inst = instances.get(a.name)
    accels = [AccelRow(a.name, str(u.get("unit", "")), inst.brm.name if inst else None,
                       inst.implementation if inst else None,
                       dict(inst.params) if inst else {}, int(u.get("latency", 0)),
                       int(u.get("ii", 1)), [f"{m.port} ← {m.name}" for m in a.movers])]  # fmt: skip
    streamers = [
        StreamerRow(m.name, a.name, m.port, "write" if m.write else "read", m.lanes, m.fifo_depth,
                    point.streamers[m.name].options.temporal_dims, containers.get(m.name))
        for m in a.movers
    ]  # fmt: skip
    return DesignReport(name, header, components, accels, streamers,
                        _memory_tables(cluster, point), _notes(cluster, point, streamers))  # fmt: skip


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
    out += table(
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
    )
    out += ["", "## Streamers", ""]
    out += table(
        ["Streamer", "Serves", "Direction", "Lanes", "FIFO depth", "Temporal loops", "Container"],
        [
            [
                s.name,
                f"{s.instance}.{s.port}" if s.instance else None,
                s.direction,
                s.lanes,
                s.fifo_depth,
                s.temporal_dims,
                s.container,
            ]
            for s in r.streamers
        ],
        {3, 4, 5},
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
                "Banks" if m.mem == "l1" else "Words",
                "Rows" if m.mem == "l1" else "Beats",
            ],
            [[x.name, x.base, x.end, x.bytes, pct(x.share), x.banks, x.rows] for x in m.regions],
            {1, 2, 3, 4},
        )
        out += [
            "",
            f"Used: {m.used_bytes} of {m.size_bytes} bytes ({pct(m.used_bytes / m.size_bytes)}).",
        ]
    out += ["", "## Notes", ""]
    out += [f"- {n}" for n in r.notes] or ["- None."]
    return "\n".join(out) + "\n"
