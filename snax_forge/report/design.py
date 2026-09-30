"""The design report, ``design.md`` (REP1, D99): what was built, before any cycle is counted.

Everything here comes from the design point, its recipe and the cluster file
SNAX-LOWER derives from it; the model is not run. For a plain run directory
(a scenario run, no design point) the report has the cluster part only, from
``run.json``'s cluster and regions.

    header       kernel, recipe and its params, symbols, platform (base and
                 changes), memory plan passes and pins      (design point only)
    components   the cluster file's components in order
    accelerators instance, BRM and implementation (design point only), params,
                 latency, target II, ports, drain (D103)
    streamers    instance port, direction, lanes, FIFO depth, temporal loops and
                 the container it streams (from the graph's memlets, else from
                 the task list's base address, else unknown)
    memories     per memory its regions: base, end, bytes, share, banks, rows
                 (the memory tab's numbers, viz/memory.py)
    notes        facts about addresses only: streamers of one accelerator that
                 start in the same banks, or whose containers lie a whole number
                 of bank rows apart, so that element i of both is in one bank;
                 and containers the DMA moves as padded beats (D105)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from snax_forge.snax_model.scenario import ClusterConfig, Region

from ..viz import memory as memview
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
    drain: int = 0  # cycles from a push in which no input is taken (D103)


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


def _component_detail(c: Any) -> str:
    cfg = c.config
    if c.kind == "streamer":
        return (
            f"{'writer' if cfg.get('write') else 'reader'}, {cfg.get('n_ports', 1)} lanes, "
            f"FIFO depth {cfg.get('fifo_depth')}, {cfg.get('temporal_dims', 1)} temporal loop(s)"
        )
    if c.kind == "accel":
        attach = ", ".join(f"{p} ← {s}" for p, s in c.attach.items())
        return f"{c.accel}; {attach}"
    if c.kind == "dma":
        return (
            f"startup {cfg.get('startup', 0)}, a beat every {cfg.get('beat_interval', 1)} cycle(s)"
        )
    if c.kind == "controller":
        return f"write cost {cfg.get('write_cost', 1)}, poll every {cfg.get('poll_interval', 1)}"
    return ", ".join(f"{k} {v}" for k, v in cfg.items())


def _memory_tables(cluster: ClusterConfig, regions: list[Region]) -> list[MemoryTable]:
    out = []
    for mem in ["l1"] + (["l2"] if cluster.l2 is not None else []):
        s = memview.summary(memview.layout_of(mem, cluster, regions))
        g = s["geometry"]
        col = "banks" if mem == "l1" else "words"
        rows = [
            RegionRow(
                r["name"],
                r["start"],
                r["end"],
                r["bytes"],
                r["share"],
                spans(r[col]),
                span_text(r["rows"]),
            )
            for r in s["regions"]
        ]
        out.append(
            MemoryTable(
                mem,
                g["columns"],
                g["rows"],
                g["word_bytes"],
                g["size_bytes"],
                s["used"]["words"] * g["word_bytes"],
                rows,
            )
        )
    return out


def _notes(
    cluster: ClusterConfig, streamers: list[StreamerRow], regions: list[Region]
) -> list[str]:
    """Address facts about the streamers of each accelerator (module doc)."""
    lay = memview.layout_of("l1", cluster, [])
    row_bytes = lay.word_bytes * lay.columns
    by_name = {r.name: r for r in regions if r.mem == "l1"}
    notes = []
    groups: dict[str, list[StreamerRow]] = {}
    for s in streamers:
        if s.instance and s.container in by_name:
            groups.setdefault(s.instance, []).append(s)
    for ss in groups.values():
        for i, a in enumerate(ss):
            for b in ss[i + 1 :]:
                ra, rb = by_name[a.container], by_name[b.container]  # type: ignore[index]

                def first_banks(r: Region, lanes: int) -> list[int]:
                    words = [lay.element_word(r, k) for k in range(min(lanes, r.size))]
                    return sorted({lay.place(w)[0] for w in words})

                ba, bb = first_banks(ra, a.lanes), first_banks(rb, b.lanes)
                both = sorted(set(ba) & set(bb))
                rows_apart = (
                    ra.strides == rb.strides
                    and ra.shape == rb.shape
                    and (rb.base - ra.base) % row_bytes == 0
                )
                if rows_apart:  # the stronger fact: every element, not only the first beat
                    notes.append(
                        f"{a.name} and {b.name}: {a.container} and {b.container} lie "
                        f"{abs(rb.base - ra.base)} B apart, a whole number of {row_bytes}-byte bank "
                        f"rows, so {a.container}[i] and {b.container}[i] are in the same bank for every i."
                    )
                elif both:
                    notes.append(
                        f"{a.name} ({a.container}) and {b.name} ({b.container}) start in the same "
                        f"banks, {spans(both)}."
                    )
    if not notes and groups:
        notes.append("No two streamers of one accelerator start in the same banks.")
    return notes


def _padding_notes(cluster: ClusterConfig, regions: list[Region]) -> list[str]:
    """Containers the DMA moves (in L2 and L1) whose size is not whole beats (D105)."""
    if cluster.l2 is None:
        return []
    beat = cluster.l1.wide_bits // 8
    word = cluster.l1.width_bits // 8
    in_l2 = {r.name for r in regions if r.mem == "l2"}
    notes = []
    for r in regions:
        if r.mem != "l1" or r.name not in in_l2:
            continue
        size = r.size * word
        if size % beat:
            moved = -(-size // beat) * beat
            notes.append(
                f"{r.name}: {size} B, moved by the DMA as {moved // beat} whole {beat}-byte "
                f"beat{'s' if moved > beat else ''}; bytes {r.base + size}–{r.base + moved} are "
                "kept free in L1 and L2."
            )
    return notes


def _container_by_task_base(tasks: Any, regions: list[Region]) -> dict[str, str]:
    """Streamer -> the L1 region holding the base of its first task (a task list), if any."""
    out: dict[str, str] = {}
    if tasks is None:
        return out
    for st in tasks.steps:
        if getattr(st, "type", None) != "streamer" or st.component in out:
            continue
        base = st.values.get("base") if isinstance(st.values, dict) else None
        for r in regions:
            lo, hi = r.span()
            if r.mem == "l1" and base is not None and lo <= base <= hi:
                out[st.component] = r.name
                break
    return out


def build_design(
    name: str,
    cluster: ClusterConfig,
    regions: list[Region],
    point: Any = None,
    recipe: Any = None,
    tasks: Any = None,
) -> DesignReport:
    """The design report of a flow (``point`` a DesignPoint and its ``recipe``) or, with
    neither, of a plain run's cluster and regions; ``tasks`` names streamed containers."""
    header = None
    instances: dict[str, Any] = {}
    containers: dict[str, str] = {}
    if point is not None:
        from snax_forge.design import run_checks
        from snax_forge.design.streamers import accelerated, streamer_name

        design = point.design()
        run_checks(design)  # resolves the instances (BRM, implementation, params)
        instances = dict(design.instances)
        header = Header(
            kernel=recipe.kernel if recipe is not None else "",
            recipe=recipe.name if recipe is not None else point.name,
            params=dict(recipe.params) if recipe is not None else {},
            symbols={k: v for k, v in point.graph.symbols.items() if v is not None},
            platform=point.platform.name,
            platform_base=point.platform.base,
            platform_changes=dict(point.platform.changes),
            passes=dict(point.memory.passes),
            pins=dict(point.memory.changes),
        )
        for n in accelerated(point.graph):
            for side in (n.inputs, n.outputs):
                for port, m in side.items():
                    containers[streamer_name(n.attrs["instance"], port)] = m.data
    else:
        containers = _container_by_task_base(tasks, regions)

    components = [ComponentRow(c.name, c.kind, _component_detail(c)) for c in cluster.components]
    attach = {
        s: (c.name, p) for c in cluster.components if c.kind == "accel" for p, s in c.attach.items()
    }
    accels = []
    for c in cluster.components:
        if c.kind != "accel":
            continue
        inst = instances.get(c.name)
        params = {k: v for k, v in c.params.items() if k not in ("latency", "ii", "drain")}
        ports = [f"{p} ← {s}" for p, s in c.attach.items()]
        accels.append(
            AccelRow(
                c.name,
                c.accel or "",
                inst.brm.name if inst else None,
                inst.implementation if inst else None,
                dict(inst.params) if inst else params,
                int(c.params.get("latency", 0)),
                int(c.params.get("ii", 1)),
                ports,
                int(c.params.get("drain", 0)),
            )
        )
    streamers = []
    for c in cluster.components:
        if c.kind != "streamer":
            continue
        cfg = c.config
        inst, port = attach.get(c.name, (None, None))
        streamers.append(
            StreamerRow(
                c.name,
                inst,
                port,
                "write" if cfg.get("write") else "read",
                int(cfg.get("n_ports", 1)),
                int(cfg.get("fifo_depth", 0)),
                int(cfg.get("temporal_dims", 1)),
                containers.get(c.name),
            )
        )
    return DesignReport(
        name,
        header,
        components,
        accels,
        streamers,
        _memory_tables(cluster, regions),
        _notes(cluster, streamers, regions) + _padding_notes(cluster, regions),
    )


# =============================================================================
# Rendering
# =============================================================================


def render_design(r: DesignReport) -> str:
    """``design.md``: the report as Markdown, one section per part."""
    out = [f"# Design report: {r.name}", ""]
    if r.header is None:
        out += [
            "From the run's cluster file and regions only: this directory has no design point.",
            "",
        ]
    else:
        h = r.header
        base = f", base {h.platform_base}" if h.platform_base else ""
        out += table(
            ["", ""],
            [
                ["Kernel", h.kernel],
                ["Recipe", f"{h.recipe} ({kv(h.params)})"],
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
            "Drain",
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
                a.drain,
                "; ".join(a.ports),
            ]
            for a in r.accelerators
        ],
        {5, 6, 7},
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
