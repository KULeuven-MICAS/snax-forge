"""The design checks: does this platform fit this bound graph (DP1a, D85)?

The design step pairs a bound graph (the last step of a recipe) with a
platform. Before it writes anything it runs every registered check and
collects every problem (problems.py), each with a code, what it is about,
why it fails and the change that fixes it. Checks belong to stages:

    platform   the platform alone: keys, values, L1 and L2 agreeing, banks
    graph      the graph alone: symbols bound, everything bound to an
               accelerator, BRMs and implementations known, instance names
    connect    the two together: streamer entries, lanes, dtypes, registers
    memory     the memory plan (DP1b, D86): pins and passes, then the plan
               the passes make, its residency, shapes, alignment, fit and
               overlaps

``platform`` and ``graph`` run independently; ``connect`` runs only when
both passed, since its checks read what they established, and ``memory``
only when ``connect`` passed. ``memory.pin`` makes the plan (memory.py)
when its pins and passes name what exists, unless the Design already holds
one (a design point, judged as stored). A new check is registered with
``register_check(code, stage, fn)``; ``fn(design)`` yields Problems.

The model's own checks (scenario.py, when a cluster is built) stay as a
backstop; these run earlier and speak in platform, graph and recipe terms.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from snax_forge import expr
from snax_forge.brm import BrmError, Instance
from snax_forge.dfg import DfgError, Graph, Node
from snax_forge.dfg.subset import parse_dim
from snax_forge.snax_model.ctrl import STATUS, DmaAdapter, StreamerAdapter

from .memory import MemoryContext, MemoryPlan, MemorySpec, pin_problems, plan
from .platform import DERIVED, Platform
from .problems import DesignError, Problem
from .streamers import (
    LoopError,
    accelerated,
    firing_loops,
    instance_nodes,
    instance_of,
    nest_spatial_bounds,
    streamer_name,
)

# Stage -> the stages that must have passed before it runs.
STAGES: dict[str, tuple[str, ...]] = {
    "platform": (),
    "graph": (),
    "connect": ("platform", "graph"),
    "memory": ("connect",),
}
# Component names of the cluster besides the shell and the accelerators (LOW1c).
FIXED = ("xbar", "dma", "ctl")


@dataclass
class Design:
    """What the checks read: a platform, a bound graph, and what loading them found."""

    platform: Platform | None
    graph: Graph | None
    load_problems: list[Problem] = field(default_factory=list)
    graph_path: str = "graph"
    instances: dict[str, Instance] = field(default_factory=dict)  # filled by graph.brm
    memory_spec: MemorySpec | None = field(default_factory=MemorySpec)
    memory: MemoryPlan | None = None  # made by the memory.pin check, or given (a design point)

    def context(self) -> MemoryContext:
        return MemoryContext(self.graph, self.platform)


Check = Callable[[Design], Iterable[Problem]]


@dataclass(frozen=True)
class Registered:
    code: str
    stage: str
    fn: Check


CHECKS: dict[str, Registered] = {}


def register_check(code: str, stage: str, fn: Check) -> None:
    """Add a check; it runs in its stage, in registration order."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r} ({list(STAGES)})")
    if code in CHECKS:
        raise ValueError(f"check {code!r} already registered")
    CHECKS[code] = Registered(code, stage, fn)


def load(
    graph_path: str | Path,
    platform_path: str | Path,
    sets: Sequence[tuple[str, Any]] = (),
    memory_path: str | Path | None = None,
    memory_sets: Sequence[tuple[str, Any]] = (),
) -> Design:
    """Load the inputs; what fails to load becomes a problem, never an exception.

    The platform is the working copy (``Platform.with_changes``), with ``sets``
    applied when they are given. The memory spec starts from ``memory_path``
    (a memory working copy) or the default passes, with ``memory_sets``
    applied; the plan itself is made by the memory checks.
    """
    problems: list[Problem] = []
    platform = None
    try:
        platform = Platform.load(platform_path).with_changes(sets)
    except DesignError as e:
        problems += e.problems
    graph = None
    try:
        graph = Graph.load(graph_path)
    except (OSError, json.JSONDecodeError, DfgError) as e:
        problems.append(Problem("graph.load", str(graph_path), str(e)))
    spec = None
    try:
        start = MemorySpec() if memory_path is None else MemorySpec.load(memory_path)
        spec = start.with_changes(memory_sets)
    except DesignError as e:
        problems += e.problems
    return Design(platform, graph, problems, str(graph_path), memory_spec=spec)


def run_checks(design: Design, stages: Sequence[str] | None = None) -> list[Problem]:
    """Every problem found, stage by stage; a stage whose prerequisites failed is skipped."""
    stages = list(stages or STAGES)
    failed: set[str] = set()
    out: list[Problem] = []
    for stage in stages:
        if any(s in failed or s not in stages for s in STAGES[stage]):
            failed.add(stage)
            continue
        found = [p for c in CHECKS.values() if c.stage == stage for p in c.fn(design)]
        if found:
            failed.add(stage)
            out += found
    return out


def check(design: Design) -> None:
    """Raise DesignError with every problem, or return when the pairing fits."""
    problems = run_checks(design)
    if problems:
        raise DesignError(problems)


# =============================================================================
# Helpers for messages
# =============================================================================


def _params(inst: Instance) -> str:
    return ", ".join(f"{k}={v}" for k, v in inst.params.items())


def _about(design: Design, instance: str) -> str:
    inst = design.instances[instance]
    return f"instance {instance} ({inst.brm.name}, {_params(inst)})"


def _set(path: str, value: Any) -> str:
    return f"--set platform.{path}={json.dumps(value)}"


def _pow2_at_least(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _subtree(node: Node) -> Iterator[Node]:
    for n in node.body or []:
        yield n
        yield from _subtree(n)


def _streamers(design: Design) -> dict[str, tuple[str, str]]:
    """Every streamer the graph needs: name -> (instance, port)."""
    return {
        streamer_name(i, p.name): (i, p.name)
        for i, inst in design.instances.items()
        for p in inst.brm.interface.ports
    }


# =============================================================================
# Stage platform
# =============================================================================


def _platform_load(design: Design) -> Iterable[Problem]:
    return [p for p in design.load_problems if p.code.startswith("platform.")]


def _platform_l2(design: Design) -> Iterable[Problem]:
    pf = design.platform
    if pf is None or pf.l2 is None:
        return
    l1, l2 = pf.l1, pf.l2
    if l2.beat_bits != l1.wide_bits:
        yield Problem(
            "platform.l2", "l2.beat_bits",
            f"{l2.beat_bits}, but the L1's wide port (the DMA's) is {l1.wide_bits} bits",
            _set("l2.beat_bits", l1.wide_bits),
        )  # fmt: skip
    for f in ("width_bits", "dtype", "elems_per_word"):
        a, b = getattr(l1, f), getattr(l2, f)
        if a != b:
            yield Problem(
                "platform.l2", f"l2.{f}", f"{b!r}, but the L1's is {a!r}; L1 and L2 must agree",
                _set(f"l2.{f}", a),
            )  # fmt: skip


def _platform_banks(design: Design) -> Iterable[Problem]:
    pf = design.platform
    if pf is None or pf.l2 is None:
        return
    group = pf.l1.wide_bits // pf.l1.width_bits
    if pf.l1.n_banks % group:
        up = (pf.l1.n_banks // group + 1) * group
        yield Problem(
            "platform.banks", "l1.n_banks",
            f"{pf.l1.n_banks} is not a multiple of the DMA's port group of {group} banks "
            f"(wide_bits {pf.l1.wide_bits} / width_bits {pf.l1.width_bits})",
            _set("l1.n_banks", up),
        )  # fmt: skip


# =============================================================================
# Stage graph
# =============================================================================


def _graph_load(design: Design) -> Iterable[Problem]:
    return [p for p in design.load_problems if p.code.startswith("graph.")]


def _graph_symbols(design: Design) -> Iterable[Problem]:
    g = design.graph
    if g is None:
        return
    for s, v in g.symbols.items():
        if v is None:
            yield Problem(
                "graph.symbols", f"symbol {s}", "is not bound",
                f'bind it in the recipe: "symbols": {{"{s}": ...}}',
            )  # fmt: skip


def _graph_unbound(design: Design) -> Iterable[Problem]:
    g = design.graph
    if g is None:
        return
    found = False
    for n, outer in g.walk():
        inside = f" in {'/'.join(o.id for o in outer)}" if outer else ""
        if n.kind == "tasklet":
            found = True
            yield Problem(
                "graph.unbound", f"tasklet {n.id}{inside}",
                "is not bound to an accelerator: SNAX-MODEL has no core to run it",
                "bind it in the recipe: "
                f'{{"transform": "bind", "params": {{"node": "{n.id}", ...}}}}',
            )  # fmt: skip
        elif n.kind == "map" and not any(m.kind in ("accelerated", "tasklet") for m in _subtree(n)):
            found = True
            yield Problem("graph.unbound", f"map {n.id}{inside}", "holds nothing to run")
    if not found and not accelerated(g):
        yield Problem(
            "graph.unbound", g.name, "has no accelerated node",
            "bind a tasklet to a BRM in the recipe (split_map, then bind)",
        )  # fmt: skip


def _graph_brm(design: Design) -> Iterable[Problem]:
    g = design.graph
    if g is None:
        return
    for n in accelerated(g):
        where = f"node {n.id}"
        try:
            inst = instance_of(n)
        except (BrmError, ValueError) as e:
            yield Problem("graph.brm", where, str(e))
            continue
        design.instances.setdefault(n.attrs["instance"], inst)
        ports = inst.brm.interface.ports
        ins = [p.name for p in ports if p.direction == "in"]
        outs = [p.name for p in ports if p.direction == "out"]
        if sorted(n.inputs) != sorted(ins) or sorted(n.outputs) != sorted(outs):
            yield Problem(
                "graph.brm", where,
                f"connectors {sorted(n.inputs)} -> {sorted(n.outputs)} are not the ports of "
                f"{inst.brm.name}: {ins} -> {outs}",
            )  # fmt: skip
        code = inst.brm.function.code
        if code is None:
            yield Problem("graph.brm", where, f"{inst.brm.name} has no function.code to bind to")
        elif n.attrs.get("code") != code:
            yield Problem(
                "graph.brm", where,
                f"computes {n.attrs.get('code')!r}, but {inst.brm.name} computes {code!r}",
            )  # fmt: skip


def _graph_instance(design: Design) -> Iterable[Problem]:
    g = design.graph
    if g is None or len(design.instances) != len(instance_nodes(g)):
        return  # graph.brm reported the instances it could not resolve
    names = _streamers(design)
    taken: dict[str, str] = {}
    for s, (i, p) in names.items():
        if s in taken:
            yield Problem(
                "graph.instance", f"streamer {s}",
                f"is port {p} of instance {i} and also {taken[s]}",
                "rename one of the instances in the recipe's bind",
            )  # fmt: skip
        taken[s] = f"port {p} of instance {i}"
    for i in design.instances:
        clash = "a fixed component" if i in FIXED else taken.get(i)
        if clash:
            yield Problem(
                "graph.instance", f"instance {i}", f"has the name of {clash}",
                f"rename it in the recipe's bind (the names {list(FIXED)} are taken)",
            )  # fmt: skip


# =============================================================================
# Stage connect
# =============================================================================


def _connect_streamer_key(design: Design) -> Iterable[Problem]:
    pf = design.platform
    names = list(_streamers(design))
    for e in pf.entries:
        if e not in names:
            close = difflib.get_close_matches(e, names, n=1)
            yield Problem(
                "connect.streamer_key", f"platform.streamers.{e}",
                f"no port of the graph has this streamer; the graph's streamers are {names}",
                f"did you mean {close[0]}?" if close else "remove the entry",
            )  # fmt: skip


def _connect_derived(design: Design) -> Iterable[Problem]:
    pf = design.platform
    names = _streamers(design)
    for e, entry in pf.entries.items():
        for k in DERIVED:
            if k not in entry:
                continue
            if e in names:
                i, p = names[e]
                port = design.instances[i].brm.port(p)
                src = (
                    f"direction {port.direction!r}"
                    if k == "write"
                    else f"{design.instances[i].lanes(p)} lanes"
                )
                why = f"comes from port {p} of {_about(design, i)}: {src}"
            else:
                why = "comes from the accelerator port, not the platform"
            yield Problem(
                "connect.derived",
                f"platform.streamers.{e}.{k}",
                why,
                f"remove {k!r} from the entry",
            )


def _connect_lanes(design: Design) -> Iterable[Problem]:
    pf = design.platform
    for s, (i, p) in _streamers(design).items():
        sb = pf.spatial_bounds(s)
        if sb is None:
            continue
        inst = design.instances[i]
        where = f"platform.streamers.{s}.spatial_bounds"
        lanes = inst.lanes(p)
        if prod(sb) != lanes:
            yield Problem(
                "connect.lanes", where,
                f"{sb} gives {prod(sb)} lanes, but port {p} of {_about(design, i)} has {lanes}",
                "remove spatial_bounds (it follows the port), or change the lanes in the recipe",
            )  # fmt: skip
            continue
        try:
            want = nest_spatial_bounds(inst, p)
        except ValueError as e:
            yield Problem("connect.lanes", where, str(e))
            continue
        if list(sb) != want:
            yield Problem(
                "connect.lanes", where,
                f"{sb} are not the spatial loops of port {p}'s nest, {want} (fastest first)",
                f"remove spatial_bounds, or {_set(f'streamers.{s}.spatial_bounds', want)}",
            )  # fmt: skip


def _connect_dtype(design: Design) -> Iterable[Problem]:
    pf, g = design.platform, design.graph
    l1 = pf.l1
    if l1.elems_per_word != 1:
        yield Problem(
            "connect.dtype", "platform.l1.elems_per_word",
            f"the L1 packs {l1.elems_per_word} elements per word; "
            "only 1 is supported (open item 21)",
            _set("l1.elems_per_word", 1),
        )  # fmt: skip
    seen: set[str] = set()
    for n in accelerated(g):
        inst = design.instances[n.attrs["instance"]]
        for conn, m in {**n.inputs, **n.outputs}.items():
            dt = g.containers[m.data].dtype
            port = inst.brm.port(conn).dtype
            if dt != port and (n.id, conn) not in seen:
                seen.add((n.id, conn))
                yield Problem(
                    "connect.dtype", f"node {n.id}.{conn}",
                    f"container {m.data} is {dt}, port {conn} of {inst.brm.name} takes {port}",
                )  # fmt: skip
            if dt != l1.dtype and m.data not in seen:
                seen.add(m.data)
                fix = _set("l1.dtype", dt) + (f" {_set('l2.dtype', dt)}" if pf.l2 else "")
                yield Problem(
                    "connect.dtype", f"container {m.data}",
                    f"is {dt}, but the L1 holds {l1.dtype}", fix,
                )  # fmt: skip


def _connect_temporal(design: Design) -> Iterable[Problem]:
    """Each port needs one streamer temporal loop per firing loop around its node (LOW1a)."""
    pf, g = design.platform, design.graph
    for n in accelerated(g):
        try:
            loops = firing_loops(g, n)
        except LoopError as e:
            yield Problem("connect.temporal", f"node {n.id}", str(e))
            continue
        i = n.attrs["instance"]
        for conn, m in {**n.inputs, **n.outputs}.items():
            s = streamer_name(i, conn)
            try:
                for dim in m.subset:
                    for part in parse_dim(dim, "subset")[:2]:
                        expr.linear(part, [lp.var for lp in loops])
            except expr.ExprError as e:
                yield Problem(
                    "connect.temporal", f"node {n.id}.{conn}",
                    f"subset {m.subset} is not affine in the firing loops: {e}",
                )  # fmt: skip
                continue
            have = pf.options(s).temporal_dims
            if len(loops) > have:
                yield Problem(
                    "connect.temporal", f"streamer {s}",
                    f"node {n.id} fires over {len(loops)} temporal loops "
                    f"({', '.join(lp.map for lp in loops)}), the streamer has {have}",
                    _set(f"streamers.{s}.temporal_dims", len(loops)),
                )  # fmt: skip


def _connect_regmap(design: Design) -> Iterable[Problem]:
    pf = design.platform
    window = pf.register_window

    def over(block: str, regs: list[str], alternative: str | None = None) -> Iterable[Problem]:
        n = len(STATUS) + len(regs)
        if n > window:
            fix = _set("register_window", _pow2_at_least(n))
            yield Problem(
                "connect.regmap", block,
                f"needs {n} registers ({len(STATUS)} status + {len(regs)} configuration), "
                f"the register window is {window}",
                f"{fix}, or {alternative}" if alternative else fix,
            )  # fmt: skip

    for s, (i, p) in _streamers(design).items():
        sb = pf.spatial_bounds(s) or nest_spatial_bounds(design.instances[i], p)
        cfg = pf.options(s).config(write=False, n_ports=prod(sb))
        regs = StreamerAdapter(SimpleNamespace(name=s, cfg=cfg), spatial_bounds=sb).registers
        yield from over(f"streamer {s}", regs, f"lower platform.streamers.{s}.temporal_dims")
    for i, inst in design.instances.items():
        yield from over(f"accelerator {i}", list(inst.brm.registers))
    if pf.l2 is not None:
        regs = DmaAdapter(SimpleNamespace(name="dma", cfg=pf.dma)).registers
        yield from over("dma", regs, "lower platform.dma.dims")


# =============================================================================
# Stage memory
# =============================================================================


def _memory_pin(design: Design) -> Iterable[Problem]:
    """Pins and passes name what exists; then the plan is made (unless one was given)."""
    yield from (p for p in design.load_problems if p.code.startswith("memory."))
    if design.memory_spec is None:
        return
    if design.memory is not None:  # a design point's stored plan is judged as it is
        return
    ctx = design.context()
    found = pin_problems(ctx, design.memory_spec)
    if found:
        yield from found
        return
    try:
        design.memory = plan(ctx, design.memory_spec)
    except (ValueError, KeyError, TypeError) as e:
        yield Problem("memory.pin", "memory.passes", f"the passes failed: {e}")


def _layouts(design: Design) -> Iterator[tuple[str, str, Any]]:
    if design.memory is None:
        return
    for c, mems in design.memory.layouts.items():
        for m, lay in mems.items():
            yield c, m, lay


def _memory_residency(design: Design) -> Iterable[Problem]:
    if design.memory is None:
        return
    ctx, lays = design.context(), design.memory.layouts
    for c, m, _ in _layouts(design):
        if c not in design.graph.containers:
            yield Problem("memory.residency", f"container {c}", "is not a container of the graph")
        elif m not in ctx.memories:
            yield Problem(
                "memory.residency", f"{c} in {m}", f"the platform has no {m} ({ctx.memories})"
            )
    for c, cont in design.graph.containers.items():
        if not cont.transient and not lays.get(c):
            yield Problem("memory.residency", f"container {c}", "lives in no memory")
    for n in accelerated(design.graph):
        for conn, mlet in {**n.inputs, **n.outputs}.items():
            if "l1" not in lays.get(mlet.data, {}):
                yield Problem(
                    "memory.residency", f"container {mlet.data}",
                    f"is read or written by {n.id}.{conn}, but has no L1 layout: "
                    "the streamers reach only the L1",
                )  # fmt: skip


def _memory_layout(design: Design) -> Iterable[Problem]:
    ctx = design.context() if design.memory is not None else None
    for c, m, lay in _layouts(design):
        if c not in design.graph.containers:
            continue
        want = ctx.shape(c)
        if tuple(lay.shape) != want:
            yield Problem(
                "memory.layout", f"{c} in {m}",
                f"shape {list(lay.shape)} is not the container's {list(want)}",
            )  # fmt: skip


def _memory_align(design: Design) -> Iterable[Problem]:
    if design.memory is None:
        return
    ctx = design.context()
    for c, m, lay in _layouts(design):
        word = ctx.word_bytes(m)
        where = f"{c} in {m}"
        if lay.base % word or any(s % word for s in lay.strides):
            yield Problem(
                "memory.align", where,
                f"base {lay.base} and strides {list(lay.strides)} must be multiples of the "
                f"{word}-byte word",
                f"--set memory.{c}.{m}.base={_align_up(lay.base, word)}",
            )  # fmt: skip
            continue
        mems = design.memory.layouts.get(c, {})
        if not ("l1" in mems and "l2" in mems):
            continue
        beat = ctx.beat_bytes
        size = prod(lay.shape) * word
        contiguous = type(lay).contiguous(lay.base, lay.shape, word)
        if lay.strides != contiguous.strides:
            yield Problem(
                "memory.align", where,
                f"strides {list(lay.strides)} are not contiguous {list(contiguous.strides)}; "
                "the DMA moves a container as whole contiguous beats (open item 38)",
            )  # fmt: skip
        elif size % beat:
            yield Problem(
                "memory.align", where,
                f"{prod(lay.shape)} elements are {size} bytes, not whole {beat}-byte beats: "
                "the DMA moves whole beats (open item 38)",
                f"make the size a multiple of {beat // word} elements in the recipe's symbols",
            )  # fmt: skip
        elif lay.base % beat:
            yield Problem(
                "memory.align", where,
                f"base {lay.base} is not on a {beat}-byte beat, and the DMA moves whole beats",
                f"--set memory.{c}.{m}.base={_align_up(lay.base, beat)}",
            )  # fmt: skip


def _align_up(x: int, a: int) -> int:
    return -(-x // a) * a


def _extent(lay: Any, word: int) -> tuple[int, int]:
    lo, hi = lay.span()
    return lo, hi + word


def _memory_fit(design: Design) -> Iterable[Problem]:
    if design.memory is None:
        return
    ctx = design.context()
    pf = design.platform
    for c, m, lay in _layouts(design):
        if m not in ctx.memories:
            continue
        start, end = ctx.span(m)
        lo, hi = _extent(lay, ctx.word_bytes(m))
        if lo < start or hi > end:
            if m == "l1":
                l1 = pf.l1
                row = l1.n_banks * (l1.width_bits // 8)
                rows = -(-(hi - l1.base_addr) // row)
                fix = f"--set platform.l1.rows={max(rows, l1.rows)}, or tile the loop in the recipe"
                size = f"L1 is {end - start} B ({l1.n_banks} banks x {l1.rows} rows x {row // l1.n_banks} B)"  # fmt: skip
            else:
                fix = f"--set platform.l2.size_bytes={_align_up(hi - start, pf.l2.beat_bytes)}"
                size = f"L2 is {end - start} B"
            if (c, m) in design.memory.spec.pins():
                fix = f"pin it lower (--set memory.{c}.{m}.base=...), or {fix}"
            yield Problem("memory.fit", f"{c} in {m}", f"spans bytes [{lo}, {hi}); {size}", fix)


def _memory_overlap(design: Design) -> Iterable[Problem]:
    if design.memory is None:
        return
    ctx = design.context()
    pins = design.memory.spec.pins()
    for m in ctx.memories:
        word = ctx.word_bytes(m)
        items = [(c, lay) for c, mm, lay in _layouts(design) if mm == m]
        for (a, la), (b, lb) in _pairs(items):
            alo, ahi = _extent(la, word)
            blo, bhi = _extent(lb, word)
            if alo >= bhi or blo >= ahi:
                continue
            shared = _shared(la, lb, word)
            if not shared:
                continue
            pinned = [x for x in (a, b) if (x, m) in pins]
            fix = None
            if pinned:
                order = list(design.memory.changes)
                p = max(pinned, key=lambda x: order.index(f"{x}.{m}.base"))  # the latest pin
                end = max(_extent(lay, word)[1] for _, mm, lay in _layouts(design) if mm == m)
                free = _align_up(end, ctx.beat_bytes)
                fix = (
                    f"move the pin past everything: --set memory.{p}.{m}.base={free}, or remove it"
                )
            yield Problem(
                "memory.overlap", f"{a} and {b} in {m}",
                f"share {shared} bytes: {a} [{alo}, {ahi}), {b} [{blo}, {bhi})"
                + (f"; pinned: {', '.join(pinned)}" if pinned else ""),
                fix,
            )  # fmt: skip


def _pairs(items: list[Any]) -> Iterator[tuple[Any, Any]]:
    for i, x in enumerate(items):
        for y in items[i + 1 :]:
            yield x, y


def _shared(la: Any, lb: Any, word: int) -> int:
    """Bytes two layouts both use (element addresses, each a word)."""
    import numpy as np

    def addrs(lay: Any) -> Any:
        idx = np.indices(lay.shape).reshape(len(lay.shape), -1).T
        return lay.address(idx)

    return int(np.intersect1d(addrs(la), addrs(lb)).size) * word


for _code, _stage, _fn in (
    ("platform.load", "platform", _platform_load),
    ("platform.l2", "platform", _platform_l2),
    ("platform.banks", "platform", _platform_banks),
    ("graph.load", "graph", _graph_load),
    ("graph.symbols", "graph", _graph_symbols),
    ("graph.unbound", "graph", _graph_unbound),
    ("graph.brm", "graph", _graph_brm),
    ("graph.instance", "graph", _graph_instance),
    ("connect.streamer_key", "connect", _connect_streamer_key),
    ("connect.derived", "connect", _connect_derived),
    ("connect.lanes", "connect", _connect_lanes),
    ("connect.dtype", "connect", _connect_dtype),
    ("connect.temporal", "connect", _connect_temporal),
    ("connect.regmap", "connect", _connect_regmap),
    ("memory.pin", "memory", _memory_pin),
    ("memory.residency", "memory", _memory_residency),
    ("memory.layout", "memory", _memory_layout),
    ("memory.align", "memory", _memory_align),
    ("memory.fit", "memory", _memory_fit),
    ("memory.overlap", "memory", _memory_overlap),
):
    register_check(_code, _stage, _fn)
