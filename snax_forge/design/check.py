"""The design checks: does this platform fit this bound graph (DP1a, D85)?

The design step pairs a bound graph (the last step of a recipe) with a
platform. Before it writes anything it runs every registered check and
collects every problem (problems.py), each with a code, what it is about,
why it fails and the change that fixes it. Checks belong to stages:

    platform   the platform alone: keys, values, L1 and L2 agreeing, banks
    graph      the graph alone: symbols bound, everything bound to an
               accelerator, BRMs and implementations known, instance names
    connect    the two together: streamer entries, lanes, dtypes, registers

``platform`` and ``graph`` run independently; ``connect`` runs only when
both passed, since its checks read what they established. DP1b adds a
``memory`` stage after ``connect``. A new check is registered with
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

from snax_forge.brm import BrmError, Instance
from snax_forge.dfg import DfgError, Graph, Node
from snax_forge.snax_model.ctrl import STATUS, DmaAdapter, StreamerAdapter

from .platform import DERIVED, Platform
from .problems import DesignError, Problem
from .streamers import (
    accelerated,
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
    graph_path: str | Path, platform_path: str | Path, sets: Sequence[tuple[str, Any]] = ()
) -> Design:
    """Load both inputs; what fails to load becomes a problem, never an exception.

    The platform is the working copy (``Platform.with_changes``), with ``sets``
    applied when they are given.
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
    return Design(platform, graph, problems, str(graph_path))


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
    ("connect.regmap", "connect", _connect_regmap),
):
    register_check(_code, _stage, _fn)
