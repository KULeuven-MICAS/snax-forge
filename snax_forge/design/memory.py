"""The memory plan: where every container lives (DP1b, D86).

The plan gives each container a layout per memory it lives in: ``base``
(byte address of index (0, ..., 0)), ``shape`` and one byte stride per
dimension (``snax_forge.lower.layout.Layout``, D70, D74). Banks follow from
addresses under the fixed word-interleaved map (open item 18) and are not
stated. It is made by three passes, each registered by name so a policy can
replace any one of them (``register_memory_pass``):

    residency   which memories a container lives in      default: a
                container that is not transient in L2 and L1 when the
                platform has an L2 (the DMA loads and stores it), else in
                L1; a transient in L1
    layout      shape and strides per container and      contiguous:
                memory, base 0                           row-major, one
                                                         element per word
    placement   base per container and memory            contiguous: per
                memory, in the graph's container order,
                from the memory's first byte, no gaps; a layout the DMA
                moves (in L2 and L1) starts on a wide beat and keeps its
                last beat to itself (a size that is not whole beats is
                padded, D105), others start on a word

**Pins.** ``--set memory.<container>.<memory>.base=N`` fixes a base; the
placement puts everything else around the pinned layouts, never over them.
``--set memory.passes.<pass>=<name>`` picks another registered pass. Both
are ``changes`` of the ``MemorySpec``, kept in the working copy
``out/design/<name>/memory.json`` so a run can continue from it
(``--memory``), as the platform's are.

**What passes see** (``MemoryContext``): the graph with its symbols bound,
the platform, each container's shape and each memory's word and range, and,
for later policies, every accelerator port's element-index stream
(``accesses``: the elements in the order the port takes them, beat by beat,
from its memlet over the temporal maps around it; independent of any
layout) grouped by the node that runs them together. A policy that avoids
bank conflicts turns candidate layouts into bank sequences for the ports of
one group and compares them, without running the model (DSE4). The
defaults read only shapes and sizes.

Nothing here judges the plan: the ``memory`` checks (check.py) do, so a
plan that leaves a memory or overlaps a pin is reported with its fix.
"""

from __future__ import annotations

import difflib
import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from snax_forge import expr
from snax_forge.dfg import Graph, Node
from snax_forge.dfg.subset import parse_dim
from snax_forge.lower.layout import Layout, moved_bytes, round_up
from snax_forge.snax_model.config import plain, to_json

from .platform import Platform
from .problems import DesignError, Problem

PASS_KINDS = ("residency", "layout", "placement")
DEFAULT_PASSES = {"residency": "default", "layout": "contiguous", "placement": "contiguous"}
MEMORIES = ("l2", "l1")  # the order layouts are written in


# =============================================================================
# What the user controls: passes and pins
# =============================================================================


@dataclass(frozen=True)
class MemorySpec:
    """The passes to run and the changes (pins) made with ``--set memory.PATH=VALUE``."""

    passes: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PASSES))
    changes: dict[str, Any] = field(default_factory=dict)

    def pins(self) -> dict[tuple[str, str], int]:
        """``(container, memory) -> base`` for every pinned base."""
        out = {}
        for path, v in self.changes.items():
            parts = path.split(".")
            if len(parts) == 3 and parts[2] == "base":
                out[(parts[0], parts[1])] = v
        return out

    def with_changes(self, sets: Sequence[tuple[str, Any]] = ()) -> MemorySpec:
        """The spec with ``sets`` applied; the shape of each path is checked here,
        what it names (containers, memories, passes) by the memory.pin check."""
        problems = []
        passes, changes = dict(self.passes), dict(self.changes)
        for path, value in sets:
            where = f"memory.{path}"
            parts = path.split(".")
            if len(parts) == 2 and parts[0] == "passes":
                if parts[1] not in PASS_KINDS:
                    problems.append(Problem("memory.pin", where, f"no pass {parts[1]!r} (passes: {list(PASS_KINDS)})"))  # fmt: skip
                elif not isinstance(value, str):
                    problems.append(Problem("memory.pin", where, f"must be a pass name, got {value!r}"))  # fmt: skip
                else:
                    passes[parts[1]] = value
                    changes.pop(path, None)
                    changes[path] = value
                continue
            if len(parts) != 3 or not all(parts):
                problems.append(
                    Problem(
                        "memory.pin", where,
                        "give <container>.<memory>.base or passes.<pass>",
                    )
                )  # fmt: skip
                continue
            if parts[2] != "base":
                problems.append(Problem("memory.pin", where, "only a base can be pinned (<container>.<memory>.base)"))  # fmt: skip
                continue
            if isinstance(value, bool) or type(value) is not int:
                problems.append(Problem("memory.pin", where, f"must be an int byte address, got {value!r}"))  # fmt: skip
                continue
            changes.pop(path, None)
            changes[path] = value
        if problems:
            raise DesignError(problems, "memory --set")
        return MemorySpec(passes, changes)

    def to_dict(self) -> dict[str, Any]:
        return {"passes": dict(self.passes), "changes": plain(self.changes)}

    @classmethod
    def from_dict(cls, d: Any) -> MemorySpec:
        """From a memory working copy or a design point's ``memory`` (its layouts ignored)."""
        if not isinstance(d, Mapping):
            raise DesignError([Problem("memory.pin", "memory", f"must be an object, got {d!r}")])
        passes = d.get("passes", DEFAULT_PASSES)
        changes = d.get("changes", {})
        if not isinstance(passes, Mapping) or not isinstance(changes, Mapping):
            raise DesignError([Problem("memory.pin", "memory", "passes and changes must be objects")])  # fmt: skip
        spec = cls(dict(DEFAULT_PASSES))
        sets = [(f"passes.{k}", v) for k, v in passes.items()] + list(changes.items())
        new = spec.with_changes(sets)
        return MemorySpec(new.passes, dict(changes))

    @classmethod
    def load(cls, path: str | Path) -> MemorySpec:
        try:
            d = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise DesignError([Problem("memory.pin", str(path), str(e))]) from None
        return cls.from_dict(d)


# =============================================================================
# What passes see
# =============================================================================


@dataclass(frozen=True)
class Access:
    """One accelerator port's elements in the order it takes them."""

    node: str
    connector: str
    container: str
    instance: str
    direction: str
    indices: np.ndarray  # (beats, lanes, ndim) element indices


@dataclass
class MemoryContext:
    """The graph (symbols bound) and the platform, with what passes ask about them."""

    graph: Graph
    platform: Platform

    @property
    def symbols(self) -> dict[str, int]:
        return {k: v for k, v in self.graph.symbols.items() if v is not None}

    @property
    def containers(self) -> list[str]:
        return list(self.graph.containers)

    def shape(self, container: str) -> tuple[int, ...]:
        c = self.graph.containers[container]
        return tuple(int(expr.evaluate(s, self.symbols)) for s in c.shape)

    @property
    def memories(self) -> list[str]:
        return ["l2", "l1"] if self.platform.l2 is not None else ["l1"]

    def word_bytes(self, mem: str) -> int:
        cfg = self.platform.l2 if mem == "l2" else self.platform.l1
        return cfg.width_bits // 8

    @property
    def beat_bytes(self) -> int:
        """The DMA's wide beat: the alignment of what it moves."""
        return self.platform.l1.wide_bits // 8

    def span(self, mem: str) -> tuple[int, int]:
        """The memory's bytes, ``[start, end)``."""
        if mem == "l2":
            l2 = self.platform.l2
            return l2.base_addr, l2.base_addr + l2.size_bytes
        l1 = self.platform.l1
        return l1.base_addr, l1.base_addr + l1.n_banks * l1.rows * (l1.width_bits // 8)

    @cached_property
    def accesses(self) -> list[Access]:
        """Every accelerator port's element-index stream, in execution order."""
        out = []
        for node, outer in self.graph.walk():
            if node.kind != "accelerated":
                continue
            for conn, m in {**node.inputs, **node.outputs}.items():
                out.append(
                    Access(
                        node.id, conn, m.data, node.attrs["instance"],
                        "in" if conn in node.inputs else "out",
                        _stream(m.subset, outer, self.symbols),
                    )
                )  # fmt: skip
        return out

    def groups(self) -> dict[str, list[Access]]:
        """The accesses that run together: one group per accelerated node."""
        out: dict[str, list[Access]] = {}
        for a in self.accesses:
            out.setdefault(a.node, []).append(a)
        return out


def _stream(subset: list[Any], outer: tuple[Node, ...], symbols: dict[str, int]) -> np.ndarray:
    """A memlet's element indices over the maps around it, outermost first: (beats, lanes, ndim)."""
    loops = []
    for m in outer:
        if m.kind != "map":
            continue
        b, e, s = parse_dim(m.attrs["range"], m.id)
        loops.append(
            (m.attrs["var"], range(expr.evaluate(b, symbols), expr.evaluate(e, symbols), s))
        )
    beats = []
    for values in itertools.product(*(r for _, r in loops)):
        env = {**symbols, **{v: x for (v, _), x in zip(loops, values, strict=True)}}
        axes = []
        for dim in subset:
            parts = parse_dim(dim, "subset")
            if len(parts) == 1:
                axes.append(np.array([expr.evaluate(parts[0], env)]))
            else:
                lo, hi = expr.evaluate(parts[0], env), expr.evaluate(parts[1], env)
                axes.append(np.arange(lo, hi, parts[2]))
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))
        beats.append(mesh)
    if not beats:
        return np.zeros((0, 0, len(subset)), dtype=np.int64)
    return np.stack(beats).astype(np.int64)


# =============================================================================
# The passes
# =============================================================================

Residency = Callable[[MemoryContext], dict[str, list[str]]]
LayoutPass = Callable[[MemoryContext, dict[str, list[str]]], dict[str, dict[str, Layout]]]
Placement = Callable[
    [MemoryContext, dict[str, dict[str, Layout]], dict[tuple[str, str], int]],
    dict[str, dict[str, int]],
]

MemoryPass = Residency | LayoutPass | Placement
MEMORY_PASSES: dict[str, dict[str, MemoryPass]] = {k: {} for k in PASS_KINDS}


def register_memory_pass(kind: str, name: str, fn: MemoryPass) -> None:
    """Add a pass of ``kind`` (residency, layout, placement) under ``name``."""
    if kind not in PASS_KINDS:
        raise ValueError(f"unknown pass kind {kind!r} ({list(PASS_KINDS)})")
    if name in MEMORY_PASSES[kind]:
        raise ValueError(f"{kind} pass {name!r} already registered")
    MEMORY_PASSES[kind][name] = fn


def default_residency(ctx: MemoryContext) -> dict[str, list[str]]:
    out = {}
    for name, c in ctx.graph.containers.items():
        out[name] = ["l1"] if c.transient or ctx.platform.l2 is None else ["l2", "l1"]
    return out


def contiguous_layout(
    ctx: MemoryContext, residency: dict[str, list[str]]
) -> dict[str, dict[str, Layout]]:
    return {
        c: {m: Layout.contiguous(0, ctx.shape(c), ctx.word_bytes(m)) for m in mems}
        for c, mems in residency.items()
    }


def _extent(lay: Layout, word: int) -> tuple[int, int]:
    """Bytes ``[lo, hi)`` a layout covers."""
    lo, hi = lay.span()
    return lo, hi + word


def contiguous_placement(
    ctx: MemoryContext,
    layouts: dict[str, dict[str, Layout]],
    pins: dict[tuple[str, str], int],
) -> dict[str, dict[str, int]]:
    bases: dict[str, dict[str, int]] = {c: {} for c in layouts}
    for mem in ctx.memories:
        word = ctx.word_bytes(mem)
        taken: list[tuple[int, int]] = []
        for (c, m), b in pins.items():
            if m == mem and c in layouts and mem in layouts[c]:
                lo, hi = _extent(replace(layouts[c][mem], base=b), word)
                taken.append((lo, hi))
                bases[c][mem] = b
        cursor = ctx.span(mem)[0]
        for c in ctx.containers:
            if mem not in layouts.get(c, {}) or (c, mem) in pins:
                continue
            lay = layouts[c][mem]
            moved = "l1" in layouts[c] and "l2" in layouts[c]
            a = ctx.beat_bytes if moved else word
            lo0, hi0 = _extent(lay, word)  # with base 0
            if moved:  # the DMA moves whole beats: keep the padding free (D105)
                hi0 = lo0 + moved_bytes(lay, word, a)
            base = round_up(cursor - lo0, a)
            clash = True
            while clash:
                clash = False
                for tlo, thi in sorted(taken):
                    if base + lo0 < thi and tlo < base + hi0:
                        base, clash = round_up(thi - lo0, a), True
            bases[c][mem] = base
            taken.append((base + lo0, base + hi0))
            cursor = base + hi0
    return bases


register_memory_pass("residency", "default", default_residency)
register_memory_pass("layout", "contiguous", contiguous_layout)
register_memory_pass("placement", "contiguous", contiguous_placement)


# =============================================================================
# The plan
# =============================================================================


@dataclass(frozen=True)
class MemoryPlan:
    """The passes run, the pins, and the layouts they gave: container -> memory -> Layout."""

    passes: dict[str, str]
    changes: dict[str, Any]
    layouts: dict[str, dict[str, Layout]]

    @property
    def spec(self) -> MemorySpec:
        return MemorySpec(dict(self.passes), dict(self.changes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "passes": dict(self.passes),
            "changes": plain(self.changes),
            "layouts": {
                c: {m: lay.to_dict() for m, lay in mems.items()} for c, mems in self.layouts.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Any) -> MemoryPlan:
        spec = MemorySpec.from_dict(d)
        layouts = d.get("layouts", {})
        problems, out = [], {}
        if not isinstance(layouts, Mapping):
            raise DesignError([Problem("memory.layout", "memory.layouts", "must be an object")])
        for c, mems in layouts.items():
            out[c] = {}
            for m, lay in (mems or {}).items():
                try:
                    out[c][m] = Layout.from_dict(lay)
                except (TypeError, ValueError) as e:
                    problems.append(Problem("memory.layout", f"memory.layouts.{c}.{m}", str(e)))
        if problems:
            raise DesignError(problems)
        return cls(spec.passes, spec.changes, out)

    def to_json(self) -> str:
        return to_json(self.to_dict())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())


def plan(ctx: MemoryContext, spec: MemorySpec) -> MemoryPlan:
    """Run the spec's passes; the plan is judged by the memory checks, not here."""
    residency = MEMORY_PASSES["residency"][spec.passes["residency"]](ctx)
    shapes = MEMORY_PASSES["layout"][spec.passes["layout"]](ctx, residency)
    pins = {k: v for k, v in spec.pins().items() if k[0] in shapes and k[1] in shapes[k[0]]}
    bases = MEMORY_PASSES["placement"][spec.passes["placement"]](ctx, shapes, pins)
    layouts = {
        c: {
            m: replace(shapes[c][m], base=bases[c][m])
            for m in MEMORIES
            if m in shapes[c]
        }
        for c in shapes
    }  # fmt: skip
    return MemoryPlan(dict(spec.passes), dict(spec.changes), layouts)


def pin_problems(ctx: MemoryContext, spec: MemorySpec) -> list[Problem]:
    """Pins and passes that name nothing: an unknown container, memory or pass."""
    out = []
    for kind, name in spec.passes.items():
        if name not in MEMORY_PASSES[kind]:
            out.append(
                Problem(
                    "memory.pin", f"memory.passes.{kind}",
                    f"no {kind} pass {name!r} (registered: {sorted(MEMORY_PASSES[kind])})",
                )
            )  # fmt: skip
    if out:
        return out
    residency = MEMORY_PASSES["residency"][spec.passes["residency"]](ctx)
    for c, m in spec.pins():
        where = f"memory.{c}.{m}.base"
        if c not in ctx.graph.containers:
            close = difflib.get_close_matches(c, ctx.containers, n=1)
            out.append(
                Problem(
                    "memory.pin", where, f"no container {c!r} (containers: {ctx.containers})",
                    f"did you mean memory.{close[0]}.{m}.base?" if close else None,
                )
            )  # fmt: skip
        elif m not in residency.get(c, []):
            out.append(
                Problem(
                    "memory.pin", where,
                    f"{c} does not live in {m!r} (it lives in {residency.get(c, [])})",
                )
            )  # fmt: skip
    return out
