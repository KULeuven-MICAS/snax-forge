"""Structural pattern detection on SDFGs.

A pattern is a DaCe PatternTransformation whose apply() is empty. It answers
one question -- does this scope look like something we can build hardware for
-- and, when it does, hands back the parameters the hardware needs. Detection
never mutates the graph, so the census can run over every kernel safely.

Each pattern supplies:

    expressions() / can_be_applied()   inherited DaCe machinery: the subgraph
                                       SHAPE and the PREDICATE, kept separate
    describe(state)                    the match turned into a descriptor

describe() is the seam where detection hands off to descriptor generation.
Its field names deliberately mirror analysis.extract_compute, so a census row
and a descriptor line up instead of being two vocabularies for one graph.

The elementwise family
----------------------

FlatElementwise and TiledElementwise describe the SAME hardware -- a datapath
of W lanes fed T times -- and differ only in how the graph spells the split:

    flat, not unrolled    W = 1        T = N              fully temporal
    flat, unrolled        W = N        T = 1              fully spatial
    tiled                 W = tile     T = ceil(N/tile)   the middle ground

with W * T = N throughout. The `shape` section states that split directly, so
a backend reads one pair of numbers instead of inferring them from whichever
pattern happened to fire. That is also why both raise to one library node.

W is a hardware width, so it has to be a concrete integer. A flat unrolled map
over symbolic N asks for unbounded lanes; `shape.bounded` records that rather
than letting it pass as a plausible-looking descriptor.

Because DaCe auto-registers PatternTransformation subclasses on import, these
also appear in recipes.transformation_matches alongside the stock ones, and in
the VS Code panel when the daemon runs in an interpreter that can import them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import dace
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.symbolic import SymExpr
from dace.transformation import transformation
from dace.transformation.optimizer import Optimizer

from snax_forge.sdfg.analysis import _node_ref, _sym, tasklet_signature

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class PatternMatch:
    """One detected occurrence, decoupled from the live DaCe match object.

    Bound PatternNode attributes are only meaningful while the match object is
    alive and paired with its state. Snapshotting here means nothing
    downstream has to hold one.
    """

    pattern: str
    state: str
    roles: dict[str, dict]
    descriptor: dict[str, Any] = field(default_factory=dict)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(asdict(self), **kwargs)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _map_facts(state: dace.SDFGState, entry: nodes.MapEntry) -> dict:
    """The hardware-relevant facts about one map.

    Same keys as the map rows in analysis.extract_compute. Recomputed rather
    than looked up because map labels are not unique -- after tiling, the
    outer and inner map share one label (`_Add__map`), so nothing can be
    keyed by it.
    """
    trip = entry.map.range.num_elements()
    trip_value = _sym(trip, None)
    uf = entry.map.unroll_factor or 0
    ((_, _, step),) = entry.map.range.ndrange()
    return {
        "label": entry.map.label,
        "params": list(entry.map.params),
        "range": str(entry.map.range),
        # Not in extract_compute's map rows: the stride is what the tile size
        # becomes after tiling, and parsing it back out of `range` would be a
        # string-shaped way to lose it.
        "step": str(step),
        "step_value": _sym(step, None),
        "schedule": str(entry.map.schedule).split(".")[-1],
        "unroll": entry.map.unroll,
        # unroll_factor is a PARTIAL-unroll width; 0 means "all".
        "unroll_factor": uf,
        "trip_count": str(trip),
        "trip_count_value": trip_value,
        # Lanes of hardware this dimension implies. unroll is the declaration
        # of spatial replication; without it the dimension is temporal.
        "lanes": (uf or trip_value) if entry.map.unroll else 1,
        "top_level": state.entry_node(entry) is None,
        # SymExpr only exists when main != approx, so its presence means a
        # clamped bound -> partial-tile predication needed.
        "ragged": any(isinstance(e, SymExpr) for _, e, _ in entry.map.range.ndrange()),
    }


def _variant(temporal: dict | None, spatial: dict | None) -> str:
    """Name the design point from which scopes are present.

        loop           temporal only    W = 1     T = N
        spatial        spatial only     W = N     T = 1
        tiled_spatial  both             W = tile  T = ceil(N/tile)

    Derived from the graph rather than from which pattern class fired, so two
    routes to the same structure get the same name. The three cases are
    exhaustive: a flat scope is temporal or spatial by its unroll flag, and a
    tiled scope always has both (TiledElementwise requires the inner unroll).
    """
    if temporal and spatial:
        return "tiled_spatial"
    if spatial:
        return "spatial"
    if temporal:
        return "loop"
    # Unreachable through the current patterns; a scope with neither would
    # mean a map that is somehow both unrolled and not.
    raise ValueError("elementwise match has no temporal or spatial scope")


def _elementwise_tasklet(graph, tasklet, *boundary) -> bool:
    """Exactly one value out per iteration, and no accumulation.

    A WCR edge is a reduction: different hardware, and a chaining barrier, so
    it must not land in an elementwise pattern.
    """
    if len(tasklet.out_connectors) != 1:
        return False
    return all(e.data.wcr is None for node in (tasklet, *boundary) for e in graph.out_edges(node))


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class ForgePattern(transformation.SingleStateTransformation):
    """Base for all SNAX-FORGE detectors.

    Subclasses declare PatternNode roles, expressions() and can_be_applied()
    exactly as any DaCe transformation would, then add describe(). apply() is
    inert here: these classify, they do not rewrite. Rewrites subclass the
    detector and override apply(), so shape and predicate are stated once.
    """

    #: Stable identifier, written into the descriptor and the census.
    pattern_name: str = "forge_pattern"

    #: Attribute names of the declared PatternNode roles, in match order.
    role_names: tuple[str, ...] = ()

    def apply(self, graph, sdfg) -> None:  # deliberately inert
        return None

    def describe(self, state: dace.SDFGState) -> dict[str, Any]:
        """Return the hardware-relevant parameters of this match."""
        raise NotImplementedError

    def bound_roles(self) -> dict[str, dict]:
        return {r: _node_ref(getattr(self, r)) for r in self.role_names}


class ElementwisePattern(ForgePattern):
    """Shared machinery for the elementwise family.

    A subclass supplies scopes() -- the map entries and exits enclosing the
    tasklet, outermost first -- and everything else follows: which scope is
    temporal, which is spatial, and how the ports are wired. One or two scopes
    are handled identically, which is what keeps flat and tiled from drifting
    into two descriptions of one piece of hardware.
    """

    #: The tasklet role is common to the whole family.
    tasklet = transformation.PatternNode(nodes.Tasklet)

    def scopes(self, state) -> tuple[list, list]:
        """Enclosing map entries and exits, outermost first."""
        raise NotImplementedError

    # -- ports -------------------------------------------------------------
    def ports(self, state: dace.SDFGState) -> tuple[dict[str, dict], dict[str, dict]]:
        """Array name -> {var, subset}, for reads and writes respectively.

        `var` is the datapath variable the array feeds; nothing else in the
        toolchain records that binding, and an RTL emitter cannot wire a
        streamer to a lane input without it.

        `subset` is the memlet subset INSIDE the scope -- the per-iteration
        access, not the whole-array footprint. It is what a streamer's address
        generator has to reproduce, and it is what lets the expansion rebuild
        the scope rather than approximate it.
        """
        reads = {
            e.data.data: {"var": e.dst_conn, "subset": str(e.data.subset)}
            for e in state.in_edges(self.tasklet)
        }
        writes = {
            e.data.data: {"var": e.src_conn, "subset": str(e.data.subset)}
            for e in state.out_edges(self.tasklet)
        }
        return reads, writes

    # -- describe ----------------------------------------------------------
    def describe(self, state: dace.SDFGState) -> dict[str, Any]:
        sdfg = state.sdfg if hasattr(state, "sdfg") else state.parent
        entries, exits = self.scopes(state)
        facts = [_map_facts(state, e) for e in entries]
        reads, writes = self.ports(state)

        # A map is spatial exactly when it declares itself unrolled; the rest
        # is temporal. Either may be absent: a flat unrolled map has no
        # temporal dimension, a flat sequential one has no spatial dimension.
        spatial = next((f for f in facts if f["unroll"]), None)
        temporal = next((f for f in facts if not f["unroll"]), None)

        elements = 1
        for entry in entries:
            elements = elements * entry.map.range.num_elements()
        lanes = spatial["lanes"] if spatial else 1

        def stream(edges, ports):
            seen: dict[str, dict] = {}
            for e in edges:
                name = e.data.data
                if not name or name in seen:
                    continue
                d = sdfg.arrays[name]
                port = ports.get(name, {})
                seen[name] = {
                    "name": name,
                    # The datapath variable this array feeds.
                    "port": port.get("var"),
                    # The per-iteration access inside the scope.
                    "subset": port.get("subset"),
                    "dtype": str(d.dtype.as_numpy_dtype()),
                    "bytes_per_element": d.dtype.bytes,
                    "shape": [str(x) for x in d.shape],
                    "transient": bool(d.transient),
                }
            return [seen[k] for k in sorted(seen)]

        return {
            "pattern": self.pattern_name,
            # The allocation split, stated once so a backend never has to
            # infer it from which pattern fired.
            "shape": {
                # Which hardware this becomes: loop, spatial, tiled_spatial.
                # Named here rather than at the raise, so the classification
                # is a fact about the graph and not about the call site.
                "variant": _variant(temporal, spatial),
                "lanes": lanes,
                "trips": temporal["trip_count"] if temporal else "1",
                "elements": str(elements),
                # Lanes are physical hardware, so an unresolved width is not a
                # buildable design. Flat + unrolled + symbolic bound is the
                # case this exists to catch.
                "bounded": isinstance(lanes, int),
            },
            # Outer scope: the sequential driver, how often the datapath runs.
            "temporal": temporal,
            # Inner scope: the datapath itself, how wide it is.
            "spatial": spatial,
            # Tasklet: the functional unit. tasklet_signature gives the op
            # histogram and AST depth -- the units needed and the depth of the
            # combinational chain.
            "datapath": {
                "label": self.tasklet.label,
                "code": self.tasklet.code.as_string,
                "language": self.tasklet.code.language.name,
                "in_connectors": {k: str(v) for k, v in self.tasklet.in_connectors.items()},
                "out_connectors": {k: str(v) for k, v in self.tasklet.out_connectors.items()},
                **tasklet_signature(self.tasklet),
            },
            # Arrays crossing the outermost scope become streamer ports.
            "streams": {
                "in": stream(state.in_edges(entries[0]), reads),
                "out": stream(state.out_edges(exits[0]), writes),
            },
        }


# ---------------------------------------------------------------------------
# Flat elementwise
# ---------------------------------------------------------------------------


class FlatElementwise(ElementwisePattern):
    """An untiled 1-D elementwise scope.

        entry -> tasklet -> exit

    A kernel before any allocation decision has been made. The single map
    carries the whole iteration space, so its unroll declaration alone decides
    whether this is one lane run N times or N lanes run once -- there is no
    second scope for a width to come from.

    Cannot collide with TiledElementwise: the top-level requirement rejects a
    tiled graph's inner scope, and a tiled graph's outer scope is followed by
    a MapEntry rather than a Tasklet, so the path shape does not match either.
    """

    pattern_name = "flat_elementwise"
    role_names = ("entry", "tasklet", "exit")

    entry = transformation.PatternNode(nodes.MapEntry)
    exit = transformation.PatternNode(nodes.MapExit)

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.entry, cls.tasklet, cls.exit)]

    def scopes(self, state):
        return [self.entry], [self.exit]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False) -> bool:
        # One top-level map scope is one accelerator. A nested hit is not a
        # scope we can hand to a cluster, and would double-count the inner
        # half of a tiled graph.
        if graph.entry_node(self.entry) is not None:
            return False
        if len(self.entry.map.params) != 1:
            return False
        # node_path_graph does not guarantee the matched exit is the one DaCe
        # pairs with this entry.
        if graph.exit_node(self.entry) is not self.exit:
            return False
        return _elementwise_tasklet(graph, self.tasklet, self.exit)


# ---------------------------------------------------------------------------
# Tiled elementwise
# ---------------------------------------------------------------------------


class TiledElementwise(ElementwisePattern):
    """A tiled, unrolled 1-D elementwise scope.

        outer_entry -> inner_entry -> tasklet -> inner_exit -> outer_exit

    What MapTiling plus an unroll declaration leave behind, read as (temporal
    loop of stride S) x (spatial datapath of W lanes). Neither number is a
    knob to search over: both were decided by whoever wrote the recipe.
    """

    pattern_name = "tiled_elementwise"
    role_names = ("outer_entry", "inner_entry", "tasklet", "inner_exit", "outer_exit")

    outer_entry = transformation.PatternNode(nodes.MapEntry)
    inner_entry = transformation.PatternNode(nodes.MapEntry)
    inner_exit = transformation.PatternNode(nodes.MapExit)
    outer_exit = transformation.PatternNode(nodes.MapExit)

    @classmethod
    def expressions(cls):
        return [
            sdutil.node_path_graph(
                cls.outer_entry,
                cls.inner_entry,
                cls.tasklet,
                cls.inner_exit,
                cls.outer_exit,
            )
        ]

    def scopes(self, state):
        return [self.outer_entry, self.inner_entry], [self.outer_exit, self.inner_exit]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False) -> bool:
        outer, inner = self.outer_entry, self.inner_entry

        if graph.entry_node(outer) is not None:
            return False

        # 1-D only for now. Higher rank is real (matadd) but it changes the
        # streamer configuration, so it gets its own pattern rather than a
        # loosened predicate here.
        if len(outer.map.params) != 1 or len(inner.map.params) != 1:
            return False

        if graph.exit_node(outer) is not self.outer_exit:
            return False
        if graph.exit_node(inner) is not self.inner_exit:
            return False

        # No unroll, no datapath. Without it the inner map is a sequential
        # walk through shared hardware, which is not what this pattern claims.
        if not inner.map.unroll:
            return False

        ((_, _, ostep),) = outer.map.range.ndrange()
        ((_, _, istep),) = inner.map.range.ndrange()

        # Outer strided (it steps by the tile size), inner unit-stride (it
        # walks the lanes). An untiled map has ostep == 1 and is rejected.
        if ostep == 1 or istep != 1:
            return False

        if not _elementwise_tasklet(graph, self.tasklet, self.inner_exit, self.outer_exit):
            return False

        # The clincher: the inner range must be written in terms of the outer
        # parameter (`__i0 = tile___i0 : tile___i0 + 64`). That is what makes
        # this a TILING rather than any two adjacent nested maps.
        return outer.map.params[0] in {str(x) for x in inner.map.range.free_symbols}


#: Every pattern the toolchain knows about. Adding a class here is all that is
#: needed for it to appear in the census.
PATTERNS: tuple[type[ForgePattern], ...] = (FlatElementwise, TiledElementwise)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def detect(sdfg: dace.SDFG, patterns=None) -> list[PatternMatch]:
    """Enumerate pattern occurrences without modifying the SDFG."""
    out: list[PatternMatch] = []
    for match in Optimizer(sdfg).get_pattern_matches(
        patterns=list(PATTERNS if patterns is None else patterns)
    ):
        state = _state_of(sdfg, match)
        out.append(
            PatternMatch(
                pattern=match.pattern_name,
                state=state.label,
                roles=match.bound_roles(),
                descriptor=match.describe(state),
            )
        )
    return out


def detect_file(path, **kwargs) -> list[PatternMatch]:
    """detect() on a stored .sdfg."""
    return detect(dace.SDFG.from_file(str(path)), **kwargs)


def _state_of(sdfg: dace.SDFG, match) -> dace.SDFGState:
    """Resolve the state a match was found in.

    state_id indexes the top-level graph, which holds while one SDFG state is
    one cluster configuration. Guarded so a future nested control-flow region
    fails loudly rather than silently mislabelling.
    """
    state = sdfg.node(match.state_id)
    if not isinstance(state, dace.SDFGState):
        raise TypeError(f"expected SDFGState at index {match.state_id}, got {type(state).__name__}")
    return state