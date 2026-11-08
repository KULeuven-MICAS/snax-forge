"""Sandbox transforms (D72, D73): each takes a graph and returns a new one.

A transform is registered with ``register_transform(name, fn, ints=...,
site=...)``; ``fn(graph, **params) -> graph`` never changes its input.
``ints`` names the params that are ints, which a recipe may give as
expressions over its own params (recipe.py). The result is a new ``Graph``,
validated as any graph is; the reference check after the step is run.py's.

**Sites** (D72). ``site(graph, selected)`` is the graph-shape test of where a
transform applies, for a node id, a container name, a group's or an accelerator's name
(``site_of`` takes the editor's selection, one item or several: a transform that takes
several, ``encapsulate`` and ``fuse_map``, registers a ``multi`` site over the list, any
other refuses several with ``takes one selection``): it returns the params
the selection fills (the selection itself and what follows from it) or
raises the error the transform raises there, since the transform calls it
first, so each check lives in one place. A site decides from the graph
alone; the checks on the user's params (a factor, a kind, a BRM, a depth, a
name) stay in the transform.

``split(map, factor)`` and ``split_map(map, factor)``
    ``split`` splits a map ``v`` over ``b:e`` (step 1, no ``loop.kind``) into
    an outer map ``v_1`` over ``0:(e - b) // factor``, keeping the map's id
    and attrs and recording the split (``loop.split``), around an inner map
    ``<id>_inner`` with ``v_2`` over ``0:factor``, both untagged, neutral
    names that suggest no tag (D73); a later split takes the next free pair,
    ``v_3`` and ``v_4``, then ``v_5`` and ``v_6``; every use of ``v`` in
    the body becomes ``b + factor * v_1 + v_2``. A length n that is not a
    multiple of the factor (D144) runs ``v_1`` over ceil(n / factor) and
    records n in ``loop.split`` (``valid``): only the lanes with
    ``factor * v_1 + v_2`` below n are the kernel's, the rest pad lanes
    (dfg/pad.py), which the reference executor neither reads nor writes and
    the containers hold in memory as padding. ``split_map`` is ``split``,
    then ``tag_map`` ``temporal`` on the outer map and ``spatial`` on the
    inner, in one step, with the names of the tags it gives: ``v_t``,
    ``v_s`` and ``<id>_s`` (``SPLIT_TAGGED``), which the recipes bind and
    compose.py's ``split_reduce`` uses too.
``tag_map(map, kind)`` and ``untag_map(map)``
    ``tag_map`` tags an untagged map ``temporal``, ``tile`` or ``spatial`` as
    it is, without splitting it (matadd's row map); a tagged map is an
    error. ``untag_map`` removes the tag, whatever set it, but not from a map
    a bound unit is inside (unbind first).
``bind(node, brm, implementation, instance, params={})``
    replaces a tasklet and the spatial map around it (the map's only
    child), or a fused nest's spatial map and its two tasklets
    (compose.py's ``fuse_map``; the first's inputs, the second's output,
    patterns.py's ``fused_view``), or a spatial map whose one tasklet
    folds, given by the map (``lane_fold`` without a front op,
    ``lane_accumulate``), by an accelerated node with the
    (first) tasklet's id, bound to an
    instance of a library BRM. The BRM's pattern matcher (patterns.py) says
    which connector feeds which port. Every map around the spatial map
    must be ``temporal``: they are the firing loops (an untagged one is
    tagged with ``tag_map``; a ``tile`` one is not lowered yet, LOW3b). The
    lanes design param is read off the
    spatial bound, the other design params come from ``params`` or
    the BRM's defaults, and ``Brm.resolve`` checks them all. Each
    memlet becomes the port's lanes as a range over the spatial variable,
    the container dtype must be the port's, and, for a BRM in the
    ``affine`` notation, the memlets together with the temporal maps around
    must give the elements in the order of the port's nest (D70); in
    the ``stream`` notation the memlet is the order. The tasklet's code, on
    the BRM's port names, must be the BRM's ``function.code``: a code table (brm.py's ``Code``,
    ``simd``'s one entry per ``op`` and ``streams``, then ``axpby``'s two) is tried in order,
    and the first entry whose present input ports (``Port.when`` at its params) take the
    tasklet's array inputs, in some assignment (port order first, then every other), and
    whose code equals the tasklet's on them up to its slots and to ``+``, ``*``, ``min`` and
    ``max`` commuting (``expr.unify``: a slot takes an int sub-expression with no names or a
    scalar operand, the same wherever it appears) is taken; its params and int slot values
    are set on the instance. A **scalar operand** is an input whose memlet names no variable
    of the maps around it, on a kernel's scalar argument (``sdfg.scalar``, axpy's
    ``alpha[0]``): it maps to no port, only to a slot, which the instance leaves unset (a CSR)
    and the accelerated node keeps as an input connector named after the slot (``k1:
    alpha[0]``); such an input on another container is refused, naming it, and an input on a
    scalar whose subset uses a map variable is an array input. Refused, each naming what it
    found: a ``params`` value that contradicts the entry's params or a slot's value (or gives
    a slot read from a scalar), a slot value that does not fit the op type (NumPy's result type
    of the input ports' dtypes, the literal weak, NEP 50) or a scalar that the op type does not
    hold, and no matching entry, listing every entry's code with its params. A fused nest of
    two plain tasklets (``fuse_map``, the second no fold) binds as one, the second's code with
    the first's expression in place of the local value (axpy's ``out = y + alpha * x``,
    patterns.py's ``fused_view``). The
    accelerated node carries that code and the replaced subtree (D82), and, for a fused nest, the nest's local value
    (``loop.local``), which only its two tasklets named: it leaves the graph's containers for
    the node's ``locals`` (dfg/kinds.py), and ``unbind`` puts it back at its index. A folding
    output (``reduce``) whose memlet does not use the spatial variable (dot's
    ``out[0]``) folds every lane into one: its port must have one lane, its
    index becomes the range of that lane (``out[0:1]``, no ``reduce`` on
    the accelerated node), and its order is checked over the temporal maps
    it uses, one beat per T firings, T from ``named_rates`` (D102).
``unbind(node)`` and ``join_map(map)``
    the inverses: the subtree an accelerated node replaced, and the
    map a split pair was split from (``loop.split``), whatever their tags,
    so a graph at any step can go back to the imported one.
``interchange_map(map)``
    a map ``X`` whose body is one map ``Y`` and nothing else trades places
    with it: ``Y`` becomes the parent, ``X`` its only child, and ``Y``'s
    body ``X``'s; each keeps its id, variable, range and attrs (its tag
    among them). Applied to ``Y`` it undoes itself. rowsum's lane
    accumulator puts the spatial row map inside the column map this way.
    Refused, naming what it found: a body other than one map; a range that
    names the other's variable; ``X`` inside an accelerated node or a bound
    unit under ``Y`` (unbind first); ``X`` or ``Y`` a fused nest's outer map
    (``loop.fused``: unfuse_map first); a fold under ``Y`` over ``X``'s
    variable whose op is not order-free (dfg/reduce.py's ``order_free``;
    every registered op is, D28).
``encapsulate(accelerator, units)`` and ``unencapsulate(accelerator)``
    put the groups of the instances ``units`` into one accelerator of that
    name (D148): every accelerated node of each carries ``accelerator``. A
    group is one instance, named after it, unless ``link`` merged several
    (compose.py). The
    groups keep the graph's order, which ``units`` must give (the CSRs and
    the tasks follow it, C§17); a unit already in an accelerator, a name an
    instance or an accelerator has, or a unit the graph lacks is an error.
    ``unencapsulate`` takes the attr off again: each group its own
    accelerator. Their sites take a unit (a node id or an instance) and
    fill ``units`` with its instance, or ``accelerator`` with its
    accelerator (``unencapsulate``'s also takes an accelerator's name).
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from snax_forge import expr
from snax_forge.common import fresh
from snax_forge.dfg import (
    REDUCE_OPS,
    Container,
    DfgError,
    Graph,
    Memlet,
    Node,
    named_rates,
    parse_dim,
)
from snax_forge.dfg.graph import SCALAR
from snax_forge.dfg.subset import dim_names, format_dim, map_range
from snax_forge.expr import ExprError

from .patterns import fused_view, match
from .recipe import SandboxError

Transform = Callable[..., Graph]
Site = Callable[[Graph, str], dict[str, Any]]
MultiSite = Callable[[Graph, list[str]], dict[str, Any]]


@dataclass(frozen=True)
class TransformSpec:
    fn: Transform
    ints: tuple[str, ...] = ()
    site: Site | None = None
    multi: MultiSite | None = None  # its site over a selection of several items (D72)
    # the inputs the step fixes, {container: array} from (graph, **params): its reference check
    # runs the graph before it and after it on them (run.py's checked_step; bake_scalar's value)
    fixes: Callable[..., dict[str, Any]] | None = None


TRANSFORMS: dict[str, TransformSpec] = {}


def register_transform(
    name: str,
    fn: Transform,
    ints: tuple[str, ...] = (),
    site: Site | None = None,
    multi: MultiSite | None = None,
    fixes: Callable[..., dict[str, Any]] | None = None,
) -> None:
    """Make ``{"transform": name}`` usable in a recipe (principle 6), with its site (D72),
    for a transform that takes several items its site over a selection (``multi``), and for
    one that fixes an input (``bake_scalar``) the inputs it fixes (``fixes``)."""
    if name in TRANSFORMS:
        raise ValueError(f"transform {name!r} already registered")
    TRANSFORMS[name] = TransformSpec(fn, tuple(ints), site, multi, fixes)


def site_of(name: str, g: Graph, selected: str | list[str]) -> dict[str, Any]:
    """Transform ``name``'s site on a selection, one item or several (D72): one item goes to its
    ``site`` as before (or its ``multi`` when it has one, which must agree); several to its
    ``multi``, and a transform without one refuses them (``site.one``, "takes one
    selection")."""
    spec = TRANSFORMS[name]
    items = [selected] if isinstance(selected, str) else list(selected)
    if len(items) == 1 and spec.site is not None and spec.multi is None:
        return spec.site(g, items[0])
    if spec.multi is not None:
        return spec.multi(g, items)
    raise SiteError(f"{name}: takes one selection, not {len(items)}")


class SiteError(SandboxError):
    """A selection a transform's site cannot take: several items for a one-item site."""

    code = "site.one"


def load_brm(name: str) -> Any:
    """The library BRM called ``name`` (a seam for tests)."""
    from snax_forge.brm import load_brm as load

    return load(name)


# =============================================================================
# Helpers
# =============================================================================


def _copy(g: Graph) -> Graph:
    return copy.deepcopy(g)


def _locate(g: Graph, node_id: str) -> tuple[list[Node], int, tuple[Node, ...]]:
    """The body list holding ``node_id``, its index there, and its enclosing nodes."""

    def go(body: list[Node], outer: tuple[Node, ...]) -> Any:
        for i, n in enumerate(body):
            if n.id == node_id:
                return body, i, outer
            if n.body:
                found = go(n.body, (*outer, n))
                if found:
                    return found
        return None

    found = go(g.body, ())
    if not found:
        raise SandboxError(f"{g.name}: no node {node_id!r}")
    return found


def _map_dims(body: list[Node], fn: Callable[[Any], Any], splits: bool = False) -> None:
    """``fn`` on every memlet subset dimension and map range under ``body`` (an accelerated
    node's ``replaced`` subtree included); with ``splits``, on each split's range too."""
    for n in body:
        for m in (*n.inputs.values(), *n.outputs.values()):
            m.subset = [fn(d) for d in m.subset]
        if n.kind == "map":
            n.attrs["range"] = fn(n.attrs["range"])
            if splits and n.attrs.get("loop.split"):
                split = n.attrs["loop.split"]
                n.attrs["loop.split"] = {**split, "range": fn(split["range"])}
                if "valid" in split:
                    n.attrs["loop.split"]["valid"] = fn(split["valid"])
        if n.kind == "accelerated" and n.attrs.get("replaced"):
            old = Node.from_dict(n.attrs["replaced"], n.id)
            _map_dims([old], fn, splits)
            n.attrs["replaced"] = old.to_dict()
        if n.body:
            _map_dims(n.body, fn, splits)


def _substitute_body(body: list[Node], values: Mapping[str, Any]) -> None:
    """Replace names in every memlet subset, map range and split range under ``body``."""
    _map_dims(body, lambda d: _substitute_dim(d, values), splits=True)


def _substitute_dim(d: Any, values: Mapping[str, Any]) -> Any:
    return format_dim(tuple(expr.substitute(p, values) for p in parse_dim(d, "dimension")))


def _bound(g: Graph, v: Any, what: str) -> int:
    env = g.bound_symbols
    try:
        return expr.evaluate(v, env)
    except ExprError as e:
        raise SandboxError(f"{what}: {e} (bind the symbol in the recipe)") from None


def _result(g: Graph) -> Graph:
    """``g`` rebuilt from its dict, so the new graph is validated as a file would be."""
    try:
        return Graph.from_dict(g.to_dict())
    except DfgError as e:
        raise SandboxError(f"the result is not a valid graph: {e}") from None


def _map_node(g: Graph, map: str, what: str) -> Node:
    body, i, _ = _locate(g, map)
    if body[i].kind != "map":
        raise SandboxError(f"{what}: a {body[i].kind!r} is not a map")
    return body[i]


# =============================================================================
# split and split_map
# =============================================================================


def _split_site(g: Graph, map: str, what: str) -> dict[str, Any]:
    """An untagged map with a step of 1."""
    node = _map_node(g, map, what)
    if "loop.kind" in node.attrs:
        raise SandboxError(f"{what}: already {node.attrs['loop.kind']}")
    if parse_dim(node.attrs["range"], what)[2] != 1:
        raise SandboxError(f"{what}: only a step of 1 is split, the range is {node.attrs['range']}")
    return {"map": map}


# The suffixes of a split's outer variable, inner variable and inner map id: ``split``'s
# neutral ones and ``split_map``'s (and ``split_reduce``'s), which tag in the same step (D73).
SPLIT_NEUTRAL = ("_1", "_2", "_inner")
SPLIT_TAGGED = ("_t", "_s", "_s")


def split(g: Graph, map: str, factor: int) -> Graph:
    return _split(g, map, factor, f"split {map!r}", SPLIT_NEUTRAL)[0]


def split_map(g: Graph, map: str, factor: int) -> Graph:
    g, inner = _split(g, map, factor, f"split_map {map!r}", SPLIT_TAGGED)
    return tag_map(tag_map(g, map, "temporal"), inner, "spatial")


def _split(g: Graph, map: str, factor: int, what: str,
           names: tuple[str, str, str]) -> tuple[Graph, str]:  # fmt: skip
    """``split`` with ``names`` (outer var, inner var, inner id suffixes): the graph and the
    inner map's id."""
    _split_site(g, map, what)
    if type(factor) is not int or factor < 1:
        raise SandboxError(f"{what}: factor {factor!r} must be a positive int")
    g = _copy(g)
    body, i, _ = _locate(g, map)
    node = body[i]
    begin = parse_dim(node.attrs["range"], what)[0]
    length = _length(node)
    n = _bound(g, length, what)
    tail = n % factor != 0
    var = node.attrs["var"]
    taken = (
        set(g.symbols)
        | set(g.containers)
        | {nd.attrs["var"] for nd, _ in g.walk() if nd.kind == "map"}
    )
    if names == SPLIT_NEUTRAL:  # the next free pair: v_1 and v_2, then v_3 and v_4, ... (D73)
        k = 1
        while f"{var}_{k}" in taken or f"{var}_{k + 1}" in taken:
            k += 2
        var_t, var_s = f"{var}_{k}", f"{var}_{k + 1}"
    else:
        var_t, var_s = fresh(f"{var}{names[0]}", taken), fresh(f"{var}{names[1]}", taken)
    ids = {nd.id for nd, _ in g.walk()}
    inner = Node(
        fresh(f"{map}{names[2]}", ids),
        "map",
        attrs={"var": var_s, "range": f"0:{factor}"},
        body=node.body,
    )
    start = "" if begin == 0 else f"{begin} + "
    _substitute_body(inner.body, {var: f"{start}{factor} * {var_t} + {var_s}"})
    firings = f"({length} + {factor - 1}) // {factor}" if tail else f"({length}) // {factor}"
    split = {"var": var, "range": node.attrs["range"]}  # for join_map
    if tail:  # the lanes past n are padding
        split["valid"] = expr.canonical(length)
    node.attrs = {
        "var": var_t,
        "range": format_dim((0, expr.canonical(firings), 1)),
        **{k: v for k, v in node.attrs.items() if k not in ("var", "range")},
        "loop.split": split,
    }
    node.body = [inner]
    return _result(g), inner.id


# =============================================================================
# tag_map and untag_map
# =============================================================================

TAGS = ("temporal", "tile", "spatial")


def _tag_site(g: Graph, map: str, what: str) -> dict[str, Any]:
    """An untagged map."""
    node = _map_node(g, map, what)
    if "loop.kind" in node.attrs:
        raise SandboxError(f"{what}: already {node.attrs['loop.kind']}")
    return {"map": map}


def tag_map(g: Graph, map: str, kind: str) -> Graph:
    what = f"tag_map {map!r}"
    _tag_site(g, map, what)
    if kind not in TAGS:
        raise SandboxError(f"{what}: kind {kind!r} is not one of {list(TAGS)}")
    g = _copy(g)
    node = g.node(map)
    attrs: dict[str, Any] = {}
    for k, v in node.attrs.items():  # the kind before loop.split, as split_map writes them
        if k == "loop.split":
            attrs["loop.kind"] = kind
        attrs[k] = v
    node.attrs = {**attrs, "loop.kind": kind}
    return _result(g)


def _untag_site(g: Graph, map: str, what: str) -> dict[str, Any]:
    """A tagged map no bound unit is inside."""
    body, i, _ = _locate(g, map)
    node = body[i]
    if node.kind != "map" or "loop.kind" not in node.attrs:
        raise SandboxError(f"{what}: not a tagged map")
    inside = [n.id for n, outer in g.walk()
              if n.kind == "accelerated" and any(o is node for o in outer)]  # fmt: skip
    if inside:
        raise SandboxError(f"{what}: the bound unit {inside[0]!r} is inside it (unbind first)")
    return {"map": map}


def untag_map(g: Graph, map: str) -> Graph:
    """The inverse of tag_map: the map without its tag."""
    _untag_site(g, map, f"untag_map {map!r}")
    g = _copy(g)
    node = g.node(map)
    node.attrs = {k: v for k, v in node.attrs.items() if k != "loop.kind"}
    return _result(g)


# =============================================================================
# bind
# =============================================================================


def bind(
    g: Graph,
    node: str,
    brm: str,
    implementation: str,
    instance: str,
    params: Mapping[str, Any] | None = None,
) -> Graph:
    g = _copy(g)
    what = f"bind {node!r}"
    site = _bind_site(g, node, what)
    bound = _bind_check(g, site, brm, implementation, params, what)
    return _bind_rewrite(g, site, bound, brm, implementation, instance, what)


@dataclass(frozen=True)
class _Site:
    """Where ``bind`` binds: the tasklet (a fused nest's two as one, ``pattern`` the node its
    ports are matched on), its spatial map and the firing loops around it, outermost first."""

    tasklet: Node
    pattern: Node
    smap: Node
    temporal: list[Node]


def _bind_params(g: Graph, node: str) -> dict[str, Any]:
    """bind's site: ``_bind_site``."""
    _bind_site(g, node, f"bind {node!r}")
    return {"node": node}


@dataclass(frozen=True)
class _Bound:
    """What ``bind`` checked: the BRM, connector -> port, the instance, the spatial map's
    variable and bound, the tasklet's memlets and its folding outputs."""

    brm: Any
    ports: dict[str, str]
    inst: Any
    var_s: str
    end: int
    memlets: dict[str, Memlet]
    folds: set[str]
    code: str  # the matched code entry's
    scalars: dict[str, Memlet]  # slot -> the scalar operand it is read from (a CSR)


def _bind_site(g: Graph, node: str, what: str) -> _Site:
    """The match: the node is the only node of a spatial map, or a fused nest's spatial map,
    and every map around it is tagged temporal."""
    body, i, outer = _locate(g, node)
    if body[i].kind == "map":  # a fused nest's spatial map (fuse_map): its two tasklets as one
        smap, found = body[i], fused_view(body[i])
        if found is not None:
            tasklet, pattern_node = found[2], smap
        elif _lane_fold_map(smap):  # a spatial map of one folding tasklet (lane_accumulate)
            tasklet, pattern_node = smap.body[0], smap
        else:
            raise SandboxError(f"{what}: a map binds only as a fused nest, a spatial map of two "
                               "tasklets (fuse_map), or as a spatial map of one tasklet whose "
                               "output folds (a lane fold)")  # fmt: skip
    else:
        tasklet = pattern_node = body[i]
        smap = outer[-1] if outer else None
        if smap is None or smap.attrs.get("loop.kind") != "spatial" or len(smap.body or []) != 1:
            raise SandboxError(f"{what}: must be the only node of a spatial map (split the map "
                               "first)")  # fmt: skip
        outer = outer[:-1]
    temporal = list(outer)
    for m in temporal:
        kind = m.attrs.get("loop.kind")
        if kind == "tile":
            raise SandboxError(f"{what}: map {m.id!r} is a tile; tiles are not lowered yet (LOW3b)")
        if kind != "temporal":
            raise SandboxError(
                f"{what}: map {m.id!r} around it is untagged; the maps around the spatial map are "
                "its firing loops: tag it temporal with tag_map"
            )
    return _Site(tasklet, pattern_node, smap, temporal)


def _lane_fold_map(m: Node) -> bool:
    """A spatial map whose body is one tasklet with a folding output: ``bind`` takes the map
    (patterns.py's ``lane_fold`` without a front op)."""
    kids = m.body or []
    return (
        m.attrs.get("loop.kind") == "spatial"
        and len(kids) == 1
        and kids[0].kind == "tasklet"
        and any(o.reduce is not None for o in kids[0].outputs.values())
    )


def _bind_check(g: Graph, site: _Site, brm: str, implementation: str,
                params: Mapping[str, Any] | None, what: str) -> _Bound:  # fmt: skip
    """The check: the BRM's ports match the tasklet's, their dtypes are accepted, it computes
    what the tasklet computes, and its lanes are the spatial bound (the design params: the
    lanes param from that bound, the rest given or default)."""
    from snax_forge.brm import BrmError, fits, result_type

    tasklet, smap = site.tasklet, site.smap
    try:
        b = load_brm(brm)
    except BrmError as e:
        raise SandboxError(f"{what}: {e}") from None
    conn_dtypes = {
        c: g.containers[m.data].dtype for c, m in (tasklet.inputs | tasklet.outputs).items()
    }
    scalars = _scalar_inputs(g, site, what)
    ports = match(site.pattern, b, conn_dtypes, frozenset(scalars))
    _check_port_dtypes(b, ports, conn_dtypes, what)
    entries = b.function.entries()
    if not entries:
        raise SandboxError(f"{what}: brm {brm!r} does not say what it computes (function.code)")
    given = dict(params or {})
    names = _scalar_names(b, scalars)
    entry, ports, slots = _entry(b, entries, ports, tasklet, names, given)
    if entry is None:
        computes = _computes(tasklet, ports, names)
        listed = (
            repr(entries[0].code)
            if isinstance(b.function.code, str)
            else "one of " + ", ".join(f"{e.code!r} ({_params_text(e.params)})" for e in entries)
        )
        raise SandboxError(
            f"{what}: the tasklet computes {computes!r} on the BRM's ports, {brm!r} computes "
            f"{listed}"
        )
    _check_port_dtypes(b, ports, conn_dtypes, what)  # the entry's assignment of the inputs
    computes = _computes(tasklet, ports, names)
    for k, v in entry.params.items():
        if k in given and given[k] != v:
            raise SandboxError(
                f"{what}: {k} = {given[k]!r} given, but the tasklet computes {computes!r}, "
                f"{brm!r}'s code at {k} = {v!r}"
            )
        given[k] = v
    op_type = None
    if slots:
        try:
            op_type = result_type([conn_dtypes[c] for c in tasklet.inputs if c in ports])
        except ValueError:
            op_type = None  # no integer result type: check_dtypes refused it above
    from_scalar: dict[str, Memlet] = {}
    for k, v in slots.items():
        if isinstance(v, str):  # a scalar operand: the slot is read from its container (a CSR)
            c = names[v]
            m = tasklet.inputs[c]
            if k in given:
                raise SandboxError(
                    f"{what}: {k} = {given[k]!r} given, but the tasklet computes {computes!r}, "
                    f"{brm!r}'s {entry.code!r} with its slot {k} from the scalar {m.data}"
                )
            dt = conn_dtypes[c]
            if op_type is not None and not np.can_cast(dt, op_type, "safe"):
                raise SandboxError(
                    f"{what}: {k} reads the scalar {m.data}, {dt}, which the op type {op_type} "
                    "does not hold (NumPy's result type of the input ports' dtypes)"
                )
            from_scalar[k] = m
            continue
        if k in given and given[k] != v:
            raise SandboxError(
                f"{what}: {k} = {given[k]!r} given, but the tasklet computes {computes!r}, "
                f"{brm!r}'s {entry.code!r} at its slot {k} = {v!r}"
            )
        if op_type is not None and not fits(v, op_type):
            raise SandboxError(
                f"{what}: {k} = {v} does not fit the op type {op_type} (NumPy's result type of "
                "the input ports' dtypes; the literal is weak, NEP 50)"
            )
        given[k] = v

    begin, end, step = parse_dim(smap.attrs["range"], what)
    if begin != 0 or step != 1:
        raise SandboxError(
            f"{what}: the spatial map must run over 0:<end>, not {smap.attrs['range']}"
        )
    end = _bound(g, end, what)  # a symbol (tag_map spatial on an unsplit map) at its value
    var_s = smap.attrs["var"]
    memlets = {c: m for c, m in (tasklet.inputs | tasklet.outputs).items() if c in ports}
    # a folding output that does not use the spatial variable takes one lane
    folds = {c for c, m in tasklet.outputs.items() if m.reduce is not None and var_s not in m.names}
    for c, m in memlets.items():
        if c not in folds and var_s not in m.names:
            raise SandboxError(
                f"{what}: {c} {m.data}{m.subset} does not use the spatial variable {var_s!r}"
            )
    lanes = {b.port(ports[c]).lanes for c in memlets if c not in folds}
    if len(lanes) != 1:
        raise SandboxError(
            f"{what}: the BRM's ports have different lanes {sorted(map(str, lanes))}"
        )
    (lanes_v,) = lanes
    if expr.is_name(lanes_v):
        if lanes_v in given and given[lanes_v] != end:
            raise SandboxError(
                f"{what}: {lanes_v} = {given[lanes_v]} given, the spatial bound is {end}"
            )
        given[lanes_v] = end
    try:
        inst = b.resolve(implementation, given, from_scalar)
    except BrmError as e:
        raise SandboxError(f"{what}: {e}") from None
    if inst.code != entry.code:  # a slot given that the matched entry leaves unset
        extra = [k for k in b.all_slots() if k in inst.params and k not in slots]
        raise SandboxError(
            f"{what}: {', '.join(f'{k} = {inst.params[k]!r}' for k in extra)} given, but the "
            f"tasklet computes {computes!r}, {brm!r}'s {entry.code!r}, which leaves "
            f"{' and '.join(extra)} unset"
        )
    lane_port = next(ports[c] for c in memlets if c not in folds)
    if inst.lanes(lane_port) != end:
        raise SandboxError(f"{what}: the BRM has {inst.lanes(lane_port)} lanes, the map {end}")
    for c in folds:
        if inst.lanes(ports[c]) != 1:
            raise SandboxError(
                f"{what}: {c} folds every lane into one, port {ports[c]!r} has "
                f"{inst.lanes(ports[c])} lanes"
            )
    return _Bound(b, ports, inst, var_s, end, memlets, folds, entry.code, from_scalar)


def _scalar_inputs(g: Graph, site: _Site, what: str) -> dict[str, Memlet]:
    """The tasklet's scalar operands: each input whose memlet names no variable of the maps
    around it (the same element every firing), on a kernel's scalar argument (``sdfg.scalar``),
    which goes to a slot, not a port; such an input on any other container is refused,
    naming it. An input on a scalar whose subset uses a map variable stays an array input."""
    variables = {m.attrs["var"] for m in [*site.temporal, site.smap]}
    out: dict[str, Memlet] = {}
    for c, m in site.tasklet.inputs.items():
        if m.names & variables:
            continue
        if SCALAR not in g.containers[m.data].attrs:
            raise SandboxError(
                f"{what}: {c} reads {m.data}{m.subset}, the same element in every firing; a "
                f"slot takes a kernel's scalar argument (sdfg.scalar), and {m.data} is not one"
            )
        out[c] = m
    return out


def _scalar_names(b: Any, scalars: Mapping[str, Memlet]) -> dict[str, str]:
    """Each scalar operand's name in the code ``bind`` matches: its container's (``alpha``), or
    with ``_s`` appended while that is a port's or a design param's name; name -> connector.
    Two connectors on one container share the name: they read the same element."""
    taken = {p.name for p in b.interface.ports} | set(b.params("design"))
    out: dict[str, str] = {}
    for c, m in scalars.items():
        name = m.data
        while name in taken:
            name = f"{name}_s"
        out.setdefault(name, c)
    return out


def _computes(tasklet: Node, ports: Mapping[str, str], names: Mapping[str, str]) -> str:
    """The tasklet's code on the BRM's ports, a scalar operand under its name (``_scalar_names``)."""
    by_conn = {tasklet.inputs[c].data: n for n, c in names.items()}
    rename = dict(ports) | {c: by_conn[m.data] for c, m in tasklet.inputs.items()
                            if c not in ports and m.data in by_conn}  # fmt: skip
    return expr.rename_code(tasklet.attrs["code"], rename)


def _check_port_dtypes(b: Any, ports: Mapping[str, str], conn_dtypes: Mapping[str, str],
                       what: str) -> None:  # fmt: skip
    from snax_forge.brm import accepted, check_dtypes

    dtypes = {p: conn_dtypes[c] for c, p in ports.items()}
    if refused := check_dtypes(b, dtypes):
        raise SandboxError(
            f"{what}: {'; '.join(refused)}; accepted output dtypes for these inputs: "
            f"{accepted(b, dtypes)}"
        )


def _entry(b: Any, entries: list[Any], ports: Mapping[str, str], tasklet: Node,
           names: Mapping[str, str], given: Mapping[str, Any],
           ) -> tuple[Any, dict[str, str], dict[str, int | str]]:  # fmt: skip
    """The first code entry whose present ports (at the defaults, ``given`` and its params) take
    the tasklet's array inputs, in some assignment, and whose code is the tasklet's on them up
    to its slots (``expr.unify``, a slot an int or a scalar operand's name, ``names``): the
    entry, connector -> port, and the slot values; or ``(None, ports, {})``. The assignments
    are tried port order first: the matcher's (the inputs in the order they first appear onto
    the ports in order), then every other."""
    arrays = [c for c in ports if c in tasklet.inputs]
    outs = {c: p for c, p in ports.items() if c in tasklet.outputs}
    for e in entries:
        present = b.present(b.defaults() | dict(given) | e.params)
        ins = [p.name for p in present if p.direction == "in"]
        if {p.name for p in present if p.direction == "out"} != set(outs.values()):
            continue
        if len(ins) != len(arrays):
            continue
        for perm in itertools.permutations(ins):
            mapping = dict(zip(arrays, perm, strict=True)) | outs
            found = expr.unify(e.code, _computes(tasklet, mapping, names), b.slots(e), names)
            if found is not None:
                return e, mapping, found
    return None, dict(ports), {}


def _params_text(params: Mapping[str, Any]) -> str:
    """A code entry's params as the refusals write them: ``op = 'add'``."""
    return ", ".join(f"{k} = {v!r}" for k, v in params.items()) or "any params"


def _bind_rewrite(g: Graph, site: _Site, bound: _Bound, brm: str, implementation: str,
                  instance: str, what: str) -> Graph:  # fmt: skip
    """The rewrite: each memlet's spatial variable becomes the lanes' range (a fold's last
    index one lane), its order checked against the BRM's nest unless the port is a stream,
    and the spatial map replaced by the accelerated node."""
    from snax_forge.brm import memlet_ordered, task_nest

    tasklet, smap, temporal = site.tasklet, site.smap, site.temporal
    b, ports, inst, folds = bound.brm, bound.ports, bound.inst, bound.folds
    counts = [_bound(g, _length(m), what) for m in temporal]
    n = int(np.prod(counts, dtype=int))
    loops = [(m.attrs["var"], k) for m, k in zip(temporal, counts, strict=True)]
    try:
        rates = named_rates(
            {ports[c]: (m, b.port(ports[c]).rate) for c, m in bound.memlets.items()}, loops, what
        )
    except DfgError as e:
        raise SandboxError(str(e)) from None
    task = {"n": n} | rates
    by_nest = not memlet_ordered(b)  # stream: the memlet is the order, no nest to check
    new_in: dict[str, Memlet] = {}
    new_out: dict[str, Memlet] = {}
    for side, conns, out in (
        ("inputs", tasklet.inputs, new_in),
        ("outputs", tasklet.outputs, new_out),
    ):
        for c, m in conns.items():
            if c not in ports:  # a scalar operand: its slot's connector, below
                continue
            port = ports[c]
            nest = None
            if by_nest:
                try:
                    nest = task_nest(inst, port, task)
                except ValueError as e:
                    raise SandboxError(f"{what}: {e}") from None
            if c in folds:  # one lane: the last index becomes a range of one
                dims = [*m.subset[:-1], _one_lane(m.subset[-1], f"{what}.{side}.{c}")]
                out[port] = Memlet(m.data, dims)
                beats = [lp for lp in temporal if lp.attrs["var"] in m.names]
                if nest is not None:
                    _check_order(g, m, beats, None, nest.indices(), f"{what} port {port!r}")
                continue
            out[port] = Memlet(m.data, _lanes_dims(m, bound.var_s, bound.end, f"{what}.{side}.{c}"))
            if nest is not None:
                _check_order(g, m, temporal, smap, nest.indices(), f"{what} port {port!r}")
    for slot, m in bound.scalars.items():  # a slot read from a scalar: a connector of its name
        new_in[slot] = Memlet(m.data, list(m.subset))
    acc = Node(
        tasklet.id,
        "accelerated",
        inputs=new_in,
        outputs=new_out,
        attrs={
            "instance": instance,
            "brm": brm,
            "implementation": implementation,
            "code": bound.code,
            "params": dict(inst.params),
            "replaced": smap.to_dict(),  # for unbind
        },
        body=[],
    )
    fused = site.pattern is smap and fused_view(smap) is not None
    if fused and (taken := _take_locals(g, smap)):  # a fused nest's local value
        acc.attrs["locals"] = taken  # goes with its two tasklets
    parent_body, j, _ = _locate(g, smap.id)
    parent_body[j] = acc
    return _result(g)


def _take_locals(g: Graph, smap: Node) -> list[dict[str, Any]]:
    """The fused nest's local values (``loop.local``: the first tasklet's output, which only the
    two tasklets name) taken out of the graph, each with its index among the containers, for
    ``unbind`` to put back."""
    found = fused_view(smap)
    assert found is not None  # _bind_site took the map as a fused nest
    (m,) = found[0].outputs.values()
    names = list(g.containers)
    if "loop.local" not in g.containers[m.data].attrs:
        return []
    cont = g.containers.pop(m.data)
    return [{"name": m.data, "index": names.index(m.data), "container": cont.to_dict()}]


def _length(m: Node) -> str:
    begin, end, step = parse_dim(m.attrs["range"], m.id)
    if step != 1:
        raise SandboxError(f"map {m.id!r}: a temporal map with a step is not bound yet")
    return end if begin == 0 else f"({end}) - ({begin})"


def _one_lane(d: Any, what: str) -> Any:
    """An index as the range of one lane, ``c`` -> ``c:c + 1``."""
    parts = parse_dim(d, what)
    if len(parts) != 1:
        raise SandboxError(f"{what}: {d!r} is already a range")
    return format_dim((parts[0], expr.canonical(f"({parts[0]}) + 1"), 1))


def _lanes_dims(m: Memlet, var: str, lanes: int, what: str) -> list[Any]:
    """The memlet with the one dimension over the spatial variable as the lanes' range; the
    other dimensions (a row index in 2-D) stay as they are."""
    over = [k for k, d in enumerate(m.subset) if var in dim_names(d)]
    if len(over) != 1:
        raise SandboxError(
            f"{what}: {m.data}{m.subset} uses the spatial variable {var!r} in {len(over)} "
            "dimensions; the lanes run along one"
        )
    return [_lanes_dim(d, var, lanes, what) if k in over else d for k, d in enumerate(m.subset)]


def _lanes_dim(d: Any, var: str, lanes: int, what: str) -> Any:
    """A dimension over the spatial variable as the range of the lanes."""
    parts = parse_dim(d, what)
    if len(parts) != 1:
        raise SandboxError(f"{what}: {d!r} is already a range")
    try:
        const, coeff = expr.linear(parts[0], [var])
    except ExprError as e:
        raise SandboxError(f"{what}: {e}") from None
    c = coeff[var]
    if c == 0:
        raise SandboxError(f"{what}: {d!r} does not use the spatial variable {var!r}")
    if c < 0:
        raise SandboxError(f"{what}: {d!r} runs the lanes backwards")
    return format_dim((const, expr.canonical(f"{const} + {c * lanes}"), c))


def _check_order(
    g: Graph, m: Memlet, temporal: list[Node], smap: Node | None, want: np.ndarray, what: str
) -> None:
    """The memlet over the temporal maps and the lanes gives the nest's order.

    ``smap`` None: a folding output, one lane, over the temporal maps its
    memlet uses (one beat per T firings).
    """
    env: dict[str, Any] = g.bound_symbols
    loops = [*temporal, *([smap] if smap is not None else [])]
    for d, lp in enumerate(loops):
        try:
            values = np.array(map_range(lp.attrs["range"], g.bound_symbols, lp.id))
        except ExprError as e:
            raise SandboxError(f"{what}: {e} (bind the symbol in the recipe)") from None
        shape = [1] * len(loops)
        shape[d] = len(values)
        env[lp.attrs["var"]] = values.reshape(shape)
    full = tuple(len(np.asarray(env[lp.attrs["var"]]).reshape(-1)) for lp in loops)
    lanes = full[-1] if smap is not None else 1
    got = np.stack(
        [np.broadcast_to(np.asarray(expr.evaluate(dim, env)), full) for dim in m.subset], axis=-1
    ).reshape(-1, lanes, len(m.subset))
    if got.shape != want.shape or not np.array_equal(got - got[0, 0], want - want[0, 0]):
        raise SandboxError(f"{what}: the memlet {m.data}{m.subset} does not give the BRM's order")


# =============================================================================
# unbind and join_map: the inverses
# =============================================================================


def _unbind_site(g: Graph, node: str, what: str) -> dict[str, Any]:
    """An accelerated node that records what it replaced, on no link."""
    body, i, _ = _locate(g, node)
    acc = body[i]
    if acc.kind != "accelerated":
        raise SandboxError(f"{what}: a {acc.kind!r} is not an accelerated node")
    if acc.attrs.get("replaced") is None:
        raise SandboxError(f"{what}: it records nothing it replaced (written by hand?)")
    inst = acc.attrs["instance"]
    linked = [n.attrs["container"] for n, _ in g.walk()
              if n.kind == "link" and inst in (n.attrs["from"], n.attrs["to"])]  # fmt: skip
    if linked:
        raise SandboxError(
            f"{what}: its instance {inst!r} is on the link of {linked} (unlink first)"
        )
    return {"node": node}


def unbind(g: Graph, node: str) -> Graph:
    """The accelerated node ``node`` replaced by the subtree ``bind`` replaced."""
    what = f"unbind {node!r}"
    _unbind_site(g, node, what)
    g = _copy(g)
    body, i, _ = _locate(g, node)
    taken = body[i].attrs.get("locals", [])
    body[i] = Node.from_dict(body[i].attrs["replaced"], what)
    if taken:  # a fused nest's local values back where they were among the containers
        items = list(g.containers.items())
        for rec in sorted(taken, key=lambda r: r["index"]):
            items.insert(rec["index"], (rec["name"], Container.from_dict(rec["container"], what)))
        g.containers = dict(items)
    return _result(g)


def _join_site(g: Graph, map: str, what: str) -> dict[str, Any]:
    """A map with ``loop.split`` whose only child is a map."""
    body, i, _ = _locate(g, map)
    outer = body[i]
    if outer.kind != "map" or outer.attrs.get("loop.split") is None:
        raise SandboxError(f"{what}: not a map split by split or split_map (no loop.split)")
    kids = outer.body or []
    if len(kids) != 1 or kids[0].kind != "map":
        raise SandboxError(f"{what}: its only child must be the inner map (unbind first)")
    return {"map": map}


def join_map(g: Graph, map: str) -> Graph:
    """The inverse of split: a split map and its inner map become the map they were.

    Needs ``loop.split`` (the var and range before the split) and the
    inner map as the outer map's only child, whatever their tags; the
    joined map has none. Every index under it is
    written back over the old variable: an index ``k + c * (b + f * v_t + v_s)``
    that split wrote becomes ``k + c * v``; an index in another form is
    an error, since it is not what split wrote.
    """
    what = f"join_map {map!r}"
    _join_site(g, map, what)
    g = _copy(g)
    body, i, _ = _locate(g, map)
    outer = body[i]
    split = outer.attrs["loop.split"]
    (inner,) = outer.body
    v_t, v_s, var = outer.attrs["var"], inner.attrs["var"], split["var"]
    begin, _, _ = parse_dim(split["range"], what)
    b0, factor, step = parse_dim(inner.attrs["range"], what)
    if b0 != 0 or step != 1 or type(factor) is not int:
        raise SandboxError(f"{what}: the inner map must run over 0:<int>")

    def back(part: Any) -> Any:
        try:
            const, co = expr.linear(part, [v_t, v_s])
        except ExprError as e:
            raise SandboxError(f"{what}: {e}") from None
        c = co[v_s]
        if co[v_t] != factor * c:
            raise SandboxError(f"{what}: {part!r} is not an index split_map wrote")
        if c == 0:
            return part
        k = expr.linear(f"({const}) - {c} * ({begin})", [])[0]
        term = var if c == 1 else f"-{var}" if c == -1 else f"{c} * {var}"
        if k == 0:
            return expr.canonical(term)
        if type(k) is int and k < 0:
            return expr.canonical(f"{term} - {-k}")
        return expr.canonical(f"{term} + {k}")

    def back_dim(d: Any) -> Any:
        return format_dim(tuple(back(p) for p in parse_dim(d, what)))

    _map_dims(inner.body or [], back_dim)
    rest = {
        k: v for k, v in outer.attrs.items() if k not in ("var", "range", "loop.kind", "loop.split")
    }
    body[i] = Node(
        outer.id, "map", attrs={"var": var, "range": split["range"], **rest}, body=inner.body
    )
    return _result(g)


# =============================================================================
# interchange_map: its own inverse
# =============================================================================


def _interchange_site(g: Graph, map: str, what: str) -> dict[str, Any]:
    """A map whose body is one map, the two free to trade places: neither range names the
    other's variable, no accelerated node around or under them, neither a fused nest's map,
    and no fold under them over the outer map with an op whose result depends on the order."""
    body, i, outer = _locate(g, map)
    x = body[i]
    if x.kind != "map":
        raise SandboxError(f"{what}: a {x.kind!r} is not a map")
    kids = x.body or []
    if len(kids) != 1 or kids[0].kind != "map":
        held = ", ".join(f"the {k.kind} {k.id!r}" for k in kids) or "nothing"
        raise SandboxError(f"{what}: its body must be one map and nothing else; it holds {held}")
    (y,) = kids
    for a, b in ((x, y), (y, x)):
        if b.attrs["var"] in dim_names(a.attrs["range"]):
            raise SandboxError(f"{what}: {a.id}'s range {a.attrs['range']} names {b.id}'s "
                               f"variable {b.attrs['var']!r}; the two cannot trade places")  # fmt: skip
    for o in outer:
        if o.kind == "accelerated":
            raise SandboxError(f"{what}: it is inside the accelerated node {o.id!r} (unbind "
                               "first)")  # fmt: skip
    under = list(_nodes_under(y.body or []))
    for n in under:
        if n.kind == "accelerated":
            raise SandboxError(f"{what}: the bound unit {n.id!r} is inside it (unbind first)")
    for m in (x, y):
        if m.attrs.get("loop.fused") is not None:
            raise SandboxError(f"{what}: {m.id!r} is a fused nest's outer map (unfuse_map "
                               "first)")  # fmt: skip
    var = x.attrs["var"]
    for n in under:
        for c, m in n.outputs.items():
            red = m.reduce
            if red is not None and var not in m.names and not REDUCE_OPS[red.op].order_free:
                raise SandboxError(
                    f"{what}: {n.id}'s {c} folds over {var!r} with {red.op!r}, whose result "
                    "depends on the order (not associative and commutative); the interchange "
                    "would change it"
                )
    return {"map": map}


def _nodes_under(body: list[Node]) -> Iterator[Node]:
    """Every live node in ``body``, depth first (not an accelerated node's ``replaced``)."""
    for n in body:
        yield n
        yield from _nodes_under(n.body or [])


def interchange_map(g: Graph, map: str) -> Graph:
    """The map ``map`` and its only child map trade places (module doc); its own inverse."""
    _interchange_site(g, map, f"interchange_map {map!r}")
    g = _copy(g)
    body, i, _ = _locate(g, map)
    x = body[i]
    (y,) = x.body
    x.body, y.body = y.body, [x]
    body[i] = y
    return _result(g)


# =============================================================================
# encapsulate and unencapsulate
# =============================================================================


def _accelerated(g: Graph) -> list[Node]:
    """The live accelerated nodes, in execution order (not the ``replaced`` history)."""
    return [n for n, _ in g.walk() if n.kind == "accelerated"]


def _unit(g: Graph, selected: str) -> list[Node]:
    """The nodes of the unit ``selected`` names: an accelerated node's id or an instance."""
    nodes = _accelerated(g)
    by_id = [n for n in nodes if n.id == selected]
    inst = by_id[0].attrs["instance"] if by_id else selected
    return [n for n in nodes if n.attrs["instance"] == inst]


def _encapsulate_site(g: Graph, selected: str) -> dict[str, Any]:
    """A unit in no accelerator."""
    unit = _unit(g, selected)
    if not unit:
        order = list(dict.fromkeys(n.attrs["instance"] for n in _accelerated(g)))
        raise SandboxError(f"encapsulate: no instance {selected!r} (instances: {order})")
    inst = unit[0].attrs["instance"]
    inside = sorted({n.attrs["accelerator"] for n in unit if "accelerator" in n.attrs})
    if inside:
        raise SandboxError(f"encapsulate: {inst!r} is in accelerator {inside[0]!r} already "
                           "(unencapsulate it first)")  # fmt: skip
    return {"units": [inst]}


def selected_units(g: Graph, selected: list[str]) -> list[str]:
    """The instances a selection stands for, in the graph's order: a unit's node id or instance
    itself, a group's name its units, an accelerator's name its groups' units (D148); an item
    that names none of these is a SandboxError."""
    nodes = _accelerated(g)
    order = list(dict.fromkeys(n.attrs["instance"] for n in nodes))
    picked: set[str] = set()
    for s in selected:
        mine = {n.attrs["instance"] for n in nodes
                if s in (n.id, n.attrs["instance"], n.attrs.get("group"), n.attrs.get("accelerator"))}  # fmt: skip
        if not mine:
            raise SandboxError(f"encapsulate: {s!r} is no unit, group or accelerator "
                               f"(instances: {order})")  # fmt: skip
        picked |= mine
    return [u for u in order if u in picked]


def _encapsulate_multi(g: Graph, selected: list[str]) -> dict[str, Any]:
    """Units and groups (outlines or boxes), all in no accelerator: their units, in the graph's
    order."""
    units = selected_units(g, selected)
    for u in units:
        _encapsulate_site(g, u)
    return {"units": units}


def encapsulate(g: Graph, accelerator: str, units: list[str]) -> Graph:
    """The groups of ``units`` (instances) as one accelerator named ``accelerator``."""
    for u in units if isinstance(units, list) else []:
        if isinstance(u, str):
            _encapsulate_site(g, u)
    g = _copy(g)
    what = f"encapsulate {accelerator!r}"
    if not isinstance(accelerator, str) or not accelerator.isidentifier():
        raise SandboxError(f"{what}: the accelerator must be a name")
    if not isinstance(units, list) or not units or not all(isinstance(u, str) for u in units):
        raise SandboxError(f"{what}: units must be a non-empty list of instances")
    if len(set(units)) != len(units):
        raise SandboxError(f"{what}: a unit is listed twice in {units}")
    nodes = _accelerated(g)
    order = list(dict.fromkeys(n.attrs["instance"] for n in nodes))
    taken = {n.attrs["accelerator"] for n in nodes if "accelerator" in n.attrs}
    if accelerator in order or accelerator in taken:
        what_has = "an instance" if accelerator in order else "an accelerator"
        raise SandboxError(f"{what}: {what_has} is already named {accelerator!r}")
    for u in units:
        if u not in order:  # a unit's node id: the site took it, units name instances
            raise SandboxError(f"{what}: no instance {u!r} (instances: {order})")
    graph_order = [u for u in order if u in units]
    if graph_order != units:
        raise SandboxError(f"{what}: units {units} in another order than the graph runs them, "
                           f"{graph_order}; the groups keep the graph's order (C§17)")  # fmt: skip
    for n in nodes:
        if n.attrs["instance"] in units:
            n.attrs["accelerator"] = accelerator
    return _result(g)


def _unencapsulate_site(g: Graph, selected: str) -> dict[str, Any]:
    """A unit in an accelerator, or the accelerator's name."""
    what = f"unencapsulate {selected!r}"
    if any(n.attrs.get("accelerator") == selected for n in _accelerated(g)):
        return {"accelerator": selected}
    unit = _unit(g, selected)
    if not unit:
        raise SandboxError(f"{what}: no accelerator of that name")
    if "accelerator" not in unit[0].attrs:  # a unit's nodes share it (DfgError otherwise)
        raise SandboxError(f"{what}: the unit {unit[0].attrs['instance']!r} is in no accelerator")
    return {"accelerator": unit[0].attrs["accelerator"]}


def unencapsulate(g: Graph, accelerator: str) -> Graph:
    """The inverse of encapsulate: each group of ``accelerator`` its own accelerator again."""
    if _unencapsulate_site(g, accelerator)["accelerator"] != accelerator:  # a unit's name
        raise SandboxError(f"unencapsulate {accelerator!r}: no accelerator of that name")
    g = _copy(g)
    for n in _accelerated(g):
        if n.attrs.get("accelerator") == accelerator:
            del n.attrs["accelerator"]
    return _result(g)


register_transform(
    "split", split, ints=("factor",), site=lambda g, s: _split_site(g, s, f"split {s!r}")
)
register_transform("split_map", split_map, ints=("factor",),
                   site=lambda g, s: _split_site(g, s, f"split_map {s!r}"))  # fmt: skip
register_transform("tag_map", tag_map, site=lambda g, s: _tag_site(g, s, f"tag_map {s!r}"))
register_transform("untag_map", untag_map, site=lambda g, s: _untag_site(g, s, f"untag_map {s!r}"))
register_transform("bind", bind, site=_bind_params)
register_transform("unbind", unbind, site=lambda g, s: _unbind_site(g, s, f"unbind {s!r}"))
register_transform("join_map", join_map, site=lambda g, s: _join_site(g, s, f"join_map {s!r}"))
register_transform("interchange_map", interchange_map,
                   site=lambda g, s: _interchange_site(g, s, f"interchange_map {s!r}"))  # fmt: skip
register_transform("encapsulate", encapsulate, site=_encapsulate_site, multi=_encapsulate_multi)
register_transform("unencapsulate", unencapsulate, site=_unencapsulate_site)
