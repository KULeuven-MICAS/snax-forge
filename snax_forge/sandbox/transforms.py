"""Sandbox transforms (D72, D73): each takes a graph and returns a new one.

A transform is registered with ``register_transform(name, fn, ints=...,
site=...)``; ``fn(graph, **params) -> graph`` never changes its input.
``ints`` names the params that are ints, which a recipe may give as
expressions over its own params (recipe.py). The result is a new ``Graph``,
validated as any graph is; the reference check after the step is run.py's.

**Sites** (D72). ``site(graph, selected)`` is the graph-shape test of where a
transform applies, for a node id or a container name: it returns the params
the selection fills (the selection itself and what follows from it) or
raises the error the transform raises there, since the transform calls it
first, so each check lives in one place. A site decides from the graph
alone; the checks on the user's params (a factor, a kind, a BRM, a depth, a
name) stay in the transform.

``split(map, factor)`` and ``split_map(map, factor)``
    ``split`` splits a map ``v`` over ``b:e`` (step 1, no ``loop.kind``) into
    an outer map ``v_t`` over ``0:(e - b) // factor``, keeping the map's id
    and attrs and recording the split (``loop.split``), around an inner map
    ``<id>_s`` over ``0:factor``, both untagged (D73); every use of ``v`` in
    the body becomes ``b + factor * v_t + v_s``. A length n that is not a
    multiple of the factor (D144) runs ``v_t`` over ceil(n / factor) and
    records n in ``loop.split`` (``valid``): only the lanes with
    ``factor * v_t + v_s`` below n are the kernel's, the rest pad lanes
    (dfg/pad.py), which the reference executor neither reads nor writes and
    the containers hold in memory as padding. ``split_map`` is ``split``,
    then ``tag_map`` ``temporal`` on the outer map and ``spatial`` on the
    inner, in one step.
``tag_map(map, kind)`` and ``untag_map(map)``
    ``tag_map`` tags an untagged map ``temporal``, ``tile`` or ``spatial`` as
    it is, without splitting it (matadd's row map); a tagged map is an
    error. ``untag_map`` removes the tag, whatever set it, but not from a map
    a bound unit is inside (unbind first).
``bind(node, brm, implementation, instance, params={})``
    replaces a tasklet and the spatial map around it (the map's only
    child), or a fused nest's spatial map and its two tasklets
    (compose.py's ``fuse_map``; the first's inputs, the second's output,
    patterns.py's ``fused_view``), by an accelerated node with the
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
    the BRM's port names, must be the BRM's ``function.code``; the accelerated node carries that
    code and the replaced subtree (D82). A folding output (``reduce``)
    whose memlet does not use the spatial variable (dot's
    ``out[0]``) folds every lane into one: its port must have one lane, its
    index becomes the range of that lane (``out[0:1]``, no ``reduce`` on
    the accelerated node), and its order is checked over the temporal maps
    it uses, one beat per T firings, T from ``named_rates`` (D102).
``unbind(node)`` and ``join_map(map)``
    the inverses: the subtree an accelerated node replaced, and the
    map a split pair was split from (``loop.split``), whatever their tags,
    so a graph at any step can go back to the imported one.
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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from snax_forge import expr
from snax_forge.common import fresh
from snax_forge.dfg import DfgError, Graph, Memlet, Node, named_rates, parse_dim
from snax_forge.dfg.subset import dim_names, format_dim, map_range
from snax_forge.expr import ExprError

from .patterns import fused_view, match
from .recipe import SandboxError

Transform = Callable[..., Graph]
Site = Callable[[Graph, str], dict[str, Any]]


@dataclass(frozen=True)
class TransformSpec:
    fn: Transform
    ints: tuple[str, ...] = ()
    site: Site | None = None


TRANSFORMS: dict[str, TransformSpec] = {}


def register_transform(
    name: str, fn: Transform, ints: tuple[str, ...] = (), site: Site | None = None
) -> None:
    """Make ``{"transform": name}`` usable in a recipe (principle 6), with its site (D72)."""
    if name in TRANSFORMS:
        raise ValueError(f"transform {name!r} already registered")
    TRANSFORMS[name] = TransformSpec(fn, tuple(ints), site)


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


def split(g: Graph, map: str, factor: int) -> Graph:
    return _split(g, map, factor, f"split {map!r}")[0]


def split_map(g: Graph, map: str, factor: int) -> Graph:
    g, inner = _split(g, map, factor, f"split_map {map!r}")
    return tag_map(tag_map(g, map, "temporal"), inner, "spatial")


def _split(g: Graph, map: str, factor: int, what: str) -> tuple[Graph, str]:
    """``split``: the graph and the inner map's id."""
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
    var_t, var_s = fresh(f"{var}_t", taken), fresh(f"{var}_s", taken)
    ids = {nd.id for nd, _ in g.walk()}
    inner = Node(
        fresh(f"{map}_s", ids),
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


def _bind_site(g: Graph, node: str, what: str) -> _Site:
    """The match: the node is the only node of a spatial map, or a fused nest's spatial map,
    and every map around it is tagged temporal."""
    body, i, outer = _locate(g, node)
    if body[i].kind == "map":  # a fused nest's spatial map (fuse_map): its two tasklets as one
        smap, found = body[i], fused_view(body[i])
        if found is None:
            raise SandboxError(f"{what}: a map binds only as a fused nest, a spatial map of two "
                               "tasklets (fuse_map)")  # fmt: skip
        tasklet, pattern_node = found[2], smap
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


def _bind_check(g: Graph, site: _Site, brm: str, implementation: str,
                params: Mapping[str, Any] | None, what: str) -> _Bound:  # fmt: skip
    """The check: the BRM's ports match the tasklet's, their dtypes are accepted, it computes
    what the tasklet computes, and its lanes are the spatial bound (the design params: the
    lanes param from that bound, the rest given or default)."""
    from snax_forge.brm import BrmError, accepted, check_dtypes

    tasklet, smap = site.tasklet, site.smap
    try:
        b = load_brm(brm)
    except BrmError as e:
        raise SandboxError(f"{what}: {e}") from None
    conn_dtypes = {
        c: g.containers[m.data].dtype for c, m in (tasklet.inputs | tasklet.outputs).items()
    }
    ports = match(site.pattern, b, conn_dtypes)
    dtypes = {ports[c]: dt for c, dt in conn_dtypes.items()}
    if refused := check_dtypes(b, dtypes):
        raise SandboxError(
            f"{what}: {'; '.join(refused)}; accepted output dtypes for these inputs: "
            f"{accepted(b, dtypes)}"
        )
    if b.function.code is None:
        raise SandboxError(f"{what}: brm {brm!r} does not say what it computes (function.code)")
    computes = expr.rename_code(tasklet.attrs["code"], ports)
    if computes != b.function.code:
        raise SandboxError(
            f"{what}: the tasklet computes {computes!r} on the BRM's ports, "
            f"{brm!r} computes {b.function.code!r}"
        )

    begin, end, step = parse_dim(smap.attrs["range"], what)
    if begin != 0 or step != 1:
        raise SandboxError(
            f"{what}: the spatial map must run over 0:<end>, not {smap.attrs['range']}"
        )
    end = _bound(g, end, what)  # a symbol (tag_map spatial on an unsplit map) at its value
    var_s = smap.attrs["var"]
    memlets = tasklet.inputs | tasklet.outputs
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
    given = dict(params or {})
    if expr.is_name(lanes_v):
        if lanes_v in given and given[lanes_v] != end:
            raise SandboxError(
                f"{what}: {lanes_v} = {given[lanes_v]} given, the spatial bound is {end}"
            )
        given[lanes_v] = end
    try:
        inst = b.resolve(implementation, given)
    except BrmError as e:
        raise SandboxError(f"{what}: {e}") from None
    lane_port = next(ports[c] for c in memlets if c not in folds)
    if inst.lanes(lane_port) != end:
        raise SandboxError(f"{what}: the BRM has {inst.lanes(lane_port)} lanes, the map {end}")
    for c in folds:
        if inst.lanes(ports[c]) != 1:
            raise SandboxError(
                f"{what}: {c} folds every lane into one, port {ports[c]!r} has "
                f"{inst.lanes(ports[c])} lanes"
            )
    return _Bound(b, ports, inst, var_s, end, memlets, folds)


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
    acc = Node(
        tasklet.id,
        "accelerated",
        inputs=new_in,
        outputs=new_out,
        attrs={
            "instance": instance,
            "brm": brm,
            "implementation": implementation,
            "code": b.function.code,
            "params": dict(inst.params),
            "replaced": smap.to_dict(),  # for unbind
        },
        body=[],
    )
    parent_body, j, _ = _locate(g, smap.id)
    parent_body[j] = acc
    return _result(g)


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
    body[i] = Node.from_dict(body[i].attrs["replaced"], what)
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
register_transform("encapsulate", encapsulate, site=_encapsulate_site)
register_transform("unencapsulate", unencapsulate, site=_unencapsulate_site)
