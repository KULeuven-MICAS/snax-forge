"""Sandbox transforms (SBX1, D72, D73, D80): each takes a graph and returns a new one.

A transform is registered with ``register_transform(name, fn, ints=...)``;
``fn(graph, **params) -> graph`` never changes its input. ``ints`` names the
params that are ints, which a recipe may give as expressions over its own
params (recipe.py). The result is a new ``Graph``, validated as any graph
is (D77); the reference check after the step is run.py's.

``split_map(map, factor)``
    splits a map ``v`` over ``b:e`` (step 1, no ``loop.kind`` yet) into an
    outer map ``v_t`` over ``0:(e - b) // factor`` tagged ``temporal``,
    keeping the map's id and attrs, and an inner map ``<id>_s`` over
    ``0:factor`` tagged ``spatial``; every use of ``v`` in the body becomes
    ``b + factor * v_t + v_s`` (D73). The length must be a multiple of the
    factor for the graph's bound symbols (open item 31).
``bind(node, brm, implementation, instance, params={})``
    replaces a tasklet and the spatial map around it (the map's only
    child) by an accelerated node with the tasklet's id, bound to an
    instance of a library BRM. The BRM's pattern matcher (patterns.py) says
    which connector feeds which port. The lanes design param is read off the
    spatial bound (D73), the other design params come from ``params`` or
    the BRM's defaults, and ``Brm.resolve`` checks them all (D68). Each
    memlet becomes the port's lanes as a range over the spatial variable,
    the container dtype must be the port's, and the memlets together with
    the temporal maps around must give the elements in the order of the
    port's nest (D70, D73). The tasklet's code, on the BRM's port names,
    must be the BRM's ``function.code``; the accelerated node carries that
    code and the replaced subtree (D82).
``unbind(node)`` and ``join_map(map)``
    the inverses (D82): the subtree an accelerated node replaced, and the
    map a temporal / spatial pair was split from (``loop.split``), so a
    graph at any step can go back to the imported one.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from snax_forge import expr
from snax_forge.dfg import DfgError, Graph, Memlet, Node, parse_dim
from snax_forge.dfg.subset import format_dim
from snax_forge.expr import ExprError

from .patterns import match
from .recipe import SandboxError

Transform = Callable[..., Graph]


@dataclass(frozen=True)
class TransformSpec:
    fn: Transform
    ints: tuple[str, ...] = ()


TRANSFORMS: dict[str, TransformSpec] = {}


def register_transform(name: str, fn: Transform, ints: tuple[str, ...] = ()) -> None:
    """Make ``{"transform": name}`` usable in a recipe (principle 6)."""
    if name in TRANSFORMS:
        raise ValueError(f"transform {name!r} already registered")
    TRANSFORMS[name] = TransformSpec(fn, tuple(ints))


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


def _substitute_body(body: list[Node], values: Mapping[str, Any]) -> None:
    """Replace names in every memlet subset and map range under ``body``."""
    for n in body:
        for m in (*n.inputs.values(), *n.outputs.values()):
            m.subset = [_substitute_dim(d, values) for d in m.subset]
        if n.kind == "map":
            n.attrs["range"] = _substitute_dim(n.attrs["range"], values)
            if n.attrs.get("loop.split"):
                split = n.attrs["loop.split"]
                n.attrs["loop.split"] = {**split, "range": _substitute_dim(split["range"], values)}
        if n.kind == "accelerated" and n.attrs.get("replaced"):
            old = Node.from_dict(n.attrs["replaced"], n.id)
            _substitute_body([old], values)
            n.attrs["replaced"] = old.to_dict()
        if n.body:
            _substitute_body(n.body, values)


def _substitute_dim(d: Any, values: Mapping[str, Any]) -> Any:
    return format_dim(tuple(expr.substitute(p, values) for p in parse_dim(d, "dimension")))


def _bound(g: Graph, v: Any, what: str) -> int:
    env = {k: x for k, x in g.symbols.items() if x is not None}
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


# =============================================================================
# split_map
# =============================================================================


def split_map(g: Graph, map: str, factor: int) -> Graph:
    g = _copy(g)
    body, i, _ = _locate(g, map)
    node = body[i]
    what = f"split_map {map!r}"
    if node.kind != "map":
        raise SandboxError(f"{what}: a {node.kind!r} is not a map")
    if "loop.kind" in node.attrs:
        raise SandboxError(f"{what}: already {node.attrs['loop.kind']}")
    if type(factor) is not int or factor < 1:
        raise SandboxError(f"{what}: factor {factor!r} must be a positive int")
    begin, end, step = parse_dim(node.attrs["range"], what)
    if step != 1:
        raise SandboxError(f"{what}: only a step of 1 is split, the range is {node.attrs['range']}")
    length = end if begin == 0 else f"({end}) - ({begin})"
    n = _bound(g, length, what)
    if n % factor:
        raise SandboxError(
            f"{what}: {n} iterations are not a multiple of {factor} (a tail is open item 31)"
        )
    var = node.attrs["var"]
    taken = (
        set(g.symbols)
        | set(g.containers)
        | {nd.attrs["var"] for nd, _ in g.walk() if nd.kind == "map"}
    )
    var_t, var_s = _fresh(f"{var}_t", taken), _fresh(f"{var}_s", taken)
    ids = {nd.id for nd, _ in g.walk()}
    inner = Node(
        _fresh(f"{map}_s", ids),
        "map",
        attrs={"var": var_s, "range": f"0:{factor}", "loop.kind": "spatial"},
        body=node.body,
    )
    start = "" if begin == 0 else f"{begin} + "
    _substitute_body(inner.body, {var: f"{start}{factor} * {var_t} + {var_s}"})
    node.attrs = {
        "var": var_t,
        "range": format_dim((0, expr.canonical(f"({length}) // {factor}"), 1)),
        **{k: v for k, v in node.attrs.items() if k not in ("var", "range")},
        "loop.kind": "temporal",
        "loop.split": {"var": var, "range": node.attrs["range"]},  # for join_map (D82)
    }
    node.body = [inner]
    return _result(g)


def _fresh(base: str, taken: set[str]) -> str:
    name, k = base, 0
    while name in taken:
        k += 1
        name = f"{base}_{k}"
    taken.add(name)
    return name


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
    from snax_forge.brm import BrmError, task_nest

    g = _copy(g)
    what = f"bind {node!r}"
    body, i, outer = _locate(g, node)
    tasklet = body[i]
    smap = outer[-1] if outer else None
    if smap is None or smap.attrs.get("loop.kind") != "spatial" or len(smap.body or []) != 1:
        raise SandboxError(f"{what}: must be the only node of a spatial map (split the map first)")
    temporal = []
    for m in reversed(outer[:-1]):
        if m.attrs.get("loop.kind") != "temporal":
            break
        temporal.insert(0, m)
    try:
        b = load_brm(brm)
    except BrmError as e:
        raise SandboxError(f"{what}: {e}") from None
    ports = match(tasklet, b)
    if b.function.code is None:
        raise SandboxError(f"{what}: brm {brm!r} does not say what it computes (function.code)")
    computes = expr.rename_code(tasklet.attrs["code"], ports)
    if computes != b.function.code:
        raise SandboxError(
            f"{what}: the tasklet computes {computes!r} on the BRM's ports, "
            f"{brm!r} computes {b.function.code!r}"
        )

    # design params: the lanes param from the spatial bound, the rest given or default
    begin, end, step = parse_dim(smap.attrs["range"], what)
    if begin != 0 or step != 1 or type(end) is not int:
        raise SandboxError(
            f"{what}: the spatial map must run over 0:<int>, not {smap.attrs['range']}"
        )
    lanes = {b.port(p).lanes for p in ports.values()}
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
    if inst.lanes(next(iter(ports.values()))) != end:
        raise SandboxError(
            f"{what}: the BRM has {inst.lanes(next(iter(ports.values())))} lanes, the map {end}"
        )

    # memlets: the spatial variable becomes the lanes' range; dtype and order checked
    var_s = smap.attrs["var"]
    n = int(np.prod([_bound(g, _length(m), what) for m in temporal], dtype=int))
    new_in: dict[str, Memlet] = {}
    new_out: dict[str, Memlet] = {}
    for side, conns, out in (
        ("inputs", tasklet.inputs, new_in),
        ("outputs", tasklet.outputs, new_out),
    ):
        for c, m in conns.items():
            port = ports[c]
            want = b.port(port).dtype
            if g.containers[m.data].dtype != want:
                raise SandboxError(
                    f"{what}: port {port!r} is {want}, container {m.data!r} is {g.containers[m.data].dtype}"
                )
            out[port] = Memlet(
                m.data, [_lanes_dim(d, var_s, end, f"{what}.{side}.{c}") for d in m.subset]
            )
            try:
                nest = task_nest(inst, port, {"n": n})
            except ValueError as e:
                raise SandboxError(f"{what}: {e}") from None
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
            "replaced": smap.to_dict(),  # for unbind (D82)
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
    g: Graph, m: Memlet, temporal: list[Node], smap: Node, want: np.ndarray, what: str
) -> None:
    """The memlet over the temporal maps and the lanes gives the nest's order."""
    env: dict[str, Any] = {k: v for k, v in g.symbols.items() if v is not None}
    loops = [*temporal, smap]
    for d, lp in enumerate(loops):
        begin, end, step = parse_dim(lp.attrs["range"], lp.id)
        values = np.arange(_bound(g, begin, what), _bound(g, end, what), step)
        shape = [1] * len(loops)
        shape[d] = len(values)
        env[lp.attrs["var"]] = values.reshape(shape)
    full = tuple(len(np.asarray(env[lp.attrs["var"]]).reshape(-1)) for lp in loops)
    got = np.stack(
        [np.broadcast_to(np.asarray(expr.evaluate(dim, env)), full) for dim in m.subset], axis=-1
    ).reshape(-1, full[-1], len(m.subset))
    if got.shape != want.shape or not np.array_equal(got - got[0, 0], want - want[0, 0]):
        raise SandboxError(f"{what}: the memlet {m.data}{m.subset} does not give the BRM's order")


# =============================================================================
# unbind and join_map: the inverses (D82)
# =============================================================================


def unbind(g: Graph, node: str) -> Graph:
    """The accelerated node ``node`` replaced by the subtree ``bind`` replaced."""
    g = _copy(g)
    body, i, _ = _locate(g, node)
    acc = body[i]
    what = f"unbind {node!r}"
    if acc.kind != "accelerated":
        raise SandboxError(f"{what}: a {acc.kind!r} is not an accelerated node")
    if acc.attrs.get("replaced") is None:
        raise SandboxError(f"{what}: it records nothing it replaced (written by hand?)")
    body[i] = Node.from_dict(acc.attrs["replaced"], what)
    return _result(g)


def join_map(g: Graph, map: str) -> Graph:
    """The inverse of split_map: a temporal map and its spatial map become the map they were.

    Needs ``loop.split`` (the var and range before the split) and the
    spatial map as the temporal map's only child. Every index under it is
    written back over the old variable: an index ``k + c * (b + f * v_t + v_s)``
    that split_map wrote becomes ``k + c * v``; an index in another form is
    an error, since it is not what split_map wrote.
    """
    g = _copy(g)
    body, i, _ = _locate(g, map)
    outer = body[i]
    what = f"join_map {map!r}"
    split = outer.attrs.get("loop.split") if outer.kind == "map" else None
    if split is None:
        raise SandboxError(f"{what}: not a map split by split_map (no loop.split)")
    kids = outer.body or []
    if len(kids) != 1 or kids[0].kind != "map" or kids[0].attrs.get("loop.kind") != "spatial":
        raise SandboxError(f"{what}: its only child must be the spatial map (unbind first)")
    inner = kids[0]
    v_t, v_s, var = outer.attrs["var"], inner.attrs["var"], split["var"]
    begin, _, _ = parse_dim(split["range"], what)
    b0, factor, step = parse_dim(inner.attrs["range"], what)
    if b0 != 0 or step != 1 or type(factor) is not int:
        raise SandboxError(f"{what}: the spatial map must run over 0:<int>")

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

    def walk(nodes: list[Node]) -> None:
        for n in nodes:
            for m in (*n.inputs.values(), *n.outputs.values()):
                m.subset = [back_dim(d) for d in m.subset]
            if n.kind == "map":
                n.attrs["range"] = back_dim(n.attrs["range"])
            if n.kind == "accelerated" and n.attrs.get("replaced"):
                old = Node.from_dict(n.attrs["replaced"], n.id)
                walk([old])
                n.attrs["replaced"] = old.to_dict()
            if n.body:
                walk(n.body)

    walk(inner.body or [])
    rest = {
        k: v for k, v in outer.attrs.items() if k not in ("var", "range", "loop.kind", "loop.split")
    }
    body[i] = Node(
        outer.id, "map", attrs={"var": var, "range": split["range"], **rest}, body=inner.body
    )
    return _result(g)


register_transform("split_map", split_map, ints=("factor",))
register_transform("bind", bind)
register_transform("unbind", unbind)
register_transform("join_map", join_map)
