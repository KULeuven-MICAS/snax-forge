"""The transforms that compose units into one accelerator (D148, D150), each with its inverse.

The mapping is spelled step by step, one recipe per variant (D150): a
reduction split into lane partials and a final sum, two nests fused into one,
a transient between two bound units made a link. Registered as any transform
(transforms.py); ``ints`` as there.

``split_reduce(map, factor)`` and ``join_reduce(map)``
    an untagged map ``v`` over ``b:e`` whose one tasklet folds (``reduce``)
    into an element that does not move with ``v`` is split as ``split_map``
    splits it: ``<map>`` temporal around a spatial ``<map>_s``; its tasklet
    now writes one partial per lane, into a new transient
    ``<output>_partial`` indexed by the output's dimensions that move with an
    enclosing map, then the spatial variable (dot's ``out_partial[v_s]``,
    matvec's ``q_partial[i, v_s]``), folding as before. A final sum follows
    the temporal map: tasklet ``<tasklet>_final`` in a spatial map
    ``<map>_final`` over ``0:factor``, folding the partials into the output
    as the tasklet did. The temporal map records both
    (``loop.split_reduce``); ``join_reduce`` takes them back and joins the
    map (``join_map``). A length that is not a multiple of the factor is
    refused: the partials' padding is not placed yet (D144).
``fuse_map(first, second)`` and ``unfuse_map(map)``
    two adjacent nests, ``second`` right after ``first``, with the same
    ranges and tags level by level down to one tasklet each, joined by a
    transient the first writes and the second reads at the same index,
    become one nest of two tasklets: the first's ids and variables (the
    second's renamed onto them), the second's other nodes after the fused
    one at their level (matvec's final sum inside the row map). The
    transient becomes a local value: ``loop.local`` (the fused map's id) on
    the container, so it gets no L1 space; nothing else may name it. The
    fused map records both nests (``loop.fused``), and ``unfuse_map`` gives
    them back when nothing changed since (unbind first). ``bind`` takes the
    fused nest's spatial map (patterns.py's ``fused_view``).
``link(container, depth, group)`` and ``unlink(container)``
    a transient with one writer and one reader, both bound (``link.unbound``
    otherwise; ``link.container`` for another container), becomes a link
    node of that depth before the reader, in the body the two share, once
    the writer's beats equal the reader's in the same order (each port's
    memlet over its firing loops, at the recipe's symbols, one beat per T
    firings for a named rate: ``link.order``). The two units' groups merge:
    ``group`` is written on every unit of both; a unit whose group has
    another name, a name another instance has, or units in two
    accelerators is ``link.group``. The container stays, so the reference
    executor runs the link as it (dfg/execute.py); the design step gives it
    no L1 space and no streamer. ``unlink`` removes the node and takes the
    group off every unit left alone; two groups of several units left under
    one name is ``unlink.group``.
"""

from __future__ import annotations

import copy
from itertools import product

from snax_forge import expr
from snax_forge.common import CodedError, fresh
from snax_forge.dfg import Container, Graph, Memlet, Node
from snax_forge.dfg.subset import dim_names, map_range, parse_dim

from .recipe import SandboxError
from .transforms import (
    _accelerated,
    _bound,
    _copy,
    _length,
    _locate,
    _result,
    _substitute_body,
    join_map,
    load_brm,
    register_transform,
    split_map,
)


class LinkError(CodedError, SandboxError):
    """A link the sandbox refuses; ``code`` names the rule (``link.<rule>``, ``unlink.<rule>``)."""


def _taken(g: Graph) -> set[str]:
    """Every name a new container or variable must avoid."""
    return (
        set(g.symbols)
        | set(g.containers)
        | {n.attrs["var"] for n, _ in g.walk() if n.kind == "map"}
    )


# =============================================================================
# split_reduce and join_reduce
# =============================================================================


def split_reduce(g: Graph, map: str, factor: int) -> Graph:
    what = f"split_reduce {map!r}"
    body, i, outer = _locate(g, map)
    node = body[i]
    kids = node.body or []
    if node.kind != "map" or len(kids) != 1 or kids[0].kind != "tasklet":
        raise SandboxError(f"{what}: must be a map around one tasklet")
    t = kids[0]
    folds = [(c, m) for c, m in t.outputs.items() if m.reduce is not None]
    if len(t.outputs) != 1 or len(folds) != 1 or node.attrs["var"] in folds[0][1].names:
        raise SandboxError(f"{what}: its tasklet must fold its one output into an element that "
                           f"does not move with {node.attrs['var']!r} (a reduction, D102)")  # fmt: skip
    if type(factor) is int and factor > 0 and _bound(g, _length(node), what) % factor:
        raise SandboxError(f"{what}: the length is not a multiple of {factor}; the partials' "
                           "padding is not placed yet (D144)")  # fmt: skip
    g = split_map(g, map, factor)
    body, i, outer = _locate(g, map)
    node = body[i]
    (smap,) = node.body
    (t,) = smap.body
    (cout, om), (cin,) = next(iter(t.outputs.items())), t.inputs
    enclosing = {m.attrs["var"] for m in outer if m.kind == "map"}
    kept = [k for k, d in enumerate(om.subset) if dim_names(d) & enclosing]
    taken = _taken(g)
    partial = fresh(f"{om.data}_partial", taken)
    out = g.containers[om.data]
    g.containers[partial] = Container([*(out.shape[k] for k in kept), factor], out.dtype, True)
    index = [om.subset[k] for k in kept]
    t.outputs[cout] = Memlet(partial, [*index, smap.attrs["var"]], om.reduce)
    var = fresh(f"{node.attrs['loop.split']['var']}_f", taken | {partial})
    ids = {n.id for n, _ in g.walk()}
    final_t = Node(fresh(f"{t.id}_final", ids), "tasklet", inputs={cin: Memlet(partial, [*index, var])},
                   outputs={cout: om}, attrs={"code": t.attrs["code"]})  # fmt: skip
    final = Node(fresh(f"{map}_final", ids | {final_t.id}), "map",
                 attrs={"var": var, "range": f"0:{factor}", "loop.kind": "spatial"}, body=[final_t])  # fmt: skip
    node.attrs["loop.split_reduce"] = {"partial": partial, "final": final.id}
    body.insert(i + 1, final)
    return _result(g)


def join_reduce(g: Graph, map: str) -> Graph:
    """The inverse of split_reduce: the partials and the final sum gone, the map joined."""
    g = _copy(g)
    what = f"join_reduce {map!r}"
    body, i, _ = _locate(g, map)
    node = body[i]
    rec = node.attrs.get("loop.split_reduce") if node.kind == "map" else None
    if rec is None:
        raise SandboxError(f"{what}: not a map split by split_reduce (no loop.split_reduce)")
    try:
        fbody, j, _ = _locate(g, rec["final"])
    except SandboxError:
        raise SandboxError(
            f"{what}: its final sum {rec['final']!r} is bound (unbind it first)"
        ) from None
    final = fbody[j]
    kids = node.body or []
    inner = kids[0].body if len(kids) == 1 and kids[0].kind == "map" else None
    if not inner or len(inner) != 1 or inner[0].kind != "tasklet" or fbody is not body:
        raise SandboxError(f"{what}: its spatial map must hold its tasklet again (unbind and "
                           "unfuse_map first)")  # fmt: skip
    ft = (final.body or [None])[0]
    if ft is None or ft.kind != "tasklet":
        raise SandboxError(f"{what}: the final sum {final.id!r} is bound (unbind it first)")
    users = [n.id for n, _ in g.walk() if any(m.data == rec["partial"] for m in
             (*n.inputs.values(), *n.outputs.values()))] + [n.id for n, _ in g.walk()
             if n.kind == "link" and n.attrs["container"] == rec["partial"]]  # fmt: skip
    if sorted(users) != sorted([inner[0].id, ft.id]):
        raise SandboxError(f"{what}: {rec['partial']} is named by {users} (unlink first)")
    (cout,) = inner[0].outputs
    inner[0].outputs[cout] = ft.outputs[cout]
    del body[j]
    del g.containers[rec["partial"]]
    del node.attrs["loop.split_reduce"]
    return join_map(_result(g), map)


# =============================================================================
# fuse_map and unfuse_map
# =============================================================================


def _fuse(g: Graph, first: Node, second: Node, what: str) -> tuple[Node, str]:
    """``first`` and ``second`` fused (module doc) and the local container; SandboxError."""
    rename: dict[str, str] = {}
    f, s = first, second
    while f.kind == "map":
        if s.kind != "map" or len(f.body or []) != 1 or not s.body:
            raise SandboxError(f"{what}: {f.id} and {s.id} are not nests of the same depth down "
                               "to one tasklet each")  # fmt: skip
        rename[s.attrs["var"]] = f.attrs["var"]
        f, s = f.body[0], s.body[0]
    if f.kind != "tasklet" or s.kind != "tasklet":
        raise SandboxError(f"{what}: the nests must end in one tasklet each")
    second = copy.deepcopy(second)
    _substitute_body([second], rename)
    f, s = first, second
    levels = []
    while f.kind == "map":
        same = (f.attrs["range"], f.attrs.get("loop.kind")) == (
            s.attrs["range"],
            s.attrs.get("loop.kind"),
        )
        if not same:
            raise SandboxError(f"{what}: {f.id} runs {f.attrs['range']} "
                               f"{f.attrs.get('loop.kind') or 'untagged'}, {s.id} "
                               f"{s.attrs['range']} {s.attrs.get('loop.kind') or 'untagged'}; "
                               "fused nests have the same ranges and tags")  # fmt: skip
        levels.append((f, s.body[1:]))
        f, s = f.body[0], s.body[0]
    t1, t2 = f, s
    writes = list(t1.outputs.values())
    reads = [m for m in t2.inputs.values() if writes and m.data == writes[0].data]
    if len(writes) != 1 or len(reads) != 1 or (reads[0].subset != writes[0].subset):
        raise SandboxError(f"{what}: {t1.id} and {t2.id} are not joined by a transient the first "
                           "writes and the second reads at the same index")  # fmt: skip
    local = writes[0].data
    if not g.containers[local].transient or writes[0].reduce is not None:
        raise SandboxError(f"{what}: {local} is not a transient written plainly")
    users = {n.id for n, _ in g.walk()
             for m in (*n.inputs.values(), *n.outputs.values()) if m.data == local}  # fmt: skip
    others = users - {t1.id, t2.id}
    if others:
        raise SandboxError(f"{what}: {local} is named by {sorted(others)} too; a local value is "
                           "the two tasklets' only")  # fmt: skip
    inner: list[Node] = [copy.deepcopy(t1), t2]
    for f, extra in reversed(levels):
        node = Node(f.id, "map", attrs=copy.deepcopy(f.attrs), body=[*inner, *extra])
        inner = [node]
    return inner[0], local


def fuse_map(g: Graph, first: str, second: str) -> Graph:
    g = _copy(g)
    what = f"fuse_map {first!r} {second!r}"
    body, i, _ = _locate(g, first)
    if i + 1 >= len(body) or body[i + 1].id != second:
        raise SandboxError(f"{what}: {second!r} must come right after {first!r} in one body")
    f, s = body[i], body[i + 1]
    if f.kind != "map" or s.kind != "map":
        raise SandboxError(f"{what}: both must be maps")
    fused, local = _fuse(g, f, s, what)
    fused.attrs["loop.fused"] = {"first": f.to_dict(), "second": s.to_dict(), "local": local}
    body[i : i + 2] = [fused]
    g.containers[local].attrs["loop.local"] = fused.id
    return _result(g)


def unfuse_map(g: Graph, map: str) -> Graph:
    """The inverse of fuse_map: the two nests it fused, when nothing changed since."""
    g = _copy(g)
    what = f"unfuse_map {map!r}"
    body, i, _ = _locate(g, map)
    node = body[i]
    rec = node.attrs.get("loop.fused") if node.kind == "map" else None
    if rec is None:
        raise SandboxError(f"{what}: not a nest fuse_map made (no loop.fused)")
    f, s = Node.from_dict(rec["first"], what), Node.from_dict(rec["second"], what)
    again, _ = _fuse(g, f, s, what)
    again.attrs["loop.fused"] = rec
    if again.to_dict() != node.to_dict():
        raise SandboxError(f"{what}: the nest changed since fuse_map (unbind first)")
    body[i : i + 1] = [f, s]
    del g.containers[rec["local"]].attrs["loop.local"]
    return _result(g)


# =============================================================================
# link and unlink
# =============================================================================


def _path(body: list[Node], node_id: str) -> list[tuple[list[Node], int]] | None:
    """The body lists and indices from the top down to ``node_id``."""
    for k, n in enumerate(body):
        if n.id == node_id:
            return [(body, k)]
        if n.body:
            sub = _path(n.body, node_id)
            if sub:
                return [(body, k), *sub]
    return None


def beats(g: Graph, node: Node, port: str) -> list[tuple[tuple[int, ...], ...]]:
    """The elements an accelerated node's ``port`` moves, beat by beat: its memlet over its
    firing loops, outermost first (for a named rate the loops it uses: one beat per T
    firings), at the graph's symbols; each beat its lanes' indices, row-major."""
    m = (node.inputs | node.outputs)[port]
    rate = load_brm(node.attrs["brm"]).port(port).rate
    outer = next(o for n, o in g.walk() if n is node)
    loops = [o for o in outer if o.kind == "map"]
    if rate != 1:
        loops = [lp for lp in loops if lp.attrs["var"] in m.names]
    symbols = g.bound_symbols
    ranges = [list(map_range(lp.attrs["range"], symbols, lp.id)) for lp in loops]
    out = []
    for values in product(*ranges):
        env = symbols | dict(zip((lp.attrs["var"] for lp in loops), values, strict=True))
        dims = []
        for d in m.subset:
            parts = parse_dim(d, m.data)
            ends = [expr.evaluate(p, env) for p in parts[:2]]
            dims.append(ends[:1] if len(parts) == 1 else list(range(ends[0], ends[1], parts[2])))
        out.append(tuple(product(*dims)))
    return out


def _group(n: Node) -> str:
    return n.attrs.get("group", n.attrs["instance"])


def link(g: Graph, container: str, depth: int, group: str) -> Graph:
    g = _copy(g)
    what = f"link {container!r}"
    c = g.containers.get(container)
    if c is None or not c.transient or "loop.local" in c.attrs:
        raise LinkError("link.container", f"{what}: not a transient container of the graph "
                        "(a fused nest's local value is no link)")  # fmt: skip
    if any(n.kind == "link" and n.attrs["container"] == container for n, _ in g.walk()):
        raise LinkError("link.container", f"{what}: already a link")
    writers = [n for n, _ in g.walk() if any(m.data == container for m in n.outputs.values())]
    readers = [n for n, _ in g.walk() if any(m.data == container for m in n.inputs.values())]
    if len(writers) != 1 or len(readers) != 1:
        raise LinkError("link.container", f"{what}: written by {[n.id for n in writers]}, read "
                        f"by {[n.id for n in readers]}; a link has one writer and one reader")  # fmt: skip
    (w,), (r,) = writers, readers
    if w.kind != "accelerated" or r.kind != "accelerated":
        raise LinkError("link.unbound", f"{what}: {w.id} ({w.kind}) to {r.id} ({r.kind}); both "
                        "ends of a link are bound units (bind them first)")  # fmt: skip
    if type(depth) is not int or depth < 0:
        raise LinkError("link.container", f"{what}: depth {depth!r}; 0 is a wire, >= 1 a buffer")
    (wp,) = [p for p, m in w.outputs.items() if m.data == container]
    (rp,) = [p for p, m in r.inputs.items() if m.data == container]
    wb, rb = beats(g, w, wp), beats(g, r, rp)
    if wb != rb:
        raise LinkError("link.order", f"{what}: {w.id}.{wp} gives {len(wb)} beats of "
                        f"{len(wb[0]) if wb else 0} lanes, {r.id}.{rp} takes {len(rb)} of "
                        f"{len(rb[0]) if rb else 0}, or another order; a link passes the beats "
                        "as they are")  # fmt: skip
    nodes = _accelerated(g)
    merged = {_group(w), _group(r)}
    members = {n.attrs["instance"] for n in nodes if _group(n) in merged}
    if not isinstance(group, str) or not group.isidentifier():
        raise LinkError("link.group", f"{what}: group {group!r} is not a name")
    named = {
        n.attrs["group"] for n in nodes if n.attrs["instance"] in members and "group" in n.attrs
    }
    if named - {group}:
        raise LinkError("link.group", f"{what}: a unit's group is already {sorted(named)}, not "
                        f"{group!r}")  # fmt: skip
    others = {n.attrs["instance"] for n in nodes} - members
    if group in others or group in {_group(n) for n in nodes if n.attrs["instance"] in others}:
        raise LinkError("link.group", f"{what}: {group!r} names another instance or group")
    accs = {n.attrs.get("accelerator") for n in nodes if n.attrs["instance"] in members}
    if len(accs) > 1:
        raise LinkError(
            "link.group", f"{what}: its units are in accelerators {sorted(map(str, accs))}"
        )
    wpath, rpath = _path(g.body, w.id), _path(g.body, r.id)
    k = 0
    while (
        k + 1 < min(len(wpath), len(rpath))
        and wpath[k][0] is rpath[k][0]
        and wpath[k][1] == rpath[k][1]
    ):
        k += 1
    shared, wi, ri = rpath[k][0], wpath[k][1], rpath[k][1]
    if wpath[k][0] is not shared or wi >= ri:
        raise LinkError("link.order", f"{what}: {w.id} does not run before {r.id}")
    for n in nodes:
        if n.attrs["instance"] in members:
            n.attrs["group"] = group
    ids = {n.id for n, _ in g.walk()}
    node = Node(fresh(container, ids), "link", attrs={"container": container, "depth": depth,
                "from": w.attrs["instance"], "to": r.attrs["instance"]})  # fmt: skip
    shared.insert(ri, node)
    return _result(g)


def unlink(g: Graph, container: str) -> Graph:
    """The inverse of link: the link node gone, the group taken off every unit left alone."""
    g = _copy(g)
    what = f"unlink {container!r}"
    found = [n for n, _ in g.walk() if n.kind == "link" and n.attrs["container"] == container]
    if not found:
        raise SandboxError(f"{what}: no link of that container")
    body, i, _ = _locate(g, found[0].id)
    gone = body.pop(i)
    nodes = _accelerated(g)
    name = next(_group(n) for n in nodes if n.attrs["instance"] == gone.attrs["from"])
    members = {n.attrs["instance"] for n in nodes if _group(n) == name}
    parent = {u: u for u in members}

    def root(u: str) -> str:
        while parent[u] != u:
            u = parent[u]
        return u

    for n, _ in g.walk():
        if n.kind == "link" and n.attrs["from"] in members:
            parent[root(n.attrs["from"])] = root(n.attrs["to"])
    parts: dict[str, set[str]] = {}
    for u in members:
        parts.setdefault(root(u), set()).add(u)
    if sum(len(p) > 1 for p in parts.values()) > 1:
        raise LinkError("unlink.group", f"{what}: group {name!r} would fall into "
                        f"{sorted(map(sorted, parts.values()))}; unlink the others first")  # fmt: skip
    alone = {u for p in parts.values() if len(p) == 1 for u in p}
    for n in nodes:
        if n.attrs["instance"] in alone:
            n.attrs.pop("group", None)
    return _result(g)


register_transform("split_reduce", split_reduce, ints=("factor",))
register_transform("join_reduce", join_reduce)
register_transform("fuse_map", fuse_map)
register_transform("unfuse_map", unfuse_map)
register_transform("link", link, ints=("depth",))
register_transform("unlink", unlink)
