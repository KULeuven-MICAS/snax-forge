"""The fold's seed: the reduction x' = x (+) fold_j(a_j) done in the fold itself (M13 B2 R5).

``fuse_seed(first, second)`` and ``unfuse_seed(map)``
    A fuse, as ``fuse_map`` (compose.py) is: ``first`` is a map of a fold's
    nest, any map of the chain of single-child maps down to one tasklet whose
    one output folds (``reduce``, D102), or a map ``split_reduce`` split, which
    stands for its final sum (the fold that writes the output); ``second`` a
    map of the elementwise nest that reads the fold's result, any map of its
    chain down to the reader (its outermost the site's default). The fold
    writes a transient ``t`` with op (+), and ``t``'s only reader is that
    nest, computing ``t (+) x`` with the same op and writing ``x`` in place,
    over ``t``'s whole subset, each element once, ``x`` at ``t``'s index
    (matvec_acc: the sum's ``tmp1[m]``, then ``augassign_19_4``'s ``q[m] =
    q[m] + tmp1[m]``): the fold's kept loops (those its output moves with) and
    the nest's loops run over the same elements, ``m_t`` and ``m`` (0:8)
    after ``tag_map sum_map temporal``, the fold's other loops free.
    ``fuse_seed`` removes that nest and ``t``: the fold writes ``x`` at its own
    subset, and its reduce holds ``"seed": {"data": "x", "subset": [...]}`` in
    place of ``identity`` (dfg/reduce.py), so the fold starts each element
    from ``x``'s. That is exact for any associative and commutative op on
    integers (D28), which every registered op is. Refused, naming what it
    found: a fold whose transient has another reader (or writer, or a link);
    a reader computing another op; a reader over a different subset (an
    element twice or not at all, ``x`` at another index, ``x`` not written in
    place, the fold not writing every element of ``t``); a target that is not
    a transient; a float dtype; a fold seeded already; a node between the
    fold and its reader naming ``x``, or the fold's nest reading it; a bound
    fold (unbind first); a ``second`` that is not a map of the reader's nest.
    The subsets are compared at the graph's symbol values (a recipe binds
    them at step 0). Registered as ``fuse_map`` is: its site a map of a fold's
    nest, filling ``second``; over two maps selected, ``first`` and
    ``second`` in the order selected.

    The fold's tasklet records what went (``sandbox.seeded``: the transient's
    name, definition, index and identity, and the nest with where it stood: ``after`` null
    when right after the node holding the fold, whatever that node's id becomes),
    which ``split_reduce`` moves to the final sum with the seed (the partials
    keep the identity) and ``join_reduce`` back. ``unfuse_seed`` (``map`` a map
    of the seeded fold's nest) puts the nest
    and the transient back, names as at import, and the fold writes ``t`` at
    its subset again from the identity: ``fuse_seed`` then ``unfuse_seed`` is
    the graph before. split, split_map, tag_map, untag_map, interchange_map and
    join_map carry the seed as any subset (``Memlet.map_dims``), so the steps
    before ``bind`` go in any order.

    ``seed_seconds`` gives the maps ``second`` takes for a ``first`` (the
    editor's drop-down), ``seed_pairs`` every (first, second) of a graph and
    ``seed_link`` the transient between them with the loops matched (the
    viewer's partners, viz/dfg/api.py), all through the site's check.

``bind`` takes a seeded fold only on a BRM with a ``seed`` port
(``accumulate``: TreeAccumulate and SerialAccumulate, D149), with the design
param ``seed`` 1 and the port on the seed's element, one beat per job.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from snax_forge import expr
from snax_forge.dfg import Container, Graph, Memlet, Node, Reduction, Seed
from snax_forge.dfg.graph import SEEDED
from snax_forge.dfg.subset import map_range
from snax_forge.expr import ExprError

from .recipe import SandboxError
from .transforms import _copy, _locate, _result, register_transform

# the reader's code for each registered op, on the transient's and x's connectors
_OP_CODE = {"add": "{t} + {x}", "mul": "{t} * {x}", "min": "min({t}, {x})",
            "max": "max({t}, {x})"}  # fmt: skip


def fold_of(g: Graph, map: str, what: str) -> tuple[Node, tuple[Node, ...]]:
    """The folding tasklet of ``map``'s nest and its enclosing nodes (module doc): ``map`` a map
    of the chain down to it, or a map split_reduce split, standing for its final sum."""
    body, i, _ = _locate(g, map)
    node = body[i]
    if node.kind != "map":
        raise SandboxError(f"{what}: a {node.kind!r} is not a map")
    rec = node.attrs.get("loop.split_reduce")
    if rec is not None:  # the final sum writes the output
        try:
            body, i, _ = _locate(g, rec["final"])
        except SandboxError:
            raise SandboxError(f"{what}: its final sum {rec['final']!r} is bound (unbind it "
                               "first)") from None  # fmt: skip
        node = body[i]
    while True:
        kids = node.body or []
        if len(kids) != 1:
            raise SandboxError(f"{what}: must be a map of a nest around one folding tasklet, and "
                               f"{node.id} holds {len(kids)} nodes")  # fmt: skip
        (node,) = kids
        if node.kind == "map":
            continue
        if node.kind == "accelerated":
            raise SandboxError(f"{what}: its fold is bound as {node.attrs['instance']!r} (unbind "
                               "first)")  # fmt: skip
        if node.kind != "tasklet":
            raise SandboxError(f"{what}: must be a map of a nest around one folding tasklet, not "
                               f"a {node.kind!r}")  # fmt: skip
        break
    folds = [m for m in node.outputs.values() if m.reduce is not None]
    if len(node.outputs) != 1 or len(folds) != 1:
        raise SandboxError(f"{what}: its tasklet {node.id} must fold its one output (a reduce, "
                           "D102)")  # fmt: skip
    _, _, outer = _locate(g, node.id)
    return node, outer


def _text(m: Memlet) -> str:
    return f"{m.data}[{', '.join(str(d) for d in m.subset)}]"


def _indices(g: Graph, outer: tuple[Node, ...], m: Memlet, what: str) -> np.ndarray:
    """The elements ``m`` names over every iteration of the maps ``outer``, one row each."""
    env: dict[str, Any] = dict(g.bound_symbols)
    maps = [o for o in outer if o.kind == "map"]
    try:
        ranges = [np.array(map_range(o.attrs["range"], env, o.id)) for o in maps]
        full = tuple(len(r) for r in ranges)
        for k, (o, r) in enumerate(zip(maps, ranges, strict=True)):  # one axis per map
            env[o.attrs["var"]] = r.reshape([len(r) if j == k else 1 for j in range(len(maps))])
        dims = [np.broadcast_to(np.asarray(expr.evaluate(d, env)), full) for d in m.subset]
    except ExprError as e:
        raise SandboxError(f"{what}: {e} (bind the symbols first: a recipe's step 0)") from None
    return np.stack(dims, axis=-1).reshape(-1, len(m.subset)).astype(int)


def _shape(g: Graph, c: str, what: str) -> tuple[int, ...]:
    try:
        return tuple(expr.evaluate(s, g.bound_symbols) for s in g.containers[c].shape)
    except ExprError as e:
        raise SandboxError(f"{what}: {e} (bind the symbols first: a recipe's step 0)") from None


def _valid(rows: np.ndarray, shape: tuple[int, ...], outer: tuple[Node, ...]) -> np.ndarray:
    """The rows inside ``shape``; under a tailed split (D144) the pad lanes' rows past it go."""
    inside = np.all((rows >= 0) & (rows < np.array(shape)), axis=1)
    if any(o.kind == "map" and (o.attrs.get("loop.split") or {}).get("valid") for o in outer):
        return rows[inside]
    return rows


def _nest_root(g: Graph, node: Node) -> tuple[Node, Node | None, list[Node]]:
    """The outermost map of ``node``'s chain of single-child maps, its parent (None at the top)
    and the body holding it."""
    _, _, outer = _locate(g, node.id)
    root = node
    k = len(outer)
    while k and outer[k - 1].kind == "map" and len(outer[k - 1].body or []) == 1:
        root = outer[k - 1]
        k -= 1
    parent = outer[k - 1] if k else None
    return root, parent, parent.body if parent is not None else g.body


def _reader_chain(g: Graph, reader: Node) -> list[Node]:
    """The maps of the elementwise nest around ``reader``, outermost first: ``second``'s
    choices (its chain of single-child maps)."""
    root, _, _ = _nest_root(g, reader)
    chain: list[Node] = []
    node = root
    while node.kind == "map":
        chain.append(node)
        node = (node.body or [node])[0]
    return chain


def seed_seconds(g: Graph, first: str, what: str | None = None) -> list[str]:
    """The maps ``second`` takes for ``first`` (module doc), outermost first; SandboxError
    with the site's refusal when ``first`` takes none."""
    what = what or f"fuse_seed {first!r}"
    found = _check(g, first, what)
    chain = _reader_chain(g, found["reader"])
    if not chain:
        raise SandboxError(f"{what}: {found['t']}'s reader {found['reader'].id} is in no map")
    return [m.id for m in chain]


def _site(g: Graph, first: str) -> dict[str, Any]:
    """fuse_seed's site: a map of a fold's nest the checks of the module doc pass; ``second``
    the elementwise nest's outermost map."""
    return {"first": first, "second": seed_seconds(g, first)[0]}


def _multi(g: Graph, selected: list[str]) -> dict[str, Any]:
    """One map, ``_site``'s; or two, ``first`` and ``second`` in the order selected."""
    if len(selected) == 1:
        return _site(g, selected[0])
    if len(selected) != 2:
        raise SandboxError(f"fuse_seed: takes one map or two, not {len(selected)}")
    first, second = selected
    _second(g, first, second)
    return {"first": first, "second": second}


def _second(g: Graph, first: str, second: str) -> None:
    what = f"fuse_seed {first!r} {second!r}"
    seconds = seed_seconds(g, first, what)
    if second not in seconds:
        raise SandboxError(f"{what}: {second!r} is not a map of the elementwise nest that reads "
                           f"the fold's result ({', '.join(seconds)})")  # fmt: skip


def _check(g: Graph, map: str, what: str) -> dict[str, Any]:
    fold, outer = fold_of(g, map, what)
    ((cout, om),) = fold.outputs.items()
    red = om.reduce
    assert red is not None
    t = om.data
    if red.seed is not None:
        raise SandboxError(f"{what}: {fold.id} is seeded from {red.seed.data} already "
                           "(unfuse_seed first)")  # fmt: skip
    if not g.containers[t].transient:
        raise SandboxError(f"{what}: {fold.id} folds into {t}, which is not a transient; the seed "
                           "takes a transient that one elementwise nest adds to x")  # fmt: skip
    if np.issubdtype(np.dtype(g.containers[t].dtype), np.floating):
        raise SandboxError(f"{what}: {t} is {g.containers[t].dtype}, a float op type; seeding "
                           "reorders the fold, exact on integers only (D28)")  # fmt: skip
    readers, others = [], []
    for n, _ in g.walk():
        if n is fold:
            continue
        if n.kind == "link" and n.attrs.get("container") == t:
            others.append(f"the link {n.id}")
        if any(m.data == t for m in n.outputs.values()):
            others.append(f"{n.id} (a writer)")
        if any(m.data == t for m in n.inputs.values()):
            readers.append(n)
    if others or len(readers) != 1:
        named = [*(r.id for r in readers), *others]
        raise SandboxError(f"{what}: {t} has another reader: {named}; the seed takes a transient "
                           "whose only reader is the elementwise nest t (+) x")  # fmt: skip
    (r,) = readers
    if r.kind == "accelerated":
        raise SandboxError(f"{what}: {t}'s reader {r.id} is bound (unbind first)")
    cts = [c for c, m in r.inputs.items() if m.data == t]
    if r.kind != "tasklet" or len(r.inputs) != 2 or len(r.outputs) != 1 or len(cts) != 1:
        raise SandboxError(f"{what}: {t}'s reader {r.id} must be a tasklet computing {t} "
                           f"{red.op} x from two inputs")  # fmt: skip
    (ct,) = cts
    (cx,) = [c for c in r.inputs if c != ct]
    ((rout, wm),) = r.outputs.items()
    x = r.inputs[cx].data
    want = f"{rout} = " + _OP_CODE[red.op].format(t=ct, x=cx)
    if expr.unify(want, r.attrs["code"], ()) is None:
        raise SandboxError(f"{what}: {t}'s reader {r.id} computes {r.attrs['code']!r}, a "
                           f"different op from the fold's {red.op} ({want!r})")  # fmt: skip
    if wm.reduce is not None or wm.data != x:
        raise SandboxError(f"{what}: {r.id} writes {wm.data}, not {x} in place: a different "
                           f"subset (x' = x {red.op} {t})")  # fmt: skip
    _, _, r_outer = _locate(g, r.id)
    shape = _shape(g, t, what)
    if _shape(g, x, what) != shape:
        raise SandboxError(f"{what}: {x} has shape {_shape(g, x, what)}, {t} {shape}: a "
                           "different subset")  # fmt: skip
    rt = _valid(_indices(g, r_outer, r.inputs[ct], what), shape, r_outer)
    rx = _valid(_indices(g, r_outer, r.inputs[cx], what), shape, r_outer)
    rw = _valid(_indices(g, r_outer, wm, what), shape, r_outer)
    every = {tuple(i) for i in np.ndindex(*shape)}
    once = len(rt) == len(every) and {tuple(i) for i in rt} == every
    if not (once and np.array_equal(rt, rx) and np.array_equal(rx, rw)):
        raise SandboxError(f"{what}: {r.id} reads {_text(r.inputs[ct])} and {_text(r.inputs[cx])}, "
                           f"writes {_text(wm)}: a different subset; the seed needs each element "
                           f"of {t} once, {x} at its index")  # fmt: skip
    written = _valid(_indices(g, outer, om, what), shape, outer)
    if {tuple(i) for i in written} != every:
        raise SandboxError(f"{what}: {fold.id} writes part of {_text(om)} only: a different "
                           "subset from its reader's")  # fmt: skip
    order = [n for n, _ in g.walk()]
    between = order[order.index(fold) : order.index(r)]
    for n in between:
        names = [m.data for m in (*n.inputs.values(), *n.outputs.values())]
        if x in names or (n.kind == "link" and n.attrs.get("container") == x):
            raise SandboxError(f"{what}: {n.id} names {x} between the fold and its reader; the "
                               f"seed would read {x} before it")  # fmt: skip
    return {"fold": fold, "cout": cout, "reader": r, "x": x, "t": t}


def fuse_seed(g: Graph, first: str, second: str) -> Graph:
    what = f"fuse_seed {first!r} {second!r}"
    _site(g, first)  # the site's refusal first, as every transform's
    _second(g, first, second)
    g = _copy(g)
    found = _check(g, first, what)
    fold, cout, r, x, t = (found[k] for k in ("fold", "cout", "reader", "x", "t"))
    om = fold.outputs[cout]
    assert om.reduce is not None
    root, parent, body = _nest_root(g, r)
    k = next(i for i, n in enumerate(body) if n is root)
    names = list(g.containers)
    fold.attrs[SEEDED] = {
        "container": t,
        "definition": g.containers[t].to_dict(),
        "index": names.index(t),
        "identity": om.reduce.identity,
        "nest": root.to_dict(),
        "parent": parent.id if parent is not None else None,
        # null: right after the node holding the fold (or first), whatever its id becomes
        "after": None if k == 0 or _holds(body[k - 1], fold) else body[k - 1].id,
    }
    fold.outputs[cout] = Memlet(x, list(om.subset), Reduction(om.reduce.op, None,
                                                              Seed(x, tuple(om.subset))))  # fmt: skip
    del body[k]
    del g.containers[t]
    return _result(g)


def _unseed_site(g: Graph, map: str) -> dict[str, Any]:
    _unseed_check(g, map, f"unfuse_seed {map!r}")
    return {"map": map}


def _unseed_check(g: Graph, map: str, what: str) -> tuple[Node, dict[str, Any]]:
    fold, _ = fold_of(g, map, what)
    ((_, om),) = fold.outputs.items()
    rec = fold.attrs.get(SEEDED)
    if om.reduce is None or om.reduce.seed is None or rec is None:
        raise SandboxError(f"{what}: {fold.id}'s fold is not seeded by fuse_seed")
    t = rec["container"]
    taken = set(g.symbols) | set(g.containers) | {n.attrs["var"] for n, _ in g.walk()
                                                  if n.kind == "map"}  # fmt: skip
    if t in taken:
        raise SandboxError(f"{what}: the graph has a name {t!r} again")
    ids = {n.id for n, _ in g.walk()}
    nest = Node.from_dict(rec["nest"], what)
    back = [n.id for n in _ids(nest) if n.id in ids]
    if back:
        raise SandboxError(f"{what}: the graph has a node {back[0]!r} again")
    if rec["parent"] is not None and rec["parent"] not in ids:
        raise SandboxError(f"{what}: the map {rec['parent']!r} that held the nest is gone")
    return fold, rec


def _holds(n: Node, fold: Node) -> bool:
    """``n`` is ``fold`` or holds it."""
    return any(x is fold for x in _ids(n))


def _ids(n: Node) -> list[Node]:
    return [n, *(k for c in n.body or [] for k in _ids(c))]


def unfuse_seed(g: Graph, map: str) -> Graph:
    """The inverse of fuse_seed: the elementwise nest and the transient back."""
    what = f"unfuse_seed {map!r}"
    _unseed_check(g, map, what)
    g = _copy(g)
    fold, rec = _unseed_check(g, map, what)
    ((cout, om),) = fold.outputs.items()
    assert om.reduce is not None
    t = rec["container"]
    items = list(g.containers.items())
    items.insert(min(rec["index"], len(items)), (t, Container.from_dict(rec["definition"], what)))
    g.containers = dict(items)
    fold.outputs[cout] = Memlet(t, list(om.subset), Reduction(om.reduce.op, rec["identity"]))
    body = g.body if rec["parent"] is None else (g.node(rec["parent"]).body or [])
    ids = [n.id for n in body]
    holder = next((i for i, n in enumerate(body) if _holds(n, fold)), None)
    if rec["after"] in ids:
        at = ids.index(rec["after"]) + 1
    else:  # after the node holding the fold (or first)
        at = 0 if holder is None else holder + 1
    body.insert(at, Node.from_dict(rec["nest"], what))
    del fold.attrs[SEEDED]
    return _result(g)


def fold_map(g: Graph, fold: Node) -> str:
    """The map that names ``fold`` in a recipe and the editor: the map split_reduce split, for
    its final sum, else the outermost map of the fold's chain whose variable the fold's output
    does not move with (matvec's ``sum_map_1``, before and after its split)."""
    _, _, outer = _locate(g, fold.id)
    srs = {n.attrs["loop.split_reduce"]["final"]: n.id for n, _ in g.walk()
           if n.kind == "map" and n.attrs.get("loop.split_reduce")}  # fmt: skip
    chain: list[Node] = []
    for o in reversed(outer):
        if o.kind != "map" or (chain and len(o.body or []) != 1):
            break
        chain.insert(0, o)
        if o.id in srs:
            return srs[o.id]
    ((_, om),) = fold.outputs.items()
    return next((o.id for o in chain if o.attrs["var"] not in om.names), chain[-1].id)


def seeded_choices(g: Graph) -> list[dict[str, str]]:
    """The folds ``unfuse_seed`` takes, one each in the graph's order: its map (``fold_map``),
    its tasklet, what it folded into, the seed's container and the reader's code (the
    editor's drop-down)."""
    out: list[dict[str, str]] = []
    for n, _ in g.walk():
        if n.kind != "tasklet" or not any(m.reduce is not None for m in n.outputs.values()):
            continue
        try:
            m = fold_map(g, n)
            _unseed_check(g, m, "")
        except (StopIteration, IndexError, SandboxError):
            continue
        ((_, om),) = n.outputs.items()
        assert om.reduce is not None
        into, x = n.attrs[SEEDED]["container"], om.data
        reads = f"{x} = " + _OP_CODE[om.reduce.op].format(t=into, x=x)
        out.append({"map": m, "fold": n.id, "into": into, "x": x, "reads": reads})
    return out


def seed_pairs(g: Graph) -> list[tuple[str, str]]:
    """Every (first, second) ``fuse_seed`` takes in ``g`` (``seed_seconds`` on each map), in the
    graph's order."""
    out: list[tuple[str, str]] = []
    for n, _ in g.walk():
        if n.kind != "map":
            continue
        try:
            seconds = seed_seconds(g, n.id)
        except SandboxError:
            continue
        out += [(n.id, s) for s in seconds]
    return out


def range_text(g: Graph, rng: str) -> str:
    """A map's range at the graph's symbols, ``0:8``; as written where they are unknown."""
    try:
        return ":".join(str(expr.evaluate(p, g.bound_symbols)) for p in rng.split(":"))
    except ExprError:
        return rng


def seed_link(g: Graph, first: str, second: str) -> dict[str, Any]:
    """The transient between ``first``'s fold and ``second``'s nest and the loops matched: the
    fold's kept loops (the maps around it its output moves with) against the nest's, level by
    level, ``{"container": "tmp1", "subset": ["m"], "loops": [["m_t", "m", "0:8"]]}``; the
    two sides' ranges joined by `` = `` where they differ (``0:8 = 0:4``), and the leftover
    loops of the longer side with an empty partner."""
    _second(g, first, second)
    found = _check(g, first, "")
    fold, r = found["fold"], found["reader"]
    ((_, om),) = fold.outputs.items()
    _, _, outer = _locate(g, fold.id)
    kept = [o for o in outer if o.kind == "map" and o.attrs["var"] in om.names]
    mine = _reader_chain(g, r)
    loops = []
    for k in range(max(len(kept), len(mine))):
        a = kept[k] if k < len(kept) else None
        b = mine[k] if k < len(mine) else None
        ra = range_text(g, a.attrs["range"]) if a else None
        rb = range_text(g, b.attrs["range"]) if b else None
        text = ra if ra == rb or rb is None else rb if ra is None else f"{ra} = {rb}"
        loops.append([a.attrs["var"] if a else "", b.attrs["var"] if b else "", text])
    (read,) = [m for m in r.inputs.values() if m.data == found["t"]]
    return {"container": found["t"], "subset": [str(d) for d in read.subset], "loops": loops}


register_transform("fuse_seed", fuse_seed, site=_site, multi=_multi)
register_transform("unfuse_seed", unfuse_seed, site=_unseed_site)
