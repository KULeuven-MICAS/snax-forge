"""The transforms that compose units into one accelerator (D148, D150), each with its inverse.

The mapping is spelled step by step, one recipe per variant: a
reduction split into lane partials and a final sum, two nests fused into one,
a transient between two bound units made a link. Registered as any transform
(transforms.py); ``ints`` and the sites as there: ``split_reduce``'s an
untagged map around one folding tasklet, ``join_reduce``'s and
``unfuse_map``'s a map that records what made it, ``fuse_map``'s a map that
fuses with its next sibling (filling ``second``) or its previous one (filling
``first``; ``fuse_choices`` both, the editor's drop-down), ``link``'s a transient
between two bound units (filling ``group`` with an end's, else the graph's
name), ``unlink``'s a link node or its container.

``split_reduce(map, factor)`` and ``join_reduce(map)``
    an untagged map ``v`` over ``b:e`` whose one tasklet folds (``reduce``)
    into an element that does not move with ``v`` is split as ``split_map``
    splits it: ``<map>`` temporal around a spatial ``<map>_s``; its tasklet
    now writes one partial per lane, into a new transient
    ``<output>_partial`` indexed by the output's dimensions that move with an
    enclosing map, then the spatial variable (dot's ``out_partial[n_2_s]``,
    matvec's ``q_partial[m_t, n_2_s]``), folding as before. A final sum follows
    the temporal map: tasklet ``<tasklet>_final`` in a spatial map
    ``<map>_final`` over ``0:factor``, a dimension of its own, ``<dim>_f``
    (dot's ``n_f_s``, dfg/loops.py), folding the partials into the output
    as the tasklet did. The temporal map records both
    (``loop.split_reduce``); ``join_reduce`` takes them back and joins the
    map (``join_map``). A length that is not a multiple of the factor is
    refused: the partials' padding is not placed yet (D144). A seeded fold
    (seed.py's ``fuse_seed``) keeps its seed on the final sum only, which
    takes the fold's record too; the partials start from the identity and are
    named after the transient the seed took (matvec_acc's ``tmp1_partial``),
    as split before the seed.
``fuse_map(first, second)`` and ``unfuse_map(map)``
    two adjacent nests, ``second`` right after ``first``, with the same
    ranges level by level down to one tasklet each, and the same tags, an
    untagged map taking the other side's tag (``tag_map`` on it first, so
    tagging then fusing and fusing then tagging give one graph; temporal
    against spatial is refused), joined by a transient the first writes and
    the second reads at the same index,
    become one nest of two tasklets: the first's ids, variables, dimensions
    and pieces (the second's renamed onto them), the second's other nodes after the fused
    one at their level (matvec's final sum inside the row map). The
    transient becomes a local value: ``loop.local`` (the fused map's id) on
    the container, so it gets no L1 space; nothing else may name it. The
    fused map records both nests (``loop.fused``), and ``unfuse_map`` gives
    them back when nothing changed since (unbind first). ``bind`` takes the
    fused nest's spatial map (patterns.py's ``fused_view``) and the local
    value with it: the container leaves the graph for the accelerated node's
    ``locals``, and ``unbind`` puts it back where it was.
    Different ranges at a level are refused naming the fix where one side is
    split and the other not (``FuseRangeError``, its ``fix`` the split_map
    that makes them agree: "mult_map_1 runs 0:512, sum_map_1 runs 0:128 and
    its child 0:4: split_map mult_map_1 4 first"); ``fuse_near`` gives the
    pairs that fix alone makes fuse (the viewer's near partners). ``tag_map``
    and ``untag_map`` on a map of a fused nest whose record still holds
    (``unfuse_map`` takes it) tag or untag that level on both nests and fuse
    them again (``retag``), so the record follows the tag.
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
    one name is ``unlink.group``. The merged group must be a chain in the
    graph's order (each unit's output linked to the next unit's input),
    SNAX-GVSoC's limit until tier A (fork and join): ``link.chain``, gen.chain's
    check made early.
"""

from __future__ import annotations

import copy
from itertools import pairwise, product
from typing import Any

from snax_forge import expr
from snax_forge.common import CodedError, fresh
from snax_forge.dfg import Container, Graph, Memlet, Node, Reduction, loops
from snax_forge.dfg.graph import SEEDED
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
    tag_map,
)


class FuseRangeError(SandboxError):
    """fuse_map's refusal of two nests whose ranges differ at a level because one side is split
    and the other not; ``fix`` the split that makes them agree, ``{"map", "factor"}``."""

    def __init__(self, message: str, fix: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.fix = fix


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


def _split_reduce_site(g: Graph, map: str) -> dict[str, Any]:
    """An untagged map around one tasklet that folds into an element not moving with it."""
    what = f"split_reduce {map!r}"
    body, i, _ = _locate(g, map)
    node = body[i]
    if node.kind == "map" and "loop.kind" in node.attrs:
        raise SandboxError(f"{what}: already {node.attrs['loop.kind']}")
    kids = node.body or []
    if node.kind != "map" or len(kids) != 1 or kids[0].kind != "tasklet":
        raise SandboxError(f"{what}: must be a map around one tasklet")
    t = kids[0]
    folds = [(c, m) for c, m in t.outputs.items() if m.reduce is not None]
    if len(t.outputs) != 1 or len(folds) != 1 or node.attrs["var"] in folds[0][1].names:
        raise SandboxError(f"{what}: its tasklet must fold its one output into an element that "
                           f"does not move with {node.attrs['var']!r} (a reduction, D102)")  # fmt: skip
    return {"map": map}


def split_reduce(g: Graph, map: str, factor: int) -> Graph:
    what = f"split_reduce {map!r}"
    _split_reduce_site(g, map)
    body, i, outer = _locate(g, map)
    node = body[i]
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
    red = om.reduce
    seeded = t.attrs.pop(SEEDED, None) if red is not None and red.seed is not None else None
    # a seeded fold's partials are named after the transient the seed took, as before the seed
    partial = fresh(f"{seeded['container'] if seeded else om.data}_partial", taken)
    out = g.containers[om.data]
    g.containers[partial] = Container([*(out.shape[k] for k in kept), factor], out.dtype, True)
    index = [om.subset[k] for k in kept]
    if seeded is not None:  # the partials start from the identity, the final sum from the seed
        red = Reduction(red.op, seeded["identity"])
    t.outputs[cout] = Memlet(partial, [*index, smap.attrs["var"]], red)
    dim = fresh(f"{node.attrs[loops.DIM]}_f", taken | {partial})  # its own dimension
    var = fresh(f"{dim}_s", taken)  # named by the rule below
    ids = {n.id for n, _ in g.walk()}
    final_t = Node(fresh(f"{t.id}_final", ids), "tasklet", inputs={cin: Memlet(partial, [*index, var])},
                   outputs={cout: om}, attrs={"code": t.attrs["code"],
                                              **({SEEDED: seeded} if seeded else {})})  # fmt: skip
    final = Node(fresh(f"{map}_final", ids | {final_t.id}), "map",
                 attrs={"var": var, "range": f"0:{factor}", loops.DIM: dim, loops.PIECE: 1,
                        "loop.kind": "spatial"}, body=[final_t])  # fmt: skip
    node.attrs["loop.split_reduce"] = {"partial": partial, "final": final.id}
    body.insert(i + 1, final)
    loops.rename(g)
    return _result(g)


def _join_reduce_site(g: Graph, map: str) -> dict[str, Any]:
    """A map with ``loop.split_reduce``."""
    body, i, _ = _locate(g, map)
    node = body[i]
    if node.kind != "map" or node.attrs.get("loop.split_reduce") is None:
        raise SandboxError(
            f"join_reduce {map!r}: not a map split by split_reduce (no loop.split_reduce)"
        )
    return {"map": map}


def join_reduce(g: Graph, map: str) -> Graph:
    """The inverse of split_reduce: the partials and the final sum gone, the map joined."""
    _join_reduce_site(g, map)
    g = _copy(g)
    what = f"join_reduce {map!r}"
    body, i, _ = _locate(g, map)
    node = body[i]
    rec = node.attrs["loop.split_reduce"]
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
             n.memlets())] + [n.id for n, _ in g.walk()
             if n.kind == "link" and n.attrs["container"] == rec["partial"]]  # fmt: skip
    if sorted(users) != sorted([inner[0].id, ft.id]):
        raise SandboxError(f"{what}: {rec['partial']} is named by {users} (unlink first)")
    (cout,) = inner[0].outputs
    inner[0].outputs[cout] = ft.outputs[cout]
    if SEEDED in ft.attrs:  # the seed's record back with the seed
        inner[0].attrs[SEEDED] = ft.attrs[SEEDED]
    del body[j]
    del g.containers[rec["partial"]]
    del node.attrs["loop.split_reduce"]
    return join_map(_result(g), map)


# =============================================================================
# fuse_map and unfuse_map
# =============================================================================


def _count(g: Graph, n: Node) -> int | None:
    try:
        return len(map_range(n.attrs["range"], g.bound_symbols, n.id))
    except (expr.ExprError, SandboxError, ValueError):
        return None


def _range_refusal(g: Graph, f: Node, s: Node, what: str) -> SandboxError:
    """Different ranges at one level (``f`` the first's map, ``s`` the second's): with the fix
    when one is split and the other not (its count times its one child's is the other's)."""
    text = {f.id: f"{f.id} runs {range_text(g, f.attrs['range'])}",
            s.id: f"{s.id} runs {range_text(g, s.attrs['range'])}"}  # fmt: skip
    for x, y in ((s, f), (f, s)):  # x split, y not
        kids = x.body or []
        if len(kids) != 1 or kids[0].kind != "map":
            continue
        nx, nc, ny = _count(g, x), _count(g, kids[0]), _count(g, y)
        if None in (nx, nc, ny) or nx * nc != ny:  # type: ignore[operator]
            continue
        text[x.id] += f" and its child {range_text(g, kids[0].attrs['range'])}"
        return FuseRangeError(f"{what}: {text[f.id]}, {text[s.id]}: split_map {y.id} {nc} first",
                              {"map": y.id, "factor": nc})  # fmt: skip
    return SandboxError(f"{what}: {text[f.id]}, {text[s.id]}; fused nests have the same "
                        "ranges")  # fmt: skip


def _fuse(g: Graph, first: Node, second: Node, what: str) -> tuple[Node, str, list[list[str]]]:
    """``first`` and ``second`` fused (module doc), the local container and the tags an
    untagged side takes first (``[map, kind]`` each; the fused node is then the first's
    untagged at those levels: fuse_map tags and fuses again); SandboxError."""
    rename: dict[str, str] = {}
    pairs: list[tuple[Node, Node]] = []
    f, s = first, second
    while f.kind == "map" and s.kind == "map":
        if len(f.body or []) != 1 or not s.body:
            raise SandboxError(f"{what}: {f.id} and {s.id} are not nests of the same depth down "
                               "to one tasklet each")  # fmt: skip
        rename[s.attrs["var"]] = f.attrs["var"]
        pairs.append((f, s))
        f, s = f.body[0], s.body[0]
    second = copy.deepcopy(second)
    _substitute_body([second], rename)
    f, s = first, second
    levels, tags = [], []
    for of, os_ in pairs:
        if f.attrs["range"] != s.attrs["range"]:
            raise _range_refusal(g, of, os_, what)
        kf, ks = f.attrs.get("loop.kind"), s.attrs.get("loop.kind")
        if kf != ks and kf is not None and ks is not None:
            raise SandboxError(f"{what}: {f.id} is {kf}, {s.id} {ks}; fused nests have the same "
                               "tags (an untagged map takes the other's)")  # fmt: skip
        if kf != ks:
            tags.append([of.id, ks] if kf is None else [os_.id, kf])
        levels.append((f, s.body[1:]))
        f, s = f.body[0], s.body[0]
    if f.kind != "tasklet" or s.kind != "tasklet":
        raise SandboxError(f"{what}: {f.id} and {s.id} are not nests of the same depth down to "
                           "one tasklet each")  # fmt: skip
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
             for m in n.memlets() if m.data == local}  # fmt: skip
    others = users - {t1.id, t2.id}
    if others:
        raise SandboxError(f"{what}: {local} is named by {sorted(others)} too; a local value is "
                           "the two tasklets' only")  # fmt: skip
    inner: list[Node] = [copy.deepcopy(t1), t2]
    for f, extra in reversed(levels):
        node = Node(f.id, "map", attrs=copy.deepcopy(f.attrs), body=[*inner, *extra])
        inner = [node]
    return inner[0], local, tags


def _fuse_next(g: Graph, first: str) -> dict[str, Any]:
    """``first`` fused with its next sibling, a map it fuses with (``_fuse``); ``second`` that
    map."""
    body, i, _ = _locate(g, first)
    if i + 1 >= len(body):
        raise SandboxError(f"fuse_map {first!r}: nothing comes right after it in one body")
    f, s = body[i], body[i + 1]
    what = f"fuse_map {first!r} {s.id!r}"
    if f.kind != "map" or s.kind != "map":
        raise SandboxError(f"{what}: both must be maps")
    _fuse(g, f, s, what)
    return {"first": first, "second": s.id}


def _fuse_site(g: Graph, map: str) -> dict[str, Any]:
    """A map that fuses with its next sibling (``first``) or its previous one (``second``), the
    next first; refused as the map fused with its next sibling is."""
    try:
        return _fuse_next(g, map)
    except SandboxError as e:
        why = e
    body, i, _ = _locate(g, map)
    if i > 0 and body[i - 1].kind == "map" and body[i].kind == "map":
        try:
            return _fuse_next(g, body[i - 1].id)
        except SandboxError:
            pass
    raise why


def fuse_choices(g: Graph, map: str) -> list[dict[str, str]]:
    """The pairs ``fuse_map`` takes with ``map`` on one side, next sibling first then previous,
    each ``{"first", "second"}`` in body order (the editor's drop-down)."""
    out = []
    body, i, _ = _locate(g, map)
    for k in (i, i - 1):
        if 0 <= k < len(body) and body[k].kind == "map":
            try:
                out.append(_fuse_next(g, body[k].id))
            except SandboxError:
                continue
    return [c for c in out if map in c.values()]


def _fuse_multi(g: Graph, selected: list[str]) -> dict[str, Any]:
    """One map, ``_fuse_site``'s; or two, ``first`` and ``second`` in the order selected, the
    second right after the first in one body."""
    if len(selected) == 1:
        return _fuse_site(g, selected[0])
    if len(selected) != 2:
        raise SandboxError(f"fuse_map: takes one map or two, not {len(selected)}")
    first, second = selected
    got = _fuse_next(g, first)
    if got["second"] != second:
        raise SandboxError(f"fuse_map {first!r} {second!r}: {second!r} must come right after "
                           f"{first!r} in one body (after it comes {got['second']!r})")  # fmt: skip
    return got


def fuse_map(g: Graph, first: str, second: str) -> Graph:
    what = f"fuse_map {first!r} {second!r}"
    if _fuse_next(g, first)["second"] != second:
        raise SandboxError(f"{what}: {second!r} must come right after {first!r} in one body")
    body, i, _ = _locate(g, first)
    for m, kind in _fuse(g, body[i], body[i + 1], what)[2]:  # an untagged side takes the tag
        g = tag_map(g, m, kind)
    g = _copy(g)
    body, i, _ = _locate(g, first)
    f, s = body[i], body[i + 1]
    fused, local, _ = _fuse(g, f, s, what)
    fused.attrs["loop.fused"] = {"first": f.to_dict(), "second": s.to_dict(), "local": local}
    body[i : i + 2] = [fused]
    g.containers[local].attrs["loop.local"] = fused.id
    return _result(g)


def fuse_pairs(g: Graph) -> list[tuple[str, str]]:
    """Every (first, second) ``fuse_map`` takes in ``g`` (``_fuse_next`` on each map), in the
    graph's order."""
    out: list[tuple[str, str]] = []
    for n, _ in g.walk():
        if n.kind != "map":
            continue
        try:
            out.append((n.id, _fuse_next(g, n.id)["second"]))
        except SandboxError:
            continue
    return out


def fuse_link(g: Graph, first: str, second: str) -> dict[str, Any]:
    """The transient between the two nests ``fuse_map`` fuses and the loops matched, level by
    level: ``{"container": "tmp0", "subset": ["m", "n"], "writer": "mult", "loops": [["m",
    "m", "0:8"], ["n", "n", "0:512"]]}`` (matvec_acc's mult and sum), the subset the second
    reads; SandboxError as the site."""
    if _fuse_next(g, first)["second"] != second:
        raise SandboxError(f"fuse_map {first!r} {second!r}: {second!r} must come right after "
                           f"{first!r} in one body")  # fmt: skip
    f, s = g.node(first), g.node(second)
    matched = []
    while f.kind == "map":
        matched.append([f.attrs["var"], s.attrs["var"], range_text(g, f.attrs["range"])])
        f, s = (f.body or [])[0], (s.body or [])[0]
    ((_, write),) = f.outputs.items()
    (read,) = [m for m in s.inputs.values() if m.data == write.data]
    return {"container": write.data, "subset": [str(d) for d in read.subset], "writer": f.id,
            "loops": matched}  # fmt: skip


def fuse_near(g: Graph) -> list[tuple[str, str, str]]:
    """Every (first, second, refusal) ``fuse_map`` refuses only for different ranges (a
    ``FuseRangeError``) and takes after its fix alone: the viewer's near partners."""
    out: list[tuple[str, str, str]] = []
    for n, _ in g.walk():
        if n.kind != "map":
            continue
        try:
            _fuse_next(g, n.id)
        except FuseRangeError as e:
            assert e.fix is not None
            body, i, _ = _locate(g, n.id)
            second = body[i + 1].id
            try:
                if (
                    _fuse_next(split_map(g, e.fix["map"], e.fix["factor"]), n.id)["second"]
                    == second
                ):
                    out.append((n.id, second, str(e)))
            except SandboxError:
                continue
        except SandboxError:
            continue
    return out


def _fused_level(g: Graph, map: str) -> tuple[Node, int] | None:
    """The fused nest whose chain holds ``map`` (``loop.fused``) and ``map``'s level in it."""
    for n, _ in g.walk():
        if n.kind != "map" or n.attrs.get("loop.fused") is None:
            continue
        x, k = n, 0
        while x.kind == "map":
            if x.id == map:
                return n, k
            if not x.body:
                break
            x, k = x.body[0], k + 1
    return None


def retag(g: Graph, map: str, fn: Any) -> Graph | None:
    """``fn`` (tag_map's or untag_map's) on ``map``'s level of both nests of the fused nest
    holding it, fused again (module doc); None when no fused nest holds it or its record no
    longer holds (``unfuse_map`` refuses)."""
    found = _fused_level(g, map)
    if found is None:
        return None
    fused, k = found
    rec = fused.attrs["loop.fused"]
    try:
        apart = unfuse_map(g, fused.id)
    except SandboxError:
        return None
    other = rec["second"]
    for _ in range(k):
        other = other["body"][0]
    return fuse_map(fn(fn(apart, map), other["id"]), fused.id, rec["second"]["id"])


def range_text(g: Graph, rng: str) -> str:
    """A map's range at the graph's symbols, ``0:8``; as written where they are unknown."""
    try:
        return ":".join(str(expr.evaluate(p, g.bound_symbols)) for p in rng.split(":"))
    except expr.ExprError:
        return rng


def link_text(link: dict[str, Any]) -> str:
    """``fuse_link``'s or seed.py's ``seed_link``'s as one line: ``tmp0[m, n] · m = m (0:8), n
    = n (0:512)``; a loop of one side only by its name and range."""
    loops = [f"{a} = {b} ({r})" if a and b else f"{a or b} ({r})" for a, b, r in link["loops"]]
    return f"{link['container']}[{', '.join(link['subset'])}] · {', '.join(loops)}"


def _unfuse_site(g: Graph, map: str) -> dict[str, Any]:
    """A map with ``loop.fused``."""
    body, i, _ = _locate(g, map)
    if body[i].kind != "map" or body[i].attrs.get("loop.fused") is None:
        raise SandboxError(f"unfuse_map {map!r}: not a nest fuse_map made (no loop.fused)")
    return {"map": map}


def unfuse_map(g: Graph, map: str) -> Graph:
    """The inverse of fuse_map: the two nests it fused, when nothing changed since."""
    _unfuse_site(g, map)
    g = _copy(g)
    what = f"unfuse_map {map!r}"
    body, i, _ = _locate(g, map)
    node = body[i]
    rec = node.attrs["loop.fused"]
    if rec["local"] not in g.containers:  # bind took the local value with the nest
        raise SandboxError(f"{what}: the nest changed since fuse_map (unbind first)")
    f, s = Node.from_dict(rec["first"], what), Node.from_dict(rec["second"], what)
    again, _, _ = _fuse(g, f, s, what)
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


def _ends(g: Graph, container: str) -> tuple[Node, Node]:
    """The writer and the reader of a transient two bound units share: link's site."""
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
    return w, r


def _link_site(g: Graph, container: str) -> dict[str, Any]:
    """``container`` and a ``group``: an end's group name, else the graph's, or the next free
    one (``mvt_2``) when another unit's instance or group has it (mvt's second half)."""
    w, r = _ends(g, container)
    named = w.attrs.get("group", r.attrs.get("group"))
    if named is not None:
        return {"container": container, "group": named}
    mine = {w.attrs["instance"], r.attrs["instance"]}
    others = [n for n in _accelerated(g) if n.attrs["instance"] not in mine]
    taken = {n.attrs["instance"] for n in others} | {_group(n) for n in others}
    return {"container": container, "group": fresh(g.name, taken)}


def link(g: Graph, container: str, depth: int, group: str) -> Graph:
    g = _copy(g)
    w, r = _ends(g, container)  # the site
    what = f"link {container!r}"
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
    order = list(dict.fromkeys(n.attrs["instance"] for n in nodes
                               if n.attrs["instance"] in members))  # fmt: skip
    hops = [(k.attrs["from"], k.attrs["to"]) for k, _ in g.walk()
            if k.kind == "link" and k.attrs["from"] in members]  # fmt: skip
    hops.append((w.attrs["instance"], r.attrs["instance"]))
    if sorted(hops) != sorted(pairwise(order)):
        raise LinkError("link.chain", f"{what}: the group's links {hops} over its units {order}: "
                        "a group is a chain until tier A (fork and join)")  # fmt: skip
    for n in nodes:
        if n.attrs["instance"] in members:
            n.attrs["group"] = group
    ids = {n.id for n, _ in g.walk()}
    node = Node(fresh(container, ids), "link", attrs={"container": container, "depth": depth,
                "from": w.attrs["instance"], "to": r.attrs["instance"]})  # fmt: skip
    shared.insert(ri, node)
    return _result(g)


def _unlink_site(g: Graph, selected: str) -> dict[str, Any]:
    """A link node, or its container."""
    for n, _ in g.walk():
        if n.kind == "link" and selected in (n.id, n.attrs["container"]):
            return {"container": n.attrs["container"]}
    raise SandboxError(f"unlink {selected!r}: no link of that container")


def unlink(g: Graph, container: str) -> Graph:
    """The inverse of link: the link node gone, the group taken off every unit left alone."""
    if _unlink_site(g, container)["container"] != container:  # a link node's id
        raise SandboxError(f"unlink {container!r}: no link of that container")
    g = _copy(g)
    what = f"unlink {container!r}"
    found = [n for n, _ in g.walk() if n.kind == "link" and n.attrs["container"] == container]
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


register_transform("split_reduce", split_reduce, ints=("factor",), site=_split_reduce_site)
register_transform("join_reduce", join_reduce, site=_join_reduce_site)
register_transform("fuse_map", fuse_map, site=_fuse_site, multi=_fuse_multi)
register_transform("unfuse_map", unfuse_map, site=_unfuse_site)
register_transform("link", link, ints=("depth",), site=_link_site)
register_transform("unlink", unlink, site=_unlink_site)
