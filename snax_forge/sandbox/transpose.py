"""The transpose marker: a tasklet's input read across its loop order, marked as such.

``mark_transpose(node, container)`` and ``unmark_transpose(node, container)``
    ``node`` is any map of a nest (the chain of single-child maps down to one tasklet) or its
    tasklet. For each input memlet of that tasklet, ``loop_orders`` gives π: each container
    dimension's index names exactly one loop dimension (a map's ``loop.dim``, so the
    pieces of a split dimension and a tag keep it), and π orders the container's dimensions by
    the depth of the outermost map of their loop dimension. mvt's ``A[m, n]`` under the loops
    (n, m) gives π = (1, 0), under (m, n) (0, 1). An index naming two loop dimensions or none
    (a constant, a scalar's ``alpha[0]``) has no π, and its input is no candidate.
    ``mark_transpose`` takes the inputs whose π is not the identity: with one container among
    them it marks it (``container`` filled), with several ``container`` names one (the
    editor's choice), with none it refuses ("tmp0_mult reads A, y1 in loop order"). The mark
    is the tasklet's attr ``sandbox.transpose``, ``{connector: π}`` for each of that
    container's connectors; no memlet, map or range changes, so the reference check is
    unchanged. split, split_map, tag_map, untag_map, interchange_map and join_map leave the
    tasklet's attrs as they are and keep it; ``bind`` carries it onto the accelerated node,
    keyed by the connector's port, and ``unbind`` gives back the tasklet that holds it.
    ``unmark_transpose`` removes a container's mark, the attr with its last: mark then unmark
    is the graph before. Refused, both: a selection that is no map of a nest nor its tasklet,
    a bound unit (unbind first); ``mark_transpose`` a container marked already
    (``unmark_transpose`` first) or not a candidate, ``unmark_transpose`` a tasklet that marks
    none.

The viewer (viz/dfg/api.py) draws a marked edge with its label (``Aᵀ``).
"""

from __future__ import annotations

from typing import Any

from snax_forge.dfg import Graph, Node, loops
from snax_forge.dfg.graph import TRANSPOSE
from snax_forge.dfg.subset import dim_names

from .recipe import SandboxError
from .transforms import _copy, _locate, _result, register_transform


def nest_tasklet(g: Graph, selected: str, what: str) -> Node:
    """The tasklet of the nest ``selected`` is a map of (down its single-child maps), or
    ``selected`` itself when it is a tasklet."""
    body, i, _ = _locate(g, selected)
    n = body[i]
    while n.kind == "map":
        if len(n.body) != 1:
            raise SandboxError(f"{what}: {n.id} holds {len(n.body)} nodes, not one nest down to "
                               "a tasklet")  # fmt: skip
        n = n.body[0]
    if n.kind == "accelerated":
        raise SandboxError(f"{what}: the bound unit {n.id} (unbind first)")
    if n.kind != "tasklet":
        raise SandboxError(f"{what}: a {n.kind!r} is neither a map of a nest nor its tasklet")
    return n


def loop_orders(g: Graph, tasklet: str) -> list[tuple[str, str, list[int] | None]]:
    """Each input connector of ``tasklet``, its container and π (module doc), or None when an
    index names two loop dimensions or none."""
    _, _, around = _locate(g, tasklet)
    maps = [m for m in around if m.kind == "map"]
    dim = {m.attrs["var"]: m.attrs.get(loops.DIM, m.attrs["var"]) for m in maps}
    depth: dict[str, int] = {}
    for k, m in enumerate(maps):
        depth.setdefault(dim[m.attrs["var"]], k)
    out: list[tuple[str, str, list[int] | None]] = []
    for c, m in g.node(tasklet).inputs.items():
        named = [{dim[v] for v in dim_names(d) if v in dim} for d in m.subset]
        if not named or any(len(x) != 1 for x in named):
            out.append((c, m.data, None))
            continue
        ds = [x.pop() for x in named]
        out.append((c, m.data, sorted(range(len(ds)), key=lambda i: depth[ds[i]])))
    return out


def _transposed(g: Graph, t: Node) -> list[tuple[str, str, list[int]]]:
    """The inputs of ``t`` whose π is not the identity: (connector, container, π)."""
    return [(c, data, p) for c, data, p in loop_orders(g, t.id)
            if p is not None and p != sorted(p)]  # fmt: skip


def transpose_choices(g: Graph, node: str, mark: bool = True) -> list[dict[str, Any]]:
    """The containers ``mark_transpose`` (``mark``) or ``unmark_transpose`` takes at ``node``,
    each ``{"container", "transpose"}`` (its first connector's π), in the tasklet's input
    order (the editor's choice)."""
    what = f"{'mark' if mark else 'unmark'}_transpose {node!r}"
    t = nest_tasklet(g, node, what)
    marked = t.attrs.get(TRANSPOSE, {})
    if mark:
        found = [(data, p) for c, data, p in _transposed(g, t) if c not in marked]
    else:
        found = [(t.inputs[c].data, p) for c, p in marked.items() if c in t.inputs]
    out: dict[str, list[int]] = {}
    for data, p in found:
        out.setdefault(data, p)
    return [{"container": c, "transpose": p} for c, p in out.items()]


def _mark_site(g: Graph, selected: str) -> dict[str, Any]:
    """A map of a nest or its tasklet reading a container across its loop order, not marked
    yet; ``container`` filled when it is the only one."""
    what = f"mark_transpose {selected!r}"
    t = nest_tasklet(g, selected, what)
    found = transpose_choices(g, selected)
    if not found:
        marked = t.attrs.get(TRANSPOSE, {})
        already = list(dict.fromkeys(t.inputs[c].data for c in marked if c in t.inputs))
        if already:
            raise SandboxError(f"{what}: {t.id} marks {', '.join(already)} already "
                               "(unmark_transpose first)")  # fmt: skip
        read = list(dict.fromkeys(data for _, data, p in loop_orders(g, t.id) if p is not None))
        raise SandboxError(f"{what}: {t.id} reads {', '.join(read) or 'no array'} in loop order")
    return {"node": selected} | ({"container": found[0]["container"]} if len(found) == 1 else {})


def _pick(found: list[dict[str, Any]], container: str | None, what: str) -> str:
    names = [x["container"] for x in found]
    if container is None:
        if len(names) > 1:
            raise SandboxError(f"{what}: give container, one of {', '.join(names)}")
        return names[0]
    if container not in names:
        raise SandboxError(f"{what}: {container} is not one of {', '.join(names)}")
    return container


def mark_transpose(g: Graph, node: str, container: str | None = None) -> Graph:
    what = f"mark_transpose {node!r}"
    _mark_site(g, node)
    container = _pick(transpose_choices(g, node), container, what)
    g = _copy(g)
    t = nest_tasklet(g, node, what)
    marks = dict(t.attrs.get(TRANSPOSE, {}))
    for c, data, p in _transposed(g, t):
        if data == container and c not in marks:
            marks[c] = p
    t.attrs[TRANSPOSE] = marks
    return _result(g)


def _unmark_site(g: Graph, selected: str) -> dict[str, Any]:
    """A map of a nest or its tasklet that marks a transpose; ``container`` filled when it
    marks one."""
    what = f"unmark_transpose {selected!r}"
    t = nest_tasklet(g, selected, what)
    found = transpose_choices(g, selected, mark=False)
    if not found:
        raise SandboxError(f"{what}: {t.id} marks no transpose (mark_transpose)")
    return {"node": selected} | ({"container": found[0]["container"]} if len(found) == 1 else {})


def unmark_transpose(g: Graph, node: str, container: str | None = None) -> Graph:
    """The inverse of mark_transpose: ``container``'s mark gone, the attr with its last."""
    what = f"unmark_transpose {node!r}"
    _unmark_site(g, node)
    container = _pick(transpose_choices(g, node, mark=False), container, what)
    g = _copy(g)
    t = nest_tasklet(g, node, what)
    marks = {c: p for c, p in t.attrs[TRANSPOSE].items()
             if c not in t.inputs or t.inputs[c].data != container}  # fmt: skip
    if marks:
        t.attrs[TRANSPOSE] = marks
    else:
        del t.attrs[TRANSPOSE]
    return _result(g)


register_transform("mark_transpose", mark_transpose, site=_mark_site)
register_transform("unmark_transpose", unmark_transpose, site=_unmark_site)
