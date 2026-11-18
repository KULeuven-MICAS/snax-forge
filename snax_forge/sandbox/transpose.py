"""A tasklet's input read across its loop order: marked, or copied transposed.

``mark_transpose(node, container)``, ``unmark_transpose(node, container)`` and
``data_transpose(node, container, by)``, as CONTRACTS, "Recipe" defines them. ``node`` is any map of a nest or its tasklet (``nest_tasklet``).
``loop_orders`` gives each input's π: each container dimension's index names one loop
dimension (a map's ``loop.dim``), and π orders the container's dimensions by the depth of the
outermost map of theirs (mvt's ``A[m, n]`` under the loops (n, m): π = (1, 0)). An index naming
two loop dimensions or none has no π. ``mark_transpose`` takes the inputs whose π is not the
identity (``transpose_choices``) and marks one container: ``sandbox.transpose``,
``{connector: π}``. No memlet, map or range changes, so the reference check is unchanged; the
loop transforms keep the mark, ``bind`` carries it onto the accelerated node keyed by port.
``data_transpose`` takes the same choices and instead adds a transient ``<container>_T`` and a
``transpose`` node (dfg/kinds.py) filling it, just before the nest; the nest's memlets of the
container, their dimensions permuted by π, read the copy in loop order. A packed read cannot
cross the loop order, so ``mark_transpose`` refuses an element narrower than a bank word.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from snax_forge.common import fresh
from snax_forge.dfg import Container, Graph, Memlet, Node, loops
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


def _mark_site(g: Graph, selected: str, name: str = "mark_transpose") -> dict[str, Any]:
    """A map of a nest or its tasklet reading a container across its loop order, not marked
    yet; ``container`` filled when it is the only one. ``name`` the transform asking
    (``data_transpose`` takes the same sites)."""
    what = f"{name} {selected!r}"
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


# The bank word, bytes: a packed input's lanes are contiguous elements of one word, so an
# element narrower than it cannot be read across the loop order by strides
WORD_BYTES = 8


def mark_transpose(g: Graph, node: str, container: str | None = None) -> Graph:
    what = f"mark_transpose {node!r}"
    _mark_site(g, node)
    container = _pick(transpose_choices(g, node), container, what)
    dtype = g.containers[container].dtype
    if np.dtype(dtype).itemsize < WORD_BYTES:
        raise SandboxError(f"{what}: {container} is {dtype}: a packed read cannot cross the "
                           "loop order; use data_transpose")  # fmt: skip
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


DATA_TRANSPOSE_BY = ("dma", "core")  # what makes the copy; only dma is built
NOT_BUILT = {"core": "data_transpose by core is not built yet; use by=dma"}


def _data_site(g: Graph, selected: str) -> dict[str, Any]:
    """mark_transpose's sites: a map of a nest or its tasklet reading a container across its
    loop order; ``by`` dma."""
    return _mark_site(g, selected, "data_transpose") | {"by": "dma"}


def _top(g: Graph, node: str) -> Node:
    """The top-level node holding ``node``: its nest's outermost map."""
    _, i, outer = _locate(g, node)
    return outer[0] if outer else _locate(g, node)[0][i]


def _nest(n: Node) -> list[Node]:
    out = [n]
    for k in n.body or []:
        out += _nest(k)
    return out


def data_transpose(g: Graph, node: str, container: str | None = None, by: str = "dma") -> Graph:
    """``container``, read across the loop order in ``node``'s nest, copied transposed into a
    transient ``<container>_T`` by a ``transpose`` node just before the nest, which reads the
    copy in loop order (module doc)."""
    what = f"data_transpose {node!r}"
    if by in NOT_BUILT:
        raise SandboxError(NOT_BUILT[by])
    if by not in DATA_TRANSPOSE_BY:
        raise SandboxError(f"{what}: by {by!r} is not one of {', '.join(DATA_TRANSPOSE_BY)}")
    _data_site(g, node)
    found = transpose_choices(g, node)
    container = _pick(found, container, what)
    perm = next(x["transpose"] for x in found if x["container"] == container)
    name = f"{container}_T"
    if name in g.containers:
        raise SandboxError(f"{what}: a container {name} exists already; rename it, or undo the "
                           f"data_transpose that made it")  # fmt: skip
    t = nest_tasklet(g, node, what)
    orders = {c: p for c, data, p in loop_orders(g, t.id) if data == container}
    if any(p != perm for p in orders.values()) or container in {m.data for m in t.outputs.values()}:
        raise SandboxError(f"{what}: {t.id} reads or writes {container} in more than one order; "
                           "mark_transpose marks one connector")  # fmt: skip
    g = _copy(g)
    top = _top(g, node)
    for n in _nest(top):
        for m in n.memlets():
            if m.data == container:
                m.data, m.subset = name, [m.subset[p] for p in perm]
    src = g.containers[container]
    copy = Container([src.shape[p] for p in perm], src.dtype, transient=True)
    order = list(g.containers)
    g.containers = {k: g.containers[k] if k != name else copy
                    for k in [*order[: order.index(container) + 1], name,
                              *order[order.index(container) + 1:]]}  # fmt: skip
    taken = {n.id for n, _ in g.walk()}
    node_t = Node(fresh(f"transpose_{container}", taken), "transpose",
                  {"src": Memlet(container, [f"0:{s}" for s in src.shape])},
                  {"dst": Memlet(name, [f"0:{s}" for s in copy.shape])},
                  {"perm": list(perm), "by": by})  # fmt: skip
    g.body.insert(g.body.index(top), node_t)
    return _result(g)


register_transform("mark_transpose", mark_transpose, site=_mark_site)
register_transform("data_transpose", data_transpose, site=_data_site)
register_transform("unmark_transpose", unmark_transpose, site=_unmark_site)
