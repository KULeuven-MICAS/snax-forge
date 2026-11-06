"""Baking a kernel's scalar argument into the graph, and back (C§12).

``bake_scalar(container, value)`` and ``unbake_scalar(container)``
    ``bake_scalar`` replaces a kernel's scalar argument (``sdfg.scalar``,
    axpy's ``alpha``) by a value: every tasklet reading it, at a subset with
    no names (``alpha[0]``), gets the value as an int literal in place of its
    connector, and loses the connector; the container leaves the graph. So
    ``bind`` sees ``3 * x + y`` and bakes ``k1 = 3`` into the hardware, where
    without the step it reads ``k1`` from ``alpha`` (a CSR). Refused: a
    container that is not a scalar argument, one a node writes, one read at a
    subset with names, one an accelerated node reads or a bound unit's
    replaced subtree names (unbind first), and a value that does not fit its
    dtype. Each tasklet rewritten records the bake (``sandbox.baked``: the
    container's definition and index, the connector, its place and memlet, and the code
    before and after), and ``unbake_scalar`` puts the connector and the
    container back when every such tasklet is live with the code the bake
    left (otherwise unbind first, or undo what changed since).

The step's reference check (run.py's ``checked_step``, through the
transform's ``fixes``) runs the graph before the step with the container set
to the value and the graph after it on the same other seeded inputs; the
value then stays in the inputs for the steps after it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from snax_forge import expr
from snax_forge.dfg import Container, Graph, Memlet, Node
from snax_forge.dfg.graph import BAKED, SCALAR

from .recipe import SandboxError
from .transforms import _copy, _result, register_transform


def _bake_site(g: Graph, container: str) -> dict[str, Any]:
    """A kernel's scalar argument no node writes, read by tasklets only, at no names."""
    what = f"bake_scalar {container!r}"
    c = g.containers.get(container)
    if c is None:
        raise SandboxError(f"{what}: the graph has no container {container!r}")
    if not c.attrs.get(SCALAR):
        raise SandboxError(f"{what}: {container} is not a kernel's scalar argument (sdfg.scalar)")
    for n, _ in g.walk():
        if any(m.data == container for m in n.outputs.values()):
            raise SandboxError(f"{what}: {n.id} writes {container}; a baked value is never written")
        reads = [(k, m) for k, m in n.inputs.items() if m.data == container]
        if reads and n.kind != "tasklet":
            raise SandboxError(f"{what}: the {n.kind} {n.id} reads {container} (unbind first)")
        for k, m in reads:
            if m.names:
                raise SandboxError(f"{what}: {n.id}.{k} reads {container}{m.subset}, at names; a "
                                   "baked value is one element")  # fmt: skip
        if n.kind == "accelerated" and _names_in(n.attrs.get("replaced"), container):
            raise SandboxError(f"{what}: the bound unit {n.id} replaced a node reading "
                               f"{container} (unbind first)")  # fmt: skip
    return {"container": container}


def _names_in(d: Any, container: str) -> bool:
    """Whether a node's dict (a ``replaced`` subtree) has a memlet on ``container``."""
    if isinstance(d, dict):
        if d.get("data") == container and "subset" in d:
            return True
        return any(_names_in(v, container) for v in d.values())
    if isinstance(d, list):
        return any(_names_in(v, container) for v in d)
    return False


def _fixes(g: Graph, container: str, value: int) -> dict[str, Any]:
    """The inputs the step fixes: the container at the value (its dtype, shape [1])."""
    return {container: np.array([value], g.containers[container].dtype)}


def bake_scalar(g: Graph, container: str, value: int) -> Graph:
    from snax_forge.brm import fits

    what = f"bake_scalar {container!r}"
    _bake_site(g, container)
    dtype = g.containers[container].dtype
    if type(value) is not int:
        raise SandboxError(f"{what}: value {value!r} is not an int")
    if not fits(value, dtype):
        raise SandboxError(f"{what}: {value} does not fit {container}'s dtype {dtype}")
    g = _copy(g)
    names = list(g.containers)
    record = {"container": g.containers[container].to_dict(), "index": names.index(container),
              "value": value}  # fmt: skip
    for n, _ in g.walk():
        reads = [k for k, m in n.inputs.items() if m.data == container]
        for k in reads:
            before = n.attrs["code"]
            lines = expr.statements(before, what)
            after = "\n".join(f"{t} = {expr.substitute(e, {k: value})}" for t, e in lines)
            n.attrs[BAKED] = [*n.attrs.get(BAKED, []), {
                **record, "name": container, "connector": k, "position": list(n.inputs).index(k),
                "memlet": n.inputs[k].to_dict(), "before": before, "after": after}]  # fmt: skip
            n.attrs["code"] = after
            del n.inputs[k]
    del g.containers[container]
    return _result(g)


def _unbake_site(g: Graph, selected: str) -> dict[str, Any]:
    """A baked container's name, or a live tasklet that records a bake (the editor's selection:
    the container is gone), its last bake's container."""
    if _baked(g, selected):
        return {"container": selected}
    node = next((n for n, _ in g.walk() if n.id == selected), None)
    if node is not None and node.attrs.get(BAKED):
        return {"container": node.attrs[BAKED][-1]["name"]}
    raise SandboxError(f"unbake_scalar {selected!r}: no live tasklet records {selected} as baked "
                       "(bake_scalar; a bound unit's: unbind first), nor is it a tasklet that "
                       "records a bake")  # fmt: skip


def _baked(g: Graph, container: str) -> list[tuple[Node, int, dict[str, Any]]]:
    """Each live tasklet's record of baking ``container``, with its place in the node's list."""
    return [(n, i, r) for n, _ in g.walk() for i, r in enumerate(n.attrs.get(BAKED, []))
            if r["name"] == container]  # fmt: skip


def unbake_scalar(g: Graph, container: str) -> Graph:
    """The inverse of bake_scalar, when nothing changed since."""
    what = f"unbake_scalar {container!r}"
    named = _unbake_site(g, container)["container"]
    if named != container:
        raise SandboxError(f"{what}: {container} is a tasklet; give the container it baked, "
                           f"{named}")  # fmt: skip
    if container in g.containers:
        raise SandboxError(f"{what}: the graph has a container {container!r} again")
    g = _copy(g)
    found = _baked(g, container)
    for n, _, r in reversed(found):  # the last bake on a node first
        if n.attrs["code"] != r["after"]:
            raise SandboxError(f"{what}: {n.id}'s code changed since bake_scalar ({n.attrs['code']!r}, "
                               f"baked as {r['after']!r}): undo what changed first")  # fmt: skip
        n.attrs["code"] = r["before"]
        ins = list(n.inputs.items())
        ins.insert(r["position"], (r["connector"], Memlet.from_dict(r["memlet"], f"{what}.{n.id}")))
        n.inputs = dict(ins)
        rest = [x for x in n.attrs[BAKED] if x is not r]
        if rest:
            n.attrs[BAKED] = rest
        else:
            del n.attrs[BAKED]
    r = found[0][2]
    items = list(g.containers.items())
    items.insert(r["index"], (container, Container.from_dict(r["container"], what)))
    g.containers = dict(items)
    return _result(g)


register_transform("bake_scalar", bake_scalar, site=_bake_site, fixes=_fixes)
register_transform("unbake_scalar", unbake_scalar, site=_unbake_site)
