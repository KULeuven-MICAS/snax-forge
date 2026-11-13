"""Baking a kernel's scalar argument into the graph, and back.

``bake_scalar(container, value)`` and ``unbake_scalar(container)``
    ``bake_scalar`` fixes how a kernel's scalar argument (``sdfg.scalar``,
    axpy's ``alpha``) reaches every tasklet reading it at a subset with no
    names (``alpha[0]``). With an int ``value`` the tasklet gets it as a
    literal in place of its connector and loses the connector, and the
    container leaves the graph: ``mult`` reads ``out = 3 * in2``, so ``bind``
    bakes ``k1 = 3`` into the hardware. With ``value`` ``"csr"`` the scalar
    is read at run time: the connector takes the container's name, so the
    code names it (``out = alpha * in2``), the container stays (data.h holds
    it, the program writes the CSR from it), and ``bind`` reads ``k1`` from
    it, as it does from a scalar no step marked. Either way the
    tasklet then has one streamed input, the same shape for both. Refused: a
    container that is not a scalar argument, one a node writes, one read at a
    subset with names, one an accelerated node reads or a bound unit's
    replaced subtree names (unbind first), one a tasklet reads as CSR already
    (unbake first), a connector name the tasklet uses already, and an int
    that does not fit its dtype. Each tasklet rewritten records the bake
    (``sandbox.baked``: the value, the container's definition and index, the
    connector, its place and memlet, and the code before and after), and
    ``unbake_scalar`` undoes it when every such tasklet is live with the code
    the bake left (otherwise unbind first, or undo what changed since).

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
        if any(r["name"] == container and r["value"] == CSR for r in n.attrs.get(BAKED, [])):
            raise SandboxError(f"{what}: {n.id} reads {container} as CSR already (unbake_scalar "
                               "first)")  # fmt: skip
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


# bake_scalar's value for a scalar read at run time, from a CSR
CSR = "csr"


def _fixes(g: Graph, container: str, value: int | str) -> dict[str, Any]:
    """The inputs the step fixes: the container at an int value (its dtype, shape [1]); none
    for ``csr``, which changes no value."""
    if value == CSR:
        return {}
    return {container: np.array([value], g.containers[container].dtype)}


def bake_scalar(g: Graph, container: str, value: int | str) -> Graph:
    from snax_forge.brm import fits

    what = f"bake_scalar {container!r}"
    _bake_site(g, container)
    dtype = g.containers[container].dtype
    if value == CSR:
        return _as_csr(g, container, what)
    if type(value) is not int:
        raise SandboxError(f"{what}: value {value!r} is neither an int nor {CSR!r}")
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


def _as_csr(g: Graph, container: str, what: str) -> Graph:
    """``bake_scalar`` at ``csr``: each reading connector renamed to the container (module doc)."""
    g = _copy(g)
    names = list(g.containers)
    record = {"container": g.containers[container].to_dict(), "index": names.index(container),
              "value": CSR}  # fmt: skip
    for n, _ in g.walk():
        for k in [k for k, m in n.inputs.items() if m.data == container]:
            if k != container and (container in n.inputs or container in n.outputs):
                raise SandboxError(f"{what}: {n.id} has a connector {container!r} already")
            before = n.attrs["code"]
            after = expr.rename_code(before, {k: container})
            n.attrs[BAKED] = [*n.attrs.get(BAKED, []), {
                **record, "name": container, "connector": k, "position": list(n.inputs).index(k),
                "memlet": n.inputs[k].to_dict(), "before": before, "after": after}]  # fmt: skip
            n.attrs["code"] = after
            n.inputs = {container if c == k else c: m for c, m in n.inputs.items()}
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
    g = _copy(g)
    found = _baked(g, container)
    csr = found[0][2]["value"] == CSR
    if container in g.containers and not csr:
        raise SandboxError(f"{what}: the graph has a container {container!r} again")
    for n, _, r in reversed(found):  # the last bake on a node first
        if n.attrs["code"] != r["after"]:
            raise SandboxError(f"{what}: {n.id}'s code changed since bake_scalar ({n.attrs['code']!r}, "
                               f"baked as {r['after']!r}): undo what changed first")  # fmt: skip
        n.attrs["code"] = r["before"]
        if r["value"] == CSR:  # the connector back to its name, where it is
            n.inputs = {r["connector"] if c == container else c: m for c, m in n.inputs.items()}
        else:
            ins = list(n.inputs.items())
            ins.insert(r["position"], (r["connector"], Memlet.from_dict(r["memlet"],
                                                                        f"{what}.{n.id}")))  # fmt: skip
            n.inputs = dict(ins)
        rest = [x for x in n.attrs[BAKED] if x is not r]
        if rest:
            n.attrs[BAKED] = rest
        else:
            del n.attrs[BAKED]
    r = found[0][2]
    if csr:  # the container never left
        return _result(g)
    items = list(g.containers.items())
    items.insert(r["index"], (container, Container.from_dict(r["container"], what)))
    g.containers = dict(items)
    return _result(g)


register_transform("bake_scalar", bake_scalar, site=_bake_site, fixes=_fixes)
register_transform("unbake_scalar", unbake_scalar, site=_unbake_site)
