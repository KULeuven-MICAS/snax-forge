"""One read of a container given to several bound units: ``fork_read`` and ``unfork_read``.

``fork_read(container, group)`` takes the bound units that read ``container`` (each a chain's
head, on one port each) when they read the same beats in the same order (compose.py's
``beats``, as ``link`` checks its two ends), puts their groups into one group ``group`` and
adds a fork node recording the readers, ``"<instance>.<port>"`` in the graph's order, and
every member's group before. One streamer then reads the container for all of them
(design/streamers.py's ``forked_ports``). No memlet changes, so the reference check is
unchanged. ``unfork_read(container)`` takes the fork node off and puts the groups back. The
sites take the container's name, or a reader's node; a refusal is a ``ForkError``
(``fork.<rule>``) naming its fix (``interchange_map`` when the beats come in another order).
"""

from __future__ import annotations

from typing import Any

from snax_forge.brm import SEED
from snax_forge.common import CodedError, fresh
from snax_forge.design.streamers import group_of
from snax_forge.dfg import Graph, Node

from .compose import _path, beats, chains_problem, fork_sets
from .recipe import SandboxError
from .share import SHARED
from .transforms import _accelerated, _copy, _locate, _result, register_transform


class ForkError(CodedError, SandboxError):
    """A fork the sandbox refuses; ``code`` names the rule (``fork.<rule>``)."""


def _readers(g: Graph, container: str, what: str) -> list[tuple[Node, str]]:
    """The bound units reading ``container`` and the port each reads it on, in the graph's
    order: fork_read's site."""
    if container not in g.containers:
        raise ForkError("fork.container", f"{what}: no container of that name")
    if any(n.kind == "fork" and n.attrs["container"] == container for n, _ in g.walk()):
        raise ForkError("fork.container", f"{what}: already forked")
    found = []
    for n, _ in g.walk():
        ports = [p for p, m in n.inputs.items() if m.data == container]
        if not ports:
            continue
        if n.kind != "accelerated":
            raise ForkError("fork.unbound", f"{what}: {n.id} ({n.kind}) reads it; every reader "
                            "of a forked read is a bound unit (bind it first)")  # fmt: skip
        if len(ports) > 1 or ports[0] == SEED:
            raise ForkError("fork.port", f"{what}: {n.id} reads it on {ports}; a fork gives "
                            "one input port of each unit, not a seed")  # fmt: skip
        found.append((n, ports[0]))
    if len({n.attrs["instance"] for n, _ in found}) < 2:
        raise ForkError("fork.container", f"{what}: read by {[n.id for n, _ in found]}; a fork "
                        "gives one read to two or more units")  # fmt: skip
    if len(found) != len({n.attrs["instance"] for n, _ in found}):
        raise ForkError("fork.shared", f"{what}: a unit reads it in two nests (share); a "
                        "forked read is one nest's per unit")  # fmt: skip
    return found


def _check(g: Graph, container: str, group: str, what: str) -> list[tuple[Node, str]]:
    found = _readers(g, container, what)
    first, port = found[0]
    want = beats(g, first, port)
    for n, p in found[1:]:
        got = beats(g, n, p)
        if got == want:
            continue
        same = sorted(got) == sorted(want)
        raise ForkError("fork.order", f"{what}: {first.id}.{port} reads {len(want)} beats of "
                        f"{len(want[0]) if want else 0} elements, {n.id}.{p} {len(got)} of "
                        f"{len(got[0]) if got else 0}"
                        + ("; the same beats in another order: interchange_map its loops first"
                           if same else "; other elements: a fork gives the same beats to every "
                           "unit"))  # fmt: skip
    nodes = _accelerated(g)
    linked_to = {n.attrs["to"] for n, _ in g.walk() if n.kind == "link"}
    for n, _ in found:
        if n.attrs["instance"] in linked_to:
            raise ForkError("fork.head", f"{what}: {n.attrs['instance']} is fed by a link; a "
                            "fork feeds the heads of chains")  # fmt: skip
    readers = {n.attrs["instance"] for n, _ in found}
    merged = {group_of(n) for n in nodes if n.attrs["instance"] in readers}
    members = {n.attrs["instance"] for n in nodes if group_of(n) in merged}
    if any(SHARED in n.attrs for n in nodes if n.attrs["instance"] in members):
        raise ForkError("fork.shared", f"{what}: a unit of its groups is shared (share); "
                        "unshare it first")  # fmt: skip
    if not isinstance(group, str) or not group.isidentifier():
        raise ForkError("fork.group", f"{what}: group {group!r} is not a name")
    others = {n.attrs["instance"] for n in nodes} - members
    if group in others or group in {group_of(n) for n in nodes
                                    if n.attrs["instance"] in others}:  # fmt: skip
        raise ForkError("fork.group", f"{what}: {group!r} names another instance or group")
    accs = {n.attrs.get("accelerator") for n in nodes if n.attrs["instance"] in members}
    if len(accs) > 1:
        raise ForkError("fork.group", f"{what}: its units are in accelerators "
                        f"{sorted(map(str, accs))}")  # fmt: skip
    order = list(dict.fromkeys(n.attrs["instance"] for n in nodes
                               if n.attrs["instance"] in members))  # fmt: skip
    hops = [(k.attrs["from"], k.attrs["to"]) for k, _ in g.walk()
            if k.kind == "link" and k.attrs["from"] in members]  # fmt: skip
    problem = chains_problem(order, hops, [*fork_sets(g, members), readers])
    if problem is not None:
        raise ForkError("fork.chain", f"{what}: the group's links {hops} over its units {order}: "
                        f"{problem}")  # fmt: skip
    return found


def _fork_read_site(g: Graph, selected: str) -> dict[str, Any]:
    """A container, or a bound node reading one another bound unit reads too, and a
    ``group``: the graph's name, made free of the other groups' and instances' names."""
    container = selected
    n = next((n for n, _ in g.walk() if n.id == selected), None)
    if selected not in g.containers and n is not None and n.kind == "accelerated":
        reads = [m.data for p, m in n.inputs.items() if p != SEED]
        shared = [c for c in reads
                  if sum(any(m.data == c for m in x.inputs.values())
                         for x in _accelerated(g)) > 1]  # fmt: skip
        container = shared[0] if shared else selected
        try:
            _readers(g, container, f"fork_read {container!r}")
        except ForkError:
            container = selected  # refused as the transform is, given the node's id
    found = _readers(g, container, f"fork_read {container!r}")
    readers = {n.attrs["instance"] for n, _ in found}
    nodes = _accelerated(g)
    merged = {group_of(n) for n in nodes if n.attrs["instance"] in readers}
    others = [n for n in nodes if group_of(n) not in merged]
    taken = {n.attrs["instance"] for n in others} | {group_of(n) for n in others}
    return {"container": container, "group": fresh(g.name, taken)}


def fork_read(g: Graph, container: str, group: str) -> Graph:
    what = f"fork_read {container!r}"
    found = _check(g, container, group, what)
    g = _copy(g)
    found = _readers(g, container, what)
    nodes = _accelerated(g)
    readers = {n.attrs["instance"] for n, _ in found}
    merged = {group_of(n) for n in nodes if n.attrs["instance"] in readers}
    members = [n for n in nodes if group_of(n) in merged]
    before = {n.attrs["instance"]: n.attrs.get("group") for n in members}
    for n in members:
        n.attrs["group"] = group
    paths = [_path(g.body, n.id) for n, _ in found]
    assert all(paths)
    k = 0
    while all(len(p) > k + 1 for p in paths) and all(
        p[k][0] is paths[0][k][0] and p[k][1] == paths[0][k][1] for p in paths
    ):
        k += 1
    body, at = paths[-1][k]
    ids = {n.id for n, _ in g.walk()}
    node = Node(fresh(f"{container}_fork", ids), "fork",
                attrs={"container": container,
                       "readers": [f"{n.attrs['instance']}.{p}" for n, p in found],
                       "groups": before})  # fmt: skip
    body.insert(at, node)
    return _result(g)


def _unfork_site(g: Graph, selected: str) -> dict[str, Any]:
    """A fork node, or its container."""
    for n, _ in g.walk():
        if n.kind == "fork" and selected in (n.id, n.attrs["container"]):
            return {"container": n.attrs["container"]}
    raise ForkError("fork.container", f"unfork_read {selected!r}: no fork of that container")


def unfork_read(g: Graph, container: str) -> Graph:
    """The inverse of fork_read: the fork node gone, each unit's group as it was, when the
    group's units are still those it made."""
    if _unfork_site(g, container)["container"] != container:  # a fork node's id
        raise ForkError("fork.container", f"unfork_read {container!r}: no fork of that container")
    g = _copy(g)
    what = f"unfork_read {container!r}"
    (node,) = [n for n, _ in g.walk() if n.kind == "fork" and n.attrs["container"] == container]
    before = node.attrs["groups"]
    nodes = _accelerated(g)
    group = next(group_of(n) for n in nodes
                 if n.attrs["instance"] == node.attrs["readers"][0].split(".")[0])  # fmt: skip
    members = {n.attrs["instance"] for n in nodes if group_of(n) == group}
    if members != set(before):
        raise ForkError("fork.changed", f"{what}: group {group!r} has the units "
                        f"{sorted(members)}, the fork made {sorted(before)}; unlink the others "
                        "first")  # fmt: skip
    if any(k is not node and k.kind == "fork"
           and {r.split(".")[0] for r in k.attrs["readers"]} <= members
           for k, _ in g.walk()):  # fmt: skip
        raise ForkError("fork.changed", f"{what}: another fork joins group {group!r}; unfork it "
                        "first")  # fmt: skip
    for n in nodes:
        if n.attrs["instance"] in before:
            if before[n.attrs["instance"]] is None:
                n.attrs.pop("group", None)
            else:
                n.attrs["group"] = before[n.attrs["instance"]]
    body, i, _ = _locate(g, node.id)
    body.pop(i)
    return _result(g)


register_transform("fork_read", fork_read, site=_fork_read_site)
register_transform("unfork_read", unfork_read, site=_unfork_site)
