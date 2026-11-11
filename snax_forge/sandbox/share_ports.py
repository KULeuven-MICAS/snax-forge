"""Shared movers (D169, PORT1): two accelerators that hand data over through L1 on one streamer.

``share_ports(container, accelerator)`` and ``unshare_ports(container)``
    ``container`` is written by one group's units and read by another's, both bound, through
    L1 (no link on it). ``share_ports`` puts both groups on one accelerator, as ``encapsulate``
    does (D148): every node of each carries ``accelerator``; then marks the container
    ``hw.share_ports``, so the accelerator's groups share its streamer's movers (D169,
    design/streamers.py's ``shared_movers``): the crossbar sees the movers' channels, not every
    port's. A group already in an accelerator keeps it, and the other joins it: ``accelerator``
    must then be its name. The site fills ``accelerator`` with that name, else the next free one
    after ``<writer's group>_<reader's group>`` (dot 1's ``vmul_vsum``). The marker records the
    units that got ``accelerator`` here (``{"accelerator", "added"}``), which ``unshare_ports``
    takes off again with the marker: ``share_ports`` then ``unshare_ports`` is the graph
    before. No memlet changes, so the reference check is unchanged. Refused (``share.site``),
    naming what it found: no container of that name, a container on a link (a link already puts
    both units in one group), one already shared, writers or readers that are not bound or not
    one group each, one group writing and reading it (nothing to share), a writer's group that
    does not run before the reader's, the two in different accelerators, and an
    ``accelerator`` that is no name, another instance's or group's, or another accelerator's
    than the one a group is in.

Their sites take a container's name; ``unshare_ports``'s a marked one.
"""

from __future__ import annotations

from typing import Any

from snax_forge.common import CodedError, fresh
from snax_forge.design.streamers import SHARE_PORTS, accelerator_of, group_of
from snax_forge.dfg import Graph, Node

from .recipe import SandboxError
from .transforms import _accelerated, _copy, _result, register_transform


class ShareSiteError(CodedError, SandboxError):
    """Where ``share_ports`` does not apply."""


def _ends(g: Graph, container: str) -> tuple[list[Node], list[Node]]:
    """The writers and the readers of a container two groups share: share_ports' site."""
    what = f"share_ports {container!r}"
    if container not in g.containers:
        raise ShareSiteError("share.site", f"{what}: no container of that name")
    if SHARE_PORTS in g.containers[container].attrs:
        raise ShareSiteError("share.site", f"{what}: already shared")
    if any(n.kind == "link" and n.attrs["container"] == container for n, _ in g.walk()):
        raise ShareSiteError("share.site", f"{what}: a link (its units are one group already)")
    writers = [n for n, _ in g.walk() if any(m.data == container for m in n.outputs.values())]
    readers = [n for n, _ in g.walk() if any(m.data == container for m in n.inputs.values())]
    for side, nodes in (("written", writers), ("read", readers)):
        if not nodes or any(n.kind != "accelerated" for n in nodes):
            raise ShareSiteError("share.site", f"{what}: {side} by {[n.id for n in nodes]}; both "
                                 "ends are bound units (bind them first)")  # fmt: skip
        if len({group_of(n) for n in nodes}) > 1:
            raise ShareSiteError("share.site", f"{what}: {side} by the groups "
                                 f"{sorted({group_of(n) for n in nodes})}; one group each")  # fmt: skip
    wg, rg = group_of(writers[0]), group_of(readers[0])
    if wg == rg:
        raise ShareSiteError("share.site", f"{what}: written and read by group {wg!r}; nothing "
                             "to share")  # fmt: skip
    order = list(dict.fromkeys(group_of(n) for n in _accelerated(g)))
    if order.index(wg) > order.index(rg):
        raise ShareSiteError("share.site", f"{what}: group {wg!r} writes it but runs after "
                             f"{rg!r}, which reads it")  # fmt: skip
    accs = {n.attrs["accelerator"] for n in [*writers, *readers] if "accelerator" in n.attrs}
    if len(accs) > 1:
        raise ShareSiteError("share.site", f"{what}: its groups are in the accelerators "
                             f"{sorted(accs)}; one of them unencapsulated first")  # fmt: skip
    return writers, readers


def _members(g: Graph, groups: set[str]) -> list[Node]:
    return [n for n in _accelerated(g) if group_of(n) in groups]


def _share_ports_site(g: Graph, container: str) -> dict[str, Any]:
    """``container`` and an ``accelerator``: a group's own, else the next free name after
    ``<writer's group>_<reader's group>``."""
    writers, readers = _ends(g, container)
    groups = {group_of(writers[0]), group_of(readers[0])}
    named = {n.attrs["accelerator"] for n in _members(g, groups) if "accelerator" in n.attrs}
    if named:
        return {"container": container, "accelerator": named.pop()}
    nodes = _accelerated(g)
    taken = ({n.attrs["instance"] for n in nodes} | {group_of(n) for n in nodes}
             | {accelerator_of(n) for n in nodes})  # fmt: skip
    return {"container": container,
            "accelerator": fresh(f"{group_of(writers[0])}_{group_of(readers[0])}", taken)}  # fmt: skip


def share_ports(g: Graph, container: str, accelerator: str) -> Graph:
    """Both groups on one accelerator whose movers they share (module doc)."""
    writers, readers = _ends(g, container)  # the site
    what = f"share_ports {container!r}"
    if not isinstance(accelerator, str) or not accelerator.isidentifier():
        raise ShareSiteError("share.site", f"{what}: the accelerator must be a name")
    groups = {group_of(writers[0]), group_of(readers[0])}
    g = _copy(g)
    mine = _members(g, groups)
    nodes = _accelerated(g)
    named = {n.attrs["accelerator"] for n in mine if "accelerator" in n.attrs}
    if named and accelerator not in named:
        raise ShareSiteError("share.site", f"{what}: a group is in accelerator {named.pop()!r} "
                             f"already, not {accelerator!r}")  # fmt: skip
    others = [n for n in nodes if group_of(n) not in groups]
    if not named and (accelerator in {n.attrs["instance"] for n in nodes}
                      or accelerator in {group_of(n) for n in nodes}
                      or accelerator in {accelerator_of(n) for n in others}):  # fmt: skip
        raise ShareSiteError("share.site", f"{what}: {accelerator!r} names another instance, "
                             "group or accelerator")  # fmt: skip
    added = list(dict.fromkeys(n.attrs["instance"] for n in mine if "accelerator" not in n.attrs))
    for n in mine:
        n.attrs["accelerator"] = accelerator
    g.containers[container].attrs[SHARE_PORTS] = {"accelerator": accelerator, "added": added}
    return _result(g)


def _unshare_ports_site(g: Graph, selected: str) -> dict[str, Any]:
    """A shared container."""
    c = g.containers.get(selected)
    if c is None or SHARE_PORTS not in c.attrs:
        raise ShareSiteError("share.site", f"unshare_ports {selected!r}: no shared container of "
                             "that name")  # fmt: skip
    return {"container": selected}


def unshare_ports(g: Graph, container: str) -> Graph:
    """The inverse of share_ports: the marker off, and ``accelerator`` off the units it added."""
    _unshare_ports_site(g, container)
    g = _copy(g)
    mark = g.containers[container].attrs.pop(SHARE_PORTS)
    added = set(mark.get("added", []))
    for n in _accelerated(g):
        if n.attrs["instance"] in added:
            n.attrs.pop("accelerator", None)
    return _result(g)


register_transform("share_ports", share_ports, site=_share_ports_site)
register_transform("unshare_ports", unshare_ports, site=_unshare_ports_site)
