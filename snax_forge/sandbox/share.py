"""Sharing a group's hardware: a group run on another's units, one accelerator run
once per nest.

``share(group, with)`` and ``unshare(group)``, as CONTRACTS, "Recipe" defines them. Two groups
of one shape (as many units, unit k of each with one BRM, implementation and params, the same
links at one depth, streamers.py's ``group_links``): ``share`` renames ``group``'s units, their
nodes and its links to ``with``'s, so one group has a node per nest. LOWER runs it once per
nest, the generator makes one link per pair of ports, and the estimate counts the area once
(streamers.py's ``group_runs``). No memlet changes, so the reference check is unchanged. Each
renamed node and link keeps what it was in ``sandbox.shared``, which ``unshare`` puts back.

``bind``'s "share with" (``share_choices``) does the same for one node as it is bound; ``share``
is the step for a group built and linked first.
"""

from __future__ import annotations

from typing import Any

from snax_forge.design.streamers import group_links, groups
from snax_forge.dfg import DfgError, Graph, Node

from .recipe import SandboxError
from .transforms import _copy, _result, register_transform

SHARED = "sandbox.shared"


def _nodes(g: Graph, unit: str) -> list[Node]:
    return [n for n, _ in g.walk() if n.kind == "accelerated" and n.attrs["instance"] == unit]


def _shape(g: Graph, members: list[str]) -> tuple[list[tuple[Any, ...]], set[tuple[Any, ...]]]:
    """A group's units, ``(brm, implementation, params)`` each in order, and its links as
    ``(unit index, port, unit index, port, depth)``."""
    units = []
    for u in members:
        a = _nodes(g, u)[0].attrs
        units.append((a["brm"], a["implementation"], a.get("params") or {}))
    at = {u: i for i, u in enumerate(members)}
    links = set()
    for k, frm, to in group_links(g, members):
        (fu, fp), (tu, tp) = frm.split(".", 1), to.split(".", 1)
        links.add((at[fu], fp, at[tu], tp, k.attrs["depth"]))
    return units, links


def share_choices(g: Graph, group: str) -> list[dict[str, Any]]:
    """The groups ``group`` can be shared with, in the graph's order: ``{value, label, ok,
    message}``, ``message`` what differs (the editor's choice)."""
    out = []
    for other in groups(g):
        if other == group:
            continue
        try:
            _check(g, group, other)
            out.append({"value": other, "label": other, "ok": True})
        except SandboxError as e:
            out.append({"value": other, "label": other, "ok": False,
                        "message": str(e).split(": ", 1)[-1]})  # fmt: skip
    return out


def _check(g: Graph, group: str, other: str) -> tuple[list[str], list[str]]:
    what = f"share {group!r}"
    found = groups(g)
    for x in (group, other):
        if x not in found:
            raise SandboxError(f"{what}: no group {x!r} (groups: {', '.join(found)})")
    if group == other:
        raise SandboxError(f"{what}: a group with itself")
    mine, theirs = found[group], found[other]
    for x, members in ((group, mine), (other, theirs)):
        for u in members:
            for n in _nodes(g, u):
                if "accelerator" in n.attrs:
                    raise SandboxError(f"{what}: {x} is in accelerator "
                                       f"{n.attrs['accelerator']} (unencapsulate first)")  # fmt: skip
    if any(len(_nodes(g, u)) > 1 for u in mine):
        raise SandboxError(f"{what}: {group} runs several nests already (share the other group "
                           f"with {group})")  # fmt: skip
    if len(mine) != len(theirs):
        raise SandboxError(f"{what}: {group} has {len(mine)} units, {other} {len(theirs)}")
    (ua, la), (ub, lb) = _shape(g, mine), _shape(g, theirs)
    for k, (a, b) in enumerate(zip(ua, ub, strict=True)):
        if a != b:
            differ = [n for n, x, y in zip(("brm", "implementation"), a, b, strict=False) if x != y]
            differ += [f"{p} = {a[2].get(p)} against {b[2].get(p)}"
                       for p in dict.fromkeys([*a[2], *b[2]]) if a[2].get(p) != b[2].get(p)]  # fmt: skip
            raise SandboxError(f"{what}: {mine[k]} and {theirs[k]} differ: {', '.join(differ)}")
    if la != lb:
        raise SandboxError(f"{what}: {group}'s links are not {other}'s (ports or depths)")
    return mine, theirs


def _group(g: Graph, selected: str) -> tuple[str, list[str]]:
    """``selected`` as a group (its outline, or a unit: the unit's group) and the groups of its
    shape it can be shared with (the site's and the transform's first checks, one message)."""
    found = groups(g)
    group = selected if selected in found else None
    if group is None:
        n = next((x for x, _ in g.walk() if x.id == selected and x.kind == "accelerated"), None)
        if n is None:
            raise SandboxError(f"share {selected!r}: neither a group nor a bound unit")
        group = n.attrs.get("group", n.attrs["instance"])
    ok = [c["value"] for c in share_choices(g, group) if c["ok"]]
    if not ok:
        raise SandboxError(f"share {selected!r}: no group of its shape to share with")
    return group, ok


def _site(g: Graph, selected: str) -> dict[str, Any]:
    """A group's outline, or a unit (its group): ``group`` filled, ``with`` the first group it
    can be shared with."""
    group, ok = _group(g, selected)
    return {"group": group, "with": ok[0]}


def share(g: Graph, group: str, **partner: str) -> Graph:
    """The registered transform, ``{"group": X, "with": Y}`` (``with`` is a Python keyword)."""
    if set(partner) != {"with"}:
        raise SandboxError(f"share {group!r}: takes group and with, got "
                           f"{sorted(['group', *partner])}")  # fmt: skip
    other = partner["with"]
    group, _ = _group(g, group)
    mine, theirs = _check(g, group, other)
    rename = dict(zip(mine, theirs, strict=True))
    linked = len(theirs) > 1
    g = _copy(g)
    for n, _ in g.walk():
        if n.kind == "accelerated" and n.attrs["instance"] in rename:
            n.attrs[SHARED] = {"group": group, "instance": n.attrs["instance"]}
            n.attrs["instance"] = rename[n.attrs["instance"]]
            if linked:
                n.attrs["group"] = other
        elif n.kind == "link" and n.attrs["from"] in rename:
            n.attrs[SHARED] = {"group": group, "from": n.attrs["from"], "to": n.attrs["to"]}
            n.attrs["from"], n.attrs["to"] = rename[n.attrs["from"]], rename[n.attrs["to"]]
    try:
        return _result(g)
    except DfgError as e:
        raise SandboxError(f"share {group!r}: {e}") from None


def _shared(g: Graph) -> list[str]:
    return list(dict.fromkeys(n.attrs[SHARED]["group"] for n, _ in g.walk() if SHARED in n.attrs))


def _unshare_site(g: Graph, selected: str) -> dict[str, Any]:
    """A unit of a shared nest, or the shared group: ``group`` the old name of the nest's group
    (the one selected when several share it)."""
    names = _shared(g)
    if selected in names:
        return {"group": selected}
    n = next((x for x, _ in g.walk() if x.id == selected), None)
    if n is not None and SHARED in n.attrs:
        return {"group": n.attrs[SHARED]["group"]}
    mine = [x for x, _ in g.walk() if x.kind == "accelerated"
            and x.attrs.get("group", x.attrs.get("instance")) == selected and SHARED in x.attrs]  # fmt: skip
    if mine:
        return {"group": mine[0].attrs[SHARED]["group"]}
    raise SandboxError(f"unshare {selected!r}: shares no group (share)")


def unshare(g: Graph, group: str) -> Graph:
    """The inverse of share: every node and link ``share`` renamed for ``group`` back (``group``
    as the site takes it: the old name, or a node of the shared nest)."""
    group = _unshare_site(g, group)["group"]
    g = _copy(g)
    for n, _ in g.walk():
        rec = n.attrs.get(SHARED)
        if rec is None or rec["group"] != group:
            continue
        if n.kind == "link":
            n.attrs["from"], n.attrs["to"] = rec["from"], rec["to"]
        else:
            n.attrs["instance"] = rec["instance"]
            if "group" in n.attrs:
                n.attrs["group"] = group
        del n.attrs[SHARED]
    return _result(g)


register_transform("share", share, site=_site, params=("group", "with"))
register_transform("unshare", unshare, site=_unshare_site)
