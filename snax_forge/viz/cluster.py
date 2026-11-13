"""The cluster view at one cycle, data side only, and stepping between cycles.

Queries on the run's index (index.py); the viewer only draws them.

    cluster_at(index, t)     per block at cycle t (``dma``, ``movers``, ``acc``, ``streamers``,
                             ``banks``, ``interconnect``), and the cycle's ``events``
    step(index, t, dir, by)  the next (dir 1) or previous (-1) cycle with an event (by
                             ``event``) or a task boundary (``boundary``); ``t`` when none

A field the run's trace level cannot give is absent and ``needs`` names its level: open tasks
from ``task``; classes, grants, firings and returning reads from ``beat``; stalls, FIFO counts
and conflicts from ``stall``. ``linked`` says the accelerator has a group of several linked
units, whose beat lines the converter refuses, so its runs stop at ``task``. A shared mover's
lanes name the ``user`` (``<instance>_<port>``) its port mux serves: the one whose unit's task
its open task was launched with, None between tasks. ``acc`` and ``streamers`` are in
CONTRACTS, "Profile and trace".
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any

from snax_forge.run.cluster import DMA, Mover, linked

from .index import Event, RunIndex
from .schedule import window

NEEDS = {"class": "beat", "grant": "beat", "back": "beat", "fire": "beat", "push": "beat",
         "beat": "beat", "burst": "beat", "access": "beat", "traffic": "beat", "stall": "stall",
         "fifo": "stall", "stalled": "stall", "conflicts": "stall"}  # fmt: skip


def _open(index: RunIndex, src: str, t: int) -> str | None:
    """The task of ``src`` open at ``t`` (``RunIndex.span``), or None."""
    for task in index.tasks_of.get(src, []):
        a, b = index.span(index.start[task], index.done.get(task))
        if a <= t <= b:
            return task
    return None


def class_at(index: RunIndex, src: str, t: int) -> str | None:
    """The class of ``src`` at ``t``: the class run holding it, or None (no run holds it)."""
    for task in index.tasks_of.get(src, []):
        runs = index.runs.get(task, [])
        k = bisect_right([a for a, _, _ in runs], t) - 1
        if k >= 0 and t < runs[k][1]:
            return runs[k][2]
    return None


def fifo_at(index: RunIndex, mover: str, lane: int, t: int) -> int:
    """A lane's queue count at ``t``: the last ``fifo`` line at or before it, 0 before any."""
    seq = index.fifo.get((mover, lane), [])
    k = bisect_right([c for c, _ in seq], t) - 1
    return seq[k][1] if k >= 0 else 0


def user_at(index: RunIndex, m: Mover, task: str | None) -> str | None:
    """The user a shared mover serves in its task ``task``: its user of the unit whose task
    ``task`` was launched with (the task list's pairing), or None."""
    unit = index.start.get(index.accel.get(task or "", ""), {}).get("src")
    if unit is None:
        return None
    user = index.cluster.accelerator.port_of(m, unit)
    return user if user in m.users else None


def _access(e: Event, word: int) -> dict[str, Any]:
    """What a bank shows of an access: a grant, or word ``word`` of a DMA beat."""
    if e["k"] == "grant":
        return {"port": e["src"], "op": e["op"], "row": e["row"], "region": e["region"],
                "elems": e["elems"], "vals": e["vals"], "value": e["value"],
                "task": e["task"]}  # fmt: skip
    per = len(e["elems"]) // max(len(e["value"]), 1)
    part = slice(word * per, (word + 1) * per)
    return {"port": DMA, "op": e["op"], "row": e["row"], "region": e["region"],
            "elems": e["elems"][part], "vals": e["vals"][part], "value": e["value"][word],
            "task": e["task"], "id": e["id"], "i": e["i"]}  # fmt: skip


def cluster_at(index: RunIndex, t: int) -> dict[str, Any]:
    """The blocks at cycle ``t`` (module doc)."""
    c, acc = index.cluster, index.cluster.accelerator
    beat, stall = index.on("beat"), index.on("stall")
    here = window(index, t, t + 1)
    of = {k: [e for e in here if e["k"] == k] for k in ("grant", "stall", "dma", "burst", "fire",
                                                        "push")}  # fmt: skip
    back = index.returns.get(t, []) if beat else []

    copies = []
    for i, e in index.copies():
        a, b = index.span(e, index.copy_done.get(i))
        if a <= t <= b:
            cp = {"id": i, "task": e["task"], "dir": e["dir"], "bytes": e["size"] * e["reps"]}
            if beat:
                cp["beat"] = [{**b, "n": index.dma_beats.get((i, b["op"]), 0)}
                              for b in of["dma"] if b["id"] == i]  # fmt: skip
                cp["burst"] = [b for b in of["burst"] if b["id"] == i]
                cp["back"] = [b for b in back if b["k"] in ("dma", "burst") and b["id"] == i]
            copies.append(cp)

    movers = {}
    for m in acc.movers:
        task = _open(index, m.name, t)
        user = {"user": user_at(index, m, task)} if m.users else {}
        for lane, port in enumerate(m.ports):
            x: dict[str, Any] = {"mover": m.name, "lane": lane, "task": task, **user}
            if beat:
                x["class"] = class_at(index, m.name, t)
                x["grant"] = next((e for e in of["grant"] if e["src"] == port), None)
                x["back"] = [e for e in back if e["k"] == "grant" and e["src"] == port]
            if stall:
                x["stall"] = next((e for e in of["stall"] if e["src"] == port), None)
                x["fifo"] = fifo_at(index, m.name, lane, t)
            movers[port] = x

    units = []
    for u in acc.units:
        x = {"name": u.name, "group": u.group, "task": _open(index, u.name, t)}
        if beat:
            x["class"] = class_at(index, u.name, t)
            x["fire"] = next((e for e in of["fire"] if e["src"] == u.name), None)
            x["push"] = next((e for e in of["push"] if e["src"] == u.name), None)
        units.append(x)
    active = next((x for x in units if x["task"] is not None), units[0] if units else {})
    unit: dict[str, Any] = {**active, "accelerator": acc.name, "units": units}
    group_set = {g: name for name, gs in acc.accelerators() for g in gs}
    for x in units:
        x["set"] = group_set.get(x["group"])
    streamers = []
    for s in acc.csrs.sets:
        mine = [m.name for m in acc.movers if m.set == s.name]
        task = next((k for k in (movers[f"{m}.lane0"]["task"] for m in mine) if k), None)
        streamers.append({"name": s.name, "groups": dict(acc.accelerators()).get(s.name, []),
                          "first": s.first, "last": s.last, "movers": mine,
                          "busy": task is not None, "task": task})  # fmt: skip

    per_sb = max(c.dma_beat_bytes // c.bank_bytes, 1)
    banks = []
    for b in range(c.banks):
        x = {"bank": b, "superbank": b // per_sb}
        if beat:
            acc_here = [_access(e, 0) for e in of["grant"] if e["bank"] == b]
            for e in of["dma"]:
                j = (b - e["bank"]) % c.banks
                if j < len(e["value"]):
                    acc_here.append(_access(e, j))
            x["access"] = acc_here
            x["back"] = [e for e in back if e["k"] == "grant" and e["bank"] == b] + [
                e for e in back if e["k"] == "dma" and (b - e["bank"]) % c.banks < len(e["value"])
            ]
        if stall:
            refused = [e for e in of["stall"] if e["bank"] == b]
            x["stalled"] = [{"served": refused[0]["by"]},
                            *({"port": e["src"], "addr": e["addr"], "task": e["task"]}
                              for e in refused)] if refused else []  # fmt: skip
        banks.append(x)

    xbar: dict[str, Any] = {}
    if beat:
        xbar["traffic"] = bool(of["grant"] or of["dma"])
    if stall:
        xbar["conflicts"] = [{"bank": x["bank"], "served": x["stalled"][0]["served"],
                              "refused": [s["port"] for s in x["stalled"][1:]]}
                             for x in banks if x["stalled"]]  # fmt: skip

    needs = {k: v for k, v in NEEDS.items() if not index.on(v)}
    return {"t": t, "level": index.level, "filtered": index.filtered, "needs": needs,
            "linked": linked(acc.description),
            "dma": {"copies": copies}, "movers": movers, "acc": unit, "streamers": streamers,
            "banks": banks,
            "interconnect": xbar, "events": here}  # fmt: skip


def step(index: RunIndex, t: int, dir: int, by: str) -> int:
    """The next (``dir`` 1) or previous (-1) cycle of the index's event or boundary cycles
    after ``t``; ``t`` when there is none (module doc)."""
    if dir not in (1, -1) or by not in ("event", "boundary"):
        raise ValueError(f"step: dir is 1 or -1 and by event or boundary, not {dir!r}, {by!r}")
    cycles = index.event_cycles if by == "event" else index.boundary_cycles
    if dir == 1:
        k = bisect_right(cycles, t)
        return cycles[k] if k < len(cycles) else t
    k = bisect_left(cycles, t) - 1
    return cycles[k] if k >= 0 else t


__all__ = ["class_at", "cluster_at", "fifo_at", "step"]
