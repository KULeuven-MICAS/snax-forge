"""The cluster view at one cycle (D57), data side only, and stepping between cycles. Queries on
the run's index (index.py) that read no file; the viewer only draws their answers.

    cluster_at(index, t)       -> one dict per block at cycle t, and the cycle's events
    step(index, t, dir, by)    -> the next or previous cycle with an event or a task boundary

**Blocks** at ``t`` (the cores and the controller are the schedule's core lanes):

- ``dma``: the copies open at ``t`` (issue to done, both included) with their task,
  direction and bytes; from ``beat`` on, its L1 beat at ``t`` (``i`` of the copy's ``n``
  beats of that op) and its AXI bursts at ``t``, each with its region, elements and values,
  and its reads whose data comes back at ``t`` (``back``: L1 beats and L2 bursts);
- ``movers``, per lane (crossbar port): whether its mover's task is open; from ``beat`` its
  mover's class at ``t`` (the class run holding it), its grant at ``t`` and the read data
  coming back at ``t`` (the index's ``returns``); from ``stall`` its stall at ``t`` and its
  FIFO count (the last ``fifo`` line at or before ``t``, 0 before the first);
- ``acc``: whether its task is open; from ``beat`` its class at ``t``, its firing at ``t``
  with its operands' values and its push with the result;
- ``banks``, every bank, grouped by superbank (a DMA beat's banks): the access at ``t``
  (port, op, row, region, elements and their values, the word: a grant's, or a DMA beat's
  word on it), the read
  data going back at ``t``; from ``stall`` the requesters refused at ``t``, the one it
  served first;
- ``interconnect``: whether anything crossed it at ``t``; from ``stall`` its conflicts at
  ``t`` (a bank with a refused request: the one served and the ones refused).

A field the run's level cannot give is absent, and ``needs`` names the level each needs.
``events`` is every line at ``t`` (``insn`` at ``finegrained``), in trace order.

**Stepping**: ``by`` ``event`` steps through the index's event cycles, ``boundary`` its
boundary cycles (index.py); ``dir`` 1 or -1. At the window's ends, or with no such cycle
past ``t``, the answer is ``t``.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from typing import Any

from snax_forge.run.cluster import DMA

from .index import Event, RunIndex

NEEDS = {"class": "beat", "grant": "beat", "back": "beat", "fire": "beat", "push": "beat",
         "beat": "beat", "burst": "beat", "access": "beat", "traffic": "beat", "stall": "stall",
         "fifo": "stall", "stalled": "stall", "conflicts": "stall"}  # fmt: skip


def _at(index: RunIndex, t: int) -> list[Event]:
    return index.events[bisect_left(index.times, t) : bisect_right(index.times, t)]


def _open(index: RunIndex, src: str, t: int) -> str | None:
    """The task of ``src`` open at ``t`` (start to done, both included), or None."""
    for task in index.tasks_of.get(src, []):
        s, d = index.start[task]["t"], index.done.get(task)
        if s <= t <= (d["t"] if d is not None else index.total_cycles):
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
    here = _at(index, t)
    of = {k: [e for e in here if e["k"] == k] for k in ("grant", "stall", "dma", "burst", "fire",
                                                        "push")}  # fmt: skip
    back = index.returns.get(t, []) if beat else []

    copies = []
    for i, e in sorted(index.issue.items(), key=lambda x: x[1]["t"]):
        d = index.copy_done.get(i)
        if e["t"] <= t <= (d["t"] if d is not None else index.total_cycles):
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
        for lane, port in enumerate(m.ports):
            x: dict[str, Any] = {"mover": m.name, "lane": lane, "task": task}
            if beat:
                x["class"] = class_at(index, m.name, t)
                x["grant"] = next((e for e in of["grant"] if e["src"] == port), None)
                x["back"] = [e for e in back if e["k"] == "grant" and e["src"] == port]
            if stall:
                x["stall"] = next((e for e in of["stall"] if e["src"] == port), None)
                x["fifo"] = fifo_at(index, m.name, lane, t)
            movers[port] = x

    unit: dict[str, Any] = {"name": acc.name, "task": _open(index, acc.name, t)}
    if beat:
        unit["class"] = class_at(index, acc.name, t)
        unit["fire"] = next(iter(of["fire"]), None)
        unit["push"] = next(iter(of["push"]), None)

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
            x["stalled"] = [{"served": refused[0]["by"], "served_addr": refused[0]["by_addr"]},
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
            "dma": {"copies": copies}, "movers": movers, "acc": unit, "banks": banks,
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
