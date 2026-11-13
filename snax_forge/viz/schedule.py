"""The schedule: the program's window as lanes, quiet stretches folded, and a range's detail.

Queries on the run's index (index.py); the viewer only draws them.

    lanes(index)                  one lane per core, the DMA, then per accelerator its groups'
                                  movers and units: each task a bar, start to done
    segments(index, open, cycle)  ``[0, T)`` in segments; a quiet stretch (no accelerator, mover
                                  or DMA task open) of ``FOLD_MIN`` cycles or more is folded
                                  unless opened or holding ``cycle``; the boot is one more
    schedule(index, open, cycle)  both together
    detail(index, start, stop)    the beat and stall lines of ``[start, stop)`` per lane

A mover's task with every temporal bound 0 is marked ``idle``. Below ``task`` there is no
schedule and below ``beat`` no detail: a ``ScheduleUnavailable``.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from typing import Any

from snax_forge.run.cluster import DMA
from snax_forge.run.cores import CORE_CLASSES, merge_spans

from .index import Event, RunIndex

FOLD_MIN = 16  # the shortest quiet stretch that folds


class ScheduleUnavailable(ValueError):
    """A schedule answer the run's trace level cannot give; the message says why."""


def _need(index: RunIndex, level: str, what: str) -> None:
    if not index.on(level):
        raise ScheduleUnavailable(f"{what} {index.needs(level)}")


# =============================================================================
# Lanes
# =============================================================================


def _bar(index: RunIndex, task: str) -> dict[str, Any]:
    """An accelerator or mover task's bar: start to done, its class totals and runs."""
    start, done = index.start[task], index.done.get(task)
    bar: dict[str, Any] = {"task": task, "from": start["t"], "to": None if done is None else done["t"],
                           "cycles": None if done is None else done.get("cycles"),
                           "idle": task in index.idle}  # fmt: skip
    if done is not None and "firings" in done:
        bar["firings"] = done["firings"]
    if index.on("beat"):
        bar["classes"] = [{"from": a, "to": b, "mode": m} for a, b, m in index.runs.get(task, [])]
    return bar


def lanes(index: RunIndex) -> dict[str, Any]:
    """The lanes (module doc); a ScheduleUnavailable below ``task``."""
    _need(index, "task", "the schedule")
    out: list[dict[str, Any]] = []
    for core in index.cluster.cores:
        out.append({
            "name": core, "kind": "core",
            "spans": [{"from": a, "to": b, "class": c} for a, b, c in index.core_spans[core]],
            "funcs": [{"from": a, "to": b, "name": n} for a, b, n in index.funcs.get(core, [])],
            "csr": index.core_csr.get(core, []),
        })  # fmt: skip
    out.append({"name": DMA, "kind": "dma", "bars": [
        {"task": e["task"], "id": i, "from": e["t"],
         "to": index.copy_done[i]["t"] if i in index.copy_done else None, "dir": e["dir"],
         "bytes": e["size"] * e["reps"]}
        for i, e in index.copies()]})  # fmt: skip
    acc = index.cluster.accelerator
    for set_, groups in acc.accelerators():
        for g in groups:
            units = [u.name for u in acc.units if u.group == g]
            rows = [*((m.name, "mover") for u in units for m in acc.movers_of(u)
                      if m.set == set_), *((u, "accelerator") for u in units)]  # fmt: skip
            for name, kind in rows:
                tasks = sorted(index.tasks_of.get(name, []), key=lambda t: index.start[t]["t"])
                out.append({"name": name, "kind": kind, "accelerator": set_, "group": g,
                            "bars": [_bar(index, t) for t in tasks]})  # fmt: skip
    return {"level": index.level, "total_cycles": index.total_cycles, "lanes": out}


# =============================================================================
# Segments
# =============================================================================


def active(index: RunIndex) -> list[tuple[int, int]]:
    """The window's active cycles as merged ``[first, end)`` stretches (module doc)."""
    spans = [index.span(s, index.done.get(task)) for task, s in index.start.items()]
    spans += [index.span(e, index.copy_done.get(i)) for i, e in index.issue.items()]
    clipped = ((max(a, 0), min(b + 1, index.total_cycles)) for a, b in spans)
    return merge_spans((a, b) for a, b in clipped if a < b)


def quiet(index: RunIndex) -> list[tuple[int, int]]:
    """``[0, T)`` minus the active stretches, as ``[first, end)``."""
    out, t = [], 0
    for a, b in active(index):
        if a > t:
            out.append((t, a))
        t = b
    if t < index.total_cycles:
        out.append((t, index.total_cycles))
    return out


def _clip(spans: Iterable[tuple[int, int, str]], a: int, b: int) -> list[tuple[int, int, str]]:
    return [(max(x, a), min(y, b), c) for x, y, c in spans if x < b and a < y]


def fold_summary(index: RunIndex, a: int, b: int) -> dict[str, Any]:
    """Per core, what a folded ``[a, b)`` holds: its class counts, CSR writes and reads, and
    the functions it spans."""
    out = {}
    for core in index.cluster.cores:
        counts = dict.fromkeys(CORE_CLASSES, 0)
        for x, y, c in _clip(index.core_spans[core], a, b):
            counts[c] += y - x
        writes = reads = 0
        for e in index.core_csr.get(core, []):
            if a <= e["t"] < b:
                if e["k"] == "poll":
                    reads += e["count"]
                elif e["op"] == "w":
                    writes += 1
                else:
                    reads += 1
        funcs = [n for _, _, n in _clip(index.funcs.get(core, []), a, b)]
        out[core] = {"cycles": counts, "csr_writes": writes, "csr_reads": reads,
                     "funcs": funcs}  # fmt: skip
    return out


def segments(index: RunIndex, open: Iterable[int] = (), cycle: int | None = None
             ) -> dict[str, Any]:  # fmt: skip
    """The column map over ``[0, T)`` and the boot (module doc); a ScheduleUnavailable below
    ``task``."""
    _need(index, "task", "the schedule")
    opened = set(open)
    out: list[dict[str, Any]] = []

    def plain(a: int, b: int) -> None:
        if out and not out[-1]["foldable"] and out[-1]["to"] == a:
            out[-1]["to"] = b
        else:
            out.append({"from": a, "to": b, "folded": False, "foldable": False})

    t = 0
    for a, b in quiet(index):
        if a > t:
            plain(t, a)
        if b - a >= FOLD_MIN:
            folded = a not in opened and not (cycle is not None and a <= cycle < b)
            seg = {"from": a, "to": b, "folded": folded, "foldable": True}
            if folded:
                seg["cores"] = fold_summary(index, a, b)
            out.append(seg)
        else:
            plain(a, b)
        t = b
    if t < index.total_cycles:
        plain(t, index.total_cycles)
    return {"level": index.level, "total_cycles": index.total_cycles, "fold_min": FOLD_MIN,
            "boot": {"from": -index.boot, "to": 0, "cycles": index.boot, "folded": True},
            "segments": out}  # fmt: skip


def schedule(index: RunIndex, open: Iterable[int] = (), cycle: int | None = None
             ) -> dict[str, Any]:  # fmt: skip
    """The lanes and the segments in one answer; a ScheduleUnavailable below ``task``."""
    return {**lanes(index), **segments(index, open, cycle)}


# =============================================================================
# Detail
# =============================================================================


def window(index: RunIndex, start: int, stop: int) -> list[Event]:
    """The events with ``start <= t < stop``, in trace order."""
    return index.events[bisect_left(index.times, start) : bisect_left(index.times, stop)]


def detail(index: RunIndex, start: int, stop: int) -> dict[str, Any]:
    """The beat and stall lines of ``[start, stop)`` per lane (module doc); a
    ScheduleUnavailable below ``beat``."""
    _need(index, "beat", "the detail rows")
    ports: dict[str, list[Event]] = {}
    fifo: dict[str, list[Event]] = {}
    dma: list[Event] = []
    acc: list[Event] = []
    units: dict[str, list[Event]] = {u: [] for u in index.cluster.accelerator.unit_names}
    for e in window(index, start, stop):
        k = e["k"]
        if k in ("grant", "stall"):
            ports.setdefault(e["src"], []).append(e)
        elif k in ("dma", "burst"):
            dma.append(e)
        elif k in ("fire", "push"):
            acc.append(e)
            units.setdefault(e["src"], []).append(e)
        elif k == "fifo":
            fifo.setdefault(f"{e['src']}.lane{e['lane']}", []).append(e)
    return {"from": start, "to": stop, "level": index.level, "filtered": index.filtered,
            "ports": ports, "dma": dma, "acc": acc, "units": units, "fifo": fifo}  # fmt: skip


__all__ = [
    "FOLD_MIN",
    "ScheduleUnavailable",
    "active",
    "detail",
    "lanes",
    "quiet",
    "schedule",
    "segments",
]
