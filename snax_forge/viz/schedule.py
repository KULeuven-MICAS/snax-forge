"""The schedule (D57): the program's window as lanes, its quiet stretches folded, and the beat
and stall lines of an open range. Queries on the run's index (index.py) that read no file;
the viewer only draws their answers.

    schedule(index, open, cycle)    -> lanes(index) and segments(index, open, cycle) together
    lanes(index)                    -> {"lanes": [...]}
    segments(index, open, cycle)    -> {"segments": [...], "boot": {...}}
    detail(index, start, stop)      -> the beat and stall lines of [start, stop) per lane

**Lanes**, top to bottom: one per core (``core<i>``), the DMA, one per mover as the streamer
orders them, the accelerator. A core's lane has its spans classed barrier > poll > sleep > run
(``run/cores.py``, the profile's rule), its function spans and its CSR accesses (``csr`` and
``poll`` events); the DMA's one bar per copy, issue to done; a mover's and the accelerator's
one bar per task, start to done, with the run's class totals (its ``done``'s ``cycles``) and,
from ``beat`` on, its class runs (index.py). Class names only: colours are the viewer's.

**Segments** cover ``[0, T)``, ``T`` the window, with no gap and no overlap. A cycle is
active when an accelerator or mover task is open on it (from its start to its done, both
included) or a DMA copy (issue to done, both included); the other cycles are quiet. A quiet
stretch of at least ``FOLD_MIN`` cycles is foldable: folded unless its first cycle is in
``open`` or it holds ``cycle``. The other cycles make unfoldable segments, adjacent ones
merged. A folded segment carries, per core, its class counts, its CSR writes and reads (a
``poll``'s reads all at its first cycle) and the functions it spans. The boot (the cycles
before the window, ``trace_meta.json``'s ``boot``) is one more segment before 0 that never
opens.

**Detail** is the beat and stall lines of one range, as the converter wrote them, grouped per
lane: each crossbar port's grants and stalls, the DMA's beats and bursts, the accelerator's
fires and pushes, each lane's FIFO counts (``<mover>.lane<l>``).

Below ``task`` (trace ``off``) there is no schedule, below ``beat`` no detail: a
``ScheduleUnavailable`` with the reason.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable
from typing import Any

from snax_forge.run.cluster import DMA
from snax_forge.run.cores import CORE_CLASSES

from .index import Event, RunIndex

FOLD_MIN = 16  # the shortest quiet stretch that folds
DETAIL_KINDS = ("grant", "stall", "dma", "burst", "fire", "push", "fifo")


class ScheduleUnavailable(ValueError):
    """A schedule answer the run's trace level cannot give; the message says why."""


def _need(index: RunIndex, level: str, what: str) -> None:
    if not index.on(level):
        raise ScheduleUnavailable(f"{what} needs --trace {level}; this run was traced at "
                                  f"{index.level}")  # fmt: skip


# =============================================================================
# Lanes
# =============================================================================


def _bar(index: RunIndex, task: str) -> dict[str, Any]:
    """An accelerator or mover task's bar: start to done, its class totals and runs."""
    start, done = index.start[task], index.done.get(task)
    bar: dict[str, Any] = {"task": task, "from": start["t"], "to": None if done is None else done["t"],
                           "cycles": None if done is None else done.get("cycles")}  # fmt: skip
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
        for i, e in sorted(index.issue.items(), key=lambda x: x[1]["t"])]})  # fmt: skip
    acc = index.cluster.accelerator
    for name, kind in [*((m.name, "mover") for m in acc.movers), (acc.name, "accelerator")]:
        tasks = sorted(index.tasks_of.get(name, []), key=lambda t: index.start[t]["t"])
        out.append({"name": name, "kind": kind, "bars": [_bar(index, t) for t in tasks]})
    return {"level": index.level, "total_cycles": index.total_cycles, "lanes": out}


# =============================================================================
# Segments
# =============================================================================


def active(index: RunIndex) -> list[tuple[int, int]]:
    """The window's active cycles as merged ``[first, end)`` stretches (module doc)."""
    spans = []
    for task, s in index.start.items():
        d = index.done.get(task)
        spans.append((s["t"], (d["t"] if d is not None else index.total_cycles - 1) + 1))
    for i, e in index.issue.items():
        d = index.copy_done.get(i)
        spans.append((e["t"], (d["t"] if d is not None else index.total_cycles - 1) + 1))
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        a, b = max(a, 0), min(b, index.total_cycles)
        if a >= b:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


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
    for e in window(index, start, stop):
        k = e["k"]
        if k in ("grant", "stall"):
            ports.setdefault(e["src"], []).append(e)
        elif k in ("dma", "burst"):
            dma.append(e)
        elif k in ("fire", "push"):
            acc.append(e)
        elif k == "fifo":
            fifo.setdefault(f"{e['src']}.lane{e['lane']}", []).append(e)
    return {"from": start, "to": stop, "level": index.level, "filtered": index.filtered,
            "ports": ports, "dma": dma, "acc": acc, "fifo": fifo}  # fmt: skip


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
