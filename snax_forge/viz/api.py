"""What the viewer asks for, as plain Python functions.

What this is
------------
The server (server.py) is a thin HTTP layer; everything it answers comes
from the functions here, so the tests call them directly and the server
only turns their results into JSON. Nothing here draws anything: the
viewer (static/) formats the numbers, and HTML is checked by eye (D55).

Loading
-------
A run is a flow's (D136, D137): its folder ``out/flow/<name>/`` or the
``run/`` directory in it. ``run/`` is read with ``run.outputs.read_run``
(run.json, the profile and, above trace ``off``, the trace, C§7), and the
cluster with ``run.cluster.load_cluster`` from the hjson files and the
design point ``run.json`` names, so the views use run/'s dataclasses and
never parse the files themselves (D55). Everything is read once, at start-up
or on an explicit reload (``RunSet``); nothing watches the files.

A run is named after its flow folder. Two folders with the same name get
``-2``, ``-3``, ... in the order given, so names stay stable across reloads.

Events
------
The trace events are kept as the flat dicts of trace.jsonl, sorted by
cycle, with their cycles in a separate list so a window ``[a, b)`` is two
bisects. The source filter is applied after the window, on the few events
left. A block's tasks pair each ``start`` (a DMA's ``issue``) with its
``done``.

Not yet
-------
The memory and movement views read beat-level events and memory dumps,
which come in FLOW3 round 2: their routes answer with the reason. The
schedule and the cluster view come back in round 3 (D57). The profile keeps
each FIFO's highest count only (D137), so there is no busy-window
occupancy.
"""

from __future__ import annotations

import threading
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from snax_forge.run import Cluster, RunError, RunOutputs, load_cluster, read_run
from snax_forge.run.cluster import DMA

MAX_ROWS = 256  # memory rows per request (static/memory.js); the memory view is round 2's
ROUND_2 = (
    "FLOW3 round 2: the memory and movement views need beat-level events and memory dumps, "
    "which the SNAX-GVSoC flow does not write yet"
)
NO_FIFO_WINDOW = "the profile keeps each FIFO's highest count only (D137): no busy window"


class MemoryViewError(ValueError):
    """A request the memory view cannot answer; the message says why."""


class MovementUnavailable(Exception):
    """The run cannot answer a movement question; the message is the reason shown."""


# =============================================================================
# One run
# =============================================================================


@dataclass
class RunView:
    """One flow run, read back into run/'s dataclasses."""

    name: str
    path: Path  # its run/ directory
    folder: Path  # the flow folder around it
    outputs: RunOutputs
    cluster: Cluster
    events: list[dict[str, Any]] = field(default_factory=list)  # sorted by t
    times: list[int] = field(default_factory=list)  # events[i]["t"], for bisect
    spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)  # per block

    @property
    def total_cycles(self) -> int:
        return int(self.outputs.profile.total_cycles)

    @property
    def trace_level(self) -> str:
        return str(self.outputs.run.get("trace_level", "off"))


def run_dir_of(path: str | Path) -> tuple[Path, Path]:
    """A flow folder's or its ``run/`` directory's (run directory, flow folder); a RunError
    when it is neither."""
    p = Path(path)
    if (p / "run" / "run.json").is_file():
        return p / "run", p
    if (p / "run.json").is_file():
        return p, p.parent
    raise RunError("outputs.missing", f"{p}: not a flow folder (no run/run.json) nor its run/")


def load_run(path: str | Path, name: str | None = None) -> RunView:
    """Read one flow run (module doc). Raises what read_run and load_cluster raise."""
    run_dir, folder = run_dir_of(path)
    out = read_run(run_dir)
    hjson = out.run.get("hjson") or []
    if not hjson or "design_point" not in out.run:
        raise RunError("outputs.missing", f"{run_dir / 'run.json'}: names no hjson files or "
                       "design point")  # fmt: skip
    cluster = load_cluster(folder / hjson[0], folder / out.run["design_point"])
    events = [] if out.trace is None else sorted(out.trace.events, key=lambda e: e["t"])
    return RunView(
        name=name or folder.resolve().name,
        path=run_dir,
        folder=folder,
        outputs=out,
        cluster=cluster,
        events=events,
        times=[int(e["t"]) for e in events],
        spans=_pair_tasks(events, out.profile.total_cycles),
    )


def run_names(dirs: Sequence[str | Path]) -> list[str]:
    """Flow folder names (a ``run/`` directory takes its folder's), made unique with -2, -3,
    ... in the order given."""
    names: list[str] = []
    for d in dirs:
        p = Path(d).resolve()
        base = (
            p.parent.name if p.name == "run" and (p / "run.json").is_file() else p.name
        ) or "run"
        name, i = base, 1
        while name in names:
            i += 1
            name = f"{base}-{i}"
        names.append(name)
    return names


# =============================================================================
# Memory and movement: FLOW3 round 2
# =============================================================================


def memories(rv: RunView) -> list[str]:
    """The memories the memory view shows: none until FLOW3 round 2."""
    return []


def memory_view(rv: RunView, marks: str | None = None) -> dict[str, Any]:
    raise MemoryViewError(ROUND_2)


def memory_rows(rv: RunView, mem: str, start: int, stop: int, marks: str | None = None) -> Any:
    raise MemoryViewError(ROUND_2)


def _unavailable(rv: RunView) -> dict[str, Any]:
    return {"name": rv.name, "available": False, "reason": ROUND_2}


def movement_view(rv: RunView) -> dict[str, Any]:
    return _unavailable(rv)


def journey_view(rv: RunView, region: str, index: Sequence[int] | int) -> dict[str, Any]:
    return _unavailable(rv)


def conflicts_view(rv: RunView, start: int = 0, stop: int | None = None) -> dict[str, Any]:
    return _unavailable(rv)


# =============================================================================
# Answers of the API routes
# =============================================================================


def run_summary(rv: RunView) -> dict[str, Any]:
    """One entry of /api/runs."""
    return {
        "name": rv.name,
        "path": str(rv.folder),
        "design": rv.cluster.design,
        "trace_level": rv.trace_level,
        "total_cycles": rv.total_cycles,
    }


def run_detail(rv: RunView) -> dict[str, Any]:
    """/api/run/<name>: run.json, the profile, the cluster (run/cluster.py), what the header
    needs about the trace (level, boot, window, sources, event count; None at off) and the
    tasks per block."""
    tr = rv.outputs.trace
    trace = None if tr is None else {**tr.meta(), "n_events": len(rv.events)}
    return {
        "name": rv.name,
        "path": str(rv.folder),
        "run": rv.outputs.run,
        "total_cycles": rv.total_cycles,
        "profile": rv.outputs.profile.to_dict(),
        "cluster": rv.cluster.to_dict(),
        "trace": trace,
        "tasks": tasks(rv),
    }


def tasks(rv: RunView) -> dict[str, list[dict[str, Any]]]:
    """Per block that ran a task: its tasks as ``start`` / ``done`` and ``task`` (the task
    list's name); a DMA's also carry their ``direction`` and ``bytes``. Needs the events, so
    at trace off the answer is empty."""
    named: dict[str, list[dict[str, Any]]] = {}
    for e in rv.events:
        if e["k"] in ("start", "issue"):
            named.setdefault(e["src"], []).append(e)
    out: dict[str, list[dict[str, Any]]] = {}
    for block, spans in rv.spans.items():
        out[block] = []
        for k, (start, done) in enumerate(spans):
            e = named[block][k]
            task: dict[str, Any] = {"start": start, "done": done, "task": e["task"]}
            if block == DMA:
                task.update(direction=e["dir"], bytes=e["size"] * e["reps"])
            out[block].append(task)
    return out


def events_window(
    rv: RunView,
    start: int = 0,
    stop: int | None = None,
    srcs: Iterable[str] | None = None,
    kinds: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """The events with ``start <= t < stop``, of the given sources and kinds only if given.

    Events keep their trace order. ``stop`` defaults to the end of the run.
    """
    stop = rv.total_cycles + 1 if stop is None else stop
    if stop <= start:
        return []
    lo, hi = bisect_left(rv.times, start), bisect_left(rv.times, stop)
    keep_src = set(srcs) if srcs else None  # None or empty: every source
    keep_k = set(kinds) if kinds else None  # None or empty: every kind
    if keep_src is None and keep_k is None:
        return rv.events[lo:hi]
    return [
        e
        for e in rv.events[lo:hi]
        if (keep_src is None or e["src"] in keep_src) and (keep_k is None or e["k"] in keep_k)
    ]


def _pair_tasks(events: Sequence[dict[str, Any]], total: int) -> dict[str, list[tuple[int, int]]]:
    """``(start, done)`` of every task, per block, in one pass over the events: a block's
    starts (a DMA's issues, matched by id: the iDMA queues them) closed by its dones in order;
    a task still open at the end is closed at the run's total."""
    spans: dict[str, list[tuple[int, int]]] = {}
    open_at: dict[str, list[int]] = {}
    for e in events:
        src, k = e["src"], e["k"]
        if k in ("start", "issue"):
            open_at.setdefault(src, []).append(e["t"])
            spans.setdefault(src, [])
        elif k == "done" and open_at.get(src):
            spans[src].append((open_at[src].pop(0), e["t"]))
    for src, ts in open_at.items():
        spans[src] += [(t, total) for t in ts]
    return spans


def task_spans(rv: RunView, block: str) -> list[tuple[int, int]]:
    """``(start, done)`` of every task of ``block`` (computed once, at load)."""
    return rv.spans.get(block, [])


def fifo_windows(rv: RunView) -> dict[str, Any]:
    """/api/run/<name>/fifo: no busy-window occupancy (module doc), with the reason."""
    return {"available": False, "reason": NO_FIFO_WINDOW, "streamers": {}}


# =============================================================================
# The set of runs a server shows
# =============================================================================


class RunSet:
    """The flow runs given on the command line, loaded once.

    ``reload`` reads every run again and swaps the whole set at once, so a
    request never sees half of a reload. If a run cannot be read, the old set
    stays and the error is raised to the caller.
    """

    def __init__(self, dirs: Sequence[str | Path]) -> None:
        if not dirs:
            raise ValueError("give at least one flow folder")
        self.dirs = [Path(d) for d in dirs]
        self._lock = threading.Lock()
        self._runs: dict[str, RunView] = {}
        self.reload()

    def reload(self) -> list[str]:
        """Read every run again; returns the run names."""
        names = run_names(self.dirs)
        fresh = {n: load_run(d, n) for n, d in zip(names, self.dirs, strict=True)}
        with self._lock:
            self._runs = fresh
        return names

    def get(self, name: str) -> RunView | None:
        with self._lock:
            return self._runs.get(name)

    def summaries(self) -> list[dict[str, Any]]:
        """/api/runs, in command-line order."""
        with self._lock:
            runs = list(self._runs.values())
        return [run_summary(r) for r in runs]


__all__ = [
    "MAX_ROWS",
    "ROUND_2",
    "MemoryViewError",
    "MovementUnavailable",
    "RunSet",
    "RunView",
    "conflicts_view",
    "events_window",
    "fifo_windows",
    "journey_view",
    "load_run",
    "memories",
    "memory_rows",
    "memory_view",
    "movement_view",
    "run_detail",
    "run_dir_of",
    "run_names",
    "run_summary",
    "task_spans",
    "tasks",
]
