"""What the viewer asks for, as plain Python functions (VIS1, D54, D55, D56).

What this is
------------
The server (server.py) is a thin HTTP layer; everything it answers comes
from the functions here, so the tests call them directly and the server
only turns their results into JSON. Nothing here draws anything: the
viewer (static/) formats the numbers, and HTML is checked by eye (D55).

Loading (D50, D54)
------------------
A run is one model output directory (``python -m snax_forge.snax_model run
... --out DIR``). It is read with ``snax_model.scenario.read_outputs``,
exactly as a test would, and its cluster configuration is rebuilt with
``ClusterConfig.from_dict(run["cluster"])`` from run.json (D50), so the views
use the model's own dataclasses and never parse the files themselves (D54).
Everything is read once, at start-up or on an explicit reload (``RunSet``);
nothing watches the files.

A run is named after its directory. Two directories with the same last
component get ``-2``, ``-3``, ... in the order given, so names stay stable
across reloads.

Events
------
The trace events are kept as the flat dicts of trace.jsonl, sorted by
cycle (``Trace.finish`` sorts them, D39), with their cycles in a separate
list so a window ``[a, b)`` is two bisects. The source filter is applied
after the window, on the few events left.

Memory layout (D96)
-------------------
The memory tab's rows are computed in memory.py from the run's regions
(D95) and its cluster, once per memory on first use, and folded so the
answer grows with the regions, not with the depth; folded rows come on
request, at most ``memory.MAX_ROWS`` at a time.

FIFO busy window (D56)
----------------------
The profile's FIFO statistics are over the whole run (D40), which dilutes
the mean of a FIFO that is used for a tenth of the run. The busy window of
a streamer's FIFO is the union over tasks of ``[first start, last done)``
of the streamer and the accelerators attached to it (``attach`` in the
cluster config): the k-th task of each owner is taken together. Outside
the window the FIFO is empty, so the window histogram is the run histogram
with the count-0 bucket reduced by the cycles outside the window; nothing
is recounted. It needs the task-level ``start`` / ``done`` events, so a run
traced at level ``off`` gets the whole-run numbers only. Task events are
never filtered (D49), so a filtered beat trace is as good as a task trace
here.

If the owners have different task counts, the extra tasks are windows on
their own. If a window would leave out a cycle in which the FIFO held
something (the count-0 bucket would go negative), that streamer's window
numbers are withheld with a reason rather than reported wrong.
"""

from __future__ import annotations

import threading
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from snax_forge.snax_model.dma import DIRECTIONS
from snax_forge.snax_model.scenario import ClusterConfig, Outputs, read_outputs

from . import memory, movement
from .memory import MemoryLayout, MemoryViewError
from .movement import MovementUnavailable

# =============================================================================
# One run
# =============================================================================


@dataclass
class RunView:
    """One output directory, read back into the model's dataclasses."""

    name: str
    path: Path
    outputs: Outputs
    cluster: ClusterConfig
    events: list[dict[str, Any]] = field(default_factory=list)  # sorted by t
    times: list[int] = field(default_factory=list)  # events[i]["t"], for bisect
    spans: dict[str, list[tuple[int, int]]] = field(default_factory=dict)  # task_spans, per block
    layouts: dict[str, MemoryLayout] = field(default_factory=dict, repr=False)  # per memory, lazy
    moves: Any = field(default=None, repr=False)  # movement.Index, or why it has none; lazy

    @property
    def total_cycles(self) -> int:
        return int(self.outputs.run["total_cycles"])

    @property
    def trace_level(self) -> str:
        return str(self.outputs.run["trace_level"])


def load_run(path: str | Path, name: str | None = None) -> RunView:
    """Read one output directory (module doc). Raises what read_outputs raises."""
    p = Path(path)
    out = read_outputs(p)
    cluster = ClusterConfig.from_dict(out.run["cluster"])
    events = [] if out.trace is None else [e.to_dict() for e in out.trace.events]
    return RunView(
        name=name or p.resolve().name,
        path=p,
        outputs=out,
        cluster=cluster,
        events=events,
        times=[int(e["t"]) for e in events],
        spans=_pair_tasks(events, int(out.run["total_cycles"])),
    )


def run_names(dirs: Sequence[str | Path]) -> list[str]:
    """Directory names, made unique with -2, -3, ... in the order given. A flow's ``run/``
    (a ``run`` directory beside its ``scenario.json``) takes the name of its flow folder,
    so ``out/flow/vecadd_W8/run`` is ``vecadd_W8`` (D94)."""
    names: list[str] = []
    for d in dirs:
        p = Path(d).resolve()
        flow = p.name == "run" and (p.parent / "scenario.json").is_file()
        base = (p.parent.name if flow else p.name) or "run"
        name, i = base, 1
        while name in names:
            i += 1
            name = f"{base}-{i}"
        names.append(name)
    return names


# =============================================================================
# Memory layout (VIS4a, D96)
# =============================================================================


def memories(rv: RunView) -> list[str]:
    """The run's memories, L1 first."""
    return ["l1"] + (["l2"] if rv.cluster.l2 is not None else [])


def _layout(rv: RunView, mem: str) -> MemoryLayout:
    if mem not in memories(rv):
        raise MemoryViewError(f"run {rv.name!r} has no memory {mem!r}")
    if mem not in rv.layouts:  # computed once per load; the same answer if two threads race
        rv.layouts[mem] = memory.layout_of(mem, rv.cluster, rv.outputs.regions)
    return rv.layouts[mem]


MARKS = ("conflicts",)


def _marks(rv: RunView, mem: str, marks: str | None) -> memory.Marks | None:
    """The fold marks asked for: ``conflicts`` gives L1's conflict counts per word (D97)."""
    if marks is None:
        return None
    if marks not in MARKS:
        raise MemoryViewError(f"marks must be one of {list(MARKS)}, got {marks!r}")
    return movement.conflict_marks(moves(rv)) if mem == "l1" else {}


def memory_view(rv: RunView, marks: str | None = None) -> dict[str, Any]:
    """/api/run/<name>/memory: per memory its geometry, regions, use and folded lines
    (memory.py). A run without regions gets the geometry and ``has_regions`` false.
    ``marks="conflicts"`` folds L1 with its conflict counts (needs a beat trace)."""
    return {
        "name": rv.name,
        "has_regions": bool(rv.outputs.regions),
        "marks": marks,
        "memories": [memory.summary(_layout(rv, m), _marks(rv, m, marks)) for m in memories(rv)],
    }


def memory_rows(
    rv: RunView, mem: str, start: int, stop: int, marks: str | None = None
) -> dict[str, Any]:
    """/api/run/<name>/memory/<mem>/rows: rows ``start <= r < stop`` in full."""
    rows = memory.rows(_layout(rv, mem), start, stop, _marks(rv, mem, marks))
    return {"mem": mem, "from": start, "to": stop, "rows": rows}


# =============================================================================
# Data movement (VIS4b, D97)
# =============================================================================


def moves(rv: RunView) -> movement.Index:
    """The run's movement index, built once per load; raises MovementUnavailable with the
    reason when the trace cannot give one."""
    if rv.moves is None:
        try:
            rv.moves = movement.build_index(
                rv.events,
                rv.cluster,
                _ports(rv),
                {m: _layout(rv, m) for m in memories(rv)},
                rv.spans,
                rv.outputs.trace,
            )
        except MovementUnavailable as e:
            rv.moves = e
    if isinstance(rv.moves, MovementUnavailable):
        raise rv.moves
    return rv.moves


def _ports(rv: RunView) -> dict[str, dict[str, Any]]:
    """The profile's xbar ports as plain dicts (owner, width, counts)."""
    return rv.outputs.profile.to_dict()["ports"]


def _dmas(rv: RunView) -> set[str]:
    return {c.name for c in rv.cluster.components if c.kind == "dma"}


def _unavailable(rv: RunView, e: MovementUnavailable) -> dict[str, Any]:
    return {"name": rv.name, "available": False, "reason": str(e)}


def movement_view(rv: RunView) -> dict[str, Any]:
    """/api/run/<name>/movement: residency per region and memory, patterns per port and
    task, and the conflict counts, or ``available`` false with the reason."""
    try:
        idx = moves(rv)
    except MovementUnavailable as e:
        return _unavailable(rv, e)
    c = movement.conflicts(idx)
    return {
        "name": rv.name,
        "available": True,
        "filtered": idx.filtered,
        "residency": movement.residency(idx, rv.outputs.regions, _dmas(rv)),
        "patterns": movement.patterns(idx, rv.spans, _ports(rv)),
        "conflicts": {k: c[k] for k in ("count", "banks", "ports")},
    }


def journey_view(rv: RunView, region: str, index: Sequence[int] | int) -> dict[str, Any]:
    """/api/run/<name>/journey?region=A&index=5: every hop of one element (movement.py)."""
    try:
        idx = moves(rv)
    except MovementUnavailable as e:
        return _unavailable(rv, e)
    return {
        "name": rv.name,
        "available": True,
        **movement.journey(idx, rv.outputs.regions, region, index),
    }


def conflicts_view(rv: RunView, start: int = 0, stop: int | None = None) -> dict[str, Any]:
    """/api/run/<name>/conflicts?from=A&to=B: the L1 conflicts placed on the layout."""
    try:
        idx = moves(rv)
    except MovementUnavailable as e:
        return _unavailable(rv, e)
    return {"name": rv.name, "available": True, **movement.conflicts(idx, start, stop)}


# =============================================================================
# Answers of the API routes
# =============================================================================


def run_summary(rv: RunView) -> dict[str, Any]:
    """One entry of /api/runs."""
    return {
        "name": rv.name,
        "path": str(rv.path),
        "scenario": rv.outputs.run.get("scenario"),
        "trace_level": rv.trace_level,
        "total_cycles": rv.total_cycles,
    }


def run_detail(rv: RunView) -> dict[str, Any]:
    """/api/run/<name>: run.json, the profile and the class intervals.

    ``trace`` holds what the header needs about the trace (level, D49 filter,
    sources, event count) and the class intervals, or is None at level off.
    """
    tr = rv.outputs.trace
    trace = None
    if tr is not None:
        trace = {
            "level": tr.level,
            "filter_sources": tr.filter_sources,
            "filter_window": None if tr.filter_window is None else list(tr.filter_window),
            "sources": list(tr.sources),
            "n_events": len(rv.events),
            "intervals": {k: [list(r) for r in v] for k, v in tr.intervals.items()},
        }
    return {
        "name": rv.name,
        "path": str(rv.path),
        "run": rv.outputs.run,
        "profile": rv.outputs.profile.to_dict(),
        "trace": trace,
        "tasks": tasks(rv),
    }


def tasks(rv: RunView) -> dict[str, list[dict[str, Any]]]:
    """Per block that ran a task: its tasks as ``start`` / ``done``, for the schedule (D60).

    A DMA's tasks also carry their ``direction`` (D58): the last
    ``csr_write`` to ``<dma>.direction`` that ended before the start landed,
    since a start copies the buffered registers (D36); registers reset to
    0, so with no write it is ``DIRECTIONS[0]``. Needs the task events, so
    at level off the answer is empty.
    """
    dmas = {c.name for c in rv.cluster.components if c.kind == "dma"}
    writes: dict[str, list[tuple[int, int]]] = {d: [] for d in dmas}  # (last, value)
    for e in rv.events:
        if e["k"] == "cmd" and e.get("op") == "csr_write":
            block, _, reg = str(e.get("reg") or "").partition(".")
            if reg == "direction" and block in writes:
                writes[block].append((int(e["last"]), int(e["value"])))
    out: dict[str, list[dict[str, Any]]] = {}
    for block, spans in rv.spans.items():
        out[block] = []
        for start, done in spans:
            task: dict[str, Any] = {"start": start, "done": done}
            if block in dmas:
                k = 0
                for last, value in writes[block]:  # in trace order, so the last one wins
                    if last < start:
                        k = value
                task["direction"] = DIRECTIONS[k] if 0 <= k < len(DIRECTIONS) else f"direction {k}"
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

    Events keep their trace order (D39). ``stop`` defaults to the end of the
    run. ``kinds`` lets the viewer ask for the task events or the FIFO counts
    of the whole run, once per run, without the beat events in between
    (D57, D61).
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


# =============================================================================
# FIFO busy window (D56)
# =============================================================================


def _pair_tasks(events: Sequence[dict[str, Any]], total: int) -> dict[str, list[tuple[int, int]]]:
    """``(start, done)`` of every task, per block, in one pass over the events.

    A start is closed by the next ``done`` of the same block; a zero-work
    start has its done in the next cycle (the model emits it), and a task
    still open at the end is closed at the run's total.
    """
    spans: dict[str, list[tuple[int, int]]] = {}
    open_at: dict[str, int] = {}
    for e in events:
        src, k = e["src"], e["k"]
        if k == "start":
            if src in open_at:  # cannot happen in a model run (start while busy)
                spans[src].append((open_at[src], e["t"]))
            open_at[src] = e["t"]
            spans.setdefault(src, [])
        elif k == "done" and src in open_at:
            spans[src].append((open_at.pop(src), e["t"]))
    for src, t in open_at.items():
        spans[src].append((t, total))
    return spans


def task_spans(rv: RunView, block: str) -> list[tuple[int, int]]:
    """``(start, done)`` of every task of ``block`` (computed once, at load)."""
    return rv.spans.get(block, [])


def owners_of(cluster: ClusterConfig, streamer: str) -> list[str]:
    """The streamer and every accelerator attached to it, in component order."""
    accels = [c.name for c in cluster.components if streamer in c.attach.values()]
    return [streamer, *accels]


def merge(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of half-open ranges, sorted and merged (touching ranges join)."""
    out: list[tuple[int, int]] = []
    for a, b in sorted(s for s in spans if s[1] > s[0]):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def busy_window(rv: RunView, streamer: str) -> list[tuple[int, int]]:
    """The FIFO busy window of ``streamer``: task k of every owner taken together."""
    per_owner = [task_spans(rv, o) for o in owners_of(rv.cluster, streamer)]
    n_tasks = max((len(s) for s in per_owner), default=0)
    spans = []
    for k in range(n_tasks):
        kth = [s[k] for s in per_owner if k < len(s)]
        spans.append((min(a for a, _ in kth), max(b for _, b in kth)))
    return merge(spans)


def fifo_windows(rv: RunView) -> dict[str, Any]:
    """/api/run/<name>/fifo: per streamer, its FIFO over the busy window (module doc).

    Per lane: ``max`` (the run's: the FIFO is empty outside the window),
    ``mean`` over the window and ``hist`` over the window. With no task
    events the answer says why and holds no streamers.
    """
    if rv.outputs.trace is None:
        return {
            "available": False,
            "reason": "trace level off: the busy window needs task-level start and done events",
            "streamers": {},
        }
    total = rv.total_cycles
    result: dict[str, Any] = {}
    for name, sp in rv.outputs.profile.streamers.items():
        f = sp.fifo
        window = busy_window(rv, name)
        cycles = sum(b - a for a, b in window)
        outside = total - cycles
        entry: dict[str, Any] = {
            "fifo": f.name,
            "owners": owners_of(rv.cluster, name),
            "window": [list(w) for w in window],
            "window_cycles": cycles,
            "reason": None,
            "lanes": [],
        }
        bad = [lane for lane, h in enumerate(f.hist) if h and h[0] < outside]
        if bad:
            entry["reason"] = (
                f"lanes {bad} hold data outside the window: the owners' tasks do not "
                f"pair up, so no window numbers"
            )
        else:
            for h, mx in zip(f.hist, f.max, strict=True):
                wh = list(h)
                if wh:
                    wh[0] -= outside
                mean = sum(c * n for c, n in enumerate(wh)) / cycles if cycles else 0.0
                entry["lanes"].append({"max": mx, "mean": mean, "hist": wh})
        result[name] = entry
    return {"available": True, "reason": None, "streamers": result}


# =============================================================================
# The set of runs a server shows
# =============================================================================


class RunSet:
    """The run directories given on the command line, loaded once (D55).

    ``reload`` reads every directory again and swaps the whole set at once,
    so a request never sees half of a reload. If a directory cannot be read,
    the old set stays and the error is raised to the caller.
    """

    def __init__(self, dirs: Sequence[str | Path]) -> None:
        if not dirs:
            raise ValueError("give at least one run directory")
        self.dirs = [Path(d) for d in dirs]
        self._lock = threading.Lock()
        self._runs: dict[str, RunView] = {}
        self.reload()

    def reload(self) -> list[str]:
        """Read every directory again; returns the run names."""
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
    "RunSet",
    "RunView",
    "busy_window",
    "conflicts_view",
    "events_window",
    "fifo_windows",
    "journey_view",
    "load_run",
    "memories",
    "memory_rows",
    "memory_view",
    "merge",
    "movement_view",
    "owners_of",
    "run_detail",
    "run_names",
    "run_summary",
    "task_spans",
    "tasks",
]
