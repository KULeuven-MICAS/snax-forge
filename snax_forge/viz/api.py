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
never parse the files themselves (D55). The design point gives the regions
(``run.regions.regions_of``, L2 from run.json's ``elf_symbols``) and the
events are grouped once by their ids and lookups (index.py, ``RunIndex``),
with the flow folder's task list (``tasks.json``) pairing each accelerator
task with its streamer tasks.
Everything is read once, at start-up or on an explicit reload (``RunSet``);
nothing watches the files.

A run is named after its flow folder. Two folders with the same name get
``-2``, ``-3``, ... in the order given, so names stay stable across reloads.

Events
------
The trace events are the index's (``RunIndex.events``), sorted by cycle; a route answers
from the index's queries (schedule.py's ``window`` for a range of cycles).

Memory
------
The memory tab's rows are computed in memory.py from the run's regions and
its cluster, once per memory on first use, and folded so the answer grows
with the regions; folded rows are fetched on request, at most
``memory.MAX_ROWS`` at a time. A mark (index.py's registry) is computed once
per run on first use; the view lists the marks the run has and why it lacks
the others. The cycle overlay of the tab asks for one cycle: its accesses
and, from the index's ``returns``, every read whose data comes back in it,
however many cycles after its request.

Movement
--------
The movement answers (journey, residency, patterns, conflicts) are
movement.py's queries on the index; one a run's level cannot give is
``available`` false with the reason.

Schedule and cluster
--------------------
The schedule (lanes, folded segments, detail rows) is schedule.py's queries
on the index, the cluster view at a cycle and the stepping between cycles
cluster.py's (D57); a run below the level an answer needs gets ``available``
false with the reason. The viewer draws these answers and computes nothing
of its own.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from snax_forge.common import unique_names
from snax_forge.design import DesignPoint
from snax_forge.lower.tasks import TaskList
from snax_forge.run import Cluster, Region, RunError, RunOutputs, load_cluster, read_run, regions_of

from . import cluster, memory, movement, schedule
from . import index as ix
from .index import RunIndex
from .memory import MemoryLayout, MemoryViewError, RowMarks
from .movement import MovementUnavailable
from .schedule import ScheduleUnavailable

OVERLAY_KINDS = ix.ACCESS_KINDS  # what the memory tab's cycle overlay draws: the accesses

MAX_ROWS = memory.MAX_ROWS  # memory rows per request (static/memory.js)


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
    point: DesignPoint
    regions: list[Region]
    index: RunIndex
    layouts: dict[str, MemoryLayout] = field(default_factory=dict, repr=False)  # lazy, per memory
    marks: dict[str, Any] = field(default_factory=dict, repr=False)  # lazy, per mark

    @property
    def events(self) -> list[dict[str, Any]]:
        """The trace events, sorted by cycle (the index's)."""
        return self.index.events

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
    if not (folder / "tasks.json").is_file():
        raise RunError("outputs.missing", f"{folder}: no tasks.json, the run's task list")
    cluster = load_cluster(folder / hjson[0], folder / out.run["design_point"])
    point = DesignPoint.load(folder / out.run["design_point"])
    regions = regions_of(point, out.elf_symbols)
    task_list = TaskList.load(folder / "tasks.json")
    return RunView(
        name=name or folder.resolve().name,
        path=run_dir,
        folder=folder,
        outputs=out,
        cluster=cluster,
        point=point,
        regions=regions,
        index=RunIndex.from_run(out, cluster, regions, task_list),
    )


def run_names(dirs: Sequence[str | Path]) -> list[str]:
    """Flow folder names (a ``run/`` directory takes its folder's), made unique with -2, -3,
    ... in the order given."""

    def base(p: Path) -> str:
        return (
            p.parent.name if p.name == "run" and (p / "run.json").is_file() else p.name
        ) or "run"

    return unique_names(base(Path(d).resolve()) for d in dirs)


# =============================================================================
# The memory tab (memory.py on index.py's marks)
# =============================================================================


def memories(rv: RunView) -> list[str]:
    """The run's memories, L1 first (L2 when the platform has one)."""
    return ["l1"] + (["l2"] if rv.point.platform.l2 is not None else [])


def _layout(rv: RunView, mem: str) -> MemoryLayout:
    if mem not in memories(rv):
        raise MemoryViewError(f"run {rv.name!r} has no memory {mem!r}")
    if mem not in rv.layouts:  # computed once per load; the same answer if two threads race
        rv.layouts[mem] = memory.layout_of(mem, rv.cluster, rv.point, rv.regions)
    return rv.layouts[mem]


def _marks(rv: RunView, mem: str, marks: str | None) -> RowMarks | None:
    """The mark asked for on ``mem``'s rows (index.py's registry); a MemoryViewError with the
    reason when the name is unknown or the run lacks it."""
    if marks is None:
        return None
    if marks not in rv.marks:
        try:
            rv.marks[marks] = ix.mark(rv.index, marks)
        except ValueError as e:
            raise MemoryViewError(str(e)) from None
    return memory.row_marks(_layout(rv, mem), rv.marks[marks], ix.MARKS[marks].kind)


def memory_view(rv: RunView, marks: str | None = None) -> dict[str, Any]:
    """/api/run/<name>/memory: per memory its geometry, regions, use and folded lines
    (memory.py), the marks the run has (``marks_available``, with each one's kind) and the
    reason for each it lacks (``marks_missing``). ``marks`` folds by one of them."""
    have, lack = ix.marks_of(rv.index)
    return {
        "name": rv.name,
        "has_regions": bool(rv.regions),
        "marks": marks,
        "marks_available": {m: ix.MARKS[m].kind for m in have},
        "marks_missing": lack,
        "memories": [memory.summary(_layout(rv, m), _marks(rv, m, marks)) for m in memories(rv)],
    }


def memory_rows(rv: RunView, mem: str, start: int, stop: int, marks: str | None = None
                ) -> dict[str, Any]:  # fmt: skip
    """/api/run/<name>/memory/<mem>/rows: rows ``start <= r < stop`` in full."""
    rows = memory.rows(_layout(rv, mem), start, stop, _marks(rv, mem, marks))
    return {"mem": mem, "from": start, "to": stop, "rows": rows}


def memory_cycle(rv: RunView, t: int) -> dict[str, Any]:
    """/api/run/<name>/memory/cycle?t=T: what the cycle overlay draws in cycle ``t``: the
    grants, stalls, DMA beats and bursts of the cycle (``events``) and the reads whose data
    comes back in it (``returns``, index.py), both as trace events."""
    return {"t": t, "level": rv.index.level,
            "events": [e for e in schedule.window(rv.index, t, t + 1)
                       if e["k"] in OVERLAY_KINDS],
            "returns": rv.index.returns.get(t, [])}  # fmt: skip


# =============================================================================
# Data movement (movement.py on index.py)
# =============================================================================


def _answer(rv: RunView, make: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """``make()``'s answer marked ``available``, or ``available`` false with the reason the
    run's trace cannot give it (a movement answer also says which sources it has)."""
    try:
        return {"name": rv.name, "available": True, **make()}
    except MovementUnavailable as e:
        return {"name": rv.name, "available": False, "reason": str(e),
                "filtered": rv.index.filtered}  # fmt: skip
    except ScheduleUnavailable as e:
        return {"name": rv.name, "available": False, "reason": str(e)}


def movement_view(rv: RunView) -> dict[str, Any]:
    """Residency per region and memory, patterns per port and
    task, and the conflict counts (``available`` false with the reason below ``stall``), or
    ``available`` false with the reason below ``beat``."""
    idx = rv.index

    def make() -> dict[str, Any]:
        residency, patterns = movement.residency(idx), movement.patterns(idx)
        try:
            c = movement.conflicts(idx)
            counts = {"available": True, **{k: c[k] for k in ("count", "banks", "ports")}}
        except MovementUnavailable as e:
            counts = {"available": False, "reason": str(e)}
        return {"level": idx.level, "filtered": idx.filtered, "residency": residency,
                "patterns": patterns, "conflicts": counts}  # fmt: skip

    return _answer(rv, make)


def journey_view(rv: RunView, region: str, index: Sequence[int] | int) -> dict[str, Any]:
    """/api/run/<name>/journey?region=A&index=5: every access to one element and the
    firings it fed or came from (movement.py); a ValueError for an element the run lacks."""
    return _answer(rv, lambda: movement.journey(rv.index, region, index))


def conflicts_view(rv: RunView, start: int = 0, stop: int | None = None) -> dict[str, Any]:
    """/api/run/<name>/conflicts?from=A&to=B: the L1 stalls placed on the layout."""
    return _answer(rv, lambda: movement.conflicts(rv.index, start, stop))


# =============================================================================
# The schedule and the cluster view (schedule.py, cluster.py on index.py)
# =============================================================================


def schedule_view(rv: RunView, open: Sequence[int] = (), cycle: int | None = None
                  ) -> dict[str, Any]:  # fmt: skip
    """/api/run/<name>/schedule?open=A,B&cycle=C: the lanes and the segments, the folds whose
    first cycle is in ``open`` or that hold ``cycle`` opened; ``available`` false with the
    reason below ``task``."""
    return _answer(rv, lambda: schedule.schedule(rv.index, open, cycle))


def schedule_detail(rv: RunView, start: int, stop: int) -> dict[str, Any]:
    """/api/run/<name>/schedule/detail?from=A&to=B: the beat and stall lines of ``[A, B)``
    per lane; ``available`` false with the reason below ``beat``."""
    return _answer(rv, lambda: schedule.detail(rv.index, start, stop))


def cluster_view(rv: RunView, t: int) -> dict[str, Any]:
    """/api/run/<name>/cluster?t=T: the data side of the cluster at cycle ``T``."""
    return {"name": rv.name, **cluster.cluster_at(rv.index, t)}


def step_view(rv: RunView, t: int, dir: int, by: str) -> dict[str, Any]:
    """/api/run/<name>/step?t=T&dir=D&by=B: the next (D 1) or previous (-1) cycle with an
    event (B ``event``) or a task boundary (``boundary``); T at the window's ends."""
    return {"name": rv.name, "from": t, "dir": dir, "by": by,
            "t": cluster.step(rv.index, t, dir, by)}  # fmt: skip


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
    """/api/run/<name>: run.json, the profile, the cluster (run/cluster.py) and what the
    header needs about the trace (level, boot, window, sources, event count; None at off)."""
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
    }


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
    "MemoryViewError",
    "MovementUnavailable",
    "RunSet",
    "RunView",
    "ScheduleUnavailable",
    "cluster_view",
    "conflicts_view",
    "journey_view",
    "load_run",
    "memories",
    "memory_cycle",
    "memory_rows",
    "memory_view",
    "movement_view",
    "run_detail",
    "run_dir_of",
    "run_names",
    "run_summary",
    "schedule_detail",
    "schedule_view",
    "step_view",
]
