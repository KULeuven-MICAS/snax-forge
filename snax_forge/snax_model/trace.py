"""Event trace of a SNAX-MODEL run (MOD8, D38, D39, D40).

What it is
----------
A separate, time-stamped log of what happened in a run, next to the profile
(profile.py). The profile comes from the components' own counters; the
trace is written independently, from the cycle's final wires in ``commit``,
and the tests check the two against each other (D38).

Levels, chosen per run with ``Cluster(trace=Trace(level))``:

* ``off``: nothing is recorded (same as no trace);
* ``task``: controller commands, block starts and dones, and the
  cycle-class intervals of every classified component;
* ``beat``: adds xbar grants and stalls (the L1 accesses), accelerator
  firings, DMA beats, controller polls and FIFO count changes.

Events (D39)
------------
Every event has ``t`` (cycle), ``k`` (kind) and ``src`` (emitting source),
plus the fields of its kind:

    task  cmd       pc, op, reg | block, value, mode, done, last  (t = first cycle)
    task  start     (t = cycle the start landed; busy from t + 1)
    task  done      (t = done_cycle: first cycle busy reads 0)
    beat  grant     port, mem, w, addr, banks, row
    beat  stall     port, mem, w, addr, banks, row, wider
    beat  fire      n (firing index)
    beat  dma_beat  side (src/dst), i (beat index), mem (l1/l2), addr
    beat  poll      block, value (busy as sampled)
    beat  fifo      lane, count (t = first cycle the count holds)

Timestamps: an action event carries the cycle it happens in; a state-change
event (``done``, ``fifo``) carries the first cycle the new state is visible.

``grant`` / ``stall`` are the L1 accesses: ``addr`` is the byte address of
lane 0 (lane i at ``addr + i * word_bytes``), ``banks`` and ``row`` name the
words, computed with the run's address map. There is no separate L1 access
event: the xbar serves every grant in the same cycle (D31), one access on
each bank of the group (D33).

Order inside a cycle (``finish`` sorts once, stably): state changes first,
then action events by phase (CONTROL, COMPUTE, REQUEST, ARBITRATE), then by
source in registration order (``sources``), then in emission order. Emission
order inside one source is fixed by the model (ports and lanes ascending,
DMA source beat before destination beat), so the log is identical with
skipping on and off.

Class intervals are not events: ``on_gap`` reports a skipped range only when
the component wakes, after other components' later events. They are kept per
component (``ClassLog`` in sched.py) and stored in ``intervals`` as
half-open runs ``[cls, start, stop)`` covering ``[0, total)``.

Why recording cannot change results
-----------------------------------
Hooks only write into the trace and into ``ClassLog.runs``; nothing in the
model reads either. Events are emitted in ``commit`` (or when a start lands,
which is itself a commit), so no hook adds a tick, a ``touch`` or a change to
``next_wake``, and the wake schedule is the same with tracing on or off.

New event kinds are added with ``register_event`` (principle 6).

Filter (D49)
------------
A beat-level run writes roughly a kilobyte per cycle, so a long run (ANC2,
M4) is unreadable and large. ``Trace(level, sources=..., window=...)`` drops
beat events outside the filter in ``emit``, and nowhere else:

* ``sources``: keep beat events of these sources only (component or FIFO
  names, as in ``sources``);
* ``window``: keep beat events in the half-open cycle range ``[a, b)``.

Task-level events (``cmd``, ``start``, ``done``) and the class intervals are
never filtered, so the skeleton of the run and every profile number stay
complete whatever the filter is. The filter changes what is written, never
what is simulated: emitting is still the last thing a commit does.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

from .sched import ClassLog, Phase

LEVELS = ("off", "task", "beat")

# Sort group of each kind inside a cycle: state changes first, then phases.
_STATE = -1


# =============================================================================
# Event kinds
# =============================================================================


@dataclass(frozen=True)
class Event:
    """Base of every event: cycle and source. Subclasses add their fields."""

    t: int
    src: str

    kind: ClassVar[str] = ""
    level: ClassVar[str] = "task"
    group: ClassVar[int] = _STATE  # sort group inside a cycle, see module doc

    def to_dict(self) -> dict[str, Any]:
        """Flat dict: t, k, src, then the kind's fields; None fields are left out."""
        d: dict[str, Any] = {"t": self.t, "k": self.kind, "src": self.src}
        for f in fields(self):
            if f.name in ("t", "src"):
                continue
            v = getattr(self, f.name)
            if v is None:
                continue
            d[f.name] = list(v) if isinstance(v, tuple) else v
        return d


EVENT_KINDS: dict[str, type[Event]] = {}


def register_event(cls: type[Event]) -> type[Event]:
    """Register an event kind by its ``kind`` name (usable as a decorator)."""
    if not cls.kind or cls.kind in EVENT_KINDS:
        raise ValueError(f"event kind {cls.kind!r} is empty or already registered")
    if cls.level not in LEVELS[1:]:
        raise ValueError(f"event kind {cls.kind!r}: level must be task or beat")
    EVENT_KINDS[cls.kind] = cls
    return cls


def event_from_dict(d: dict[str, Any]) -> Event:
    """Inverse of ``Event.to_dict``. Lists come back as tuples (e.g. ``banks``)."""
    cls = EVENT_KINDS.get(d.get("k", ""))
    if cls is None:
        raise ValueError(f"unknown event kind {d.get('k')!r}")
    kw = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k != "k"}
    return cls(**kw)


@register_event
@dataclass(frozen=True)
class Cmd(Event):
    """A controller command, from its first cycle ``t`` to ``last`` (D37).

    csr_write: ``reg`` (block.name), ``value`` written. csr_read: ``reg``,
    ``value`` read. wait: ``block``, ``mode``, ``done`` (the block's
    done_cycle when the wait ended).
    """

    pc: int = 0
    op: str = ""
    last: int = 0
    reg: str | None = None
    value: int | None = None
    block: str | None = None
    mode: str | None = None
    done: int | None = None

    kind: ClassVar[str] = "cmd"
    group: ClassVar[int] = int(Phase.CONTROL)


@register_event
@dataclass(frozen=True)
class Start(Event):
    """A start landed in ``t``: the block is busy from ``t + 1`` (unless zero work)."""

    kind: ClassVar[str] = "start"
    group: ClassVar[int] = int(Phase.CONTROL)


@register_event
@dataclass(frozen=True)
class Done(Event):
    """``t`` is the block's done_cycle: the first cycle ``busy`` reads 0."""

    kind: ClassVar[str] = "done"
    group: ClassVar[int] = _STATE


@register_event
@dataclass(frozen=True)
class Grant(Event):
    """A granted L1 access (one per port and cycle); ``banks`` / ``row`` name its words."""

    port: str = ""
    mem: str = "l1"
    w: bool = False
    addr: int = 0
    banks: tuple[int, ...] = ()
    row: int = 0

    kind: ClassVar[str] = "grant"
    level: ClassVar[str] = "beat"
    group: ClassVar[int] = int(Phase.ARBITRATE)


@register_event
@dataclass(frozen=True)
class Stall(Event):
    """A refused request; ``wider``: refused because a wider grant took a bank (D33)."""

    port: str = ""
    mem: str = "l1"
    w: bool = False
    addr: int = 0
    banks: tuple[int, ...] = ()
    row: int = 0
    wider: bool = False

    kind: ClassVar[str] = "stall"
    level: ClassVar[str] = "beat"
    group: ClassVar[int] = int(Phase.ARBITRATE)


@register_event
@dataclass(frozen=True)
class Fire(Event):
    """Accelerator firing number ``n`` of the current task (0-based)."""

    n: int = 0

    kind: ClassVar[str] = "fire"
    level: ClassVar[str] = "beat"
    group: ClassVar[int] = int(Phase.COMPUTE)


@register_event
@dataclass(frozen=True)
class DmaBeat(Event):
    """Beat ``i`` moved on the DMA's ``side`` (src = read, dst = write) in memory ``mem``."""

    side: str = "src"
    i: int = 0
    mem: str = "l1"
    addr: int = 0

    kind: ClassVar[str] = "dma_beat"
    level: ClassVar[str] = "beat"
    group: ClassVar[int] = int(Phase.REQUEST)


@register_event
@dataclass(frozen=True)
class Poll(Event):
    """A poll sample of ``block``'s ``busy`` register in ``t`` (D37)."""

    block: str = ""
    value: int = 0

    kind: ClassVar[str] = "poll"
    level: ClassVar[str] = "beat"
    group: ClassVar[int] = int(Phase.CONTROL)


@register_event
@dataclass(frozen=True)
class FifoCount(Event):
    """FIFO ``lane`` holds ``count`` elements from cycle ``t`` on."""

    lane: int = 0
    count: int = 0

    kind: ClassVar[str] = "fifo"
    level: ClassVar[str] = "beat"
    group: ClassVar[int] = _STATE


# =============================================================================
# The trace
# =============================================================================


@dataclass
class Trace:
    """Events and class intervals of one run, see the module doc.

    Before the run: ``Cluster(trace=Trace(level))``. The cluster calls
    ``bind`` before and ``finish`` after the scheduler runs. ``events`` is in
    the canonical order only after ``finish``.
    """

    level: str = "task"
    total_cycles: int | None = None
    sources: list[str] = field(default_factory=list)
    intervals: dict[str, list[tuple[str, int, int]]] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    filter_sources: list[str] | None = None  # beat events: these sources only
    filter_window: tuple[int, int] | None = None  # beat events: cycles [a, b)

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"trace level must be one of {LEVELS}, got {self.level!r}")
        if self.filter_sources is not None:
            self.filter_sources = list(self.filter_sources)
            if not self.filter_sources:
                raise ValueError("trace filter: sources must name at least one source")
        if self.filter_window is not None:
            a, b = (int(x) for x in self.filter_window)
            if a < 0 or b < a:
                raise ValueError(f"trace filter: window [{a}, {b}) is empty or negative")
            self.filter_window = (a, b)
        self._keep = None if self.filter_sources is None else set(self.filter_sources)
        self._logs: dict[str, ClassLog] = {}

    # -- levels (checked by the hooks before building an event) -----------------

    @property
    def task(self) -> bool:
        return self.level in ("task", "beat")

    @property
    def beat(self) -> bool:
        return self.level == "beat"

    # -- recording ----------------------------------------------------------------

    def bind(self, cluster: Any) -> None:
        """Attach to every component of ``cluster`` and to their shared elements.

        Duck-typed on purpose (no imports of component classes): a component
        gets ``_trace``; a ``fifo`` attribute (streamers) gets it too; a
        ``cycles`` ClassLog starts recording runs. Sources are listed in
        registration order, a streamer's FIFO right after it. Idempotent.
        """
        if not self.task:
            return
        origin = cluster.cycle
        for comp in cluster:
            comp._trace = self
            self._source(comp.name)
            fifo = getattr(comp, "fifo", None)
            if fifo is not None and hasattr(fifo, "occupancy"):
                fifo._trace = self
                self._source(fifo.name)
            log = getattr(comp, "cycles", None)
            if isinstance(log, ClassLog):
                log.record(origin)
                self._logs[comp.name] = log

    def _source(self, name: str) -> None:
        if name not in self.sources:
            self.sources.append(name)

    def emit(self, ev: Event) -> None:
        """Record one event, unless a filter drops it (module doc, D49).

        The hook has already checked the level. Only beat events are
        filtered: the task-level skeleton is always complete.
        """
        if ev.level == "beat":
            if self._keep is not None and ev.src not in self._keep:
                return
            w = self.filter_window
            if w is not None and not (w[0] <= ev.t < w[1]):
                return
        self.events.append(ev)

    def finish(self, total: int) -> None:
        """After the run: copy the class runs and sort the events (module doc)."""
        self.total_cycles = total
        self.intervals = {
            name: [(str(c), int(a), int(b)) for c, a, b in log.runs or ()]
            for name, log in self._logs.items()
        }
        rank = {s: i for i, s in enumerate(self.sources)}
        # Stable sort: emission order breaks the remaining ties.
        self.events.sort(key=lambda e: (e.t, e.group, rank.get(e.src, len(rank))))

    # -- queries used by tests and views ------------------------------------------

    def of_kind(self, kind: str) -> list[Event]:
        return [e for e in self.events if e.kind == kind]

    # -- serialisation (D26: plain dicts until the M6 freeze) ---------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "total_cycles": self.total_cycles,
            "filter_sources": None if self.filter_sources is None else list(self.filter_sources),
            "filter_window": None if self.filter_window is None else list(self.filter_window),
            "sources": list(self.sources),
            "intervals": {k: [list(r) for r in v] for k, v in self.intervals.items()},
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trace:
        window = d.get("filter_window")
        return cls(
            level=d["level"],
            total_cycles=d.get("total_cycles"),
            filter_sources=d.get("filter_sources"),
            filter_window=None if window is None else (int(window[0]), int(window[1])),
            sources=list(d.get("sources", [])),
            intervals={
                k: [(str(c), int(a), int(b)) for c, a, b in v]
                for k, v in d.get("intervals", {}).items()
            },
            events=[event_from_dict(e) for e in d.get("events", [])],
        )
