"""A run's events grouped by their ids and lookups, built once per run when the viewer loads
it: what the memory tab's marks (D96) and the movement view (D97) query.

    RunIndex.from_run(outputs, cluster, regions, tasks) -> RunIndex

The events are the trace's (C§7, ``run/trace.py``), as the converter wrote them: their ids
link lines directly and their lookups (``task``, ``mem``, absolute ``addr``, ``bank``,
``row``, ``region``, ``elems``, ``vals``) are done already. The index only puts each event
under its keys; it never counts lines or matches them by order. Every list keeps the trace's
time order (by ``t``, then as SNAX-GVSoC wrote them):

    elem[(container, flat)]  the grants, DMA beats, bursts and stalls whose ``elems`` hold
                             the element, in both memories
    word[(mem, addr)]        the same by bank word (``addr`` its first byte): a grant's or a
                             stall's word, every word a DMA beat or a burst covers
    beat[(task, mover, b)]   a streamer beat's grants, one per lane, by lane (``task`` the
                             streamer's)
    fire[(task, n)], push[(task, n)]   the accelerator's firings and pushes
    fired[(task, mover, b)]  the firings that name a reader's beat (several with a repeat;
                             ``task`` the accelerator's, the fire's own)
    folded[(task, n)]        the push whose ``[first, last]`` holds firing ``n``
    returns[t]               the reads whose data comes back in cycle ``t`` (``back``): a
                             grant's, a DMA beat's on L1, a burst's on L2
    dma[(id, i)]             a DMA beat on L1; ``burst[id]`` a copy's bursts
    stalls_bank[bank], stalls_port[port]   the stalls, each with its ``task``
    grants_port[port]        a crossbar port's grants
    csr[block]               the cores' CSR accesses per block (``acc_streamer``, ``acc``)
    start[task], done[task]  a component's task brackets; ``issue[id]`` a DMA copy

A beat's ``b`` and a firing's ``n`` count from 0 in each run of their component (snax-gvsoc
``docs/EVENTS.md``), and the converter puts on every line the task its component's
start/done bracket names, so ``task`` makes them unique. A full key named by two lines is a
converter bug: a RunError (``index.duplicate``) naming both. An event whose ``task`` is null
(a ``fifo`` line after a mover's last busy cycle, a core's access) is kept under its other
keys.

**The pairing** crosses components: a ``fire`` carries the accelerator's task, its operands'
and results' grants their streamers' tasks. It comes from the run's task list (``tasks.json``
in the flow folder, lower/tasks.py's ``TaskList``), whose ``start`` step launches an
accelerator task together with its streamer tasks; each task's component is its configure
step's. ``streamers[acc task] = {mover: streamer task}`` and ``accel[streamer task] = acc
task``. An accelerator task launched without a streamer task for one of its movers, or with
two for one, or with a streamer task on no mover of it, is a RunError (``index.tasks``)
naming the tasks: not expected, and never guessed.

**Marks** are the memory tab's per-word values, added by registration, as the memory passes
are: ``register_mark(name, level, fn, kind)`` with ``fn(index) -> {(mem, addr): value}``, the
lowest trace level that has the events it reads, and its kind (``count``: rows fold on equal
values; ``time``: rows fold when every value moves by one step, memory.py). ``marks_of(index)``
gives the marks a run has and, for each it lacks, the reason. The built-in ones:

    conflicts  (stall)  per L1 word the stalls that wanted it plus those in which it was the
                        word served (``by_addr``)
    arrival    (beat)   a word's first write in its memory (none: there from the start)
    use        (beat)   its first read after that by anything but the DMA (whose reads are
                        departures, movement.py's residency)
    wait       (beat)   use - arrival
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from snax_forge.lower.tasks import Configure, Start, TaskList
from snax_forge.run import Cluster, Region, RunError, RunOutputs
from snax_forge.run.cluster import DMA
from snax_forge.run.trace import LEVELS

Key = tuple[str, int]  # (mem, addr) of a word, or (container, flat) of an element
Event = dict[str, Any]
DMA_KINDS = ("dma", "burst")  # the DMA's accesses: an L1 beat and an AXI request


@dataclass
class RunIndex:
    """One run's events under their keys (module doc)."""

    level: str  # the trace level, off without a trace
    filter: list[str]  # the sources of the beat and stall lines
    cluster: Cluster
    regions: list[Region]
    events: list[Event] = field(default_factory=list)
    elem: dict[Key, list[Event]] = field(default_factory=lambda: defaultdict(list))
    word: dict[Key, list[Event]] = field(default_factory=lambda: defaultdict(list))
    beat: dict[tuple[str, str, int], dict[int, Event]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    fire: dict[tuple[str, int], Event] = field(default_factory=dict)
    fired: dict[tuple[str, str, int], list[Event]] = field(
        default_factory=lambda: defaultdict(list)
    )
    push: dict[tuple[str, int], Event] = field(default_factory=dict)
    folded: dict[tuple[str, int], Event] = field(default_factory=dict)
    returns: dict[int, list[Event]] = field(default_factory=lambda: defaultdict(list))
    dma: dict[tuple[int, int], Event] = field(default_factory=dict)
    burst: dict[int, list[Event]] = field(default_factory=lambda: defaultdict(list))
    stalls_bank: dict[int, list[Event]] = field(default_factory=lambda: defaultdict(list))
    stalls_port: dict[str, list[Event]] = field(default_factory=lambda: defaultdict(list))
    grants_port: dict[str, list[Event]] = field(default_factory=lambda: defaultdict(list))
    csr: dict[str, list[Event]] = field(default_factory=lambda: defaultdict(list))
    start: dict[str, Event] = field(default_factory=dict)
    done: dict[str, Event] = field(default_factory=dict)
    issue: dict[int, Event] = field(default_factory=dict)
    streamers: dict[str, dict[str, str]] = field(default_factory=dict)  # the pairing
    accel: dict[str, str] = field(default_factory=dict)

    def on(self, level: str) -> bool:
        """Traced at ``level`` or above."""
        return LEVELS.index(self.level) >= LEVELS.index(level)

    def word_of(self, addr: int) -> int:
        """The first byte of the bank word ``addr`` is in (L2 words are L1's, D13)."""
        w = self.cluster.bank_bytes
        return addr // w * w

    def words(self, addr: int, size: int) -> list[int]:
        """The bank words ``[addr, addr + size)`` covers."""
        w = self.cluster.bank_bytes
        return list(range(self.word_of(addr), addr + size, w))

    def mover_of(self, port: str) -> Any:
        """The mover a crossbar port is a lane of, or None (a core, AXI, the DMA)."""
        for m in self.cluster.accelerator.movers:
            if port in m.ports:
                return m
        return None

    @property
    def filtered(self) -> dict[str, Any] | None:
        """None when the beat lines cover the accelerator, its movers and the DMA; else the
        sources they cover."""
        a = self.cluster.accelerator
        whole = {a.name, *(m.name for m in a.movers), DMA}
        return None if whole <= set(self.filter) else {"sources": list(self.filter)}

    @classmethod
    def from_run(cls, outputs: RunOutputs, cluster: Cluster, regions: list[Region],
                 tasks: TaskList) -> RunIndex:  # fmt: skip
        """The index of a run (module doc); a RunError on a duplicate key or a task list
        that does not pair."""
        streamers = pairing(tasks, cluster)
        accel = {st: t for t, ms in streamers.items() for st in ms.values()}
        tr = outputs.trace
        if tr is None:
            return cls("off", [], cluster, list(regions), streamers=streamers, accel=accel)
        idx = cls(tr.level, list(tr.filter), cluster, list(regions), list(tr.events),
                  streamers=streamers, accel=accel)  # fmt: skip

        def once(table: dict, key: Any, e: Event, what: str) -> None:
            if key in table:
                raise RunError("index.duplicate", f"two lines name {what} {key}: "
                               f"{_line(table[key])} and {_line(e)}")  # fmt: skip
            table[key] = e

        for e in idx.events:
            k = e["k"]
            if k in ("grant", "stall", "dma", "burst"):
                for flat in e["elems"]:
                    if e["region"] is not None:
                        idx.elem[(e["region"], flat)].append(e)
                mem = e.get("mem", "l1")  # a stall is a crossbar line: L1
                span = (
                    idx.words(e["addr"], e["size"]) if k in DMA_KINDS else [idx.word_of(e["addr"])]
                )
                for a in span:
                    idx.word[(mem, a)].append(e)
            if k == "grant":
                idx.grants_port[e["src"]].append(e)
                m = idx.mover_of(e["src"])
                if m is not None and "b" in e:
                    once(idx.beat[(e["task"], m.name, e["b"])], m.ports.index(e["src"]), e,
                         f"{e['task']} {m.name} beat {e['b']} lane")  # fmt: skip
            elif k == "stall":
                idx.stalls_bank[e["bank"]].append(e)
                idx.stalls_port[e["src"]].append(e)
            elif k == "dma":
                once(idx.dma, (e["id"], e["i"]), e, "DMA beat")
            elif k == "burst":
                idx.burst[e["id"]].append(e)
            elif k == "fire":
                once(idx.fire, (e["task"], e["n"]), e, "firing")
                for mover, b in e["b"].items():
                    idx.fired[(e["task"], mover, b)].append(e)
            elif k == "push":
                once(idx.push, (e["task"], e["n"]), e, "push")
                for n in range(e["first"], e["last"] + 1):
                    once(idx.folded, (e["task"], n), e, "firing folded")
            elif k == "csr":
                idx.csr[e["block"]].append(e)
            elif k == "start":
                once(idx.start, e["task"], e, "task start")
            elif k == "done" and "id" not in e:
                once(idx.done, e["task"], e, "task done")
            elif k == "issue":
                once(idx.issue, e["id"], e, "DMA copy")
            if k in ("grant", "dma", "burst") and e.get("op") == "r":
                idx.returns[e["back"]].append(e)
        for table in (idx.elem, idx.word, idx.beat, idx.fired, idx.returns, idx.burst,
                      idx.stalls_bank, idx.stalls_port, idx.grants_port, idx.csr):  # fmt: skip
            table.default_factory = None  # type: ignore[attr-defined]
        return idx


def _line(e: Event) -> str:
    """A trace line as a refusal names it."""
    ids = {k: e[k] for k in ("task", "n", "b", "id", "i", "first", "last") if k in e}
    return f"(t {e['t']} {e['k']} {e['src']} {ids})"


def pairing(tasks: TaskList, cluster: Cluster) -> dict[str, dict[str, str]]:
    """Per accelerator task, the streamer task of each of its movers, from the start step
    that launches them together (module doc)."""
    acc = cluster.accelerator
    movers = [m.name for m in acc.movers]
    conf = {s.task_name: s for s in tasks.steps if isinstance(s, Configure)}
    out: dict[str, dict[str, str]] = {}
    for s in tasks.steps:
        if not isinstance(s, Start):
            continue
        launched = [conf[t] for t in s.tasks if t in conf]
        mine = [c.task_name for c in launched if c.type == "accel" and c.component == acc.name]
        if not mine:
            continue
        what = f"{tasks.name}: start {s.tasks}"
        if len(mine) > 1:
            raise RunError("index.tasks", f"{what} launches {len(mine)} tasks of {acc.name} "
                           f"({mine})")  # fmt: skip
        by: dict[str, list[str]] = {}
        for c in launched:
            if c.type != "streamer":
                continue
            if c.component not in movers:
                raise RunError("index.tasks", f"{what}: streamer task {c.task_name!r} is on "
                               f"{c.component}, no mover of {acc.name} ({movers})")  # fmt: skip
            by.setdefault(c.component, []).append(c.task_name)
        for m in movers:
            if len(by.get(m, [])) != 1:
                have = by.get(m, [])
                raise RunError("index.tasks", f"{what}: accelerator task {mine[0]!r} has "
                               f"{len(have)} streamer tasks for {m} ({have}), one expected")  # fmt: skip
        out[mine[0]] = {m: by[m][0] for m in movers}
    return out


# =============================================================================
# Marks
# =============================================================================

Marks = dict[Key, Any]  # (mem, addr) -> a value per word


@dataclass(frozen=True)
class Mark:
    name: str
    level: str  # the lowest trace level that has its events
    fn: Callable[[RunIndex], Marks]
    kind: str  # count or time (module doc)


MARKS: dict[str, Mark] = {}
KINDS = ("count", "time")


def register_mark(name: str, level: str, fn: Callable[[RunIndex], Marks],
                  kind: str = "time") -> None:  # fmt: skip
    """Add a mark (module doc); a name is registered once."""
    if name in MARKS:
        raise ValueError(f"mark {name!r} is registered already")
    if level not in LEVELS[1:] or kind not in KINDS:
        raise ValueError(f"a mark's level is one of {LEVELS[1:]} and its kind one of {KINDS}")
    MARKS[name] = Mark(name, level, fn, kind)


def marks_of(index: RunIndex) -> tuple[list[str], dict[str, str]]:
    """The marks a run has, in registration order, and the reason for each it lacks."""
    have, lack = [], {}
    for m in MARKS.values():
        if index.on(m.level):
            have.append(m.name)
        else:
            lack[m.name] = f"needs --trace {m.level}; this run was traced at {index.level}"
    return have, lack


def mark(index: RunIndex, name: str) -> Marks:
    """The values of a mark the run has; a ValueError with the reason otherwise."""
    if name not in MARKS:
        raise ValueError(f"marks must be one of {list(MARKS)}, got {name!r}")
    have, lack = marks_of(index)
    if name not in have:
        raise ValueError(f"marks {name}: {lack[name]}")
    return MARKS[name].fn(index)


def conflict_marks(index: RunIndex) -> Marks:
    out: Marks = defaultdict(int)
    for stalls in index.stalls_bank.values():
        for e in stalls:
            out[("l1", index.word_of(e["addr"]))] += 1
            out[("l1", index.word_of(e["by_addr"]))] += 1
    return dict(out)


def _is_write(e: Event) -> bool:
    return e.get("op") == "w"


def _is_use(e: Event) -> bool:
    """A read by anything but the DMA."""
    return e["k"] == "grant" and e["op"] == "r"


def arrival(index: RunIndex, key: Key) -> int | None:
    """A word's first write in its memory, or None (there from the start)."""
    return next((e["t"] for e in index.word.get(key, []) if _is_write(e)), None)


def first_use(index: RunIndex, key: Key) -> int | None:
    """A word's first read after its arrival by anything but the DMA, or None."""
    a = arrival(index, key)
    return next((e["t"] for e in index.word.get(key, [])
                 if _is_use(e) and (a is None or e["t"] > a)), None)  # fmt: skip


def _times(fn: Callable[[RunIndex, Key], int | None]) -> Callable[[RunIndex], Marks]:
    def marks(index: RunIndex) -> Marks:
        out = {key: fn(index, key) for key in index.word}
        return {k: v for k, v in out.items() if v is not None}

    return marks


def wait(index: RunIndex, key: Key) -> int | None:
    a, u = arrival(index, key), first_use(index, key)
    return None if a is None or u is None else u - a


register_mark("conflicts", "stall", conflict_marks, "count")
register_mark("arrival", "beat", _times(arrival))
register_mark("use", "beat", _times(first_use))
register_mark("wait", "beat", _times(wait))


__all__ = [
    "MARKS",
    "Mark",
    "Marks",
    "RunIndex",
    "arrival",
    "first_use",
    "mark",
    "marks_of",
    "pairing",
    "register_mark",
]
