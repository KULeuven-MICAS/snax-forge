"""The trace of a flow run (C§7, D137).

SNAX-GVSoC's ``events`` lines (``beats``, ``stalls`` and the cores' ``insn`` above ``task``),
renamed into C§7 and cut to the program's window: t = 0 is the program's ``mcycle`` read in
main on the accelerator's core, and events outside ``[0, total_cycles)`` (up to its read in
``forge_check``) are dropped. Every event is a flat dict of ``t``, ``k``, ``src`` and the
kind's own fields, by the lowest level that has it:

    task         start    task  (t = the cycle the start is taken; busy from t + 1)
    task         done     task, cycles (the run's classes), then firings, first, last for an
                          accelerator  (t = first idle)
    task         csr      pc, csr, block, op (r/w), value, held (a held write's cycles)
    task         poll     pc, csr, block, value, count, last  (t = the first read)
    task         barrier  last  (t = the barrier's read; last = the cycle it is released)
    task         state    mode (run/sleep)  (t = the first cycle in it)
    task         func     name, pc  (t = the function's first instruction)
    task         issue    task, id, dir, from, to, size, from_stride, to_stride, reps  (a copy)
    task         done     task, id  (a DMA copy; t = the cycle it completes)
    beat         grant    task, op, mem, addr, bank, row, value, back, region, elems, vals;
                          b (a SNAX port's), size (under a word)  (src = the port)
    beat         dma      task, id, i, op, mem, addr, size, bank, row, value, l2, region,
                          elems, vals; back (a read's)  (a DMA beat on L1)
    beat         burst    task, id, op, mem, addr, size, back, region, elems  (a request to AXI)
    beat         fire     task, n, b, value, vals  (b, value, vals keyed by reader mover)
    beat         push     task, n, first, last, value, vals
    beat         class    task, mode  (src = acc or the mover; t = the first cycle in it)
    stall        stall    task, addr, bank, row, by, by_addr, region, elems; b (a SNAX
                          port's)  (src = the port; one per cycle refused)
    stall        fifo     task, lane, count  (src = the mover; t = the first cycle it holds)
    finegrained  insn     pc, func, text  (src = the core)

``LEVELS`` are cumulative, each the one before plus its kinds. ``KINDS`` holds each kind's
fields, required and optional; ``Trace.from_dict`` refuses an event with an unknown kind or
field. The events are kept as the dicts they are written as, in time order (by ``t``, then as
SNAX-GVSoC wrote them). ``trace_meta.json`` is the rest: the ``level``, ``total_cycles``, the
``boot`` shift (the first ``mcycle`` read in SNAX-GVSoC's cycles), the ``sources`` in the
cluster's order, the ``filter``, the sources the beat, stall and instruction lines were
written for (empty at ``task``), and ``start``, per core in the cluster's order what is in
force at t = 0: ``func``, the name of its last ``func`` line at or before 0 (None without
one), and ``mode``, its state (cores.py's ``start_mode``). No ``func`` event is written at 0:
a ``func`` event's t is the function's first instruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .events import RunError, fits, forms

LEVELS = ("off", "task", "beat", "stall", "finegrained")


def at_least(level: str, least: str) -> bool:
    """``level`` is ``least`` or a level above it."""
    return LEVELS.index(level) >= LEVELS.index(least)


# kind -> the alternatives of its fields: (required, optional)
KINDS: dict[str, list[tuple[frozenset[str], frozenset[str]]]] = {
    "start": forms(("task", "")),
    "done": forms(("task cycles firings first last", ""), ("task cycles", ""), ("task id", "")),
    "csr": forms(("pc csr block op value", "held")),
    "poll": forms(("pc csr block value count last", "")),
    "barrier": forms(("last", "")),
    "state": forms(("mode", "")),
    "func": forms(("name pc", "")),
    "issue": forms(("task id dir from to size from_stride to_stride reps", "")),
    "grant": forms(("task op mem addr bank row value back region elems vals", "b size")),
    "dma": forms(("task id i op mem addr size bank row value l2 region elems vals", "back")),
    "burst": forms(("task id op mem addr size back region elems", "")),
    "fire": forms(("task n b value vals", "")),
    "push": forms(("task n first last value vals", "")),
    "class": forms(("task mode", "")),
    "stall": forms(("task addr bank row by by_addr region elems", "b")),
    "fifo": forms(("task lane count", "")),
    "insn": forms(("pc func text", "")),
}
HEAD = ("t", "k", "src")


def check_event(e: dict[str, Any]) -> None:
    """A RunError (``trace.kind``, ``trace.field``) unless ``e`` is an event of a known kind
    with exactly one alternative's fields."""
    k = e.get("k")
    if k not in KINDS or any(h not in e for h in HEAD):
        raise RunError("trace.kind", f"not an event of a known kind ({sorted(KINDS)}): {e}")
    own = set(e) - set(HEAD)
    if not fits(own, KINDS[k]):
        raise RunError("trace.field", f"a {k} event's fields are {sorted(own)}, not one of "
                       f"{[sorted(req) for req, _ in KINDS[k]]} (C§7): {e}")  # fmt: skip


@dataclass
class Trace:
    """The events of one run in its window, and what ``trace_meta.json`` holds."""

    level: str
    total_cycles: int
    boot: int  # the program's first mcycle read in SNAX-GVSoC's cycles: every t is shifted by it
    sources: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    filter: list[str] = field(default_factory=list)  # the sources of the beat lines
    start: dict[str, dict[str, Any]] = field(default_factory=dict)  # core -> func, mode at 0

    def __post_init__(self) -> None:
        if self.level not in LEVELS[1:]:
            raise RunError("trace.level", f"a trace's level is one of {LEVELS[1:]}, not "
                           f"{self.level!r} (at off a run has none)")  # fmt: skip
        for e in self.events:
            check_event(e)

    def of_kind(self, kind: str, src: str | None = None) -> list[dict[str, Any]]:
        return [e for e in self.events if e["k"] == kind and (src is None or e["src"] == src)]

    def meta(self) -> dict[str, Any]:
        return {"level": self.level, "total_cycles": self.total_cycles, "boot": self.boot,
                "sources": list(self.sources), "filter": list(self.filter),
                "start": {c: dict(v) for c, v in self.start.items()}}  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        return {**self.meta(), "events": [dict(e) for e in self.events]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trace:
        return cls(d["level"], d["total_cycles"], d["boot"], list(d.get("sources", [])),
                   [dict(e) for e in d.get("events", [])], list(d.get("filter", [])),
                   {c: dict(v) for c, v in d.get("start", {}).items()})  # fmt: skip
