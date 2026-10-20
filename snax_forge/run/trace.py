"""The trace of a flow run (C§7, D137).

SNAX-GVSoC's ``events`` lines, renamed into C§7 and cut to the program's window: t = 0 is
main's entry on the accelerator's core, and events outside ``[0, total_cycles)`` are dropped.
Every event is a flat dict of ``t``, ``k``, ``src`` and the kind's own fields:

    task  start    task  (t = the cycle the start is taken; busy from t + 1)
    task  done     task, then firings, first, last for an accelerator  (t = first idle cycle)
    task  csr      pc, csr, block, op (r/w), value, held (a held write's cycles)
    task  poll     pc, csr, block, value, count, last  (t = the first read)
    task  barrier  last  (t = the barrier's read; last = the cycle it is released)
    task  state    mode (run/sleep)  (t = the first cycle in it)
    task  func     name, pc  (t = the function's first instruction)
    task  issue    task, id, dir, from, to, size, from_stride, to_stride, reps  (a DMA copy)
    task  done     task, id  (a DMA copy; t = the cycle it completes)

``KINDS`` holds each kind's fields, required and optional; ``Trace.from_dict`` refuses an
event with an unknown kind or field. The events are kept as the dicts they are written as, in
time order (by ``t``, then as SNAX-GVSoC wrote them). ``trace_meta.json`` is the rest: the
``level``, ``total_cycles``, the ``boot`` shift (main's entry in SNAX-GVSoC's cycles), the
``window`` (main's entry to ``forge_check``'s, the cycles the events cover) and the
``sources`` in the cluster's order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .events import RunError

LEVELS = ("off", "task", "beat")

# kind -> the alternatives of its fields: (required, optional)
KINDS: dict[str, list[tuple[frozenset[str], frozenset[str]]]] = {
    "start": [(frozenset({"task"}), frozenset())],
    "done": [
        (frozenset({"task", "firings", "first", "last"}), frozenset()),
        (frozenset({"task"}), frozenset()),
        (frozenset({"task", "id"}), frozenset()),
    ],
    "csr": [(frozenset({"pc", "csr", "block", "op", "value"}), frozenset({"held"}))],
    "poll": [(frozenset({"pc", "csr", "block", "value", "count", "last"}), frozenset())],
    "barrier": [(frozenset({"last"}), frozenset())],
    "state": [(frozenset({"mode"}), frozenset())],
    "func": [(frozenset({"name", "pc"}), frozenset())],
    "issue": [
        (
            frozenset(
                {"task", "id", "dir", "from", "to", "size", "from_stride", "to_stride", "reps"}
            ),
            frozenset(),
        ),
    ],
}
HEAD = ("t", "k", "src")


def check_event(e: dict[str, Any]) -> None:
    """A RunError (``trace.kind``, ``trace.field``) unless ``e`` is an event of a known kind
    with exactly one alternative's fields."""
    k = e.get("k")
    if k not in KINDS or any(h not in e for h in HEAD):
        raise RunError("trace.kind", f"not an event of a known kind ({sorted(KINDS)}): {e}")
    own = set(e) - set(HEAD)
    if not any(req <= own <= req | opt for req, opt in KINDS[k]):
        raise RunError("trace.field", f"a {k} event's fields are {sorted(own)}, not one of "
                       f"{[sorted(req) for req, _ in KINDS[k]]} (C§7): {e}")  # fmt: skip


@dataclass
class Trace:
    """The events of one run in its window, and what ``trace_meta.json`` holds."""

    level: str
    total_cycles: int
    boot: int  # main's entry in SNAX-GVSoC's cycles: every t is shifted by it
    window: int  # main's entry to forge_check's: the events' t are in [0, window)
    sources: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

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
                "window": self.window, "sources": list(self.sources)}  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        return {**self.meta(), "events": [dict(e) for e in self.events]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trace:
        return cls(d["level"], d["total_cycles"], d["boot"], d["window"],
                   list(d.get("sources", [])), [dict(e) for e in d.get("events", [])])  # fmt: skip
