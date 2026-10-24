"""A core's cycles in the program's window, classed (C§7): the one rule the profile counts and
the viewer's schedule draws.

    core_spans(events, width) -> [(first, end, class)]
    start_mode(events)        -> the core's state at 0: run or sleep
    merge_spans(spans)        -> [lo, hi) spans, sorted, the ones that touch or overlap merged

``events`` are one core's ``state``, ``poll`` and ``barrier`` events in C§7's form, shifted to
the window (t = 0 its first cycle) and not cut to it: a ``state`` at ``t`` with its ``mode``
(``run``, ``sleep``), a ``poll`` or ``barrier`` from ``t`` (its first cycle) to ``last``. A
cycle of ``[0, width)`` is ``barrier`` (in a barrier's ``[t, last]``), else ``poll`` (in a
poll's), else ``sleep`` (the core's last ``state`` at or before it is ``sleep``; a state
before the window gives the class at 0, ``start_mode``), else ``run``. The spans cover ``[0, width)`` with no
gap or overlap, each the longest run of one class, ``end`` exclusive.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise
from typing import Any

CORE_CLASSES = ("run", "poll", "barrier", "sleep")
RANK = {"run": 0, "sleep": 1, "poll": 2, "barrier": 3}  # the higher wins a cycle


def core_spans(events: Iterable[dict[str, Any]], width: int) -> list[tuple[int, int, str]]:
    """The core's classes over ``[0, width)`` as spans (module doc)."""
    marks: list[tuple[int, int, str]] = []  # [first, end) of a class other than run
    states = sorted((e for e in events if e["k"] == "state"), key=lambda e: e["t"])
    sleep_from = None
    for e in [*states, {"t": width, "mode": "run"}]:
        t = min(max(e["t"], 0), width)
        if sleep_from is not None and e["mode"] != "sleep":
            marks.append((sleep_from, t, "sleep"))
            sleep_from = None
        elif sleep_from is None and e["mode"] == "sleep":
            sleep_from = t
    for e in events:
        if e["k"] in ("poll", "barrier"):
            a, b = max(e["t"], 0), min(e["last"] + 1, width)
            if a < b:
                marks.append((a, b, e["k"]))
    cuts = sorted({0, width, *(a for a, _, _ in marks), *(b for _, b, _ in marks)})
    out: list[tuple[int, int, str]] = []
    for a, b in pairwise(cuts):
        cls = max((c for x, y, c in marks if x <= a and b <= y), key=RANK.__getitem__,
                  default="run")  # fmt: skip
        if out and out[-1][2] == cls and out[-1][1] == a:
            out[-1] = (out[-1][0], b, cls)
        else:
            out.append((a, b, cls))
    return out


def start_mode(events: Iterable[dict[str, Any]]) -> str:
    """The core's state in force at 0: the mode of its last ``state`` at or before 0, ``run``
    without one (``core_spans``' rule)."""
    states = sorted((e for e in events if e["k"] == "state" and e["t"] <= 0), key=lambda e: e["t"])
    return states[-1]["mode"] if states else "run"


def core_counts(spans: Iterable[tuple[int, int, str]]) -> dict[str, int]:
    """The cycles per class of ``core_spans``'s spans, in ``CORE_CLASSES``' order."""
    out = dict.fromkeys(CORE_CLASSES, 0)
    for a, b, cls in spans:
        out[cls] += b - a
    return out


def merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """``[lo, hi)`` spans in order, the ones that touch or overlap merged into one."""
    out: list[tuple[int, int]] = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out
