"""SNAX-GVSoC's events, read (D137): ``run/events.log`` as snax-gvsoc's ``docs/EVENTS.md``
writes it, one ``Line`` per line.

A line is GVSoC's header, ``<ps>: <cycles>: [<path>] `` with ANSI colour codes around the
path, then ``<cycle> <kind> <key>=<value> ...``. The path is the component's, then ``/`` and
the trace's name (``summary``, ``events``, ``beats``, ``stalls``); only the path is kept from
the header. ``GRAMMAR`` holds, per trace and kind, the fields of each form a line may take; a
value is an integer (decimal, or hexadecimal with ``0x``), a list of integers joined by ``,``
(``LISTS``, per trace and kind) or a name (``NAMES``).

A core's ``insn`` line (GVSoC's instruction trace, ``pe<i>/insn``) is GVSoC's own, not this
grammar: after the header, the function and line, the mode, the PC in hexadecimal without
``0x``, then the instruction (mnemonic, operands, the registers it writes and reads). It reads
as kind ``insn`` at the header's cycle, with ``pc`` and ``text``, the instruction with its runs
of spaces made one.

A line that does not read, a trace, kind or field not in the grammar, or a value of the wrong
type is a ``RunError`` naming the line (``events.<rule>``). Nothing is renamed or reordered
here: that is convert.py's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RunError(ValueError):
    """A run the converter cannot read or convert; ``code`` names the rule (``<module>.<rule>``)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


Fields = frozenset[str]


def _forms(*forms: tuple[set[str], set[str]]) -> list[tuple[Fields, Fields]]:
    return [(frozenset(req), frozenset(opt)) for req, opt in forms]


# trace -> kind -> the forms of its fields: (required, optional)
GRAMMAR: dict[str, dict[str, list[tuple[Fields, Fields]]]] = {
    "summary": {
        "run": _forms(({"index", "firings", "busy", "stall_in", "stall_out", "flush"}, set())),
        "mover": _forms(
            ({"mover", "run", "busy", "stall_xbar", "stall_fifo", "idle", "qmax"}, set())
        ),
        "input": _forms(({"index", "name", "grants", "stalls", "stalls_dma", "whole_run"}, set())),
        "bank": _forms(({"index", "reads", "writes", "conflicts"}, set())),
        "csrs": _forms(({"writes", "reads"}, set())),
        "dma": _forms(
            ({"transfers", "busy", "r_beats", "w_beats", "bytes_to_l1", "bytes_from_l1"}, set())
        ),
    },
    "events": {
        "start": _forms(({"count", "rate"}, set()), ({"mover"}, set())),
        "done": _forms(({"firings", "first", "last"}, set()), ({"mover"}, set()), ({"id"}, set())),
        "csr": _forms(({"pc", "csr", "op", "value"}, {"held"})),
        "poll": _forms(({"pc", "csr", "value", "first", "last", "count"}, set())),
        "barrier": _forms(({"first", "last"}, set())),
        "state": _forms(({"mode"}, set())),
        "func": _forms(({"name", "pc"}, set())),
        "mcycle": _forms(({"pc", "value"}, set())),
        "mcycleh": _forms(({"pc", "value"}, set())),
        "issue": _forms(({"id", "src", "dst", "size", "src_stride", "dst_stride", "reps"}, set())),
    },
    "beats": {
        "grant": _forms(({"in", "op", "addr", "bank", "row", "value", "back"}, {"b", "size"})),
        "dma": _forms(({"id", "i", "op", "addr", "size", "bank", "row", "value", "l2"}, {"back"})),
        "burst": _forms(({"id", "op", "addr", "size", "back"}, set())),
        "fire": _forms(({"n", "b", "value"}, set())),
        "push": _forms(({"n", "first", "last", "value"}, set())),
        "class": _forms(({"mode"}, set()), ({"mover", "mode"}, set())),
    },
    "stalls": {
        "stall": _forms(({"in", "addr", "bank", "row", "by", "by_addr"}, {"b"})),
        "fifo": _forms(({"mover", "lane", "count"}, set())),
    },
    "insn": {"insn": _forms(({"pc", "text"}, set()))},
}
NAMES = {"mover", "name", "op", "mode"}
# (trace, kind) -> its fields that are lists
LISTS = {
    ("summary", "mover"): {"qmax"},
    ("beats", "dma"): {"value"},
    ("beats", "fire"): {"b", "value"},
    ("beats", "push"): {"value"},
}

HEADER = re.compile(r"^-?\d+: (-?\d+): \[(?:\x1b\[[0-9;]*m)?(.*?)\s*(?:\x1b\[[0-9;]*m)?\] (.*)$")


@dataclass(frozen=True)
class Line:
    """One line: the component's path, the trace, the cycle (the component's own), the kind and
    its fields."""

    path: str
    trace: str
    cycle: int
    kind: str
    fields: dict[str, Any]


def _value(key: str, text: str, lists: set[str], where: str) -> Any:
    try:
        if key in NAMES:
            if not text:
                raise ValueError
            return text
        if key in lists:
            return [int(x, 0) for x in text.split(",")]
        return int(text, 0)
    except ValueError:
        want = "name" if key in NAMES else "list of integers" if key in lists else "integer"
        raise RunError("events.value", f"{where}: {key}={text!r} is not a {want}") from None


def _insn(path: str, cycles: str, rest: str, where: str) -> Line:
    """A core's instruction line (module doc)."""
    words = rest.split(None, 3)
    try:
        if len(words) < 4:
            raise ValueError
        pc = int(words[2], 16)
    except ValueError:
        raise RunError("events.line", f"{where}: not an instruction line (function, mode, pc, "
                       f"instruction): {rest!r}") from None  # fmt: skip
    return Line(path, "insn", int(cycles), "insn", {"pc": pc, "text": " ".join(words[3].split())})


def parse_line(text: str, where: str = "line") -> Line:
    """One line of events.log; see the module doc."""
    m = HEADER.match(text.rstrip("\n"))
    if m is None:
        raise RunError("events.line", f"{where}: not a GVSoC trace line: {text.rstrip()!r}")
    path, trace = m[2].rsplit("/", 1) if "/" in m[2] else ("", m[2])
    if trace not in GRAMMAR:
        raise RunError("events.trace", f"{where}: trace {trace!r} of {path} is not one of "
                       f"{sorted(GRAMMAR)}")  # fmt: skip
    if trace == "insn":
        return _insn(path, m[1], m[3], where)
    words = m[3].split()
    if len(words) < 2:
        raise RunError("events.line", f"{where}: no `<cycle> <kind>` in {m[3]!r}")
    try:
        cycle = int(words[0])
    except ValueError:
        raise RunError("events.line", f"{where}: cycle {words[0]!r} is not an integer") from None
    kind = words[1]
    forms = GRAMMAR[trace].get(kind)
    if forms is None:
        raise RunError("events.kind", f"{where}: kind {kind!r} is not one of the {trace} trace's "
                       f"({sorted(GRAMMAR[trace])}, snax-gvsoc docs/EVENTS.md)")  # fmt: skip
    fields: dict[str, Any] = {}
    for w in words[2:]:
        key, sep, val = w.partition("=")
        if not sep or not key or key in fields:
            raise RunError("events.field", f"{where}: {w!r} is not one key=value")
        fields[key] = val
    keys = set(fields)
    if not any(req <= keys <= req | opt for req, opt in forms):
        raise RunError("events.field", f"{where}: a {trace} {kind} line has the fields "
                       f"{sorted(keys)}, not one of {[sorted(r | o) for r, o in forms]}")  # fmt: skip
    lists = LISTS.get((trace, kind), set())
    fields = {k: _value(k, v, lists, where) for k, v in fields.items()}
    return Line(path, trace, cycle, kind, fields)


def read_events(path: str | Path) -> list[Line]:
    """Every line of an events.log, in the file's order (simulation order)."""
    p = Path(path)
    return [parse_line(t, f"{p.name}:{i}") for i, t in enumerate(p.read_text().splitlines(), 1)
            if t.strip()]  # fmt: skip
