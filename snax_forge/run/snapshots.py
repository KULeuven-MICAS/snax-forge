"""SNAX-GVSoC's snapshots read, and the windows between them (CONTRACTS, "Profile and trace").

    snapshots(lines, cluster) -> list[Snapshot]
    windows(snaps, marks) -> list[Window]

Every write to CSR 0x7C5 by core 0 and the end of the simulation (mark -1) make each SNAX
component write its counters as they stand, running totals since boot, as ``snap`` lines on its
``summary`` trace (snax-gvsoc's ``docs/EVENTS.md``). ``snapshots`` gathers them in the order they
were taken, every component's k-th set of lines being snapshot k, named as the converter names
them: the crossbar's inputs by port (``acc_a.lane0``, ``core0``, ``axi``), each group by its
unit's trace name, each unit by its instance, each mover by its trace name. ``windows`` subtracts
consecutive snapshots, boot's (all 0) first; window k takes its kind and label from the quiet
build's ``obs_marks.json`` row of its mark (window 0 is ``boot``, or ``first``'s: ``WHOLE`` for a
run that wrote no mark, its one window boot to the end). Nothing is recounted (D137).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .cluster import Cluster
from .convert import port_of, source_of
from .events import Line, RunError

GROUP_CLASSES = ("busy", "stall_in", "stall_out", "flush")
MOVER_CLASSES = ("busy", "stall_xbar", "stall_fifo", "idle")
END = -1  # the mark of the end of the simulation's snapshot
BOOT = ("boot", "boot")  # window 0's kind and label
WHOLE = ("whole", "the whole run")  # window 0 of a run that wrote no mark: boot to the end


@dataclass
class Counters:
    """Counters at a snapshot, or their difference over a window: per crossbar input (by port)
    ``(grants, stalls)``, per bank ``(reads, writes, conflicts)``, per group its busy-flag
    classes, per unit ``(firings, pushes)``, per mover its classes, and the beats the DMA's
    copies inside L1 read and write on the superbank port, ``copies`` (``(0, 0)`` in a run
    without the crossbar's line)."""

    inputs: dict[str, tuple[int, int]] = field(default_factory=dict)
    banks: list[tuple[int, int, int]] = field(default_factory=list)
    groups: dict[str, dict[str, int]] = field(default_factory=dict)
    units: dict[str, tuple[int, int]] = field(default_factory=dict)
    movers: dict[str, dict[str, int]] = field(default_factory=dict)
    copies: tuple[int, int] = (0, 0)

    def minus(self, other: Counters) -> Counters:
        """``self`` less ``other``, field by field (a counter ``other`` lacks counts from 0)."""

        def sub(a: tuple[int, ...], b: tuple[int, ...] | None) -> tuple[int, ...]:
            return a if b is None else tuple(x - y for x, y in zip(a, b, strict=True))

        def subd(a: dict[str, int], b: dict[str, int] | None) -> dict[str, int]:
            return dict(a) if b is None else {k: v - b[k] for k, v in a.items()}

        banks = [sub(b, other.banks[i] if i < len(other.banks) else None)
                 for i, b in enumerate(self.banks)]  # fmt: skip
        return Counters(
            {k: sub(v, other.inputs.get(k)) for k, v in self.inputs.items()},  # type: ignore[misc]
            banks,  # type: ignore[arg-type]
            {k: subd(v, other.groups.get(k)) for k, v in self.groups.items()},
            {k: sub(v, other.units.get(k)) for k, v in self.units.items()},  # type: ignore[misc]
            {k: subd(v, other.movers.get(k)) for k, v in self.movers.items()},
            sub(self.copies, other.copies),  # type: ignore[arg-type]
        )

    def zero(self) -> Counters:
        """The same counters at 0: boot's."""
        return self.minus(self)

    def busy(self) -> int:
        """The groups' busy-flag cycles added up."""
        return sum(sum(c.values()) for c in self.groups.values())

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": {k: {"grants": g, "stalls": s} for k, (g, s) in self.inputs.items()},
                "banks": [{"reads": r, "writes": w, "conflicts": c} for r, w, c in self.banks],
                "groups": {k: dict(v) for k, v in self.groups.items()},
                "units": {k: {"firings": f, "pushes": p} for k, (f, p) in self.units.items()},
                "movers": {k: dict(v) for k, v in self.movers.items()},
                "copies": {"reads": self.copies[0], "writes": self.copies[1]}}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Counters:
        cp = d.get("copies", {"reads": 0, "writes": 0})  # none in counters written before them
        return cls({k: (v["grants"], v["stalls"]) for k, v in d["inputs"].items()},
                   [(b["reads"], b["writes"], b["conflicts"]) for b in d["banks"]],
                   {k: dict(v) for k, v in d["groups"].items()},
                   {k: (v["firings"], v["pushes"]) for k, v in d["units"].items()},
                   {k: dict(v) for k, v in d["movers"].items()},
                   (cp["reads"], cp["writes"]))  # fmt: skip


@dataclass
class Snapshot:
    """One snapshot: the mark written (-1 at the end), the cycle it was taken in, the counters."""

    mark: int
    cycle: int
    counters: Counters


@dataclass
class Window:
    """The counts between two snapshots: its index (0 from boot), the mark that opens it (0 for
    boot), its kind and label (``obs_marks.json``'s), its first and last cycles, its counters."""

    index: int
    mark: int
    kind: str
    label: str
    first: int
    last: int
    counters: Counters

    @property
    def cycles(self) -> int:
        return self.last - self.first

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "mark": self.mark, "kind": self.kind, "label": self.label,
                "first": self.first, "last": self.last, "cycles": self.cycles,
                "counters": self.counters.to_dict()}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Window:
        return cls(d["index"], d["mark"], d["kind"], d["label"], d["first"], d["last"],
                   Counters.from_dict(d["counters"]))  # fmt: skip


def _inputs(lines: Sequence[Line], cluster: Cluster) -> dict[int, str]:
    """The crossbar's inputs by index, as ports, from its whole-run ``input`` lines."""
    return {x.fields["index"]: port_of(x.fields["name"], cluster) for x in lines
            if x.trace == "summary" and x.kind == "input"}  # fmt: skip


def snapshots(lines: Sequence[Line], cluster: Cluster) -> list[Snapshot]:
    """Every snapshot of a run in the order taken (module doc); none when the run wrote no
    ``snap`` lines. A ``RunError`` (``snapshots.<rule>``) when the components did not take the
    same marks, an input is not named by an ``input`` line, or the last is not the end's."""
    inputs = _inputs(lines, cluster)
    taken: dict[str, list[tuple[int, int, Counters]]] = {}  # component path -> its snapshots
    seen: dict[str, set[tuple[str, Any]]] = {}
    for x in lines:
        if x.trace != "summary" or x.kind != "snap":
            continue
        f = x.fields
        key = next(k for k in ("input", "bank", "unit", "mover", "busy", "copy_reads") if k in f)
        ident = (key, f[key] if key not in ("busy", "copy_reads") else None)
        mine = taken.setdefault(x.path, [])
        if not mine or (mine[-1][0], mine[-1][1]) != (f["mark"], x.cycle) or ident in seen[x.path]:
            mine.append((f["mark"], x.cycle, Counters()))
            seen[x.path] = set()
        seen[x.path].add(ident)
        c = mine[-1][2]
        if key == "input":
            if f["input"] not in inputs:
                raise RunError("snapshots.input", f"snap input={f['input']}: no `input` line "
                               "names it")  # fmt: skip
            c.inputs[inputs[f["input"]]] = (f["grants"], f["stalls"])
        elif key == "bank":
            if f["bank"] != len(c.banks):
                raise RunError("snapshots.bank", f"snap bank={f['bank']} after {len(c.banks)} "
                               "banks; the banks come in order")  # fmt: skip
            c.banks.append((f["reads"], f["writes"], f["conflicts"]))
        elif key == "unit":
            c.units[f["unit"]] = (f["firings"], f["pushes"])
        elif key == "mover":
            c.movers[source_of(x, cluster)] = {k: f[k] for k in MOVER_CLASSES}
        elif key == "copy_reads":
            c.copies = (f["copy_reads"], f["copy_writes"])
        else:
            c.groups[source_of(x, cluster)] = {k: f[k] for k in GROUP_CLASSES}
    if not taken:
        return []
    order = {p: [m for m, _, _ in s] for p, s in taken.items()}
    first = next(iter(order.values()))
    if any(o != first for o in order.values()):
        raise RunError("snapshots.marks", f"the components took different marks: {order}")
    if first[-1] != END:
        raise RunError("snapshots.end", f"the last snapshot is mark {first[-1]}, not the end's "
                       f"({END})")  # fmt: skip
    out = []
    for k, mark in enumerate(first):
        merged = Counters()
        cycles = set()
        for s in taken.values():
            _, cycle, c = s[k]
            cycles.add(cycle)
            merged.inputs |= c.inputs
            merged.banks += c.banks
            merged.groups |= c.groups
            merged.units |= c.units
            merged.movers |= c.movers
            merged.copies = (merged.copies[0] + c.copies[0], merged.copies[1] + c.copies[1])
        out.append(Snapshot(mark, min(cycles), merged))
    _check_end(lines, out[-1], inputs)
    return out


def _check_end(lines: Sequence[Line], end: Snapshot, inputs: dict[int, str]) -> None:
    """The end's snapshot equals the crossbar's whole-run ``input`` lines."""
    bad = []
    for x in lines:
        if x.trace == "summary" and x.kind == "input":
            f, port = x.fields, inputs[x.fields["index"]]
            got = end.counters.inputs.get(port)
            if got is not None and got != (f["grants"], f["stalls"]):
                bad.append(f"{port}: snap {got}, input line {(f['grants'], f['stalls'])}")
    if bad:
        raise RunError(
            "snapshots.end", "the end's snapshot is not the whole run: " + "; ".join(bad)
        )


def windows(snaps: Sequence[Snapshot], marks: Sequence[dict[str, Any]],
            first: tuple[str, str] = BOOT) -> list[Window]:  # fmt: skip
    """The windows between consecutive snapshots, boot's first (module doc); ``marks`` are
    ``obs_marks.json``'s rows (``mark``, ``kind``, ``label``), ``first`` window 0's kind and
    label. Empty without snapshots."""
    if not snaps:
        return []
    named = {m["mark"]: m for m in marks}
    boot = Snapshot(0, 0, snaps[0].counters.zero())
    out = []
    for k, (a, b) in enumerate(zip([boot, *snaps[:-1]], snaps, strict=True)):
        row = named.get(a.mark) if k else {"kind": first[0], "label": first[1]}
        kind, label = (row["kind"], row["label"]) if row else ("mark", f"mark {a.mark}")
        out.append(Window(k, a.mark, kind, label, a.cycle, b.cycle, b.counters.minus(a.counters)))
    return out
