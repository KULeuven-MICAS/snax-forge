"""The run report, ``run.md``: where one flow run's cycles went.

Read from the run the way the viewer reads it (viz/api.py): the profile, the
trace's task events and the cluster (C§7, run/). Nothing is modelled here;
every number is a counter or an event SNAX-GVSoC wrote, as the converter
named it (D137).

    summary       the program's window (total cycles), passed, data errors over
                  the output words, each task's counter as the program printed
                  it, the trace level and the boot's length (the cycles before
                  the window, which the profile, the cores and the trace share)
    tasks         per block each task's start, done and length, named from the
                  task list; a DMA copy with its direction and bytes
                                                         (needs a task trace)
    accelerators  firings, target II (the datapath file's), achieved II,
                  utilisation over the window and over its tasks' span,
                  stall_in, stall_out, flush
    cores         cycles per class in the window, SNAX CSR writes and
                  reads, and each wait: block, first, last, cycles and the
                  task that starts next                  (classes and waits
                                                         need a task trace)
    DMA           transfers, busy cycles, beats each way, bytes each way, bytes
                  per busy cycle
    memory        per crossbar port grants, stalls and stalls lost to the DMA
                  (a core's or AXI's over the whole run, marked); banks with
                  conflicts as ranges; streamers' stall cycles; FIFO highest
                  count against depth
    movement      from the run index (viz/movement.py), at beat and above: per
                  region and memory its arrival, use and departure spans; per
                  streamer lane and task its configured nest and pace against
                  one beat per cycle; at stall, the L1 stalls per bank and per
                  port (and the held cycles per lane). At off and task one line
                  says why there is none

Achieved II = (last firing − first firing) / (firings − 1), over the
accelerator's tasks (their ``done`` events carry both). The report's length
grows with components, tasks and ports, never with cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..viz import api, movement
from ..viz.movement import MovementUnavailable
from .markdown import pct, spans, table
from .record import Record

DIRECTION = {"l2_to_l1": "L2 → L1", "l1_to_l2": "L1 → L2", "l1_to_l1": "L1 → L1",
             "l2_to_l2": "L2 → L2"}  # fmt: skip


@dataclass
class CheckRow(Record):
    passed: bool
    kernel: str
    data_errors: int
    words: int
    counters: dict[str, int]  # per accelerator task, as the program printed it


@dataclass
class TaskRow(Record):
    block: str
    index: int
    name: str | None
    start: int
    done: int
    cycles: int
    direction: str | None = None
    bytes: int | None = None


@dataclass
class AccelRun(Record):
    name: str
    firings: int
    target_ii: int
    achieved_ii: float | None  # None without the task events or with one firing
    utilisation: float  # busy / the window (total cycles)
    span: int | None  # its tasks' cycles, start to done
    utilisation_span: float | None  # busy / span
    stall_in: int
    stall_out: int
    flush: int


@dataclass
class WaitRow(Record):
    block: str
    first: int
    last: int
    cycles: int
    then: str | None = None  # the task that starts next


@dataclass
class CoreRun(Record):
    name: str
    cycles: dict[str, int] | None  # None at trace off
    csr_writes: int
    csr_reads: int
    waits: list[WaitRow] = field(default_factory=list)


@dataclass
class DmaRun(Record):
    name: str
    transfers: int
    busy: int
    beats_read: int
    beats_written: int
    bytes_l2_to_l1: int
    bytes_l1_to_l2: int
    bytes_per_busy_cycle: float


@dataclass
class PortRow(Record):
    port: str
    grants: int
    stalls: int
    stalls_dma: int
    whole_run: bool


@dataclass
class BankRange(Record):
    banks: str
    conflicts: int  # per bank


@dataclass
class StreamerRun(Record):
    name: str
    stall_xbar: int
    stall_fifo: int
    fifo_depth: int
    fifo_max: int


@dataclass
class ResidencyRow(Record):
    region: str
    mem: str
    elements: int
    present_at_start: int
    arrival: list[int] | None  # [first, last] over the elements
    use: list[int] | None
    departure: list[int] | None


@dataclass
class PatternRow(Record):
    port: str
    task: str
    base: int | None
    nest: list[list[int]]  # [bound, stride] per loop, innermost first
    beats: int
    first: int | None
    last: int | None
    ideal_last: int | None  # first + beats - 1: one beat per cycle
    held: int | None  # cycles; None below stall


@dataclass
class StallCount(Record):
    where: str  # banks (as ranges) or a port
    stalls: int


@dataclass
class MovementReport(Record):
    reason: str | None  # why there is none (below beat), else None
    sources: list[str] | None  # the beat lines' sources when they are not all, else None
    residency: list[ResidencyRow] = field(default_factory=list)
    patterns: list[PatternRow] = field(default_factory=list)
    conflicts: int | None = None  # L1 stalls; None below stall
    banks: list[StallCount] = field(default_factory=list)
    ports: list[StallCount] = field(default_factory=list)
    conflicts_reason: str | None = None


@dataclass
class RunReport(Record):
    name: str
    cycles: int
    trace_level: str
    boot: int | None
    check: CheckRow | None
    tasks: list[TaskRow] | None  # None: no task trace
    accelerators: list[AccelRun]
    cores: list[CoreRun]
    dmas: list[DmaRun]
    ports: list[PortRow]
    banks: list[BankRange]
    streamers: list[StreamerRun]
    movement: MovementReport


# =============================================================================
# Building
# =============================================================================


def achieved_ii(dones: list[dict[str, Any]]) -> float | None:
    """(last − first firing) / (firings − 1) over an accelerator's done events (module doc)."""
    gaps = sum(d["last"] - d["first"] for d in dones if d["firings"] > 1)
    n = sum(d["firings"] - 1 for d in dones if d["firings"] > 1)
    return gaps / n if n else None


def _bank_ranges(conflicts: list[int]) -> list[BankRange]:
    """Neighbouring banks with the same conflict count, as one range; banks without any left
    out."""
    out: list[BankRange] = []
    i = 0
    while i < len(conflicts):
        if not conflicts[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(conflicts) and conflicts[j + 1] == conflicts[i]:
            j += 1
        out.append(BankRange(spans(list(range(i, j + 1))), conflicts[i]))
        i = j + 1
    return out


def build_movement(rv: api.RunView) -> MovementReport:
    """The data movement section from the run index (module doc)."""
    idx = rv.index
    filtered = idx.filtered
    try:
        res, pats = movement.residency(idx), movement.patterns(idx)
    except MovementUnavailable:
        return MovementReport(f"Needs --trace beat; this run was traced at {idx.level}.", None)
    out = MovementReport(None, None if filtered is None else filtered["sources"])
    out.residency = [ResidencyRow(r["region"], r["mem"], r["elements"], r["present_at_start"],
                                  r["arrival"], r["use"], r["departure"]) for r in res]  # fmt: skip
    out.patterns = [PatternRow(p["port"], p["task"], p["addr"]["base"], p["addr"]["nest"],
                               p["beats"], p["first"], p["last"], p["ideal_last"],
                               p.get("held_cycles")) for p in pats if p["port"] != "dma"]  # fmt: skip
    try:
        c = movement.conflicts(idx)
    except MovementUnavailable:
        out.conflicts_reason = f"Needs --trace stall; this run was traced at {idx.level}."
        return out
    out.conflicts = c["count"]
    counts = [c["banks"].get(str(b), 0) for b in range(rv.cluster.banks)]
    out.banks = [StallCount(b.banks, b.conflicts) for b in _bank_ranges(counts)]
    out.ports = [StallCount(port, n) for port, n in c["ports"].items()]
    return out


def _next_task(rows: list[TaskRow] | None, after: int) -> TaskRow | None:
    """The first task that starts after cycle ``after`` (the one a wait held back)."""
    later = [r for r in rows or [] if r.start > after]
    return min(later, key=lambda r: (r.start, r.block)) if later else None


def build_run(rv: api.RunView) -> RunReport:
    """The run report of one loaded flow run."""
    p = rv.outputs.profile
    tr = rv.outputs.trace
    total = p.total_cycles
    fc = p.functional_check
    check = None
    if fc:
        check = CheckRow(bool(fc["passed"]), fc.get("kernel", ""), fc["data_errors"], fc["words"],
                         {t: x["counter"] for t, x in fc.get("tasks", {}).items()})  # fmt: skip
    task_rows = None
    if tr is not None:
        task_rows = []
        for block, ts in api.tasks(rv).items():
            for k, t in enumerate(ts):
                task_rows.append(TaskRow(block, k, t["task"], t["start"], t["done"],
                                         t["done"] - t["start"], t.get("direction"),
                                         t.get("bytes")))  # fmt: skip
        task_rows.sort(key=lambda r: (r.start, r.block))

    accels = []
    acc = rv.cluster.accelerator
    for name, a in p.accelerators.items():
        busy = a.cycles["busy"]
        span = sum(d - s for s, d in rv.spans.get(name, [])) or None
        dones = tr.of_kind("done", name) if tr is not None else []
        accels.append(AccelRun(
            name, a.firings, int(acc.unit.get("ii", 1)), achieved_ii(dones), a.utilisation, span,
            busy / span if span else None, a.cycles["stall_in"], a.cycles["stall_out"],
            a.cycles["flush"],
        ))  # fmt: skip

    cores = []
    for name, c in p.cores.items():
        waits = []
        for w in c.waits:
            nxt = _next_task(task_rows, w.last)
            waits.append(WaitRow(w.block, w.first, w.last, w.last - w.first + 1,
                                 None if nxt is None else nxt.name or f"{nxt.block} #{nxt.index}"))  # fmt: skip
        cores.append(CoreRun(name, c.cycles, c.csr_writes, c.csr_reads, waits))

    dmas = [
        DmaRun(name, d.transfers, d.busy, d.beats_read, d.beats_written, d.bytes_l2_to_l1,
               d.bytes_l1_to_l2, (d.bytes_l2_to_l1 + d.bytes_l1_to_l2) / d.busy if d.busy else 0.0)
        for name, d in p.dmas.items()
    ]  # fmt: skip
    ports = [PortRow(n, x.grants, x.stalls, x.stalls_dma, x.whole_run) for n, x in p.ports.items()]
    streamers = [
        StreamerRun(n, s.cycles["stall_xbar"], s.cycles["stall_fifo"], s.fifo.depth,
                    max(s.fifo.max, default=0))
        for n, s in p.streamers.items()
    ]  # fmt: skip
    return RunReport(
        rv.name, total, rv.trace_level, None if tr is None else tr.boot,
        check, task_rows, accels, cores, dmas, ports,
        _bank_ranges(p.banks.conflicts if p.banks else []), streamers, build_movement(rv),
    )  # fmt: skip


# =============================================================================
# Rendering
# =============================================================================


def render_run(r: RunReport) -> str:
    """``run.md``: the report as Markdown, one section per part."""
    out = [f"# Run report: {r.name}", ""]
    if r.check is None:
        check, data, counters = "none", "–", "–"
    else:
        check = "passed" if r.check.passed else "FAILED"
        data = f"{r.check.data_errors} errors over {r.check.words} words"
        counters = ", ".join(f"{t} {n}" for t, n in r.check.counters.items()) or "none"
    trace = r.trace_level
    if r.boot is not None:
        trace += f"; boot {r.boot} cycles (to the mcycle read in main)"
    out += table(["", ""], [
        ["Window (main to forge_check, the program's mcycle)", r.cycles],
        ["Functional check", check],
        ["Data", data],
        ["Counters", counters],
        ["Trace", trace],
    ])  # fmt: skip

    out += ["", "## Tasks", ""]
    if r.tasks is None:
        out.append("Needs at least a task trace (--trace task).")
    else:
        out += table(
            ["Block", "#", "Task", "Start", "Done", "Cycles", "Direction", "Bytes"],
            [[t.block, t.index, t.name, t.start, t.done, t.cycles,
              DIRECTION.get(t.direction or "", t.direction), t.bytes] for t in r.tasks],
            {1, 3, 4, 5, 7},
        )  # fmt: skip

    out += ["", "## Accelerators", ""]
    out += table(
        ["Accelerator", "Firings", "Target II", "Achieved II", "Utilisation (window)",
         "Task span", "Utilisation (span)", "Starved (stall_in)", "Blocked (stall_out)", "Flush"],
        [[a.name, a.firings, a.target_ii, None if a.achieved_ii is None else f"{a.achieved_ii:.2f}",
          pct(a.utilisation), a.span,
          None if a.utilisation_span is None else pct(a.utilisation_span), a.stall_in,
          a.stall_out, a.flush] for a in r.accelerators],
        {1, 2, 3, 4, 5, 6, 7, 8, 9},
    )  # fmt: skip
    out += ["", ("Achieved II is the cycles from the first firing to the last over the firings "
            "between them; utilisation is busy cycles (a firing or the II gap after it) over the "
            "window, then over the cycles its tasks ran.")]  # fmt: skip

    out += ["", "## Cores", ""]
    out += table(
        ["Core", "Run", "Poll", "Barrier", "Sleep", "CSR writes", "CSR reads"],
        [[c.name, *([c.cycles[k] for k in ("run", "poll", "barrier", "sleep")] if c.cycles
                    else [None] * 4), c.csr_writes, c.csr_reads] for c in r.cores],
        {1, 2, 3, 4, 5, 6},
    )  # fmt: skip
    if any(c.cycles is None for c in r.cores):
        out += ["", "The cores' classes and waits need at least a task trace (--trace task)."]
    for c in r.cores:
        if c.waits:
            out += ["", f"{c.name}'s waits:", ""]
            out += table(["Waits on", "From", "To", "Cycles", "Then starts"],
                         [[w.block, w.first, w.last, w.cycles, w.then] for w in c.waits],
                         {1, 2, 3})  # fmt: skip

    if r.dmas:
        out += ["", "## DMA", ""]
        out += table(
            ["DMA", "Transfers", "Busy", "Beats read", "Beats written", "Bytes L2 → L1",
             "Bytes L1 → L2", "Bytes per busy cycle"],
            [[d.name, d.transfers, d.busy, d.beats_read, d.beats_written, d.bytes_l2_to_l1,
              d.bytes_l1_to_l2, f"{d.bytes_per_busy_cycle:.1f}"] for d in r.dmas],
            {1, 2, 3, 4, 5, 6, 7},
        )  # fmt: skip

    out += ["", "## Memory", ""]
    out += table(
        ["Port", "Grants", "Stalls", "Stalls by the DMA"],
        [[f"{x.port} (whole run)" if x.whole_run else x.port, x.grants, x.stalls, x.stalls_dma]
         for x in r.ports],
        {1, 2, 3},
    )  # fmt: skip
    out += ["", ("A core's and AXI's port counts from reset, the boot included; the others' "
                 "traffic is the program's.")]  # fmt: skip
    out.append("")
    if r.banks:
        out += table(["Banks", "Conflicts per bank"], [[b.banks, b.conflicts] for b in r.banks],
                     {1})  # fmt: skip
    else:
        out.append("No bank conflicts.")
    out.append("")
    out += table(
        ["Streamer", "Held by the crossbar (cycles)", "Waiting on its FIFO (cycles)",
         "FIFO highest / depth"],
        [[s.name, s.stall_xbar, s.stall_fifo, f"{s.fifo_max} / {s.fifo_depth}"]
         for s in r.streamers],
        {1, 2},
    )  # fmt: skip

    out += ["", "## Data movement", ""] + render_movement(r.movement)
    return "\n".join(out) + "\n"


def _span(w: list[int] | None) -> str | None:
    return None if w is None else str(w[0]) if w[0] == w[1] else f"{w[0]}–{w[1]}"


def render_movement(m: MovementReport) -> list[str]:
    """The data movement section's lines (module doc)."""
    if m.reason is not None:
        return [m.reason]
    out = []
    if m.sources is not None:
        out += [f"The beat lines cover {', '.join(m.sources)} only.", ""]
    out += table(
        ["Region", "Memory", "Elements", "There from the start", "Arrival", "Use", "Departure"],
        [[r.region, r.mem.upper(), r.elements, r.present_at_start, _span(r.arrival), _span(r.use),
          _span(r.departure)] for r in m.residency],
        {2, 3},
    )  # fmt: skip
    out += ["", ("Arrival is an element's first write in that memory, use a read by anything but "
                 "the DMA after it, departure the DMA's first read after it; each column spans "
                 "the region's elements.")]  # fmt: skip
    out.append("")
    out += table(
        ["Port", "Task", "Base", "Nest (bound × stride, innermost first)", "Beats", "First",
         "Last", "Last at one a cycle", "Held (cycles)"],
        [[p.port, p.task, None if p.base is None else hex(p.base),
          ", ".join(f"{b} × {st}" for b, st in p.nest), p.beats, p.first, p.last, p.ideal_last,
          p.held] for p in m.patterns],
        {4, 5, 6, 7, 8},
    )  # fmt: skip
    out += ["", ("The nest is the one each task's CSR writes configured (the converter checked it "
                 "against the grants).")]  # fmt: skip
    out.append("")
    if m.conflicts is None:
        out.append(f"L1 stalls: {m.conflicts_reason}")
    elif not m.conflicts:
        out.append("No L1 stalls.")
    else:
        out += [f"{m.conflicts} L1 stalls (one per port and cycle a request was refused):", ""]
        out += table(["Banks", "Stalls per bank"], [[b.where, b.stalls] for b in m.banks], {1})
        out.append("")
        out += table(["Port", "Stalls"], [[p.where, p.stalls] for p in m.ports], {1})
    return out
