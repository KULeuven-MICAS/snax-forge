"""The run report, ``run.md`` (REP1, D99): where one run's cycles went.

Read from the run directory the way the viewer reads it (viz/api.py): the
profile, the class intervals and task events of the trace, and on a beat
trace the data movement answers (D97). Nothing is modelled here; every number
is a counter, an interval or an event the model wrote.

    summary       cycles, the flow's functional check, the trace level
    tasks         per block each task's start, done and length, with its name
                  from the task list when one is found   (needs a task trace)
    accelerators  firings, target II, achieved II, utilisation over its task
                  window, cycles starved and blocked
    controller    command, wait and idle cycles, and each wait: block, mode,
                  cycles and the task that starts after it; a wait on one
                  accelerator (itself or a streamer attached to it) followed
                  by a task of another is a chaining wait, listed as
                  ``mul -> sum`` (D106)
    DMAs          busy cycles, beats and bytes each way, bytes per busy cycle
    memory        banks with conflicts as ranges, streamers' stall cycles and
                  port stalls, FIFO highest count against depth
    movement      residency per region and memory, and per port and task the
                  element pattern, its pace and the first cycles held back
                  (needs a beat trace; otherwise a line says so)

Achieved II = (end of the last ``busy`` interval − start of the first) /
firings. ``busy`` includes the II gap (D59), so it equals the target II when
nothing stalls. The report's length grows with components, tasks and ports,
never with cycles or elements: held-back cycles are cut to the first few.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..viz import api
from ..viz.movement import dma_names
from .markdown import span_text, spans, table
from .record import Record

HELD_SHOWN = 3  # held-back cycles listed per pattern; the count is always given
DIRECTION = {"l2_to_l1": "L2 → L1", "l1_to_l2": "L1 → L2"}


@dataclass
class CheckRow(Record):
    passed: bool
    kernel: str
    containers: dict[str, Any]


@dataclass
class TaskRow(Record):
    block: str
    index: int
    name: str | None
    start: int
    done: int
    cycles: int
    direction: str | None = None


@dataclass
class AccelRun(Record):
    name: str
    firings: int
    target_ii: int
    achieved_ii: float | None  # None without the class intervals
    busy: int
    window: int | None  # cycles between its tasks' starts and dones
    utilisation: float  # busy / window, or busy / total cycles without tasks
    stall_in: int
    stall_out: int


@dataclass
class WaitRow(Record):
    block: str
    mode: str
    first: int
    last: int
    cycles: int
    then: str | None = None  # the task that starts next (needs a task trace)


@dataclass
class ChainRow(Record):
    """A wait between two accelerators: ``dst`` starts after ``src``'s ``block`` (D106)."""

    src: str
    dst: str
    block: str
    cycles: int
    readers: list[str] = field(default_factory=list)  # dst's streamers that read what src wrote


@dataclass
class ControllerRun(Record):
    name: str
    command: int
    wait: int
    idle: int
    waits: list[WaitRow] = field(default_factory=list)
    chains: list[ChainRow] = field(default_factory=list)


@dataclass
class DmaRun(Record):
    name: str
    busy: int
    beats_read: int
    beats_written: int
    bytes_read: int
    bytes_written: int
    bytes_per_busy_cycle: float


@dataclass
class BankRange(Record):
    banks: str
    conflicts: int  # per bank
    stalls: int  # per bank


@dataclass
class StreamerRun(Record):
    name: str
    stall_xbar: int
    stall_fifo: int
    port_stalls: int
    fifo_depth: int
    fifo_max: int


@dataclass
class ResidencyRow(Record):
    region: str
    mem: str
    arrival: list[int] | None
    use: list[int] | None
    departure: list[int] | None
    wait_in: list[int] | None
    wait_out: list[int] | None
    present_at_start: int


@dataclass
class PatternRow(Record):
    port: str
    task: int
    pattern: str
    beats: int
    first: int
    last: int
    ideal_last: int
    held_cycles: int
    held: list[str] = field(default_factory=list)  # the first few, with their cause


@dataclass
class Movement(Record):
    residency: list[ResidencyRow]
    patterns: list[PatternRow]
    conflicts: int
    filtered: dict[str, Any] | None


@dataclass
class RunReport(Record):
    name: str
    scenario: str
    cycles: int
    trace_level: str
    check: CheckRow | None
    tasks: list[TaskRow] | None  # None: no task trace
    accelerators: list[AccelRun]
    controller: ControllerRun | None
    dmas: list[DmaRun]
    banks: list[BankRange]
    streamers: list[StreamerRun]
    movement: Movement | None
    movement_reason: str | None  # why there is no movement section


# =============================================================================
# Building
# =============================================================================


def achieved_ii(intervals: list[list[Any]] | None, firings: int) -> float | None:
    """(end of the last busy interval − start of the first) / firings (module doc)."""
    if not intervals or not firings:
        return None
    busy = [(a, b) for cls, a, b in intervals if cls == "busy"]
    if not busy:
        return None
    return (busy[-1][1] - busy[0][0]) / firings


def _nest_text(name: str, base: int, nest: list[list[int]]) -> str:
    """``B[4j+1]`` for base 1, nest [[16, 4]]; loops innermost first, j, then j1, j2, ..."""
    names = ["j"] if len(nest) == 1 else [f"j{i}" for i in range(len(nest))]
    terms = [f"{s}{v}" if s != 1 else v for (_, s), v in zip(nest, names, strict=True) if s]
    if base or not terms:
        terms.append(str(base))
    return f"{name}[{'+'.join(terms)}]"


def _held_text(h: dict[str, Any]) -> str:
    """``44: B[8] waited, bank 8 served acc_a.0 for A[8]``."""
    what = h["element"] or f"a word in row {h['row']}"
    by = h["served"]["port"] if h["served"] else "another port"
    also = f" for {h['served_element']}" if h.get("served_element") else ""
    return f"{h['t']}: {what} waited, bank {h['bank']} served {by}{also}"


def _task_names(tasks: Any) -> dict[str, list[str]]:
    """Component -> its configured task names in order, from a task list."""
    out: dict[str, list[str]] = {}
    if tasks is None:
        return out
    for st in tasks.steps:
        if getattr(st, "op", None) == "configure":
            out.setdefault(st.component, []).append(st.task_name)
    return out


def _bank_ranges(banks: dict[str, list[int]]) -> list[BankRange]:
    """Neighbouring banks with the same conflict and stall counts, as one range."""
    conf, stalls = banks["conflicts"], banks["stalls"]
    out: list[BankRange] = []
    i = 0
    while i < len(conf):
        if not conf[i] and not stalls[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(conf) and (conf[j + 1], stalls[j + 1]) == (conf[i], stalls[i]):
            j += 1
        out.append(BankRange(spans(list(range(i, j + 1))), conf[i], stalls[i]))
        i = j + 1
    return out


def _owners(cluster: Any) -> dict[str, str]:
    """Component -> the accelerator it belongs to: the accelerator itself and its streamers."""
    out: dict[str, str] = {}
    for c in cluster.components:
        if c.kind == "accel":
            out[c.name] = c.name
            for s in (c.attach or {}).values():
                out[s] = c.name
    return out


def _next_task(rows: list[TaskRow] | None, after: int) -> TaskRow | None:
    """The first task that starts after cycle ``after`` (the one a wait held back)."""
    later = [r for r in rows or [] if r.start > after]
    return min(later, key=lambda r: (r.start, r.block)) if later else None


def _handed(tasks: Any, src: str, dst: str, owner: dict[str, str]) -> list[str]:
    """``dst``'s streamers whose task waits for a task of ``src`` in the task list."""
    if tasks is None:
        return []
    comp = {
        st.task_name: st.component for st in tasks.steps if getattr(st, "op", None) == "configure"
    }
    out: list[str] = []
    for st in tasks.steps:
        if getattr(st, "op", None) != "configure" or owner.get(st.component) != dst:
            continue
        for a in st.after:
            if owner.get(comp.get(a, "")) == src and st.component not in out:
                out.append(st.component)
    return out


def _chain_text(ch: ChainRow) -> str:
    """``mul → sum: 17 cycles waiting on mul_out, which sum_a reads``."""
    text = f"{ch.src} → {ch.dst}: {ch.cycles} cycles waiting on {ch.block}"
    if not ch.readers:
        return text
    verb = "reads" if len(ch.readers) == 1 else "read"
    return f"{text}, which {', '.join(ch.readers)} {verb}"


def build_run(rv: api.RunView, tasks: Any = None) -> RunReport:
    """The run report of one loaded run directory; ``tasks`` a TaskList for task names."""
    prof = rv.outputs.profile.to_dict()
    total = rv.total_cycles
    tr = rv.outputs.trace
    fc = prof.get("functional_check")
    check = None if not fc else CheckRow(bool(fc["passed"]), fc["kernel"], fc["containers"])
    names = _task_names(tasks)
    task_rows = None
    if tr is not None:
        task_rows = []
        for block, ts in api.tasks(rv).items():
            for k, t in enumerate(ts):
                nm = names.get(block, [])
                task_rows.append(
                    TaskRow(
                        block,
                        k,
                        nm[k] if k < len(nm) else None,
                        t["start"],
                        t["done"],
                        t["done"] - t["start"],
                        t.get("direction"),
                    )
                )
        task_rows.sort(key=lambda r: (r.start, r.block))
    accels = []
    for c in rv.cluster.components:
        if c.kind != "accel" or c.name not in prof["accelerators"]:
            continue
        a = prof["accelerators"][c.name]
        spans_ = rv.spans.get(c.name, [])
        window = sum(d - s for s, d in spans_) if spans_ else None
        busy = a["cycles"]["busy"]
        intervals = tr.intervals.get(c.name) if tr is not None else None
        accels.append(
            AccelRun(
                c.name,
                a["firings"],
                int(c.params.get("ii", 1)),
                achieved_ii([list(x) for x in intervals] if intervals else None, a["firings"]),
                busy,
                window,
                busy / window if window else busy / total,
                a["cycles"].get("stall_in", 0),
                a["cycles"].get("stall_out", 0),
            )
        )
    ctl = prof.get("controller")
    controller = None
    if ctl:
        owner = _owners(rv.cluster)
        waits, chains = [], []
        for w in ctl["waits"]:
            nxt = _next_task(task_rows, w["last"])
            row = WaitRow(
                w["block"], w["mode"], w["first"], w["last"], w["last"] - w["first"] + 1,
                None if nxt is None else (nxt.name or f"{nxt.block} #{nxt.index}"),
            )  # fmt: skip
            waits.append(row)
            src, dst = owner.get(w["block"]), None if nxt is None else owner.get(nxt.block)
            if src and dst and src != dst:
                chains.append(
                    ChainRow(src, dst, w["block"], row.cycles, _handed(tasks, src, dst, owner))
                )
        controller = ControllerRun(
            ctl["name"],
            ctl["cycles"]["command"],
            ctl["cycles"]["wait"],
            ctl["cycles"]["idle"],
            waits,
            chains,
        )
    dmas = []
    for name, d in prof["dmas"].items():
        busy = d["cycles"]["busy"]
        moved = d["bytes_read"] + d["bytes_written"]
        dmas.append(
            DmaRun(
                name,
                busy,
                d["beats_read"],
                d["beats_written"],
                d["bytes_read"],
                d["bytes_written"],
                moved / busy if busy else 0.0,
            )
        )
    streamers = []
    for name, s in prof["streamers"].items():
        port_stalls = sum(prof["ports"][p]["stalls"] for p in s["ports"])
        streamers.append(
            StreamerRun(
                name,
                s["cycles"].get("stall_xbar", 0),
                s["cycles"].get("stall_fifo", 0),
                port_stalls,
                s["fifo"]["depth"],
                max(s["fifo"]["max"], default=0),
            )
        )
    movement, reason = None, None
    dma_set = dma_names(rv.cluster)
    mv = api.movement_view(rv)
    if not mv["available"]:
        reason = mv["reason"]
    else:
        res = [
            ResidencyRow(
                r["region"],
                r["mem"],
                r["arrival"],
                r["use"],
                r["departure"],
                r["wait_in"],
                r["wait_out"],
                r["present_at_start"],
            )
            for r in mv["residency"]
        ]
        pats = []
        for p in mv["patterns"]:
            if p["owner"] in dma_set and "side" not in p:
                continue  # the DMA's L1 side as the xbar saw it: its own rows say it
            el = p.get("element")
            if el:
                text = _nest_text(el["region"], el["base"], el["nest"])
            elif p["addr"]:
                text = _nest_text("addr", p["addr"]["base"], p["addr"]["nest"])
            else:
                text = "irregular"
            held = [_held_text(h) for h in p.get("held", [])[:HELD_SHOWN]]
            side = (
                f" ({p['mem'].upper()} {'reads' if p.get('side') == 'src' else 'writes'})"
                if "side" in p
                else ""
            )
            pats.append(
                PatternRow(
                    p["port"] + side,
                    p["task"],
                    text,
                    p["beats"],
                    p["first"],
                    p["last"],
                    p["ideal_last"],
                    p.get("held_cycles", 0),
                    held,
                )
            )
        movement = Movement(res, pats, mv["conflicts"]["count"], mv["filtered"])
    return RunReport(
        rv.name,
        str(rv.outputs.run.get("scenario")),
        total,
        rv.trace_level,
        check,
        task_rows,
        accels,
        controller,
        dmas,
        _bank_ranges(prof["banks"]),
        streamers,
        movement,
        reason,
    )


# =============================================================================
# Rendering
# =============================================================================


def render_run(r: RunReport) -> str:
    """``run.md``: the report as Markdown, one section per part."""
    out = [f"# Run report: {r.name}", ""]
    if r.check is None:
        check = "none (not a flow run)"
    else:
        parts = [
            f"{c} ({x['elements']} elements, from {x['memory']}) "
            + (
                "equals the reference and REF1"
                if x["reference"] and x["ref1"]
                else f"DIFFERS in {x['mismatches']} elements"
            )
            for c, x in r.check.containers.items()
        ]
        check = ("passed: " if r.check.passed else "FAILED: ") + "; ".join(parts)
    out += table(
        ["", ""],
        [
            ["Scenario", r.scenario],
            ["Cycles", r.cycles],
            ["Trace", r.trace_level],
            ["Functional check", check],
        ],
    )
    out += ["", "## Tasks", ""]
    if r.tasks is None:
        out.append("Needs at least a task trace (--trace task).")
    else:
        out += table(
            ["Block", "#", "Task", "Start", "Done", "Cycles", "Direction"],
            [
                [
                    t.block,
                    t.index,
                    t.name,
                    t.start,
                    t.done,
                    t.cycles,
                    DIRECTION.get(t.direction or "", t.direction),
                ]
                for t in r.tasks
            ],
            {1, 3, 4, 5},
        )
    out += ["", "## Accelerators", ""]
    out += table(
        [
            "Accelerator",
            "Firings",
            "Target II",
            "Achieved II",
            "Busy",
            "Task window",
            "Utilisation",
            "Starved (stall_in)",
            "Blocked (stall_out)",
        ],
        [
            [
                a.name,
                a.firings,
                a.target_ii,
                None if a.achieved_ii is None else f"{a.achieved_ii:.2f}",
                a.busy,
                a.window,
                f"{100 * a.utilisation:.1f}%",
                a.stall_in,
                a.stall_out,
            ]
            for a in r.accelerators
        ],
        {1, 2, 3, 4, 5, 6, 7, 8},
    )
    out.append("")
    out.append(
        "Achieved II is the cycles from the first busy cycle to the end of the last, divided by the firings; utilisation is over the task window."
    )
    if r.controller:
        c = r.controller
        out += [
            "",
            "## Controller",
            "",
            f"{c.name}: {c.command} cycles issuing commands, {c.wait} waiting, {c.idle} idle.",
            "",
        ]
        out += table(
            ["Waits on", "Mode", "From", "To", "Cycles", "Then starts"],
            [[w.block, w.mode, w.first, w.last, w.cycles, w.then or "–"] for w in c.waits],
            {2, 3, 4},
        )
        if c.chains:
            out += [
                "",
                "Chaining waits (one accelerator's output read by the next through L1):",
                "",
            ]
            out += [f"- {_chain_text(ch)}" for ch in c.chains]
    if r.dmas:
        out += ["", "## DMA", ""]
        out += table(
            [
                "DMA",
                "Busy",
                "Beats read",
                "Beats written",
                "Bytes read",
                "Bytes written",
                "Bytes per busy cycle",
            ],
            [
                [
                    d.name,
                    d.busy,
                    d.beats_read,
                    d.beats_written,
                    d.bytes_read,
                    d.bytes_written,
                    f"{d.bytes_per_busy_cycle:.1f}",
                ]
                for d in r.dmas
            ],
            {1, 2, 3, 4, 5, 6},
        )
    out += ["", "## Memory", ""]
    if r.banks:
        out += table(
            ["Banks", "Conflicts per bank", "Stalls per bank"],
            [[b.banks, b.conflicts, b.stalls] for b in r.banks],
            {1, 2},
        )
    else:
        out.append("No bank conflicts.")
    out.append("")
    out += table(
        [
            "Streamer",
            "Held by the xbar (cycles)",
            "Waiting on its FIFO (cycles)",
            "Port stalls",
            "FIFO highest / depth",
        ],
        [
            [s.name, s.stall_xbar, s.stall_fifo, s.port_stalls, f"{s.fifo_max} / {s.fifo_depth}"]
            for s in r.streamers
        ],
        {1, 2, 3},
    )
    out += ["", "## Data movement", ""]
    if r.movement is None:
        out.append(r.movement_reason or "Not available.")
    else:
        m = r.movement
        if m.filtered:
            out += [
                f"The beat trace is filtered ({m.filtered}): the numbers cover what it kept.",
                "",
            ]
        out += table(
            [
                "Region",
                "Memory",
                "Arrival",
                "Use",
                "Departure",
                "Wait before use",
                "Wait before leaving",
                "There from the start",
            ],
            [
                [
                    x.region,
                    x.mem.upper(),
                    span_text(x.arrival),
                    span_text(x.use),
                    span_text(x.departure),
                    span_text(x.wait_in),
                    span_text(x.wait_out),
                    x.present_at_start,
                ]
                for x in m.residency
            ],
            {7},
        )
        out += ["", f"L1 conflicts: {m.conflicts}.", ""]
        out += table(
            [
                "Port",
                "Task",
                "Pattern",
                "Beats",
                "First",
                "Last",
                "Last at one beat a cycle",
                "Held back (cycles)",
                "First held back",
            ],
            [
                [
                    p.port,
                    p.task,
                    p.pattern,
                    p.beats,
                    p.first,
                    p.last,
                    p.ideal_last,
                    p.held_cycles,
                    "; ".join(p.held) or None,
                ]
                for p in m.patterns
            ],
            {1, 3, 4, 5, 6, 7},
        )
    return "\n".join(out) + "\n"
