"""SNAX-GVSoC's events -> a run's profile and trace.

    convert(lines, cluster, tasks, total_cycles, level, functional_check, regions, filter)
        -> (Profile, Trace)

``lines`` are events.log's (events.py), ``cluster`` the run's (cluster.py), ``regions`` the
design point's (regions.py), ``filter`` the sources whose beat lines the run wrote (sources.py).
The result, its names, window and fields, is CONTRACTS, "Profile and trace". Nothing is
recounted: the profile is the ``summary`` lines renamed (``profile_of``), the trace the
``events`` lines renamed, matched in order to the task list's tasks and cut to the ``window``,
and above ``task`` the beat lines too (beats.py). A DMA issue's ``direction`` comes from its
addresses: ``l2_to_l1``, ``l1_to_l2``, ``l1_to_l1`` or ``l2_to_l2``. ``invariants`` hold on every
conversion: every start and DMA issue has a done in the window, a core's classes add up to it,
the accelerator's firings are its summary's, and each container's L2 copies touch its array's
bytes exactly (``l2_copies``). Each failure is a ``RunError`` (``convert.<rule>``); a linked
group's beat lines are one (``convert.level``), not read yet.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from snax_forge.lower.tasks import TaskList

from .beats import Context, beat_events, beat_invariants, mover_tasks, pairing
from .cluster import DMA, Cluster, core_name, linked
from .cores import core_counts, core_spans, merge_spans, start_mode
from .events import Line, RunError
from .profile import (
    AccelProfile,
    BankProfile,
    CoreProfile,
    DmaProfile,
    FifoProfile,
    PortProfile,
    Profile,
    StreamerProfile,
    WaitProfile,
)
from .regions import Region
from .trace import Trace, at_least

XBAR = "xbar"
UNIT_CLASSES = ("busy", "stall_in", "stall_out", "flush")
MOVER_CLASSES = ("busy", "stall_xbar", "stall_fifo")


def source_of(line: Line, cluster: Cluster) -> str:
    """The trace name of a line's component (the crossbar is ``xbar``, the streamer its mover)."""
    segs = line.path.split("/")
    last, parent = segs[-1], segs[-2] if len(segs) > 1 else ""
    acc = cluster.accelerator
    unit = acc.unit_at(last) if parent == f"snax_acc{acc.core}" else None
    if unit is not None:
        return unit.name
    s = acc.set_at(line.path)
    if s is not None:
        mover = line.fields.get("mover")
        if mover is None:
            raise RunError("convert.path", f"a streamer line without its mover: {line}")
        return acc.mover(mover, s.name).name
    if last == "snax_xbar":
        return XBAR
    if parent == "idma" and last in ("fe", "axi_read", "axi_write"):
        return DMA
    m = re.fullmatch(r"pe(\d+)", last)
    if m and int(m[1]) < cluster.nb_cores:
        return core_name(int(m[1]))
    raise RunError("convert.path", f"{line.path}: no component of the cluster")


def port_of(name: str, cluster: Cluster) -> str:
    """A crossbar input's trace port name (CONTRACTS, "Profile and trace")."""
    acc = cluster.accelerator
    for s in acc.csrs.sets:
        m = re.fullmatch(rf"{re.escape(s.xbar)}\.(\w+)\.lane(\d+)", name)
        if m:
            return f"{acc.mover(m[1], s.name).name}.lane{m[2]}"
    if name == "axi" or name in cluster.cores:
        return name
    raise RunError("convert.path", f"crossbar input {name!r}: no port of the cluster")


def direction(src: int, dst: int, cluster: Cluster) -> str:
    """A copy's direction by its addresses against the cluster's L1 (module doc)."""
    return f"{'l1' if cluster.is_l1(src) else 'l2'}_to_{'l1' if cluster.is_l1(dst) else 'l2'}"


# =============================================================================
# The window
# =============================================================================


def _read_in(funcs: list[Line], reads: list[Line], name: str, after: int, core: str) -> Line:
    """The first ``mcycle`` read in the first span of function ``name`` that starts at or after
    cycle ``after``: from its ``func`` line to the core's next one."""
    for i, f in enumerate(funcs):
        if f.fields["name"] != name or f.cycle < after:
            continue
        end = funcs[i + 1].cycle if i + 1 < len(funcs) else None
        inside = [r for r in reads if r.cycle >= f.cycle and (end is None or r.cycle < end)]
        if inside:
            return inside[0]
    raise RunError("convert.window", f"{core}: no mcycle read in {name} (no `mcycle` line in a "
                   f"span of `func name={name}`)")  # fmt: skip


def window(lines: Sequence[tuple[str, Line]], cluster: Cluster, total_cycles: int) -> int:
    """The cycle of the program's first ``mcycle`` read, the window's start; its length checked
    against ``total_cycles`` (CONTRACTS, "Profile and trace")."""
    core = core_name(cluster.accelerator.core)
    mine = _core_lines(lines, core)
    funcs = [x for x in mine if x.kind == "func"]
    reads = [x for x in mine if x.kind == "mcycle"]
    first = _read_in(funcs, reads, "main", 0, core)
    last = _read_in(funcs, reads, "forge_check", first.cycle, core)
    if last.cycle - first.cycle != total_cycles:
        raise RunError(
            "convert.window",
            f"{core}'s mcycle reads at {first.cycle} (main) and {last.cycle} (forge_check) are "
            f"{last.cycle - first.cycle} cycles apart, the program printed forge cycles "
            f"{total_cycles}",
        )
    return first.cycle


# =============================================================================
# The trace
# =============================================================================


def _tasks(tasks: TaskList, kind: str, component: str) -> list[str]:
    return [s.task_name for s in tasks.of_type(kind) if s.component == component]


def _dma_tasks(tasks: TaskList) -> list[tuple[str, str]]:
    return [(s.task_name, str(s.values.get("direction"))) for s in tasks.of_type("dma")]


def _need(names: list[str], k: int, what: str) -> str:
    if k >= len(names):
        raise RunError("convert.tasks", f"{what}: start {k + 1} in the window, the task list has "
                       f"{len(names)} ({names})")  # fmt: skip
    return names[k]


def run_classes(lines: Sequence[tuple[str, Line]], cluster: Cluster
                ) -> dict[tuple[str, int], dict[str, int]]:  # fmt: skip
    """Per (accelerator or mover, its run index among its starts), its summary line's classes
    by name (CONTRACTS, "Profile and trace")."""
    acc = cluster.accelerator
    out = {}
    for s, x in lines:
        if x.trace != "summary":
            continue
        if x.kind == "run" and s in acc.unit_names:
            out[(s, x.fields["index"])] = {c: x.fields[c] for c in UNIT_CLASSES}
        elif x.kind == "mover":
            out[(s, x.fields["run"])] = {c: x.fields[c] for c in (*MOVER_CLASSES, "idle")}
    return out


def trace_events(lines: Sequence[tuple[str, Line]], cluster: Cluster, tasks: TaskList,
                 boot: int, width: int) -> list[tuple[int, int, dict[str, Any]]]:  # fmt: skip
    """The events lines renamed, shifted and cut (module doc), in time order, each with its
    line's index in the file."""
    acc = cluster.accelerator
    out: list[tuple[int, int, dict[str, Any]]] = []
    started: dict[str, list[str]] = {}  # src -> its tasks, in start order
    done_count: dict[str, int] = {}
    runs: dict[str, int] = {}  # src -> its starts so far, the window's or not
    run_of: dict[str, int] = {}  # task -> its run index among its component's starts
    classes = run_classes(lines, cluster)
    issued: dict[int, str] = {}  # DMA copy id -> its task
    dma_tasks = _dma_tasks(tasks)
    order = sorted(enumerate(lines), key=lambda x: (x[1][1].cycle, x[0]))
    for n, (src, x) in order:
        if x.trace != "events" or x.kind in ("mcycle", "mcycleh"):
            continue
        f, k = x.fields, x.kind
        run = None
        if k == "start":
            run = runs.get(src, 0)
            runs[src] = run + 1
        t = x.cycle - boot
        if k == "done" and src != DMA:
            t += 1
        if k in ("poll", "barrier"):
            t = f["first"] - boot
        if not 0 <= t < width:
            continue
        e: dict[str, Any] = {"t": t, "k": k, "src": src}
        if k == "start":
            kind = "accel" if src in acc.unit_names else "streamer"
            mine = started.setdefault(src, [])
            shared = next((m for m in acc.movers if m.name == src and m.users), None)
            got = _tasks(tasks, kind, src) if shared is None else mover_tasks(tasks, shared)
            e["task"] = _need(got, len(mine), f"{src}'s tasks")
            mine.append(e["task"])
            run_of[e["task"]] = run
        elif k == "done" and src == DMA:
            if f["id"] not in issued:
                raise RunError("convert.tasks", f"DMA copy {f['id']} done at {t}, never issued "
                               "in the window")  # fmt: skip
            e.update(task=issued[f["id"]], id=f["id"])
        elif k == "done":
            i = done_count.get(src, 0)
            done_count[src] = i + 1
            mine = started.get(src, [])
            if i >= len(mine):
                raise RunError("convert.tasks", f"{src}: done at {t} with no start before it")
            e["task"] = mine[i]
            cycles = classes.get((src, run_of[mine[i]]))
            if cycles is None:
                raise RunError("convert.summary", f"{src} ({mine[i]}): no summary line for its run "
                               f"{run_of[mine[i]]}")  # fmt: skip
            e["cycles"] = cycles
            if src in acc.unit_names:
                e.update(firings=f["firings"], first=f["first"] - boot, last=f["last"] - boot)
        elif k in ("csr", "poll"):
            e.update(pc=f["pc"], csr=f["csr"], block=_block(acc, f["csr"]))
            if k == "csr":
                e.update(op=f["op"], value=f["value"])
                if "held" in f:
                    e["held"] = f["held"]
            else:
                e.update(value=f["value"], count=f["count"], last=f["last"] - boot)
        elif k == "barrier":
            e["last"] = f["last"] - boot
        elif k == "state":
            e["mode"] = f["mode"]
        elif k == "func":
            e.update(name=f["name"], pc=f["pc"])
        elif k == "issue":
            i = len(issued)
            if i >= len(dma_tasks):
                raise RunError("convert.tasks", f"DMA copy {i + 1} issued in the window, the task "
                               f"list has {len(dma_tasks)} ({[d for d, _ in dma_tasks]})")  # fmt: skip
            task, want = dma_tasks[i]
            d = direction(f["src"], f["dst"], cluster)
            if d != want:
                raise RunError("convert.tasks", f"DMA copy {f['id']} ({task}) goes {d} by its "
                               f"addresses, the task list says {want}")  # fmt: skip
            issued[f["id"]] = task
            e.update(task=task, id=f["id"], dir=d, size=f["size"], reps=f["reps"])
            e.update({"from": f["src"], "to": f["dst"], "from_stride": f["src_stride"],
                      "to_stride": f["dst_stride"]})  # fmt: skip
        out.append((t, n, e))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _block(acc: Any, csr: int) -> str:
    """The block a CSR belongs to, ``-`` for none (``Accelerator.block_of``)."""
    return acc.block_of(csr) or "-"


def _core_lines(lines: Sequence[tuple[str, Line]], core: str) -> list[Line]:
    return sorted((x for s, x in lines if s == core and x.trace == "events"), key=lambda x: x.cycle)


def _states(mine: list[Line], boot: int) -> list[dict[str, Any]]:
    return [{"t": x.cycle - boot, "k": "state", "mode": x.fields["mode"]}
            for x in mine if x.kind == "state"]  # fmt: skip


def core_start(lines: Sequence[tuple[str, Line]], core: str, boot: int) -> dict[str, Any]:
    """What a core is in at t = 0: its function and its state (CONTRACTS, "Profile and trace")."""
    mine = _core_lines(lines, core)
    funcs = [x for x in mine if x.kind == "func" and x.cycle <= boot]
    return {"func": funcs[-1].fields["name"] if funcs else None,
            "mode": start_mode(_states(mine, boot))}  # fmt: skip


def core_cycles(lines: Sequence[tuple[str, Line]], core: str, boot: int, width: int
                ) -> tuple[dict[str, int], list[tuple[int, int, str, Line]]]:  # fmt: skip
    """A core's classes in the window, and its waits clipped to it: first, last, kind and
    line (CONTRACTS, "Profile and trace")."""
    mine = _core_lines(lines, core)
    events: list[dict[str, Any]] = _states(mine, boot)
    waits = []
    for x in mine:
        if x.kind in ("poll", "barrier"):
            first, last = x.fields["first"] - boot, x.fields["last"] - boot
            events.append({"t": first, "k": x.kind, "last": last})
            a, b = max(first, 0), min(last, width - 1)
            if a <= b:
                waits.append((a, b, x.kind, x))
    waits.sort(key=lambda w: (w[0], w[2] == "barrier"))  # a poll before a barrier of one cycle
    return core_counts(core_spans(events, width)), waits


# =============================================================================
# The profile
# =============================================================================


def profile_of(lines: Sequence[tuple[str, Line]], cluster: Cluster, total_cycles: int,
               functional_check: dict[str, Any] | None) -> Profile:  # fmt: skip
    """The summary lines renamed (module doc); the cores' cycles are left None."""
    acc = cluster.accelerator
    summary = [(s, x) for s, x in lines if x.trace == "summary"]
    p = Profile(total_cycles=total_cycles, functional_check=functional_check)

    for unit in acc.unit_names:
        runs = [x.fields for s, x in summary if s == unit and x.kind == "run"]
        c = {k: sum(r[k] for r in runs) for k in UNIT_CLASSES}
        p.accelerators[unit] = AccelProfile(
            {**c, "idle": total_cycles - sum(c.values())}, sum(r["firings"] for r in runs),
            c["busy"] / total_cycles if total_cycles else 0.0,
        )  # fmt: skip
    for m in acc.movers:
        runs = [x.fields for s, x in summary if s == m.name and x.kind == "mover"]
        c = {k: sum(r[k] for r in runs) for k in MOVER_CLASSES}
        qmax = [max([r["qmax"][lane] for r in runs if lane < len(r["qmax"])], default=0)
                for lane in range(m.lanes)]  # fmt: skip
        p.streamers[m.name] = StreamerProfile(
            m.write, {**c, "idle": total_cycles - sum(c.values())}, m.ports,
            FifoProfile(m.fifo, m.fifo_depth, qmax),
        )  # fmt: skip
    for core in cluster.cores:
        csrs = [x.fields for s, x in summary if s == core and x.kind == "csrs"]
        if len(csrs) != 1:
            raise RunError("convert.summary", f"{core}: {len(csrs)} `csrs` summary lines, not 1")
        p.cores[core] = CoreProfile(None, csrs[0]["writes"], csrs[0]["reads"])
    dma = [x.fields for s, x in summary if s == DMA and x.kind == "dma"]
    if len(dma) != 1:
        raise RunError("convert.summary", f"the DMA: {len(dma)} `dma` summary lines, not 1")
    d = dma[0]
    p.dmas[DMA] = DmaProfile(d["transfers"], d["busy"], d["r_beats"], d["w_beats"],
                             d["bytes_to_l1"], d["bytes_from_l1"])  # fmt: skip
    for x in sorted((x for s, x in summary if s == XBAR and x.kind == "input"),
                    key=lambda x: x.fields["index"]):  # fmt: skip
        f = x.fields
        p.ports[port_of(f["name"], cluster)] = PortProfile(
            f["grants"], f["stalls"], f["stalls_dma"], bool(f["whole_run"])
        )
    banks = sorted((x.fields for s, x in summary if s == XBAR and x.kind == "bank"),
                   key=lambda f: f["index"])  # fmt: skip
    if banks:
        if [b["index"] for b in banks] != list(range(cluster.banks)):
            raise RunError("convert.summary", f"bank lines {[b['index'] for b in banks]}, the "
                           f"cluster has {cluster.banks} banks")  # fmt: skip
        p.banks = BankProfile([b["reads"] for b in banks], [b["writes"] for b in banks],
                              [b["conflicts"] for b in banks])  # fmt: skip
    return p


# =============================================================================
# The conversion
# =============================================================================


def l2_copies(trace: Trace, regions: list[Region], cluster: Cluster) -> list[str]:
    """Each container's L2 bytes against its copies' (module doc); each that fails, said."""
    beat = cluster.dma_beat_bytes
    l2 = {r.container: r for r in regions if r.memory == "l2"}
    touched: dict[str, list[tuple[int, int]]] = {c: [] for c in l2}
    bad = []
    for e in trace.of_kind("issue"):
        sides = [s for s in ("from", "to") if not cluster.is_l1(e[s])]
        if not sides:
            continue
        m = re.fullmatch(r"(?:load|store)_(\w+)", e["task"])
        if m is None or m[1] not in l2:
            bad.append(f"DMA copy {e['id']} ({e['task']}) touches L2 and is no container's load or "
                       f"store (the containers in L2: {sorted(l2)})")  # fmt: skip
            continue
        for s in sides:
            for row in range(e["reps"]):
                lo = e[s] + row * e[f"{s}_stride"]
                touched[m[1]].append((lo, lo + e["size"]))
    for name, r in l2.items():
        want = [(r.base, r.base + r.size)]
        got = merge_spans(touched[name])
        if got != want:
            bad.append(f"container {name}: its copies touch L2 {[(hex(a), hex(b)) for a, b in got]}, "
                       f"its array ({r.size // r.itemsize} {r.dtype} elements, its padded shape "
                       f"in {beat}-byte beats) is [{want[0][0]:#x}, {want[0][1]:#x})")  # fmt: skip
    return bad


def invariants(profile: Profile, trace: Trace, cluster: Cluster,
               ctx: Context | None = None, regions: list[Region] | None = None) -> list[str]:  # fmt: skip
    """What every converted run must hold (module doc); each that fails, said. ``ctx``, the
    beat part's, adds its invariants; ``regions``, the run's, the L2 copies'."""
    bad = []
    for src in trace.sources:
        starts = len(trace.of_kind("issue" if src == DMA else "start", src))
        dones = len(trace.of_kind("done", src))
        if starts != dones:
            bad.append(f"{src}: {starts} {'issues' if src == DMA else 'starts'} and {dones} "
                       "dones in the window")  # fmt: skip
    for name, c in profile.cores.items():
        if c.cycles is not None and sum(c.cycles.values()) != trace.total_cycles:
            bad.append(f"{name}: classes add up to {sum(c.cycles.values())}, the window is "
                       f"{trace.total_cycles}")  # fmt: skip
    for acc in cluster.accelerator.unit_names:
        fired = sum(e["firings"] for e in trace.of_kind("done", acc))
        if fired != profile.accelerators[acc].firings:
            bad.append(f"{acc}: {fired} firings in its done events, "
                       f"{profile.accelerators[acc].firings} in its summary")  # fmt: skip
    if regions is not None:
        bad += l2_copies(trace, regions, cluster)
    if ctx is not None:
        bad += beat_invariants(trace, profile, ctx)
    return bad


def convert(
    lines: Sequence[Line],
    cluster: Cluster,
    tasks: TaskList,
    total_cycles: int,
    level: str,
    functional_check: dict[str, Any] | None = None,
    regions: list[Region] | None = None,
    filter: list[str] | None = None,
) -> tuple[Profile, Trace | None]:
    """The profile and, above ``off``, the trace of a run (module doc)."""
    if at_least(level, "beat") and linked(cluster.accelerator.description):
        raise RunError("convert.level", f"trace {level}: the beat lines of a group of several "
                       "linked units are not read yet; task and off are")  # fmt: skip
    named = [(source_of(x, cluster), x) for x in lines]
    profile = profile_of(named, cluster, total_cycles, functional_check)
    if level == "off":
        return profile, None
    boot = window(named, cluster, total_cycles)
    triples = trace_events(named, cluster, tasks, boot, total_cycles)
    ctx = None
    if at_least(level, "beat"):
        ctx = Context(cluster, list(regions or []), boot, total_cycles, level, list(filter or []),
                      [e for _, _, e in triples], named, pairing(tasks, cluster))  # fmt: skip
        triples = sorted(triples + beat_events(named, ctx), key=lambda x: (x[0], x[1]))
    events = [e for _, _, e in triples]
    for core in cluster.cores:
        cycles, waits = core_cycles(named, core, boot, total_cycles)
        p = profile.cores[core]
        p.cycles = cycles
        p.waits = [WaitProfile(
            "barrier" if kind == "barrier" else _block(cluster.accelerator, x.fields["csr"]),
            a, b) for a, b, kind, x in waits]  # fmt: skip
    start = {core: core_start(named, core, boot) for core in cluster.cores}
    trace = Trace(level, total_cycles, boot, cluster.sources, events, list(filter or []), start)
    bad = invariants(profile, trace, cluster, ctx, regions)
    if bad:
        raise RunError("convert.invariant", "; ".join(bad))
    return profile, trace
