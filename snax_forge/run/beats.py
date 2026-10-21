"""The converter's beat, stall and instruction part (C§7, D137): SNAX-GVSoC's ``beats``,
``stalls`` and ``insn`` lines renamed, shifted and cut to the window, with what no view should
look up, and the invariants they must hold.

    beat_events(named, ctx) -> [(t, n, event)]
    beat_invariants(trace, profile, ctx) -> [what fails]

``named`` are events.log's lines with their C§7 source (convert.py); ``ctx`` (``Context``) is
the rest of the run: the cluster, the design point's regions, the window, the filter, and
what the task level made of the events (each component's tasks, the DMA's copies, the cores'
functions).

**Names.** A crossbar line's ``in`` becomes its port (the summary's ``input`` lines name each
index; ``acc_a.lane0``, ``core0``, ``axi``), a stall's ``by`` too (-1, the DMA's superbank,
is ``dma``); a streamer line's ``mover`` its mover (``reader0`` -> ``acc_a``); a ``fire``'s
``b``, ``value`` and ``vals`` are dicts keyed by reader mover (the unit's inputs in port
order are the streamer's readers in order, each beat ``lanes * data_width / 64`` words).
Lines of a source the filter does not name are dropped (the streamer writes every mover's).

**Lookups.** Every cycle and ``back`` is shifted by the boot and lines outside the window are
dropped. ``task`` is the task whose start and done bracket the line's ``t`` on its component
(a port's mover, the accelerator), a copy's for the DMA, null for a core's or AXI's access.
``mem`` is ``l1`` for the crossbar's lines and ``l2`` for a burst, ``addr`` absolute (the
crossbar's TCDM offset plus the L1's base; ``by_addr`` likewise). ``region`` is the design
point's container the address is in (null if none), ``elems`` the flat indexes of its
elements in the word (a DMA beat's words together), ``vals`` their values by the region's
dtype; a fire's and a push's come from the grants of their beats (its reader's ``b``, the
writer's ``b = n``), null for a mover not in the trace. The program's L2 data is placed by its
linker, not at the design point's L2 layouts, so a burst's ``region`` is its copy's L1 side's
for the same bytes (through the copy's ``issue``: row, offset). An ``insn``'s ``func`` is the
function its core was in at ``t`` (its last ``func`` line at or before it, boot included),
``text`` the instruction as GVSoC writes it.

**Invariants**, each checked where the trace has its lines (its level and filter), each
failure named:

- the grants per SNAX port in the window are its summary's; per bank, the SNAX grants plus
  the DMA's words are its reads and writes;
- per copy, the ``dma`` lines into L1 are its written beats and those out of L1 its read
  beats, counted as the iDMA counts them (``dma_beat_bytes``), and over all copies those are
  the summary's ``r_beats`` and ``w_beats``;
- per accelerator run, the ``fire`` lines are its done's firings (and their sum the
  summary's), the ``push`` lines firings over its rate, the writer's grants pushes times its
  lanes;
- every ``b`` a fire names was granted on every lane of its mover with the fire's words;
  every push's words are its writer grants' (``b = n``);
- every L1 line's ``bank`` and ``row`` are its address's (``Cluster.bank_row``);
- each mover's grant addresses in a task, lane by lane in time order, are the nest its CSR
  writes gave that task: the pointer, a spatial stride per lane, the temporal bounds and
  strides (``agu_nest``, as the streamer's AGU walks it; with a repeat, loop 0 once);
- at ``stall``: the stalls per SNAX port and those lost to the DMA are its summary's; per
  bank, the cycles with two or more requesters (SNAX grants and stalls, and the DMA when it
  took the bank) its conflicts; each lane's ``fifo`` counts are the words visible (a read's
  ``back`` plus one, a push's ``t`` plus one) minus those popped (its last fire, the
  writer's grant) in every cycle, at most the depth, 0 at the end, their highest the
  summary's ``qmax``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .cluster import DMA, Cluster, Mover
from .events import Line, RunError
from .profile import Profile
from .regions import Region, region_at
from .trace import LEVELS, Trace

XBAR = "xbar"
WORD_MASK = (1 << 64) - 1


@dataclass
class Context:
    """What the beat part needs of the run (module doc)."""

    cluster: Cluster
    regions: list[Region]
    boot: int
    width: int
    level: str
    filter: list[str]
    task_events: list[dict[str, Any]] = field(default_factory=list)
    lines: Sequence[tuple[str, Line]] = ()  # every line, named

    def on(self, level: str) -> bool:
        return LEVELS.index(self.level) >= LEVELS.index(level)


# =============================================================================
# The run as the task level sees it
# =============================================================================


def brackets(events: list[dict[str, Any]]) -> dict[str, list[tuple[int, int, str]]]:
    """Per source, its tasks' ``[start, done]`` in the window, in order."""
    starts: dict[str, list[tuple[int, str]]] = defaultdict(list)
    out: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for e in events:
        if e["k"] == "start":
            starts[e["src"]].append((e["t"], e["task"]))
        elif e["k"] == "done" and e["src"] != DMA:
            for t, task in starts[e["src"]]:
                if task == e["task"]:
                    out[e["src"]].append((t, e["t"], task))
    return out


def task_at(spans: list[tuple[int, int, str]], t: int) -> tuple[int, str] | None:
    """The run index and task whose bracket holds ``t``."""
    for k, (a, b, task) in enumerate(spans):
        if a <= t <= b:
            return k, task
    return None


def copies(events: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """The DMA's copies in the window, by id: their ``issue`` events."""
    return {e["id"]: e for e in events if e["k"] == "issue"}


def beats(base: int, size: int, beat: int) -> int:
    """The beats of one row as the iDMA counts them (``IDmaFeXdma::beats``)."""
    return 0 if size == 0 else (base + size - 1) // beat - base // beat + 1


def agu_nest(ptr: int, spatial_bounds: Sequence[int], spatial_strides: Sequence[int],
             bounds: Sequence[int], strides: Sequence[int], lanes: int, mask: int,
             repeat: bool) -> list[list[int]]:  # fmt: skip
    """Each lane's addresses as the streamer's AGU walks them (``agu_push``): loop 0 fastest,
    a temporal stride one bit narrower than the address, the address cut to ``mask``; with a
    repeat (a reader's temporal stride 0 on loop 0), loop 0 once."""
    bounds = list(bounds)
    if repeat and bounds and strides[0] == 0:
        bounds[0] = 1
    out: list[list[int]] = [[] for _ in range(lanes)]
    if any(b == 0 for b in bounds):
        return out
    idx, off = [0] * len(bounds), [0] * len(bounds)
    while True:
        temporal = sum(off)
        for lane in range(lanes):
            spatial, rem = 0, lane
            for sb, ss in zip(spatial_bounds, spatial_strides, strict=True):
                spatial += ss * (rem % sb)
                rem //= sb
            out[lane].append((ptr + temporal + spatial) & mask)
        tick, all_last = True, True
        for d, b in enumerate(bounds):
            last = idx[d] == b - 1
            all_last &= last
            if tick:
                if last:
                    idx[d], off[d] = 0, 0
                else:
                    idx[d] += 1
                    off[d] = (off[d] + (strides[d] & (mask >> 1))) & mask
            tick = tick and last
        if all_last:
            return out


def mover_csrs(cluster: Cluster) -> dict[str, int]:
    """Each mover's first CSR number (its pointer's low word), in the streamer's layout."""
    out, csr = {}, cluster.accelerator.csrs.streamer_first
    for m in cluster.accelerator.movers:
        out[m.name] = csr
        csr += 2 + len(m.spatial_bounds) + 2 * m.temporal_dim
    return out


def nests(ctx: Context) -> dict[tuple[str, int], list[list[int]]]:
    """Per (mover, its run index in the window), each lane's addresses (TCDM offsets) from
    the CSR writes before its start."""
    c = ctx.cluster
    core = f"core{c.accelerator.core}"
    raw = sorted((x for s, x in ctx.lines if s == core and x.trace == "events"
                  and x.kind == "csr" and x.fields["op"] == "w"), key=lambda x: x.cycle)  # fmt: skip
    first = mover_csrs(c)
    mask = c.l1_size - 1
    out = {}
    for m in c.accelerator.movers:
        starts = sorted(x.cycle for s, x in ctx.lines
                        if s == m.name and x.trace == "events" and x.kind == "start")  # fmt: skip
        k = 0
        for s in starts:
            if not 0 <= s - ctx.boot < ctx.width:
                continue
            regs: dict[int, int] = {}
            for x in raw:
                if x.cycle <= s:
                    regs[x.fields["csr"]] = x.fields["value"]
            csr = first[m.name]
            n_sp = len(m.spatial_bounds)
            get = [regs.get(csr + i, 0) for i in range(2 + n_sp + 2 * m.temporal_dim)]
            ptr = get[0]
            sp = get[2 : 2 + n_sp]
            tb = get[2 + n_sp : 2 + n_sp + m.temporal_dim]
            ts = get[2 + n_sp + m.temporal_dim :]
            out[(m.name, k)] = agu_nest(ptr, m.spatial_bounds, sp, tb, ts, m.lanes, mask,
                                        not m.write)  # fmt: skip
            k += 1
    return out


# =============================================================================
# The events
# =============================================================================


def _input_ports(lines: Sequence[tuple[str, Line]], cluster: Cluster) -> dict[int, str]:
    from .convert import port_of  # convert imports this module

    return {x.fields["index"]: port_of(x.fields["name"], cluster) for s, x in lines
            if s == XBAR and x.trace == "summary" and x.kind == "input"}  # fmt: skip


def _bytes(words: Sequence[int], addr: int, size: int, word: int) -> bytes:
    """A beat's bytes from its words, one per bank word it covers (the first may start inside
    its word), ``size`` bytes in all."""
    out, left, a = b"", size, addr
    for w in words:
        n = min(word - a % word, left)
        out += (w & WORD_MASK).to_bytes(8, "little")[:n]
        a, left = a + n, left - n
    return out


def _lookup(ctx: Context, mem: str, addr: int, data: bytes) -> tuple[str | None, list, list]:
    r = region_at(ctx.regions, mem, addr)
    if r is None:
        return None, [], []
    return r.container, r.elems(addr, len(data)), r.vals(addr, data)


def _l1_of_l2(copy: dict[str, Any], cluster: Cluster, addr: int) -> int | None:
    """The L1 address a copy moves the byte at L2 ``addr`` to or from, if any."""
    if cluster.is_l1(copy["from"]) == cluster.is_l1(copy["to"]):
        return None
    l2, l1 = ("from", "to") if cluster.is_l1(copy["to"]) else ("to", "from")
    for r in range(copy["reps"]):
        start = copy[l2] + r * copy[f"{l2}_stride"]
        if start <= addr < start + copy["size"]:
            return copy[l1] + r * copy[f"{l1}_stride"] + addr - start
    return None


def beat_events(named: Sequence[tuple[str, Line]], ctx: Context) -> list[tuple[int, int, dict]]:
    """The beat, stall and instruction lines as events (module doc), with each line's index
    in the file for the order of a cycle."""
    c, acc = ctx.cluster, ctx.cluster.accelerator
    word = c.bank_bytes
    ports = _input_ports(named, c)
    spans = brackets(ctx.task_events)
    dma = copies(ctx.task_events)
    mover_of_port = {p: m for m in acc.movers for p in m.ports}
    readers = [m for m in acc.movers if not m.write]
    writers = [m for m in acc.movers if m.write]
    unit = acc.unit
    beat_words = unit.get("lanes", 1) * unit.get("data_width", 64) // 64
    funcs: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for s, x in named:
        if x.trace == "events" and x.kind == "func":
            funcs[s].append((x.cycle - ctx.boot, x.fields["name"]))
    for f in funcs.values():
        f.sort(key=lambda p: p[0])

    out: list[tuple[int, int, dict]] = []
    grants: dict[tuple[str, int, int], dict[int, dict]] = defaultdict(dict)  # (mover, run, b)
    fires, pushes = [], []
    for n, (src, x) in enumerate(named):
        if x.trace not in ("beats", "stalls", "insn"):
            continue
        f, k = x.fields, x.kind
        t = x.cycle - ctx.boot
        if not 0 <= t < ctx.width:
            continue
        e: dict[str, Any] = {"t": t, "k": k}
        if k in ("grant", "stall"):
            port = ports.get(f["in"])
            if port is None:
                raise RunError("convert.path", f"crossbar input {f['in']}: no `input` line names "
                               "it")  # fmt: skip
            owner = mover_of_port.get(port)
            source = owner.name if owner is not None else port
            if source not in ctx.filter:
                continue
            run = task_at(spans.get(owner.name, []), t) if owner is not None else None
            addr = c.l1_base + f["addr"]
            e.update(src=port, task=run[1] if run else None)
            if k == "grant":
                size = f.get("size", word)
                data = (f["value"] & WORD_MASK).to_bytes(8, "little")[:size]
                region, elems, vals = _lookup(ctx, "l1", addr, data)
                e.update(op=f["op"], mem="l1", addr=addr, bank=f["bank"], row=f["row"],
                         value=f["value"], back=f["back"] - ctx.boot)  # fmt: skip
                if "b" in f:
                    e["b"] = f["b"]
                if "size" in f:
                    e["size"] = f["size"]
                e.update(region=region, elems=elems, vals=vals)
                if owner is not None and run is not None and "b" in f:
                    lane = owner.ports.index(port)
                    grants[(owner.name, run[0], f["b"])][lane] = e
            else:
                region, elems, _ = _lookup(ctx, "l1", addr, bytes(word))
                by = "dma" if f["by"] == -1 else ports.get(f["by"], str(f["by"]))
                e.update(addr=addr, bank=f["bank"], row=f["row"], by=by,
                         by_addr=c.l1_base + f["by_addr"])  # fmt: skip
                if "b" in f:
                    e["b"] = f["b"]
                e.update(region=region, elems=elems)
        elif k == "dma":
            if DMA not in ctx.filter:
                continue
            copy = dma.get(f["id"])
            addr = c.l1_base + f["addr"]
            data = _bytes(f["value"], addr, f["size"], word)
            region, elems, vals = _lookup(ctx, "l1", addr, data)
            e.update(src=DMA, task=copy["task"] if copy else None, id=f["id"], i=f["i"],
                     op=f["op"], mem="l1", addr=addr, size=f["size"], bank=f["bank"],
                     row=f["row"], value=list(f["value"]), l2=f["l2"])  # fmt: skip
            if "back" in f:
                e["back"] = f["back"] - ctx.boot
            e.update(region=region, elems=elems, vals=vals)
        elif k == "burst":
            if DMA not in ctx.filter:
                continue
            copy = dma.get(f["id"])
            l1 = _l1_of_l2(copy, c, f["addr"]) if copy else None
            region = None
            if l1 is not None:
                r = region_at(ctx.regions, "l1", l1)
                region = r.container if r else None
            e.update(src=DMA, task=copy["task"] if copy else None, id=f["id"], op=f["op"],
                     mem="l2", addr=f["addr"], size=f["size"], back=f["back"] - ctx.boot,
                     region=region)  # fmt: skip
        elif k == "fire":
            if acc.name not in ctx.filter:
                continue
            run = task_at(spans.get(acc.name, []), t)
            if len(f["b"]) != len(readers) or len(f["value"]) != len(readers) * beat_words:
                raise RunError("convert.beats", f"fire at {t}: {len(f['b'])} beats of "
                               f"{len(f['value'])} words, the accelerator reads {len(readers)} "
                               f"of {beat_words}")  # fmt: skip
            e.update(src=acc.name, task=run[1] if run else None, n=f["n"],
                     b={m.name: f["b"][i] for i, m in enumerate(readers)},
                     value={m.name: f["value"][i * beat_words : (i + 1) * beat_words]
                            for i, m in enumerate(readers)})  # fmt: skip
            fires.append((e, run))
        elif k == "push":
            if acc.name not in ctx.filter:
                continue
            run = task_at(spans.get(acc.name, []), t)
            e.update(src=acc.name, task=run[1] if run else None, n=f["n"], first=f["first"],
                     last=f["last"], value=list(f["value"]))  # fmt: skip
            pushes.append((e, run))
        elif k == "fifo":
            m = acc.mover(f["mover"])
            if m.name not in ctx.filter:
                continue
            run = task_at(spans.get(m.name, []), t)
            e.update(src=m.name, task=run[1] if run else None, lane=f["lane"], count=f["count"])
        elif k == "insn":
            if src not in ctx.filter:
                continue
            spans_f = [name for ft, name in funcs.get(src, []) if ft <= t]
            e.update(src=src, pc=f["pc"], func=spans_f[-1] if spans_f else None, text=f["text"])
        else:
            raise RunError("convert.kind", f"{x.trace} {k} line: no event for it ({x})")
        out.append((t, n, e))

    def vals_of(mover: Mover, run: tuple[int, str] | None, b: int, words: list[int]) -> Any:
        g = grants.get((mover.name, run[0], b)) if run else None
        if not g or len(g) != mover.lanes:
            return None
        out: list[Any] = []
        for lane in range(mover.lanes):
            ge = g[lane]
            data = (words[lane] & WORD_MASK).to_bytes(8, "little") if lane < len(words) else b""
            r = region_at(ctx.regions, "l1", ge["addr"])
            out += r.vals(ge["addr"], data) if r else []
        return out

    for e, run in fires:
        e["vals"] = {m.name: vals_of(m, run, e["b"][m.name], e["value"][m.name]) for m in readers}
    for e, run in pushes:
        e["vals"] = vals_of(writers[0], run, e["n"], e["value"]) if writers else None
    return out


# =============================================================================
# The invariants
# =============================================================================


def _run_of(spans: dict[str, list[tuple[int, int, str]]], src: str, task: str | None) -> int:
    for k, (_, _, name) in enumerate(spans.get(src, [])):
        if name == task:
            return k
    return -1


def beat_invariants(trace: Trace, profile: Profile, ctx: Context) -> list[str]:
    """What the beat and stall events must hold (module doc); each that fails, said."""
    if not ctx.on("beat"):
        return []
    c, acc = ctx.cluster, ctx.cluster.accelerator
    bad: list[str] = []
    ev = trace.events
    spans = brackets(ev)
    of = defaultdict(list)
    for e in ev:
        of[e["k"]].append(e)
    traced = [m for m in acc.movers if m.name in ctx.filter]
    snax_ports = {p for m in traced for p in m.ports}
    all_in = len(traced) == len(acc.movers) and DMA in ctx.filter

    # grants per SNAX port, and per bank with the DMA's words
    count = defaultdict(int)
    for e in of["grant"]:
        count[e["src"]] += 1
    for p in sorted(snax_ports):
        want = profile.ports[p].grants if p in profile.ports else 0
        if count[p] != want:
            bad.append(f"{p}: {count[p]} grant lines in the window, {want} in its summary")
    if all_in and profile.banks is not None:
        rw = {"r": [0] * c.banks, "w": [0] * c.banks}
        for e in of["grant"]:
            if e["src"] in snax_ports and e["op"] in rw:
                rw[e["op"]][e["bank"]] += 1
        for e in of["dma"]:
            for j in range(len(e["value"])):
                rw[e["op"]][(e["bank"] + j) % c.banks] += 1
        for name, got, want in (("reads", rw["r"], profile.banks.reads),
                                ("writes", rw["w"], profile.banks.writes)):  # fmt: skip
            for bank, (g, w) in enumerate(zip(got, want, strict=True)):
                if g != w:
                    bad.append(f"bank {bank}: {g} {name} from the SNAX grants and the DMA's words, "
                               f"{w} in its summary")  # fmt: skip

    # the DMA's beats, per copy and over all copies
    if DMA in ctx.filter:
        lines = defaultdict(int)
        for e in of["dma"]:
            lines[(e["id"], e["op"])] += 1
        r_all = w_all = 0
        for cp in copies(ev).values():
            rows = range(cp["reps"])
            r = sum(beats(cp["from"] + i * cp["from_stride"], cp["size"], c.dma_beat_bytes)
                    for i in rows)  # fmt: skip
            w = sum(beats(cp["to"] + i * cp["to_stride"], cp["size"], c.dma_beat_bytes)
                    for i in rows)  # fmt: skip
            r_all, w_all = r_all + r, w_all + w
            for op, want, l1 in (("w", w, c.is_l1(cp["to"])), ("r", r, c.is_l1(cp["from"]))):
                got = lines[(cp["id"], op)]
                if got != (want if l1 else 0):
                    bad.append(f"DMA copy {cp['id']} ({cp['task']}): {got} `{op}` beats on L1, "
                               f"its {'writes' if op == 'w' else 'reads'} make {want if l1 else 0}")  # fmt: skip
        d = profile.dmas.get(DMA)
        if d is not None and (r_all, w_all) != (d.beats_read, d.beats_written):
            bad.append(f"the DMA's copies make {r_all} read and {w_all} written beats, its summary "
                       f"{d.beats_read} and {d.beats_written}")  # fmt: skip

    # firings and pushes per run
    if acc.name in ctx.filter:
        rates = [x.fields["rate"] for s, x in ctx.lines if s == acc.name and x.trace == "events"
                 and x.kind == "start" and 0 <= x.cycle - ctx.boot < ctx.width]  # fmt: skip
        fired = 0
        writers = [m for m in traced if m.write]
        for k, done in enumerate(e for e in ev if e["k"] == "done" and e["src"] == acc.name):
            task = done["task"]
            n_fire = sum(1 for e in of["fire"] if e["task"] == task)
            n_push = sum(1 for e in of["push"] if e["task"] == task)
            fired += n_fire
            if n_fire != done["firings"]:
                bad.append(f"{task}: {n_fire} fire lines, its done says {done['firings']} firings")
            rate = rates[k] if k < len(rates) else 1
            if n_push * rate != done["firings"]:
                bad.append(f"{task}: {n_push} push lines, {done['firings']} firings at rate {rate}")
            for m in writers:
                wtask = spans.get(m.name, [])[k][2] if k < len(spans.get(m.name, [])) else None
                got = sum(1 for e in of["grant"] if e["src"] in m.ports and e["task"] == wtask)
                if got != n_push * m.lanes:
                    bad.append(f"{m.name} ({wtask}): {got} grants, {n_push} pushes of {m.lanes} "
                               "lanes")  # fmt: skip
        if fired != profile.accelerators[acc.name].firings:
            bad.append(f"{acc.name}: {fired} fire lines, {profile.accelerators[acc.name].firings} "
                       "firings in its summary")  # fmt: skip

        # every beat a firing names, and every push, against the grants
        by_beat: dict[tuple[str, int, int], dict[int, dict]] = defaultdict(dict)
        for e in of["grant"]:
            m = next((m for m in traced if e["src"] in m.ports), None)
            if m is not None and "b" in e and e["task"] is not None:
                by_beat[(m.name, _run_of(spans, m.name, e["task"]), e["b"])][
                    m.ports.index(e["src"])
                ] = e
        for e in of["fire"]:
            k = _run_of(spans, acc.name, e["task"])
            for m in traced:
                if m.write or m.name not in e["b"]:
                    continue
                g = by_beat.get((m.name, k, e["b"][m.name]), {})
                got = [g[lane]["value"] if lane in g else None for lane in range(m.lanes)]
                if got != e["value"][m.name]:
                    bad.append(f"fire {e['n']} at {e['t']}: {m.name}'s beat {e['b'][m.name]} "
                               f"granted as {got}, fired as {e['value'][m.name]}")  # fmt: skip
        for e in of["push"]:
            k = _run_of(spans, acc.name, e["task"])
            for m in (m for m in traced if m.write):
                g = by_beat.get((m.name, k, e["n"]), {})
                got = [g[lane]["value"] if lane in g else None for lane in range(m.lanes)]
                if got != e["value"]:
                    bad.append(f"push {e['n']} at {e['t']}: written by {m.name} as {got}, "
                               f"pushed as {e['value']}")  # fmt: skip

    # the word-interleaved map
    for e in of["grant"] + of["stall"] + of["dma"]:
        if (e["bank"], e["row"]) != c.bank_row(e["addr"]):
            bad.append(f"{e['k']} at {e['t']} ({e['src']}): bank {e['bank']} row {e['row']}, "
                       f"its address {e['addr']:#x} is bank and row {c.bank_row(e['addr'])}")  # fmt: skip

    # each mover's addresses, task by task, against its CSR nest
    expected = nests(ctx)
    for m in traced:
        for k, (_, _, task) in enumerate(spans.get(m.name, [])):
            want = expected.get((m.name, k))
            if want is None:
                continue
            for lane, port in enumerate(m.ports):
                got = [e["addr"] - c.l1_base for e in of["grant"]
                       if e["src"] == port and e["task"] == task]  # fmt: skip
                if got != want[lane]:
                    bad.append(f"{port} ({task}): grant addresses {[hex(a) for a in got[:4]]}... "
                               f"({len(got)}), its CSRs' nest {[hex(a) for a in want[lane][:4]]}... "
                               f"({len(want[lane])})")  # fmt: skip

    if ctx.on("stall"):
        bad += _stall_invariants(trace, profile, ctx, of, traced, all_in)
    return bad


def _stall_invariants(trace: Trace, profile: Profile, ctx: Context, of: dict[str, list],
                      traced: list[Mover], all_in: bool) -> list[str]:  # fmt: skip
    c, acc = ctx.cluster, ctx.cluster.accelerator
    bad: list[str] = []
    spans = brackets(trace.events)
    snax_ports = {p for m in traced for p in m.ports}

    stalls, stalls_dma = defaultdict(int), defaultdict(int)
    for e in of["stall"]:
        stalls[e["src"]] += 1
        stalls_dma[e["src"]] += e["by"] == DMA
    for p in sorted(snax_ports):
        prof = profile.ports.get(p)
        want = (prof.stalls, prof.stalls_dma) if prof else (0, 0)
        if (stalls[p], stalls_dma[p]) != want:
            bad.append(f"{p}: {stalls[p]} stall lines ({stalls_dma[p]} by the DMA), its summary "
                       f"{want[0]} ({want[1]})")  # fmt: skip

    if all_in and profile.banks is not None:
        req: dict[tuple[int, int], int] = defaultdict(int)
        took: set[tuple[int, int]] = set()
        for e in of["grant"] + of["stall"]:
            if e["src"] in snax_ports:
                req[(e["bank"], e["t"])] += 1
        for e in of["stall"]:
            if e["by"] == DMA:
                took.add((e["bank"], e["t"]))
        conflicts = [0] * c.banks
        for (bank, t), n in req.items():
            if n + ((bank, t) in took) >= 2:
                conflicts[bank] += 1
        for bank, (g, w) in enumerate(zip(conflicts, profile.banks.conflicts, strict=True)):
            if g != w:
                bad.append(f"bank {bank}: {g} cycles with two or more requesters, {w} conflicts "
                           "in its summary")  # fmt: skip

    if acc.name not in ctx.filter:
        return bad
    for m in traced:
        lanes = defaultdict(list)
        for e in of["fifo"]:
            if e["src"] == m.name:
                lanes[e["lane"]].append((e["t"], e["count"]))
        words = _words(m, of, spans, acc.name)
        prof = profile.streamers.get(m.name)
        for lane in range(m.lanes):
            seq = lanes.get(lane, [])
            ws = words.get(lane, [])
            want, prev = [], 0
            if ws:
                for t in range(min(v for v, _ in ws), max(p for _, p in ws) + 2):
                    n = sum(1 for v, p in ws if v <= t <= p)
                    if n != prev:
                        want.append((t, n))
                        prev = n
            if seq != want:
                bad.append(f"{m.name} lane {lane}: fifo counts {seq[:6]}..., its words visible "
                           f"minus popped give {want[:6]}...")  # fmt: skip
            if seq and (max(n for _, n in seq) > m.fifo_depth or seq[-1][1] != 0):
                bad.append(f"{m.name} lane {lane}: fifo counts above its depth {m.fifo_depth} or "
                           "not 0 at the end")  # fmt: skip
            top = max((n for _, n in seq), default=0)
            if prof is not None and lane < len(prof.fifo.max) and top != prof.fifo.max[lane]:
                bad.append(f"{m.name} lane {lane}: fifo counts up to {top}, its summary's qmax "
                           f"{prof.fifo.max[lane]}")  # fmt: skip
    return bad


def _words(m: Mover, of: dict[str, list], spans: dict, acc: str) -> dict[int, list[tuple]]:
    """Each lane's words: the cycle each becomes visible and the cycle it is popped."""
    out = defaultdict(list)
    run_of = {task: k for k, (_, _, task) in enumerate(spans.get(m.name, []))}
    if not m.write:
        popped: dict[tuple[int, int], int] = {}
        for e in of["fire"]:
            k = _run_of(spans, acc, e["task"])
            b = e["b"].get(m.name)
            if b is not None:
                popped[(k, b)] = max(popped.get((k, b), -1), e["t"])
        for e in of["grant"]:
            if e["src"] in m.ports and "b" in e and e["task"] in run_of:
                p = popped.get((run_of[e["task"]], e["b"]))
                if p is not None:
                    out[m.ports.index(e["src"])].append((e["back"] + 1, p))
    else:
        pushed = {(_run_of(spans, acc, e["task"]), e["n"]): e["t"] for e in of["push"]}
        for e in of["grant"]:
            if e["src"] in m.ports and "b" in e and e["task"] in run_of:
                v = pushed.get((run_of[e["task"]], e["b"]))
                if v is not None:
                    out[m.ports.index(e["src"])].append((v + 1, e["t"]))
    return out
