"""The converter's beat, stall and instruction part.

    beat_events(named, ctx) -> [(t, n, event)]
    beat_invariants(trace, profile, ctx) -> [what fails]
    class_invariants(events, class_events, srcs) -> [what fails]

SNAX-GVSoC's ``beats``, ``stalls`` and ``insn`` lines, renamed, shifted and cut to the window,
with what the converter adds (CONTRACTS, "Profile and trace"). ``named`` are events.log's lines
with their trace source (convert.py); ``ctx`` (``Context``) is the rest of the run: the cluster,
the regions, the window, the filter and what the task level made of the events. A unit's lines
are paired with its streamer's tasks by ``pairing``; a run's class runs (``class_runs``) go from
each ``class`` line to the next of its task. The invariants, each checked where the trace has
its lines: grants, bank accesses, DMA beats (as the iDMA counts them), firings, each output's
pushes against its rate and its writer's words (a packed one d_out results a word) and, at
``stall``, stalls, conflicts and ``qmax`` match the summary; every fired ``b`` was granted on
every lane (a sliced input's slice of it); each L1 line's ``bank`` and ``row`` are its address's;
class runs tile each run; and each mover's grant addresses are the nest its CSR writes gave the
task (``agu_nest``). A linked group is read at its boundary: its firings the head's, each push
its sink's.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, ClassVar

from snax_forge.gen.csrs import mover_csr_count
from snax_forge.lower.tasks import Configure, Start, TaskList

from .cluster import DMA, Cluster, Mover
from .events import Line, RunError
from .profile import Profile
from .regions import Region, region_at
from .trace import Trace, at_least

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
    paired: dict[str, dict[str, str]] = field(default_factory=dict)  # unit task -> mover tasks

    def on(self, level: str) -> bool:
        return at_least(self.level, level)


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


def run_end(start: dict[str, Any], done: dict[str, Any]) -> int:
    """The cycle after a run's last counted one: its classes add up from its start + 1."""
    return start["t"] + 1 + sum(done["cycles"].values())


def class_runs(lines: list[dict[str, Any]], end: int) -> list[tuple[int, int, str]]:
    """One run's class lines (time order) as ``(first, end, mode)``: each to the next line,
    the last to ``end``, the cycle after the run's last counted one."""
    ts = [e["t"] for e in lines] + [end]
    return [(e["t"], ts[i + 1], e["mode"]) for i, e in enumerate(lines)]


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
    """Each mover's first CSR number (its pointer's low word), in its set's streamer layout
    (each set's from its own first CSR)."""
    acc = cluster.accelerator
    out, csr = {}, {s.name: s.first for s in acc.csrs.sets}
    for m in acc.movers:
        key = m.set or acc.csrs.sets[0].name
        out[m.name] = csr[key]
        csr[key] += mover_csr_count(m.spatial_bounds, m.temporal_dim, m.enabled)
    return out


def mover_values(cluster: Cluster, mover: Mover, regs: dict[int, int]
                 ) -> tuple[int, list[int], list[int], list[int]]:  # fmt: skip
    """A mover's pointer, spatial strides, temporal bounds and temporal strides from the
    streamer's CSR values (number -> the last value written; 0 when never written)."""
    csr = mover_csrs(cluster)[mover.name]
    n_sp = len(mover.spatial_bounds)
    get = [regs.get(csr + i, 0) for i in range(2 + n_sp + 2 * mover.temporal_dim)]
    return (get[0], get[2 : 2 + n_sp], get[2 + n_sp : 2 + n_sp + mover.temporal_dim],
            get[2 + n_sp + mover.temporal_dim :])  # fmt: skip


def mover_nest(cluster: Cluster, mover: Mover, regs: dict[int, int]) -> list[list[int]]:
    """Each lane's addresses (TCDM offsets) from the streamer's CSR values (``agu_nest``); a
    masked mover's lane its ENABLED_CHANNEL leaves off has none (it sends no request)."""
    ptr, sp, tb, ts = mover_values(cluster, mover, regs)
    out = agu_nest(ptr, mover.spatial_bounds, sp, tb, ts, mover.lanes, cluster.l1_size - 1,
                   not mover.write)  # fmt: skip
    if mover.enabled:
        at = (
            mover_csrs(cluster)[mover.name] + 2 + len(mover.spatial_bounds) + 2 * mover.temporal_dim
        )
        on = [regs.get(at + lane // 32, 0) >> (lane % 32) & 1 for lane in range(mover.lanes)]
        out = [addrs if on[lane] else [] for lane, addrs in enumerate(out)]
    return out


def nests(ctx: Context) -> dict[tuple[str, int], list[list[int]]]:
    """Per (mover, its run index in the window), each lane's addresses (TCDM offsets) from
    the CSR writes before its start."""
    c = ctx.cluster
    core = f"core{c.accelerator.core}"
    raw = sorted((x for s, x in ctx.lines if s == core and x.trace == "events"
                  and x.kind == "csr" and x.fields["op"] == "w"), key=lambda x: x.cycle)  # fmt: skip
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
            out[(m.name, k)] = mover_nest(c, m, regs)
            k += 1
    return out


def shared_pick(tasks: Sequence[Configure]) -> Configure:
    """Of one start's configures of a shared mover's users, the one its program writes:
    its group's own (a temporal bound not 0), else the first idle one (lower/c.py)."""
    own = [c for c in tasks if any(c.values["temporal_bounds"])]
    return own[0] if own else tasks[0]


def mover_tasks(tasks: TaskList, mover: Mover) -> list[str]:
    """A shared mover's streamer tasks in start order, one per start of its set (``shared_pick``)."""
    conf = {s.task_name: s for s in tasks.steps if isinstance(s, Configure)}
    out = []
    for s in tasks.steps:
        if isinstance(s, Start):
            mine = [conf[t] for t in s.tasks if t in conf and conf[t].component in mover.users]
            if mine:
                out.append(shared_pick(mine).task_name)
    return out


def pairing(tasks: TaskList, cluster: Cluster) -> dict[str, dict[str, str]]:
    """Per accelerator task, the streamer task of each mover of its streamer set, from the
    start step that launches them together: a unit's task with a task on every mover of its set,
    with several groups in one encapsulated set the other groups' idle (every bound 0),
    which SNAX-GVSoC starts and ends at once. An accelerator task launched without a streamer
    task for a mover of its set, with two for one, or with a streamer task on no mover of its
    set, is a RunError (``index.tasks``) naming the tasks: never guessed."""
    acc = cluster.accelerator
    set_of = {u.name: acc.accelerator_of(u.group) for u in acc.units}
    conf = {s.task_name: s for s in tasks.steps if isinstance(s, Configure)}
    out: dict[str, dict[str, str]] = {}
    for s in tasks.steps:
        if not isinstance(s, Start):
            continue
        launched = [conf[t] for t in s.tasks if t in conf]
        mine = [c.task_name for c in launched
                if c.type == "accel" and c.component in acc.unit_names]  # fmt: skip
        if not mine:
            continue
        what = f"{tasks.name}: start {s.tasks}"
        unit = next(c.component for c in launched if c.task_name == mine[0])
        movers = [m.name for m in acc.movers if m.set == set_of[unit]]
        # a port's streamer -> its mover: a shared mover's users' too
        on = {
            u: m.name for m in acc.movers if m.set == set_of[unit] for u in (m.users or (m.name,))
        }
        if len(mine) > 1:
            raise RunError("index.tasks", f"{what} launches {len(mine)} tasks of {acc.name} "
                           f"({mine})")  # fmt: skip
        by: dict[str, list[str]] = {}
        for c in launched:
            if c.type != "streamer":
                continue
            if c.component not in on:
                raise RunError("index.tasks", f"{what}: streamer task {c.task_name!r} is on "
                               f"{c.component}, no mover of set {set_of[unit]} ({movers})")  # fmt: skip
            by.setdefault(on[c.component], []).append(c.task_name)
        for m in acc.movers:
            if m.users and len(by.get(m.name, [])) > 1:  # the one its program writes
                by[m.name] = [shared_pick([conf[t] for t in by[m.name]]).task_name]
        for m in movers:
            if len(by.get(m, [])) != 1:
                have = by.get(m, [])
                raise RunError("index.tasks", f"{what}: accelerator task {mine[0]!r} has "
                               f"{len(have)} streamer tasks for {m} ({have}), one expected")  # fmt: skip
        out[mine[0]] = {m: by[m][0] for m in movers}
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


def beat_events(named: Sequence[tuple[str, Line]], ctx: Context) -> list[tuple[int, int, dict]]:
    """The beat, stall and instruction lines as events (module doc), with each line's index
    in the file for the order of a cycle."""
    return _Beats(named, ctx).events()


class _Beats:
    """``beat_events``'s context: the run's ports, task spans and DMA copies, the
    accelerator's readers and writers; one method per line kind (``KINDS``), each filling an
    event or None when its source is filtered out."""

    def __init__(self, named: Sequence[tuple[str, Line]], ctx: Context) -> None:
        self.named, self.ctx = named, ctx
        self.c, self.acc = ctx.cluster, ctx.cluster.accelerator
        self.word = self.c.bank_bytes
        self.ports = _input_ports(named, self.c)
        self.spans = brackets(ctx.task_events)
        self.dma_copies = copies(ctx.task_events)
        self.mover_of_port = {p: m for m in self.acc.movers for p in m.ports}
        # per unit the readers its firings name (a linked group's head's), each beat its num_ch
        # words
        self.readers = {u: self.acc.fire_readers(u) for u in self.acc.unit_names}
        self.words_in = {u: [self.acc.lanes_of(m, u) for m in rs] for u, rs in self.readers.items()}
        self.starts = {u: [sum(w[:i]) for i in range(len(w) + 1)] for u, w in self.words_in.items()}
        self.funcs: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for s, x in named:
            if x.trace == "events" and x.kind == "func":
                self.funcs[s].append((x.cycle - ctx.boot, x.fields["name"]))
        for f in self.funcs.values():
            f.sort(key=lambda p: p[0])
        self.fires: list[tuple[dict, tuple[int, str] | None]] = []
        self.pushes: list[tuple[dict, tuple[int, str] | None]] = []

    def events(self) -> list[tuple[int, int, dict]]:
        out: list[tuple[int, int, dict]] = []
        for n, (src, x) in enumerate(self.named):
            if x.trace not in ("beats", "stalls", "insn"):
                continue
            t = x.cycle - self.ctx.boot
            if not 0 <= t < self.ctx.width:
                continue
            kind = self.KINDS.get(x.kind)
            if kind is None:
                raise RunError("convert.kind", f"{x.trace} {x.kind} line: no event for it ({x})")
            e = kind(self, {"t": t, "k": x.kind}, src, x.fields)
            if e is not None:
                out.append((t, n, e))
        self.values(out)
        return out

    def task(self, src: str, t: int) -> tuple[int, str] | None:
        return task_at(self.spans.get(src, []), t)

    def xbar(self, e: dict, src: str, f: dict) -> dict | None:
        """A crossbar grant or stall: its port and the mover owning it, the element."""
        k, t, c = e["k"], e["t"], self.c
        port = self.ports.get(f["in"])
        if port is None:
            raise RunError("convert.path", f"crossbar input {f['in']}: no `input` line names "
                           "it")  # fmt: skip
        owner = self.mover_of_port.get(port)
        source = owner.name if owner is not None else port
        if source not in self.ctx.filter:
            return None
        run = self.task(owner.name, t) if owner is not None else None
        addr = c.l1_base + f["addr"]
        e.update(src=port, task=run[1] if run else None)
        if k == "grant":
            size = f.get("size", self.word)
            data = (f["value"] & WORD_MASK).to_bytes(8, "little")[:size]
            region, elems, vals = _lookup(self.ctx, "l1", addr, data)
            e.update(op=f["op"], mem="l1", addr=addr, bank=f["bank"], row=f["row"],
                     value=f["value"], back=f["back"] - self.ctx.boot)  # fmt: skip
            if "b" in f:
                e["b"] = f["b"]
            if "size" in f:
                e["size"] = f["size"]
            e.update(region=region, elems=elems, vals=vals)
        else:
            region, elems, _ = _lookup(self.ctx, "l1", addr, bytes(self.word))
            by = "dma" if f["by"] == -1 else self.ports.get(f["by"], str(f["by"]))
            e.update(addr=addr, bank=f["bank"], row=f["row"], by=by,
                     by_addr=c.l1_base + f["by_addr"])  # fmt: skip
            if "b" in f:
                e["b"] = f["b"]
            e.update(region=region, elems=elems)
        return e

    def dma(self, e: dict, src: str, f: dict) -> dict | None:
        """A DMA beat on L1."""
        if DMA not in self.ctx.filter:
            return None
        copy = self.dma_copies.get(f["id"])
        addr = self.c.l1_base + f["addr"]
        data = _bytes(f["value"], addr, f["size"], self.word)
        region, elems, vals = _lookup(self.ctx, "l1", addr, data)
        e.update(src=DMA, task=copy["task"] if copy else None, id=f["id"], i=f["i"],
                 op=f["op"], mem="l1", addr=addr, size=f["size"], bank=f["bank"],
                 row=f["row"], value=list(f["value"]), l2=f["l2"])  # fmt: skip
        if "back" in f:
            e["back"] = f["back"] - self.ctx.boot
        e.update(region=region, elems=elems, vals=vals)
        return e

    def burst(self, e: dict, src: str, f: dict) -> dict | None:
        """A DMA burst on L2."""
        if DMA not in self.ctx.filter:
            return None
        copy = self.dma_copies.get(f["id"])
        r = region_at(self.ctx.regions, "l2", f["addr"])
        e.update(src=DMA, task=copy["task"] if copy else None, id=f["id"], op=f["op"],
                 mem="l2", addr=f["addr"], size=f["size"], back=f["back"] - self.ctx.boot,
                 region=r.container if r else None,
                 elems=r.elems(f["addr"], f["size"]) if r else [])  # fmt: skip
        return e

    def fire(self, e: dict, src: str, f: dict) -> dict | None:
        """A firing of unit ``src``: each of its readers' beat and words; its values once the
        grants are known."""
        t = e["t"]
        if src not in self.ctx.filter:
            return None
        run = self.task(src, t)
        readers, starts = self.readers[src], self.starts[src]
        if len(f["b"]) != len(readers) or len(f["value"]) != starts[-1]:
            raise RunError("convert.beats", f"fire at {t}: {len(f['b'])} beats of "
                           f"{len(f['value'])} words, {src} reads {len(readers)} of "
                           f"{self.words_in[src]} words (its readers' num_channel)")  # fmt: skip
        e.update(src=src, task=run[1] if run else None, n=f["n"],
                 b={m.name: f["b"][i] for i, m in enumerate(readers)},
                 value={m.name: f["value"][starts[i] : starts[i + 1]]
                        for i, m in enumerate(readers)})  # fmt: skip
        self.fires.append((e, run))
        return e

    def push(self, e: dict, src: str, f: dict) -> dict | None:
        """A push of unit ``src``: the result's words and, with several outputs, its sink
        (``unit``); its values once the grants are known."""
        t = e["t"]
        if src not in self.ctx.filter:
            return None
        sinks, sink = self.acc.sinks(src), f.get("unit")
        if (sink is None) != (len(sinks) == 1) or (sink is not None and sink not in sinks):
            raise RunError("convert.beats", f"push at {t}: unit {sink!r}, where {src}'s outputs "
                           f"are {sinks}'s (a push names its unit when there are two or more)")  # fmt: skip
        w = self.acc.writer_of(src, sink)
        if w is not None and len(f["value"]) != self.acc.lanes_of(w, src):
            raise RunError("convert.beats", f"push at {t}: {len(f['value'])} words, {sink or src}'s "
                           f"writer moves {self.acc.lanes_of(w, src)} (its num_channel, or on a "
                           "shared mover its port's)")  # fmt: skip
        run = self.task(src, t)
        e.update(src=src, task=run[1] if run else None, n=f["n"], first=f["first"],
                 last=f["last"], value=list(f["value"]))  # fmt: skip
        if sink is not None:
            e["unit"] = sink
        self.pushes.append((e, run))
        return e

    def fifo(self, e: dict, src: str, f: dict) -> dict | None:
        """A queue count of a mover's lane."""
        m = next(x for x in self.acc.movers if x.name == src)  # src: the line's mover
        if m.name not in self.ctx.filter:
            return None
        run = self.task(m.name, e["t"])
        e.update(src=m.name, task=run[1] if run else None, lane=f["lane"], count=f["count"])
        return e

    def cls(self, e: dict, src: str, f: dict) -> dict | None:
        """A class line of the accelerator or a mover."""
        if src not in self.ctx.filter:
            return None
        run = self.task(src, e["t"])
        e.update(src=src, task=run[1] if run else None, mode=f["mode"])
        return e

    def insn(self, e: dict, src: str, f: dict) -> dict | None:
        """An instruction of a core, in the function it was in."""
        if src not in self.ctx.filter:
            return None
        spans_f = [name for ft, name in self.funcs.get(src, []) if ft <= e["t"]]
        e.update(src=src, pc=f["pc"], func=spans_f[-1] if spans_f else None, text=f["text"])
        return e

    KINDS: ClassVar[dict[str, Callable[..., dict | None]]] = {
        "grant": xbar, "stall": xbar, "dma": dma, "burst": burst, "fire": fire, "push": push,
        "fifo": fifo, "class": cls, "insn": insn,
    }  # fmt: skip

    def values(self, out: list[tuple[int, int, dict]]) -> None:
        """Each firing's and push's element values, from the grants of its beats (the streamer
        task launched with its unit's, ``Context.paired``)."""
        grants = beat_grants([e for _, _, e in out if e["k"] == "grant"], self.acc.movers)
        paired = self.ctx.paired

        def vals_of(mover: Mover, unit: str, run: tuple[int, str] | None, b: int,
                    words: list[int]) -> Any:  # fmt: skip
            task = paired.get(run[1], {}).get(mover.name) if run else None
            g = grants.get((mover.name, task, b)) if task else None
            lanes = self.acc.lanes_of(mover, unit)
            if not g or len(g) != lanes:
                return None
            vals: list[Any] = []
            for lane in range(lanes):
                ge = g[lane]
                data = (words[lane] & WORD_MASK).to_bytes(8, "little") if lane < len(words) else b""
                r = region_at(self.ctx.regions, "l1", ge["addr"])
                vals += r.vals(ge["addr"], data) if r else []
            return vals

        for e, run in self.fires:
            e["vals"] = {m.name: vals_of(m, e["src"], run, e["b"][m.name], e["value"][m.name])
                         for m in self.readers[e["src"]]}  # fmt: skip

        def slice_vals(mover: Mover, unit: str, run: tuple[int, str] | None, n: int, d: int,
                       words: list[int]) -> Any:  # fmt: skip
            """A packed push's values: result n is slice n % d of word n // d."""
            task = paired.get(run[1], {}).get(mover.name) if run else None
            g = grants.get((mover.name, task, n // d)) if task else None
            if not g or len(g) != 1 or not words:
                return None
            size = self.word // d
            addr = g[0]["addr"] + n % d * size
            r = region_at(self.ctx.regions, "l1", addr)
            return r.vals(addr, (words[0] & WORD_MASK).to_bytes(8, "little")[:size]) if r else []

        for e, run in self.pushes:
            w = self.acc.writer_of(e["src"], e.get("unit"))
            d = self.acc.d_out(w, e["src"]) if w is not None else 1
            e["vals"] = (None if w is None else
                         vals_of(w, e["src"], run, e["n"], e["value"]) if d == 1 else
                         slice_vals(w, e["src"], run, e["n"], d, e["value"]))  # fmt: skip


# =============================================================================
# The invariants
# =============================================================================


def beat_grants(grants: list[dict], movers: Sequence[Mover]
                ) -> dict[tuple[str, str, int], dict[int, dict]]:  # fmt: skip
    """The movers' grant events of a beat, by (mover, its streamer task, b): each lane's
    grant."""
    out: dict[tuple[str, str, int], dict[int, dict]] = defaultdict(dict)
    for e in grants:
        m = next((m for m in movers if e["src"] in m.ports), None)
        if m is not None and "b" in e and e["task"] is not None:
            out[(m.name, e["task"], e["b"])][m.ports.index(e["src"])] = e
    return out


def _paired(ctx: Context, task: str | None, mover: str) -> str | None:
    """The streamer task of ``mover`` launched with unit task ``task``."""
    return ctx.paired.get(task or "", {}).get(mover)


@dataclass
class _Seen:
    """The events ``beat_invariants`` checks: by kind, the task spans, and the movers traced
    (``all_in`` when every mover and the DMA are)."""

    ctx: Context
    profile: Profile
    ev: list[dict[str, Any]]
    spans: dict[str, list[tuple[int, int, str]]]
    of: dict[str, list[dict[str, Any]]]
    traced: list[Mover]
    snax_ports: set[str]
    all_in: bool

    @classmethod
    def of_trace(cls, trace: Trace, profile: Profile, ctx: Context) -> _Seen:
        acc = ctx.cluster.accelerator
        ev = trace.events
        of: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in ev:
            of[e["k"]].append(e)
        traced = [m for m in acc.movers if m.name in ctx.filter]
        return cls(ctx, profile, ev, brackets(ev), of, traced, {p for m in traced for p in m.ports},
                   len(traced) == len(acc.movers) and DMA in ctx.filter)  # fmt: skip


def beat_invariants(trace: Trace, profile: Profile, ctx: Context) -> list[str]:
    """What the beat and stall events must hold (module doc); each that fails, said."""
    s = _Seen.of_trace(trace, profile, ctx)
    acc = ctx.cluster.accelerator
    bad = _grant_counts(s) + _bank_counts(s)
    if DMA in ctx.filter:
        bad += _dma_beats(s)
    for unit in acc.unit_names:
        if unit in ctx.filter:
            bad += _firings(s, unit) + _fired_beats(s, unit)
    bad += _bank_map(s)
    srcs = [*acc.unit_names, *(m.name for m in s.traced)]
    bad += class_invariants(s.ev, s.of["class"], [x for x in srcs if x in ctx.filter])
    bad += _mover_nests(s)
    if ctx.on("stall"):
        bad += _stall_invariants(trace, profile, ctx, s.of, s.traced, s.all_in)
    return bad


def _grant_counts(s: _Seen) -> list[str]:
    """The grants per SNAX port against its summary."""
    bad, count = [], defaultdict(int)
    for e in s.of["grant"]:
        count[e["src"]] += 1
    for p in sorted(s.snax_ports):
        want = s.profile.ports[p].grants if p in s.profile.ports else 0
        if count[p] != want:
            bad.append(f"{p}: {count[p]} grant lines in the window, {want} in its summary")
    return bad


def _bank_counts(s: _Seen) -> list[str]:
    """Per bank, the SNAX grants and the DMA's words against its summary (every source
    traced)."""
    c, banks = s.ctx.cluster, s.profile.banks
    if not s.all_in or banks is None:
        return []
    bad: list[str] = []
    rw = {"r": [0] * c.banks, "w": [0] * c.banks}
    for e in s.of["grant"]:
        if e["src"] in s.snax_ports and e["op"] in rw:
            rw[e["op"]][e["bank"]] += 1
    for e in s.of["dma"]:
        for j in range(len(e["value"])):
            rw[e["op"]][(e["bank"] + j) % c.banks] += 1
    for name, got, want in (("reads", rw["r"], banks.reads), ("writes", rw["w"], banks.writes)):
        for bank, (g, w) in enumerate(zip(got, want, strict=True)):
            if g != w:
                bad.append(f"bank {bank}: {g} {name} from the SNAX grants and the DMA's words, "
                           f"{w} in its summary")  # fmt: skip
    return bad


def _dma_beats(s: _Seen) -> list[str]:
    """The DMA's beats on L1 per copy, and over all copies against its summary."""
    c, bad = s.ctx.cluster, []
    lines: dict[tuple[int, str], int] = defaultdict(int)
    for e in s.of["dma"]:
        lines[(e["id"], e["op"])] += 1
    r_all = w_all = 0
    for cp in copies(s.ev).values():
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
    d = s.profile.dmas.get(DMA)
    if d is not None and (r_all, w_all) != (d.beats_read, d.beats_written):
        bad.append(f"the DMA's copies make {r_all} read and {w_all} written beats, its summary "
                   f"{d.beats_read} and {d.beats_written}")  # fmt: skip
    return bad


def _firings(s: _Seen, unit: str) -> list[str]:
    """Per run of ``unit``, its fire lines against its done, each output's push lines against
    its rate, and its writer's grants (the streamer task launched with its task) against those
    pushes, d_out to a word; over all its runs, the firings against its summary."""
    ctx, acc, bad = s.ctx, s.ctx.cluster.accelerator, []
    rates = [x.fields["rate"] for src, x in ctx.lines if src == unit and x.trace == "events"
             and x.kind == "start" and 0 <= x.cycle - ctx.boot < ctx.width]  # fmt: skip
    sinks = acc.sinks(unit)
    outs = [(k, acc.writer_of(unit, k if len(sinks) > 1 else None)) for k in sinks]
    fired = 0
    for k, done in enumerate(e for e in s.ev if e["k"] == "done" and e["src"] == unit):
        task = done["task"]
        n_fire = sum(1 for e in s.of["fire"] if e["task"] == task)
        fired += n_fire
        if n_fire != done["firings"]:
            bad.append(f"{task}: {n_fire} fire lines, its done says {done['firings']} firings")
        rate = rates[k] if k < len(rates) else 1
        rate = rate if isinstance(rate, list) else [rate]
        if len(rate) != len(sinks):
            bad.append(f"{task}: {len(rate)} rates at its start, {len(sinks)} outputs ({sinks})")
            continue
        for (sink, m), r in zip(outs, rate, strict=True):
            mine = [e for e in s.of["push"] if e["task"] == task and e.get("unit", sink) == sink]
            of = f" of {sink}" if len(sinks) > 1 else ""
            if len(mine) * r != done["firings"]:
                bad.append(f"{task}: {len(mine)} push lines{of}, {done['firings']} firings at rate "
                           f"{r}")  # fmt: skip
            if m is None or m not in s.traced:
                continue
            wtask = _paired(ctx, task, m.name)
            got = sum(1 for e in s.of["grant"] if e["src"] in m.ports and e["task"] == wtask)
            lanes, d = acc.lanes_of(m, unit), acc.d_out(m, unit)
            words = -(-len(mine) // d)
            if got != words * lanes:
                packed = f" in {words} words of {d}" if d > 1 else ""
                bad.append(f"{m.name} ({wtask}): {got} grants, {len(mine)} pushes{of}{packed} of "
                           f"{lanes} lanes")  # fmt: skip
    if fired != s.profile.accelerators[unit].firings:
        bad.append(f"{unit}: {fired} fire lines, {s.profile.accelerators[unit].firings} "
                   "firings in its summary")  # fmt: skip
    return bad


def _fired_beats(s: _Seen, unit: str) -> list[str]:
    """Every beat a firing of ``unit`` names (a sliced input's slice of its word), and every
    push, against the grants of the streamer tasks launched with its task."""
    acc, bad = s.ctx.cluster.accelerator, []
    by_beat = beat_grants(s.of["grant"], s.traced)
    mine = acc.movers_of(unit)

    def granted(m: Mover, task: str | None, b: int) -> list[Any]:
        g = by_beat.get((m.name, _paired(s.ctx, task, m.name), b), {})
        return [g[lane]["value"] if lane in g else None for lane in range(acc.lanes_of(m, unit))]

    pops: dict[tuple[str, str, int], int] = defaultdict(int)
    for e in (e for e in s.of["fire"] if e["src"] == unit):
        for m in s.traced:
            if m.write or m.name not in e["b"]:
                continue
            b, fired = e["b"][m.name], e["value"][m.name]
            d = acc.d_in(m, unit)
            if d == 1 or b < 0:
                got = granted(m, e["task"], b)
                if got != fired:
                    bad.append(f"fire {e['n']} at {e['t']}: {m.name}'s beat {b} granted as "
                               f"{got}, fired as {fired}")  # fmt: skip
                continue
            # NB (D180): the k-th firing that names word b takes its slice k, slice 0 lowest
            k = pops[(m.name, e["task"], b)]
            pops[(m.name, e["task"], b)] += 1
            if k >= d:
                bad.append(f"fire {e['n']} at {e['t']}: {m.name}'s word {b} holds {d} slices, "
                           f"this is firing {k + 1} to name it")  # fmt: skip
                continue
            bits = 64 // d
            word = granted(m, e["task"], b)[0]
            got = None if word is None else word >> (k * bits) & ((1 << bits) - 1)
            if got != fired[0] & ((1 << bits) - 1):
                bad.append(f"fire {e['n']} at {e['t']}: slice {k} of {m.name}'s word {b} is "
                           f"{None if got is None else hex(got)}, fired as {hex(fired[0])}")  # fmt: skip
    for e in (e for e in s.of["push"] if e["src"] == unit):
        m = acc.writer_of(unit, e.get("unit"))
        if m is None or m not in s.traced or m not in mine:
            continue
        d = acc.d_out(m, unit)
        if d == 1:
            got = granted(m, e["task"], e["n"])
            if got != e["value"]:
                bad.append(f"push {e['n']} at {e['t']}: written by {m.name} as {got}, "
                           f"pushed as {e['value']}")  # fmt: skip
            continue
        # NB (D174): result n is slice n % d of the writer's word n // d, slice 0 lowest
        bits = 64 // d
        word = granted(m, e["task"], e["n"] // d)[0]
        got = None if word is None else word >> (e["n"] % d * bits) & ((1 << bits) - 1)
        if got != e["value"][0] & ((1 << bits) - 1):
            bad.append(f"push {e['n']} at {e['t']}: slice {e['n'] % d} of {m.name}'s word "
                       f"{e['n'] // d} is {None if got is None else hex(got)}, pushed as "
                       f"{hex(e['value'][0])}")  # fmt: skip
    return bad


def _bank_map(s: _Seen) -> list[str]:
    """Every L1 access's bank and row, the word-interleaved map's."""
    c = s.ctx.cluster
    return [f"{e['k']} at {e['t']} ({e['src']}): bank {e['bank']} row {e['row']}, "
            f"its address {e['addr']:#x} is bank and row {c.bank_row(e['addr'])}"
            for e in s.of["grant"] + s.of["stall"] + s.of["dma"]
            if (e["bank"], e["row"]) != c.bank_row(e["addr"])]  # fmt: skip


def _mover_nests(s: _Seen) -> list[str]:
    """Each mover's addresses, task by task, against its CSR nest."""
    c, bad = s.ctx.cluster, []
    expected = nests(s.ctx)
    for m in s.traced:
        for k, (_, _, task) in enumerate(s.spans.get(m.name, [])):
            want = expected.get((m.name, k))
            if want is None:
                continue
            for lane, port in enumerate(m.ports):
                got = [e["addr"] - c.l1_base for e in s.of["grant"]
                       if e["src"] == port and e["task"] == task]  # fmt: skip
                if got != want[lane]:
                    bad.append(f"{port} ({task}): grant addresses {[hex(a) for a in got[:4]]}... "
                               f"({len(got)}), its CSRs' nest {[hex(a) for a in want[lane][:4]]}... "
                               f"({len(want[lane])})")  # fmt: skip
    return bad


def class_invariants(events: list[dict[str, Any]], lines: list[dict[str, Any]],
                      srcs: list[str]) -> list[str]:  # fmt: skip
    bad: list[str] = []
    starts = {e["task"]: e for e in events if e["k"] == "start"}
    for src in srcs:
        mine = [e for e in lines if e["src"] == src]
        runs = [d for d in events if d["k"] == "done" and d["src"] == src]
        tasks = {d["task"] for d in runs}
        for e in mine:
            if e["task"] not in tasks:
                bad.append(f"{src}: class {e['mode']} at {e['t']} outside its runs")
                break
        for d in runs:
            first, end = starts[d["task"]]["t"] + 1, run_end(starts[d["task"]], d)
            seq = [e for e in mine if e["task"] == d["task"]]
            where = f"{src} ({d['task']}, cycles {first} to {end - 1})"
            if not seq or seq[0]["t"] != first:
                bad.append(f"{where}: no class line at its first cycle {first}")
                continue
            if any(a["t"] >= b["t"] or a["mode"] == b["mode"] for a, b in pairwise(seq)):
                bad.append(f"{where}: class lines overlap or repeat a class")
                continue
            if seq[-1]["t"] >= end:
                bad.append(f"{where}: class line at {seq[-1]['t']} past its run")
                continue
            got = dict.fromkeys(d["cycles"], 0)
            for a, b, mode in class_runs(seq, end):
                if mode not in got:
                    bad.append(f"{where}: class {mode} at {a}, not one of {list(got)}")
                    break
                got[mode] += b - a
            else:
                if got != d["cycles"]:
                    bad.append(f"{where}: class runs {got}, its summary {d['cycles']}")
    return bad


def _stall_invariants(trace: Trace, profile: Profile, ctx: Context, of: dict[str, list],
                      traced: list[Mover], all_in: bool) -> list[str]:  # fmt: skip
    c, acc = ctx.cluster, ctx.cluster.accelerator
    bad: list[str] = []
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

    for m in traced:
        # its units: a shared mover's every user's, each with its own tasks' words
        units = ([u for u in acc.unit_names if m in acc.movers_of(u)] if m.users
                 else [acc.unit_of_mover(m.name)])  # fmt: skip
        if not units or any(u is None or u not in ctx.filter for u in units):
            continue
        lanes = defaultdict(list)
        for e in of["fifo"]:
            if e["src"] == m.name:
                lanes[e["lane"]].append((e["t"], e["count"]))
        words: dict[int, list[tuple]] = defaultdict(list)
        unchecked: set[int] = set()
        unpopped = False  # no line names its pops: its counts checked against depth and qmax only
        for unit in units:
            if not m.write and m not in acc.fire_readers(unit):
                unpopped = True  # a linked group's outer input past its head
                continue
            for lane, ws in _words(m, of, ctx, unit).items():
                words[lane] += ws
            on = acc.lanes_of(m, unit)
            if on < m.lanes and m.write:
                # a lane ENABLED_CHANNEL leaves off sends no request: its word, visible the cycle
                # after the push, is dropped in that cycle (snax-gvsoc's streamer)
                for e in of["push"]:
                    if e["src"] == unit and acc.writer_of(unit, e.get("unit")) == m:
                        for lane in range(on, m.lanes):
                            words[lane].append((e["t"] + 1, e["t"] + 1))
            elif on < m.lanes:
                # a reader's lane off takes a 0 word at no grant: not rebuilt here, not checked
                unchecked |= set(range(on, m.lanes))
        prof = profile.streamers.get(m.name)
        for lane in range(m.lanes):
            if lane in unchecked:
                continue
            seq = lanes.get(lane, [])
            ws = sorted(words.get(lane, []))
            want, prev = [], 0
            if ws:
                for t in range(min(v for v, _ in ws), max(p for _, p in ws) + 2):
                    n = sum(1 for v, p in ws if v <= t <= p)
                    if n != prev:
                        want.append((t, n))
                        prev = n
            if seq != want and not unpopped:
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


def _words(m: Mover, of: dict[str, list], ctx: Context, unit: str) -> dict[int, list[tuple]]:
    """Each lane's words: the cycle each becomes visible and the cycle it is popped (by a
    firing of ``unit``, or the push of its sink that sends the word: a packed word's last, or
    the run's; its streamer task the one launched with the unit's)."""
    out = defaultdict(list)
    acc = ctx.cluster.accelerator
    if not m.write:
        popped: dict[tuple[str | None, int], int] = {}
        for e in of["fire"]:
            if e["src"] != unit:
                continue
            key = (_paired(ctx, e["task"], m.name), e["b"].get(m.name))
            if key[1] is not None:
                popped[key] = max(popped.get(key, -1), e["t"])
        for e in of["grant"]:
            if e["src"] in m.ports and "b" in e:
                p = popped.get((e["task"], e["b"]))
                if p is not None:
                    out[m.ports.index(e["src"])].append((e["back"] + 1, p))
    else:
        d = acc.d_out(m, unit)
        pushed: dict[tuple[str | None, int], int] = {}
        for e in of["push"]:
            if e["src"] == unit and acc.writer_of(unit, e.get("unit")) == m:
                key = (_paired(ctx, e["task"], m.name), e["n"] // d)
                pushed[key] = max(pushed.get(key, -1), e["t"])
        for e in of["grant"]:
            if e["src"] in m.ports and "b" in e:
                v = pushed.get((e["task"], e["b"]))
                if v is not None:
                    out[m.ports.index(e["src"])].append((v + 1, e["t"]))
    return out
