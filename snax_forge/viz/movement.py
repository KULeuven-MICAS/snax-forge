"""Data movement of a run, from its beat trace and its regions (VIS4b, D95, D97).

What this is
------------
Four answers, all read off the trace: nothing is modelled or predicted here
(principle 4), so every number is a count or a cycle the model wrote.

  journey     every hop of one element, in every memory it lives in, with the
              firings that consumed or produced it and the other operands of
              those firings
  residency   per region and memory: when its elements arrived, were used and
              left, and how long they waited in between
  patterns    per streamer lane and DMA side, per task: the addresses as an
              affine nest (base, then bound and stride per loop, innermost
              first), in bytes and in the region's elements, the pace (first
              and last beat against one beat per cycle), and every cycle it was
              held back, with the port its bank served instead
  conflicts   every L1 stall placed on the memory layout: the word the held-back
              request wanted and the word its bank served in that cycle; per
              word and per bank the counts, and marks for the memory tab's fold

Words and hops
--------------
An L1 access is its xbar event (``grant``, ``stall``, ``resp``): each bank in
``banks`` is one word, at (bank, ``row``) through the memory layout, so a wide
DMA beat is eight words. An L2 access is a DMA ``dma_beat`` on L2 (a read on
the ``src`` side, a write on ``dst``) and the DMA's ``resp`` for a read; it
covers one beat of words from its address. The DMA's L1 ``dma_beat`` events
repeat its xbar grants and are left out. A word's element comes from the
memory layout (memory.py).

Firings
-------
An accelerator's task k is taken with the k-th task of each streamer attached
to it (as the FIFO busy window does, D56); inside a task, beats and firings are
matched by order. Lane l of a reader carries its port's lane l, one beat per
granted read in grant order. An input port with rate r is popped at firings
0, r, 2r, ...; when a reader hands each beat out T times (the repeat of D69),
read k feeds hand-outs kT .. kT + T - 1, with T = hand-outs / reads of the
task. An output port with rate r pushes beat j at firing (j + 1) r - 1, from
firings j r .. (j + 1) r - 1; write k of lane l of its writer is beat k.
A rate given by name is read from the last ``csr_write`` to ``<acc>.<name>``
before the task started. A firing's operands on a lane are the beats of every
port on that lane, or on every lane of a port whose lane count differs.

A run traced below beat level gets a named reason instead of partial numbers.
A filtered beat trace (D49) gives what it kept, and the answer says what was
filtered.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from snax_forge.snax_model.accel import AccelConfig
from snax_forge.snax_model.scenario import ACCEL_KINDS, ClusterConfig, Region

from .memory import Marks, MemoryLayout, TimeMarks


class MovementUnavailable(Exception):
    """The run's trace cannot answer; the message is the reason shown to the user."""


# =============================================================================
# The index of one run
# =============================================================================


@dataclass
class Hop:
    """One access to one word."""

    t: int
    mem: str
    word: int
    act: str  # read, write, data back, held back
    by: str  # port (L1) or DMA name (L2)
    owner: str
    lane: int
    bank: int | None = None
    row: int | None = None
    beat: int | None = None  # DMA beat index (L2)
    served: dict[str, Any] | None = None  # held back: what the bank served instead

    def to_dict(self, elem: Callable[[str, int], str | None]) -> dict[str, Any]:
        d: dict[str, Any] = {
            "t": self.t,
            "mem": self.mem,
            "act": self.act,
            "by": self.by,
            "owner": self.owner,
            "lane": self.lane,
            "element": elem(self.mem, self.word),
        }
        for k in ("bank", "row", "beat"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.served is not None:  # the word the bank served instead, as its element
            s = self.served
            d["served"] = {
                "port": s["port"],
                "w": s["w"],
                "row": s["row"],
                "element": elem(self.mem, s["word"]),
            }
        return d


@dataclass
class Firing:
    """One firing of an accelerator and the beats it consumed and produced."""

    acc: str
    task: int
    n: int  # index inside the task, as the fire event's n
    t: int
    inputs: dict[str, dict[int, Hop]] = field(default_factory=dict)  # port -> lane -> read
    outputs: dict[str, dict[int, Hop]] = field(default_factory=dict)  # port -> lane -> write
    lanes: dict[str, int] = field(default_factory=dict)  # port -> lanes
    via: dict[str, str] = field(default_factory=dict)  # attached streamer -> port


@dataclass
class Index:
    """Everything the answers need, built once per run."""

    layouts: dict[str, MemoryLayout]
    words: dict[tuple[str, int], list[Hop]]  # (mem, word) -> hops in time order
    lane_hops: dict[str, list[Hop]]  # L1 port -> its grants and stalls in time order
    dma_hops: dict[
        tuple[str, str], list[tuple[int, int, int, str]]
    ]  # (dma, side) -> (t, beat, addr, mem)
    conflicts: list[dict[str, Any]]
    fed: dict[tuple[str, int], list[Firing]]  # (L1 port, index of its grant) -> firings
    made: dict[tuple[str, int], list[Firing]]  # (L1 port, index of its write) -> its firings
    grant_no: dict[int, tuple[str, int]]  # id(hop) -> (port, index among the port's grants)
    filtered: dict[str, Any] | None
    dmas: set[str] = field(default_factory=set)  # the DMA components: their reads are departures

    def element(self, mem: str, word: int) -> str | None:
        """``A[5]`` for the word, None when no region lives there."""
        hit = self.cell(mem, word)
        return " + ".join(_elem_text(r, f) for r, f in hit) if hit else None

    def cell(self, mem: str, word: int) -> list[tuple[Region, int]]:
        lay = self.layouts.get(mem)
        if lay is None:
            return []
        col, row = lay.place(word)
        cells = lay.occupied.get(row)
        return [(lay.regions[ri], flat) for ri, flat in cells[col]] if cells else []


def _elem_text(r: Region, flat: int) -> str:
    idx, rest = [], flat
    for e in reversed(r.shape):
        idx.append(rest % e)
        rest //= e
    return f"{r.name}[{','.join(str(i) for i in reversed(idx))}]"


def _lane_of(port: str, owner: str) -> int:
    tail = port[len(owner) + 1 :] if port.startswith(owner + ".") else ""
    return int(tail) if tail.isdigit() else 0


def build_index(
    events: Sequence[dict[str, Any]],
    cluster: ClusterConfig,
    ports: dict[str, dict[str, Any]],
    layouts: dict[str, MemoryLayout],
    spans: dict[str, list[tuple[int, int]]],
    trace: Any,
) -> Index:
    """The index of a run (module doc). ``ports`` is the profile's port table."""
    if trace is None or trace.level != "beat":
        level = "off" if trace is None else trace.level
        raise MovementUnavailable(
            f"data movement needs a beat-level trace; this run was traced at level {level}. "
            "Run it again with --trace beat."
        )
    words: dict[tuple[str, int], list[Hop]] = {}
    lane_hops: dict[str, list[Hop]] = {}
    dma_hops: dict[tuple[str, str], list[tuple[int, int, int, str]]] = {}
    conflicts: list[dict[str, Any]] = []
    grants_at: dict[tuple[int, int], dict[str, Any]] = {}  # (t, bank) -> the grant event
    l1, l2 = layouts.get("l1"), layouts.get("l2")

    def add(h: Hop) -> None:
        words.setdefault((h.mem, h.word), []).append(h)

    for e in events:
        k = e["k"]
        if k in ("grant", "stall", "resp") and e.get("mem") == "l1" and l1 is not None:
            owner = ports.get(e["port"], {}).get("owner", e["port"])
            wide = len(e["banks"]) > 1
            act = (
                "held back"
                if k == "stall"
                else "data back"
                if k == "resp"
                else "write"
                if e.get("w")
                else "read"
            )
            for i, b in enumerate(e["banks"]):
                h = Hop(
                    e["t"],
                    "l1",
                    l1.word_of(b, e["row"]),
                    act,
                    e["port"],
                    owner,
                    i if wide else _lane_of(e["port"], owner),
                    b,
                    e["row"],
                )
                add(h)
                if k == "grant":
                    grants_at[e["t"], b] = e
                if k != "resp":
                    lane_hops.setdefault(e["port"], []).append(h)
            if k == "stall":
                conflicts.append(e)
        elif k == "dma_beat" and e.get("mem") == "l2" and l2 is not None:
            side = e["side"]
            dma_hops.setdefault((e["src"], side), []).append((e["t"], e["i"], e["addr"], "l2"))
            first = (e["addr"] - l2.base_addr) // l2.word_bytes
            for j in range(l2.columns):  # one beat of words
                add(
                    Hop(
                        e["t"],
                        "l2",
                        first + j,
                        "read" if side == "src" else "write",
                        e["src"],
                        e["src"],
                        j,
                        beat=e["i"],
                    )
                )
        elif k == "resp" and e.get("mem") == "l2" and l2 is not None:
            first = (e["addr"] - l2.base_addr) // l2.word_bytes
            for j in range(l2.columns):
                add(Hop(e["t"], "l2", first + j, "data back", e["src"], e["src"], j, beat=e["i"]))
        elif k == "dma_beat" and e.get("mem") == "l1":
            dma_hops.setdefault((e["src"], e["side"]), []).append((e["t"], e["i"], e["addr"], "l1"))

    # A stall and what its bank served in that cycle.
    placed = []
    for e in conflicts:
        owner = ports.get(e["port"], {}).get("owner", e["port"])
        for b in e["banks"]:
            g = grants_at.get((e["t"], b))
            served = None
            if g is not None:
                g_owner = ports.get(g["port"], {}).get("owner", g["port"])
                served = {
                    "port": g["port"],
                    "owner": g_owner,
                    "w": bool(g.get("w")),
                    "row": g["row"],
                    "word": l1.word_of(b, g["row"]),
                }
            placed.append(
                {
                    "t": e["t"],
                    "bank": b,
                    "row": e["row"],
                    "wider": bool(e.get("wider")),
                    "waiting": {
                        "port": e["port"],
                        "owner": owner,
                        "w": bool(e.get("w")),
                        "row": e["row"],
                        "word": l1.word_of(b, e["row"]),
                    },
                    "served": served,
                }
            )
    by_stall = {(c["t"], c["bank"], c["waiting"]["port"]): c["served"] for c in placed}
    for h in (h for hs in words.values() for h in hs if h.act == "held back"):
        s = by_stall.get((h.t, h.bank, h.by))
        h.served = None if s is None else {k: s[k] for k in ("port", "w", "row", "word")}

    grant_no: dict[int, tuple[str, int]] = {}  # the n-th grant of its port, for the firings
    for port, hs in lane_hops.items():
        n = 0
        for h in hs:
            if h.act in ("read", "write"):
                grant_no[id(h)] = (port, n)
                n += 1
    fed, made = _firings(events, cluster, lane_hops, spans, grant_no)
    return Index(
        layouts,
        words,
        lane_hops,
        dma_hops,
        placed,
        fed,
        made,
        grant_no,
        None
        if not (trace.filter_sources or trace.filter_window)
        else {
            "sources": trace.filter_sources,
            "window": None if trace.filter_window is None else list(trace.filter_window),
        },
        {c.name for c in cluster.components if c.kind == "dma"},
    )


# =============================================================================
# Firings
# =============================================================================


def accel_config(spec: Any) -> AccelConfig:
    """The AccelConfig of a cluster's accelerator entry, as the model builds it (D43)."""
    return ACCEL_KINDS[spec.accel](**spec.params)


def _rate(
    port_rate: int | str, acc: str, start: int, writes: list[tuple[int, str, int]]
) -> int | None:
    if isinstance(port_rate, int):
        return port_rate
    val = None
    for last, reg, value in writes:  # in trace order: the last write before the start wins
        if last < start and reg == f"{acc}.{port_rate}":
            val = value
    return val if val and val > 0 else None


def _in_task(hops: list[Hop], span: tuple[int, int]) -> list[Hop]:
    return [h for h in hops if span[0] <= h.t <= span[1] and h.act in ("read", "write")]


def _firings(
    events: Sequence[dict[str, Any]],
    cluster: ClusterConfig,
    lane_hops: dict[str, list[Hop]],
    spans: dict[str, list[tuple[int, int]]],
    grant_no: dict[int, tuple[str, int]],
) -> tuple[dict[tuple[str, int], list[Firing]], dict[tuple[str, int], list[Firing]]]:
    """Which firings each read fed and which firing each write carried (module doc)."""
    fires: dict[str, list[tuple[int, int]]] = {}
    writes: list[tuple[int, str, int]] = []
    for e in events:
        if e["k"] == "fire":
            fires.setdefault(e["src"], []).append((e["t"], e["n"]))
        elif e["k"] == "cmd" and e.get("op") == "csr_write" and e.get("reg"):
            writes.append((int(e["last"]), str(e["reg"]), int(e["value"])))
    fed: dict[tuple[str, int], list[Firing]] = {}
    made: dict[tuple[str, int], list[Firing]] = {}
    for spec in cluster.components:
        if spec.kind != "accel" or spec.name not in fires:
            continue
        cfg = accel_config(spec)
        attach = dict(spec.attach or {})
        acc_spans = spans.get(spec.name, [])
        for k, span in enumerate(acc_spans):
            fs = [f for f in fires[spec.name] if span[0] <= f[0] <= span[1]]
            firing = [Firing(spec.name, k, n, t) for t, n in fs]
            for f in firing:
                f.lanes = {p.name: p.lanes for p in cfg.ports}
                f.via = {s: p for p, s in attach.items()}
            for p in cfg.ports:
                s = attach.get(p.name)
                s_spans = spans.get(s or "", [])
                r = _rate(p.rate, spec.name, span[0], writes)
                if s is None or k >= len(s_spans) or r is None:
                    continue
                for lane in range(p.lanes):
                    port = f"{s}.{lane}"
                    got = _in_task(lane_hops.get(port, []), s_spans[k])
                    if p.direction == "in":
                        handouts = len(firing) // r
                        rep = handouts // len(got) if got and handouts % len(got) == 0 else 1
                        for i, h in enumerate(got):
                            fed_by = [
                                firing[j * r]
                                for j in range(i * rep, (i + 1) * rep)
                                if j * r < len(firing)
                            ]
                            fed[grant_no[id(h)]] = fed_by
                            for f in fed_by:
                                f.inputs.setdefault(p.name, {})[lane] = h
                    else:
                        for j, h in enumerate(got):
                            push = (j + 1) * r - 1
                            if push < len(firing):
                                made[grant_no[id(h)]] = firing[j * r : push + 1]
                                for f in firing[j * r : push + 1]:
                                    f.outputs.setdefault(p.name, {})[lane] = h
    return fed, made


# =============================================================================
# Answers
# =============================================================================


def _firing_dict(idx: Index, f: Firing, lane: int, streamer: str) -> dict[str, Any]:
    """A firing with its operands on ``lane`` of the port ``streamer`` serves; a port with
    another lane count gives all its lanes."""
    port_lanes = f.lanes.get(f.via.get(streamer, ""), 1)

    def pick(beats: dict[int, Hop], lanes: int) -> list[str]:
        keys = [lane] if lanes == port_lanes else sorted(beats)
        return [
            idx.element(beats[k].mem, beats[k].word) or f"word {beats[k].word}"
            for k in keys
            if k in beats
        ]

    return {
        "acc": f.acc,
        "task": f.task,
        "n": f.n,
        "t": f.t,
        "lane": lane,
        "inputs": {p: pick(b, f.lanes.get(p, port_lanes)) for p, b in f.inputs.items()},
        "outputs": {p: pick(b, f.lanes.get(p, port_lanes)) for p, b in f.outputs.items()},
    }


def _flat_of(r: Region, index: Sequence[int] | int) -> int:
    if isinstance(index, int):
        if not 0 <= index < r.size:
            raise ValueError(f"{r.name} has {r.size} elements, no flat index {index}")
        return index
    if len(index) != len(r.shape) or any(
        not 0 <= i < e for i, e in zip(index, r.shape, strict=True)
    ):
        raise ValueError(f"{r.name} has shape {list(r.shape)}, no index {list(index)}")
    flat = 0
    for i, e in zip(index, r.shape, strict=True):
        flat = flat * e + i
    return flat


def _element_hops(idx: Index, regions: Sequence[Region], name: str, flat: int) -> list[Hop]:
    """Every hop of ``name[flat]``, in every memory a region of that name lives in, in time order."""
    hops: list[Hop] = []
    for r in regions:
        if r.name != name:
            continue
        lay = idx.layouts[r.mem]
        word = (r.address(_unravel(flat, r.shape)) - lay.base_addr) // lay.word_bytes
        hops += idx.words.get((r.mem, word), [])
    hops.sort(key=lambda h: (h.t, h.mem != "l2"))
    return hops


def journey(
    idx: Index, regions: Sequence[Region], name: str, index: Sequence[int] | int
) -> dict[str, Any]:
    """Every hop of element ``name[index]`` in every memory, the firings it fed or came from
    with their operands, and for a firing it fed, the hops of the results from then on."""
    mine = [r for r in regions if r.name == name]
    if not mine:
        raise ValueError(f"the run names no region {name!r}")
    flat = _flat_of(mine[0], index)
    hops = _element_hops(idx, regions, name, flat)
    firings = []
    for h in hops:
        key = idx.grant_no.get(id(h))
        if key is None:
            continue
        for f in idx.fed.get(key, []):
            results = {}
            for beats in f.outputs.values():
                for out in beats.values():
                    for r, fl in idx.cell(out.mem, out.word):
                        after = [x for x in _element_hops(idx, regions, r.name, fl) if x.t >= f.t]
                        results[_elem_text(r, fl)] = [x.to_dict(idx.element) for x in after]
            firings.append(
                {
                    **_firing_dict(idx, f, h.lane, h.owner),
                    "via": h.by,
                    "role": "consumed",
                    "results": {
                        k: v for k, v in results.items() if k in _outputs_on_lane(idx, f, h)
                    },
                }
            )
        for f in idx.made.get(key, []):  # every firing the written beat came from
            firings.append(
                {**_firing_dict(idx, f, h.lane, h.owner), "via": h.by, "role": "produced"}
            )
    return {
        "element": _elem_text(mine[0], flat),
        "flat": flat,
        "memories": [r.mem for r in mine],
        "hops": [h.to_dict(idx.element) for h in hops],
        "firings": firings,
        "filtered": idx.filtered,
    }


def _outputs_on_lane(idx: Index, f: Firing, h: Hop) -> set[str]:
    """The output elements of firing ``f`` that belong with hop ``h``'s lane."""
    d = _firing_dict(idx, f, h.lane, h.owner)
    return {x for xs in d["outputs"].values() for x in xs}


def _unravel(flat: int, shape: Sequence[int]) -> tuple[int, ...]:
    out = []
    for e in reversed(shape):
        out.append(flat % e)
        flat //= e
    return tuple(reversed(out))


def _window(xs: list[int]) -> list[int] | None:
    return [min(xs), max(xs)] if xs else None


def residency(idx: Index, regions: Sequence[Region], dmas: set[str]) -> list[dict[str, Any]]:
    """Per region and memory: arrival, use and departure windows and the waits (module doc).

    Arrival is an element's first write in that memory (none: it was there
    from the start); use is a read by anything but a DMA; departure is a DMA
    read. The waits are per element: first use less arrival, and departure
    less the later of arrival and last use.
    """
    out = []
    for r in regions:
        lay = idx.layouts[r.mem]
        arr, first_use, last_use, dep, w_in, w_out = [], [], [], [], [], []
        present = 0
        for flat in range(r.size):
            word = (r.address(_unravel(flat, r.shape)) - lay.base_addr) // lay.word_bytes
            hs = idx.words.get((r.mem, word), [])
            a = next((h.t for h in hs if h.act == "write"), None)
            uses = [h.t for h in hs if h.act == "read" and h.owner not in dmas]
            d = next(
                (h.t for h in hs if h.act == "read" and h.owner in dmas and (a is None or h.t > a)),
                None,
            )
            if a is None:
                present += 1
            else:
                arr.append(a)
            if uses:
                first_use.append(uses[0])
                last_use.append(uses[-1])
                if a is not None:
                    w_in.append(uses[0] - a)
            if d is not None:
                dep.append(d)
                last = max(
                    [x for x in (a, uses[-1] if uses else None) if x is not None], default=None
                )
                if last is not None:
                    w_out.append(d - last)
        out.append(
            {
                "region": r.name,
                "mem": r.mem,
                "elements": r.size,
                "present_at_start": present,
                "arrival": _window(arr),
                "use": _window(first_use + last_use),
                "departure": _window(dep),
                "wait_in": _window(w_in),
                "wait_out": _window(w_out),
            }
        )
    return out


def fit_nest(xs: Sequence[int]) -> tuple[int, list[list[int]]] | None:
    """``xs`` as base + an affine nest ``[[bound, stride], ...]`` innermost first, or None.

    The innermost loop is the first run of equal steps; the sequence must then
    be whole blocks of it, and the blocks' first values are fitted the same
    way one level up. Regenerating from the answer gives ``xs`` exactly.
    """
    if not xs:
        return None
    base, seq, nest = xs[0], list(xs), []
    while len(seq) > 1:
        d = seq[1] - seq[0]
        b = 2
        while b < len(seq) and seq[b] - seq[b - 1] == d:
            b += 1
        if len(seq) % b:
            return None
        blocks = [seq[i : i + b] for i in range(0, len(seq), b)]
        if any(blk[j] != blk[0] + j * d for blk in blocks for j in range(b)):
            return None
        nest.append([b, d])
        seq = [blk[0] for blk in blocks]
    return base, nest


def expand_nest(base: int, nest: Sequence[Sequence[int]]) -> list[int]:
    """Inverse of fit_nest: innermost loop first, each outer step repeats the inner block."""
    out = [0]
    for bound, stride in nest:
        out = [o + j * stride for j in range(bound) for o in out]
    return [base + o for o in out]


def _elements(idx: Index, mem: str, words: list[int]) -> dict[str, Any] | None:
    """The words as one region's flat indices, fitted as a nest, or None."""
    cells = [idx.cell(mem, w) for w in words]
    if not cells or any(len(c) != 1 for c in cells) or len({c[0][0].name for c in cells}) != 1:
        return None
    fit = fit_nest([c[0][1] for c in cells])
    return None if fit is None else {"region": cells[0][0][0].name, "base": fit[0], "nest": fit[1]}


def patterns(
    idx: Index, spans: dict[str, list[tuple[int, int]]], ports: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per L1 port and task, and per DMA side and task: the nest, the pace and the held-back cycles."""
    out = []
    l1 = idx.layouts.get("l1")
    for port, hs in sorted(idx.lane_hops.items()):
        owner = ports.get(port, {}).get("owner", port)
        for k, span in enumerate(spans.get(owner, [])):
            mine = [h for h in hs if span[0] <= h.t <= span[1]]
            beats = [h for h in mine if h.act in ("read", "write")]
            if not beats:
                continue
            wide = sum(1 for h in beats if h.t == beats[0].t) > 1
            if wide:  # a wide port: one beat is several words in one cycle; fit on the beats
                seen, lead = set(), []
                for h in beats:
                    if h.t not in seen:
                        seen.add(h.t)
                        lead.append(h)
                beats_for_fit = lead
            else:
                beats_for_fit = beats
            addrs = [l1.base_addr + h.word * l1.word_bytes for h in beats_for_fit]
            fit = fit_nest(addrs)
            held = [h for h in mine if h.act == "held back"]
            held_cycles = sorted({h.t for h in held})
            n = len(beats_for_fit)
            first, last = beats_for_fit[0].t, beats_for_fit[-1].t
            out.append(
                {
                    "port": port,
                    "owner": owner,
                    "task": k,
                    "mem": "l1",
                    "write": beats[0].act == "write",
                    "beats": n,
                    "first": first,
                    "last": last,
                    "ideal_last": first + n - 1,
                    "addr": None if fit is None else {"base": fit[0], "nest": fit[1]},
                    "element": None if wide else _elements(idx, "l1", [h.word for h in beats]),
                    "held": [
                        {
                            "t": h.t,
                            "bank": h.bank,
                            "row": h.row,
                            "element": idx.element("l1", h.word),
                            "served": h.served,
                            "served_element": None
                            if h.served is None
                            else idx.element("l1", l1.word_of(h.bank, h.served["row"])),
                        }
                        for h in held
                    ],
                    "held_cycles": len(held_cycles),
                    "idle": max(
                        last - first + 1 - n - len([t for t in held_cycles if first <= t <= last]),
                        0,
                    ),
                }
            )
    for (dma, side), beats in sorted(idx.dma_hops.items()):
        for k, span in enumerate(spans.get(dma, [])):
            mine = [b for b in beats if span[0] <= b[0] <= span[1]]
            if not mine:
                continue
            fit = fit_nest([a for _, _, a, _ in mine])
            out.append(
                {
                    "port": f"{dma}.{side}",
                    "owner": dma,
                    "task": k,
                    "side": side,
                    "mem": mine[0][3],
                    "beats": len(mine),
                    "first": mine[0][0],
                    "last": mine[-1][0],
                    "ideal_last": mine[0][0] + len(mine) - 1,
                    "addr": None if fit is None else {"base": fit[0], "nest": fit[1]},
                }
            )
    return out


def conflicts(idx: Index, start: int = 0, stop: int | None = None) -> dict[str, Any]:
    """The L1 conflicts in ``start <= t < stop`` (all by default), placed on the layout,
    with counts per word, per bank and per port over the same window."""
    rows = [c for c in idx.conflicts if c["t"] >= start and (stop is None or c["t"] < stop)]
    per_word: dict[int, dict[str, Any]] = {}
    per_bank: dict[int, int] = {}
    per_port: dict[str, int] = {}
    listed = []
    for c in rows:
        w, s = c["waiting"], c["served"]
        listed.append(
            {
                "t": c["t"],
                "bank": c["bank"],
                "row": c["row"],
                "wider": c["wider"],
                "waiting": {
                    **{k: w[k] for k in ("port", "owner", "w", "row")},
                    "element": idx.element("l1", w["word"]),
                },
                "served": None
                if s is None
                else {
                    **{k: s[k] for k in ("port", "owner", "w", "row")},
                    "element": idx.element("l1", s["word"]),
                },
            }
        )
        per_bank[c["bank"]] = per_bank.get(c["bank"], 0) + 1
        per_port[w["port"]] = per_port.get(w["port"], 0) + 1
        a = per_word.setdefault(w["word"], {"held": [], "won": []})
        a["held"].append(c["t"])
        if s is not None:
            per_word.setdefault(s["word"], {"held": [], "won": []})["won"].append(c["t"])
    words = []
    for word in sorted(per_word):
        col, row = idx.layouts["l1"].place(word)
        a = per_word[word]
        words.append(
            {
                "bank": col,
                "row": row,
                "element": idx.element("l1", word),
                "held": a["held"],
                "won": a["won"],
            }
        )
    return {
        "from": start,
        "to": stop,
        "count": len(listed),
        "conflicts": listed,
        "words": words,
        "banks": {str(b): n for b, n in sorted(per_bank.items())},
        "ports": dict(sorted(per_port.items())),
        "filtered": idx.filtered,
    }


TIMES = ("arrival", "use", "wait")


def time_marks(idx: Index, mem: str, kind: str) -> TimeMarks:
    """Per word of ``mem`` a cycle (D97): ``arrival`` its first write, ``use`` its first read by
    anything but a DMA, ``wait`` the cycles between the two; None where there is none."""
    if kind not in TIMES:
        raise ValueError(f"a time mark is one of {list(TIMES)}, got {kind!r}")
    lay = idx.layouts[mem]
    marks = TimeMarks()
    for (m, word), hs in idx.words.items():
        if m != mem:
            continue
        arrival = next((h.t for h in hs if h.act == "write"), None)
        use = next((h.t for h in hs if h.act == "read" and h.owner not in idx.dmas), None)
        t = {"arrival": arrival, "use": use}.get(kind)
        if kind == "wait":
            t = None if arrival is None or use is None else use - arrival
        if t is not None:
            col, row = lay.place(word)
            marks.setdefault(row, [None] * lay.columns)[col] = t
    return marks


def conflict_marks(idx: Index) -> Marks:
    """Per L1 word the number of conflicts it took part in (held back or served), as fold marks."""
    lay = idx.layouts["l1"]
    marks: Marks = {}
    for c in idx.conflicts:
        for side in ("waiting", "served"):
            s = c[side]
            if s is None:
                continue
            col, row = lay.place(s["word"])
            marks.setdefault(row, [0] * lay.columns)[col] += 1
    return marks


__all__ = [
    "Index",
    "MovementUnavailable",
    "accel_config",
    "build_index",
    "conflict_marks",
    "conflicts",
    "expand_nest",
    "fit_nest",
    "journey",
    "patterns",
    "residency",
    "time_marks",
]
