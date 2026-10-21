"""Data movement of a run, as queries on its index (D97, index.py).

What this is
------------
Four answers, all read off the trace's events: nothing is modelled or matched
by order here, every link is an id the events carry (snax-gvsoc
``docs/EVENTS.md``) and every number a cycle or a value SNAX-GVSoC wrote.

  journey     every access to one element in both memories, in time order;
              from a reader's grant its beat ``b`` gives the firings that name
              it, each with its other operands (their beats' grants, the fire's
              ``vals``); a firing gives the push that folds it, the push the
              writer's grants (``b = n``), and each written element its outgoing
              DMA beat and that copy's burst (``id``); a writer's grant gives its
              push and the firings it folds
  residency   per region and memory: each element's arrival (its first write
              there), uses (reads by anything but the DMA after it), departure
              (the DMA's first read after it) and wait (first use - arrival),
              and their windows
  patterns    per streamer lane and task: the address nest its CSR writes
              configured (``run/beats.mover_nest``, which the converter checked
              equal to the grants), the pace (first and last grant against one
              beat per cycle) and every cycle it was held, with the port its
              bank served and that port's address; per DMA copy its sides, as
              issued, and its L1 beats' pace
  conflicts   every L1 stall placed on the memory layout: the word the held
              request wanted and the word its bank served in that cycle; per
              word, per bank and per port the counts

Levels
------
Journey, residency and patterns need the beat level, conflicts and the held
cycles the stall level; below it an answer is ``available`` false with the
reason (``MovementUnavailable``). A filtered trace gives what it kept, and
every answer says which sources it has (``filtered``, None when the
accelerator, its movers and the DMA are all in). Ids count from 0 in each run:
the journey refuses when the index says they do not name one line each
(``RunIndex.linked``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from snax_forge.run import region_at
from snax_forge.run.beats import mover_nest, mover_values
from snax_forge.run.cluster import DMA

from .index import Event, RunIndex
from .memory import element_name


class MovementUnavailable(Exception):
    """The run's trace cannot answer; the message is the reason shown to the user."""


def need(idx: RunIndex, level: str, what: str) -> None:
    """A MovementUnavailable unless the run was traced at ``level`` or above."""
    if not idx.on(level):
        raise MovementUnavailable(f"{what} needs --trace {level}; this run was traced at "
                                  f"{idx.level}")  # fmt: skip


# =============================================================================
# Names
# =============================================================================


def _shapes(idx: RunIndex) -> dict[str, tuple[int, ...]]:
    return {r.container: r.shape for r in idx.regions}


def names(idx: RunIndex, container: str | None, elems: Sequence[int]) -> list[str]:
    """An event's elements by name (``A[5]``)."""
    if container is None:
        return []
    shape = _shapes(idx).get(container, (0,))
    return [element_name(container, shape, f) for f in elems]


def _served(idx: RunIndex, addr: int) -> str | None:
    """The element at an L1 address no event names (a stall's ``by_addr``)."""
    r = region_at(idx.regions, "l1", addr)
    if r is None:
        return None
    word = idx.word_of(addr)
    return " + ".join(names(idx, r.container, r.elems(word, idx.cluster.bank_bytes))) or None


def flat_of(idx: RunIndex, container: str, index: Sequence[int] | int) -> int:
    """A container's element as its flat index; a ValueError when it has none such."""
    shape = _shapes(idx).get(container)
    if shape is None:
        raise ValueError(f"the run names no region {container!r}")
    size = int(np.prod(shape))
    if isinstance(index, int):
        if not 0 <= index < size:
            raise ValueError(f"{container} has {size} elements, no flat index {index}")
        return index
    if len(index) != len(shape) or any(not 0 <= i < e for i, e in zip(index, shape, strict=True)):
        raise ValueError(f"{container} has shape {list(shape)}, no index {list(index)}")
    return int(np.ravel_multi_index(tuple(index), shape))


# =============================================================================
# Journey
# =============================================================================

ACTS = {"r": "read", "w": "write"}


def hop(idx: RunIndex, e: Event, flat: int | None = None) -> dict[str, Any]:
    """One access as the journey shows it: the event's own fields, its act, the port or
    the DMA, and the element's value when the event carries it."""
    k = e["k"]
    d: dict[str, Any] = {"t": e["t"], "k": k, "mem": e.get("mem", "l1"), "task": e["task"],
                         "act": "held back" if k == "stall" else ACTS.get(e["op"], e["op"]),
                         "by": e["src"], "addr": e["addr"]}  # fmt: skip
    for key in ("bank", "row", "b", "id", "i", "size", "back", "l2"):
        if key in e:
            d[key] = e[key]
    m = idx.mover_of(e["src"])
    if m is not None:
        d["owner"], d["lane"] = m.name, m.ports.index(e["src"])
    vals = e.get("vals") or []
    if flat is not None and flat in e["elems"] and len(vals) == len(e["elems"]):
        d["value"] = vals[e["elems"].index(flat)]
    d["elements"] = names(idx, e["region"], e["elems"]) if k != "burst" else None
    if k == "stall":
        d["served"] = {"port": e["by"], "addr": e["by_addr"],
                       "row": idx.cluster.bank_row(e["by_addr"])[1],
                       "element": _served(idx, e["by_addr"])}  # fmt: skip
    return d


def _beat(idx: RunIndex, mover: str, b: int) -> dict[str, Any]:
    """A streamer beat: its grants by lane, their elements and values."""
    grants = [idx.beat.get((mover, b), {}).get(lane) for lane in range(_lanes(idx, mover))]
    return {
        "b": b,
        "grants": [None if g is None else hop(idx, g) for g in grants],
        "elements": [
            x for g in grants if g is not None for x in names(idx, g["region"], g["elems"])
        ],
    }


def _lanes(idx: RunIndex, mover: str) -> int:
    return next(m.lanes for m in idx.cluster.accelerator.movers if m.name == mover)


def _writer(idx: RunIndex) -> str | None:
    return next((m.name for m in idx.cluster.accelerator.movers if m.write), None)


def _results(idx: RunIndex, grant: Event) -> dict[str, list[dict[str, Any]]]:
    """Each element a writer's grant wrote: that grant, the DMA beat that reads it out of L1
    after it, and that copy's bursts holding it in L2."""
    out: dict[str, list[dict[str, Any]]] = {}
    for flat, name in zip(grant["elems"], names(idx, grant["region"], grant["elems"]), strict=True):
        hops = [hop(idx, grant, flat)]
        evs = idx.elem.get((grant["region"], flat), [])
        out_dma = next((e for e in evs if e["k"] == "dma" and e["op"] == "r"
                        and e["t"] >= grant["t"]), None)  # fmt: skip
        if out_dma is not None:
            hops.append(hop(idx, out_dma, flat))
            hops += [hop(idx, b, flat) for b in idx.burst.get(out_dma["id"], [])
                     if b["region"] == grant["region"] and flat in b["elems"]]  # fmt: skip
        out[name] = hops
    return out


def firing(idx: RunIndex, f: Event) -> dict[str, Any]:
    """A firing: its operands (each reader's beat), the push that folds it, the writer's
    grants of that push and where each written element went."""
    operands = {m: {**_beat(idx, m, b), "vals": f["vals"].get(m)} for m, b in f["b"].items()}
    d: dict[str, Any] = {"acc": f["src"], "task": f["task"], "n": f["n"], "t": f["t"],
                         "operands": operands,
                         "inputs": {m: o["elements"] for m, o in operands.items()},
                         "push": None, "outputs": {}, "results": {}}  # fmt: skip
    p = idx.folded.get(f["n"])
    w = _writer(idx)
    if p is not None:
        d["push"] = {"n": p["n"], "t": p["t"], "first": p["first"], "last": p["last"],
                     "vals": p["vals"]}  # fmt: skip
        if w is not None:
            beat = _beat(idx, w, p["n"])
            d["push"]["writer"] = beat
            d["outputs"] = {w: beat["elements"]}
            for lane in range(_lanes(idx, w)):
                g = idx.beat.get((w, p["n"]), {}).get(lane)
                if g is not None:
                    d["results"].update(_results(idx, g))
    return d


def journey(idx: RunIndex, container: str, index: Sequence[int] | int) -> dict[str, Any]:
    """Every access to element ``container[index]`` and the firings it fed or came from
    (module doc)."""
    flat = flat_of(idx, container, index)
    need(idx, "beat", "the journey")
    if idx.linked is not None:
        raise MovementUnavailable(f"the journey: {idx.linked}")
    evs = idx.elem.get((container, flat), [])
    firings = []
    for e in evs:
        m = idx.mover_of(e["src"]) if e["k"] == "grant" else None
        if m is None or "b" not in e:
            continue
        lane = m.ports.index(e["src"])
        if not m.write:
            for f in idx.fired.get((m.name, e["b"]), []):
                firings.append({**firing(idx, f), "role": "consumed", "via": e["src"],
                                "lane": lane})  # fmt: skip
        elif e["b"] in idx.push:
            p = idx.push[e["b"]]
            for n in range(p["first"], p["last"] + 1):
                if n in idx.fire:
                    firings.append({**firing(idx, idx.fire[n]), "role": "produced",
                                    "via": e["src"], "lane": lane})  # fmt: skip
    return {
        "element": names(idx, container, [flat])[0],
        "region": container,
        "flat": flat,
        "memories": [r.memory for r in idx.regions if r.container == container],
        "hops": [hop(idx, e, flat) for e in evs],
        "firings": firings,
        "filtered": idx.filtered,
    }


# =============================================================================
# Residency
# =============================================================================


def _window(xs: list[int]) -> list[int] | None:
    return [min(xs), max(xs)] if xs else None


def residency(idx: RunIndex) -> list[dict[str, Any]]:
    """Per region and memory, in the run's region order: each element's times and their
    windows (module doc)."""
    need(idx, "beat", "residency")
    out = []
    for r in idx.regions:
        elements, arr, uses_all, dep, w_in, w_out = [], [], [], [], [], []
        for flat in range(int(np.prod(r.shape))):
            evs = [e for e in idx.elem.get((r.container, flat), [])
                   if e.get("mem", "l1") == r.memory]  # fmt: skip
            a = next((e["t"] for e in evs if e.get("op") == "w"), None)
            after = [e for e in evs if a is None or e["t"] > a]
            uses = [e["t"] for e in after if e["k"] == "grant" and e["op"] == "r"]
            d = next((e["t"] for e in after if e["k"] in ("dma", "burst") and e["op"] == "r"), None)
            wait = None if a is None or not uses else uses[0] - a
            elements.append({"flat": flat, "arrival": a, "uses": uses, "departure": d,
                             "wait": wait})  # fmt: skip
            if a is not None:
                arr.append(a)
            uses_all += uses
            if wait is not None:
                w_in.append(wait)
            if d is not None:
                dep.append(d)
                last = max([x for x in (a, uses[-1] if uses else None) if x is not None],
                           default=None)  # fmt: skip
                if last is not None:
                    w_out.append(d - last)
        out.append({
            "region": r.container, "mem": r.memory, "elements": len(elements),
            "present_at_start": sum(1 for x in elements if x["arrival"] is None),
            "arrival": _window(arr), "use": _window(uses_all), "departure": _window(dep),
            "wait_in": _window(w_in), "wait_out": _window(w_out), "per_element": elements,
        })  # fmt: skip
    return out


# =============================================================================
# Patterns
# =============================================================================


def _regs(idx: RunIndex, before: int) -> dict[int, int]:
    """The streamer's CSR values written up to cycle ``before`` (a start copies them)."""
    regs: dict[int, int] = {}
    for e in idx.csr.get(idx.cluster.accelerator.streamer, []):
        if e["op"] == "w" and e["t"] <= before:
            regs[e["csr"]] = e["value"]
    return regs


def _held(idx: RunIndex, port: str, task: str) -> list[dict[str, Any]]:
    out = []
    for e in idx.stalls_port.get(port, []):
        if e["task"] == task:
            out.append({"t": e["t"], "bank": e["bank"], "row": e["row"], "addr": e["addr"],
                        "element": " + ".join(names(idx, e["region"], e["elems"])) or None,
                        "by": e["by"], "by_addr": e["by_addr"],
                        "by_element": _served(idx, e["by_addr"])})  # fmt: skip
    return out


def patterns(idx: RunIndex) -> list[dict[str, Any]]:
    """Per streamer lane and task, then per DMA copy (module doc)."""
    need(idx, "beat", "the patterns")
    c = idx.cluster
    out = []
    for m in c.accelerator.movers:
        for s in (s for s in idx.start.values() if s["src"] == m.name):
            regs = _regs(idx, s["t"])
            ptr, sp, tb, ts = mover_values(c, m, regs)
            lanes = mover_nest(c, m, regs)
            bounds = [1 if (d == 0 and not m.write and ts[0] == 0) else b for d, b in enumerate(tb)]
            for lane, port in enumerate(m.ports):
                grants = [g for g in idx.grants_port.get(port, []) if g["task"] == s["task"]]
                p: dict[str, Any] = {
                    "port": port, "owner": m.name, "task": s["task"], "mem": "l1",
                    "write": m.write, "beats": len(grants),
                    "addr": {"base": c.l1_base + lanes[lane][0] if lanes[lane] else None,
                             "ptr": ptr, "spatial_strides": sp,
                             "nest": [[b, st] for b, st in zip(bounds, ts, strict=True)],
                             "addresses": len(lanes[lane])},
                }  # fmt: skip
                if grants:
                    first, last = grants[0]["t"], grants[-1]["t"]
                    p.update(first=first, last=last, ideal_last=first + len(grants) - 1)
                else:
                    p.update(first=None, last=None, ideal_last=None)
                if idx.on("stall"):
                    held = _held(idx, port, s["task"])
                    p["held"] = held
                    p["held_cycles"] = len({h["t"] for h in held})
                    if grants:
                        inside = len({h["t"] for h in held if first <= h["t"] <= last})
                        p["idle"] = max(last - first + 1 - len(grants) - inside, 0)
                else:
                    p["held"] = None
                    p["held_reason"] = f"needs --trace stall; this run was traced at {idx.level}"
                out.append(p)
    for ident, cp in idx.issue.items():
        beats = [idx.dma[key] for key in sorted(k for k in idx.dma if k[0] == ident)]
        p = {"port": DMA, "owner": DMA, "task": cp["task"], "copy": ident, "dir": cp["dir"],
             "from": cp["from"], "to": cp["to"], "size": cp["size"], "reps": cp["reps"],
             "from_stride": cp["from_stride"], "to_stride": cp["to_stride"],
             "beats": len(beats), "bursts": len(idx.burst.get(ident, []))}  # fmt: skip
        if beats:
            first, last = beats[0]["t"], beats[-1]["t"]
            p.update(first=first, last=last, ideal_last=first + len(beats) - 1)
        out.append(p)
    return out


# =============================================================================
# Conflicts
# =============================================================================


def _side(idx: RunIndex, port: str, addr: int, element: str | None) -> dict[str, Any]:
    m = idx.mover_of(port)
    return {"port": port, "owner": m.name if m is not None else port,
            "w": m.write if m is not None else None, "addr": addr,
            "row": idx.cluster.bank_row(addr)[1], "element": element}  # fmt: skip


def conflicts(idx: RunIndex, start: int = 0, stop: int | None = None) -> dict[str, Any]:
    """The L1 stalls in ``start <= t < stop`` (all by default), placed on the layout, with
    counts per word, per bank and per port over the same window."""
    need(idx, "stall", "the conflicts")
    stalls = sorted((e for es in idx.stalls_bank.values() for e in es
                     if e["t"] >= start and (stop is None or e["t"] < stop)),
                    key=lambda e: e["t"])  # fmt: skip
    listed, per_bank, per_port = [], {}, {}
    per_word: dict[int, dict[str, Any]] = {}
    for e in stalls:
        waiting = _side(idx, e["src"], e["addr"], " + ".join(names(idx, e["region"], e["elems"]))
                        or None)  # fmt: skip
        served = _side(idx, e["by"], e["by_addr"], _served(idx, e["by_addr"]))
        listed.append({"t": e["t"], "bank": e["bank"], "row": e["row"], "task": e["task"],
                       "b": e.get("b"), "waiting": waiting, "served": served})  # fmt: skip
        per_bank[e["bank"]] = per_bank.get(e["bank"], 0) + 1
        per_port[e["src"]] = per_port.get(e["src"], 0) + 1
        for side, key in ((waiting, "held"), (served, "won")):
            w = idx.word_of(side["addr"])
            per_word.setdefault(w, {"held": [], "won": [], "element": side["element"]})[key].append(
                e["t"]
            )
    words = []
    for w in sorted(per_word):
        bank, row = idx.cluster.bank_row(w)
        words.append({"addr": w, "bank": bank, "row": row, **per_word[w]})
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


__all__ = [
    "MovementUnavailable",
    "conflicts",
    "firing",
    "flat_of",
    "hop",
    "journey",
    "names",
    "need",
    "patterns",
    "residency",
]
