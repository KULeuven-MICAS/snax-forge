"""The dynamic energy (CONTRACTS, "Cost model"; D177): D167's five parts with a run's counts in
place of the graph's, each priced at the instance's existing price.

    prices_of(graph, est, movers) -> Prices     what one transaction of each instance costs
    price(prices, counts) -> Energy             a window's counts priced, in nJ per part
    graph_counts(graph, est) -> Counts          the static estimate's counts, for comparison

``Counts`` are what SNAX-GVSoC counts in a window (run/snapshots.py): per unit its firings and
pushes, the groups' busy-flag cycles added up, per mover its crossbar requests (grants and
stalls) and grants. A unit fires its firings times its span, a buffer link loads and a packer
pushes its producer's pushes, a slicer pops on its unit's firings (a seed's on its pushes, one a
job), the control units clock every group's busy cycle, a mover moves
its requests over its channels, the crossbar serves every request at its mover's kind (a
reader's read-only, a writer's write-only, D178) and the banks the grants. With
``graph_counts`` the parts are the static estimate's (``Estimate.parts``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from snax_forge.brm import SEED
from snax_forge.char import load
from snax_forge.char.tables import streamer_cost as _streamer
from snax_forge.design.streamers import group_links, instances

from .cost import PARTS

if TYPE_CHECKING:
    from snax_forge.dfg import Graph

    from . import Estimate


@dataclass(frozen=True)
class MoverSpec:
    """A mover as the tables price it: its name, whether it writes, its FIFO depth, loops and
    channels."""

    name: str
    write: bool
    depth: int
    loops: int
    channels: int


@dataclass(frozen=True)
class Prices:
    """One transaction of each instance, in pJ: per unit its span and energy per fire, per buffer
    link or packer (``<unit>.<port>``) its producer and energy per load or push, the control
    units' bits and e_ctrl, per mover whether it writes, its channels and its energy per beat,
    the crossbar's per request of a read-only and of a write-only input (D178), a bank's per read
    and per write; per slicer (``<unit>.<port>``) its unit, the count it pops on (``firings``, a
    seed's ``pushes``) and its energy per pop."""

    units: dict[str, tuple[int, float]]
    links: dict[str, tuple[str, float]]
    ctrl_bits: int
    e_ctrl_pj: float
    movers: dict[str, tuple[bool, int, float]]
    e_xro_pj: float
    e_xwo_pj: float
    e_rd_pj: float
    e_wr_pj: float
    slicers: dict[str, tuple[str, str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"units": {k: {"span": s, "e_fire": e} for k, (s, e) in self.units.items()},
                "links": {k: {"from": u, "e_fire": e} for k, (u, e) in self.links.items()},
                "slicers": {k: {"from": u, "count": n, "e_fire": e}
                            for k, (u, n, e) in self.slicers.items()},
                "ctrl_bits": self.ctrl_bits, "e_ctrl": self.e_ctrl_pj,
                "movers": {k: {"write": w, "channels": c, "e_beat": e}
                           for k, (w, c, e) in self.movers.items()},
                "e_xbar_ro": self.e_xro_pj, "e_xbar_wo": self.e_xwo_pj, "e_rd": self.e_rd_pj,
                "e_wr": self.e_wr_pj}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Prices:
        return cls({k: (v["span"], v["e_fire"]) for k, v in d["units"].items()},
                   {k: (v["from"], v["e_fire"]) for k, v in d["links"].items()},
                   d["ctrl_bits"], d["e_ctrl"],
                   {k: (v["write"], v["channels"], v["e_beat"]) for k, v in d["movers"].items()},
                   d["e_xbar_ro"], d["e_xbar_wo"], d["e_rd"], d["e_wr"],
                   {k: (v["from"], v["count"], v["e_fire"])
                    for k, v in d.get("slicers", {}).items()})  # fmt: skip


@dataclass
class Counts:
    """A window's counts (module doc): per unit ``(firings, pushes)``, the groups' busy-flag cycles
    added up, per mover ``(requests, grants)``."""

    units: dict[str, tuple[int, int]] = field(default_factory=dict)
    busy: int = 0
    movers: dict[str, tuple[int, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"units": {k: {"firings": f, "pushes": p} for k, (f, p) in self.units.items()},
                "busy": self.busy,
                "movers": {k: {"requests": r, "grants": g}
                           for k, (r, g) in self.movers.items()}}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Counts:
        return cls({k: (v["firings"], v["pushes"]) for k, v in d["units"].items()}, d["busy"],
                   {k: (v["requests"], v["grants"]) for k, v in d["movers"].items()})  # fmt: skip


@dataclass(frozen=True)
class Energy:
    """A window's counts priced: nJ per part (``PARTS``), and the count each part was priced on:
    ``fires`` (the units' fires, the links' loads and pushes, the slicers' pops), ``cycles`` (the busy-flag
    cycles), ``beats`` (the movers' requests over their channels), ``requests`` (the crossbar's),
    ``reads`` and ``writes`` (the banks')."""

    parts: dict[str, float]
    fires: int
    cycles: int
    beats: float
    requests: int
    reads: int
    writes: int

    @property
    def total(self) -> float:
        return sum(self.parts.values())

    def to_dict(self) -> dict[str, Any]:
        return {"parts": dict(self.parts), "total": self.total,
                "counts": {"fires": self.fires, "cycles": self.cycles, "beats": self.beats,
                           "requests": self.requests, "reads": self.reads,
                           "writes": self.writes}}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Energy:
        c = d["counts"]
        return cls(dict(d["parts"]), c["fires"], c["cycles"], c["beats"], c["requests"],
                   c["reads"], c["writes"])  # fmt: skip


def prices_of(graph: Graph, est: Estimate, movers: Sequence[MoverSpec]) -> Prices:
    """Each instance's price per transaction from a costed estimate's own (cost.py): a unit's and
    a link's energy per fire, the control units' bits and e_ctrl, each mover's energy per beat at
    its depth, loops and channels, the crossbar's energy per access, the banks' per read and
    write. Raises ValueError when ``est`` has no cost on a platform."""
    costs, x, m = est.costs, est.xbar, est.memory
    if costs is None or x is None or m is None:
        raise ValueError("the estimate is not costed on a platform: no prices")
    insts = instances(graph)
    units: dict[str, tuple[int, float]] = {}
    links: dict[str, tuple[str, float]] = {}
    slicers: dict[str, tuple[str, str, float]] = {}
    bits = 0
    for g in est.groups:
        if g.cost is None:
            raise ValueError(f"group {g.name} is not costed: no prices")
        bits += g.cost.ctrl_bits
        for u, b in g.cost.units.items():
            units[u] = (insts[u].span, b.e_fire_pj)
        src = _producers(graph, g.units)
        for k, b in g.cost.links.items():
            if b.datapath == "FifoSlicer":
                u, port = k.split(".", 1)
                slicers[k] = (u, "pushes" if port == SEED else "firings", b.e_fire_pj)
            else:
                links[k] = (src(k), b.e_fire_pj)
    beat = {s.name: (s.write, s.channels, _streamer(costs, s.depth, s.loops, s.channels).energy_pj)
            for s in movers}  # fmt: skip
    return Prices(units, links, bits, load(est.tech or "").e_ctrl_pj, beat, x.e_ro_pj, x.e_wo_pj,
                  m.e_rd_pj, m.e_wr_pj, slicers)  # fmt: skip


def _producers(graph: Graph, units: Sequence[str]) -> Callable[[str], str]:
    """A group's link or packer key to the unit whose pushes it loads: a buffer link's container to
    its producer, a packer's ``<unit>.<port>`` to its unit."""
    by = {k.attrs["container"]: frm.split(".", 1)[0] for k, frm, _ in group_links(graph, units)}
    return lambda key: by.get(key, key.split(".", 1)[0])


def price(p: Prices, c: Counts) -> Energy:
    """A window's counts priced (module doc); a unit, link or mover with no count costs 0, a count
    with no price is a ValueError."""
    for what, have, priced in (("unit", c.units, p.units), ("mover", c.movers, p.movers)):
        unknown = sorted(set(have) - set(priced))
        if unknown:
            raise ValueError(f"{what}s {unknown} have counts and no price")
    fires = {u: c.units.get(u, (0, 0))[0] * span for u, (span, _) in p.units.items()}
    loads = {k: c.units.get(u, (0, 0))[1] for k, (u, _) in p.links.items()}
    pops = {k: c.units.get(u, (0, 0))[n == "pushes"] for k, (u, n, _) in p.slicers.items()}
    dp = (sum(fires[u] * e for u, (_, e) in p.units.items())
          + sum(loads[k] * e for k, (_, e) in p.links.items())
          + sum(pops[k] * e for k, (_, _, e) in p.slicers.items()))  # fmt: skip
    beats = strm = xbar = 0.0
    requests = reads = writes = 0
    for name, (write, channels, e_beat) in p.movers.items():
        r, g = c.movers.get(name, (0, 0))
        beats += r / channels
        strm += r / channels * e_beat
        requests += r
        xbar += r * (p.e_xwo_pj if write else p.e_xro_pj)
        if write:
            writes += g
        else:
            reads += g
    parts = {
        "datapath": dp / 1000,
        "control": p.e_ctrl_pj * p.ctrl_bits * c.busy / 1000,
        "streamers": strm / 1000,
        "crossbar": xbar / 1000,
        "memory": (reads * p.e_rd_pj + writes * p.e_wr_pj) / 1000,
    }
    assert tuple(parts) == PARTS
    return Energy(parts, sum(fires.values()) + sum(loads.values()) + sum(pops.values()), c.busy,
                  beats, requests, reads, writes)  # fmt: skip


def graph_counts(graph: Graph, est: Estimate) -> Counts:
    """The static estimate's counts in ``Counts``' form: a unit's firings (its fires over its
    span) and its pushes (its links' and packer's loads, a seeded unit's seed pops), the groups' cycles (pipelined, the
    accelerator's: the longest path, D179), per port's
    streamer its words as requests and grants (none refused). Priced at ``prices_of`` with the
    ports' streamers as movers, the parts are ``est.parts``' energies."""
    insts = instances(graph)
    out = Counts()
    for g in est.groups:
        if g.cost is None:
            continue
        out.busy += g.cycles
        src = _producers(graph, g.units)
        pushes = {src(k): b.fires for k, b in g.cost.links.items() if b.datapath != "FifoSlicer"}
        for k, b in g.cost.links.items():  # a seed pops once a job, its unit's results
            if b.datapath == "FifoSlicer" and k.split(".", 1)[1] == SEED:
                pushes.setdefault(k.split(".", 1)[0], b.fires)
        for u, b in g.cost.units.items():
            out.units[u] = (b.fires // insts[u].span, pushes.get(u, 0))
    if est.schedule != "serial":  # pipelined, the control units clock the overlapped cycles (D179)
        out.busy = est.accelerator_cycles
    for s in est.streamer_counts:
        out.movers[s.name] = (s.words, s.words)
    return out


def port_movers(est: Estimate) -> list[MoverSpec]:
    """The estimate's port streamers as movers (a shared mover's users each at its own channels),
    as ``graph_counts`` names them."""
    return [MoverSpec(s.name, s.write, s.depth, s.loops, s.channels) for s in est.streamer_counts]
