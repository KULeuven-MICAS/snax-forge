"""Profile of a SNAX-MODEL run (MOD8, D38, section 5.7).

What it is
----------
The totals of one run, collected from the counters the components already
keep. ``build_profile`` reads them; it never recounts, and it changes
nothing in the cluster (the FIFO occupancy is closed on a copy). The trace
(trace.py) is written independently, and the tests check the two against
each other (D38).

Contents (section 5.7)
----------------------
* ``total_cycles``;
* ``controller``: cycles per class (``command`` = control overhead, kept
  apart from ``wait`` as section 7 needs), commands, csr reads, polls and
  every wait (D37);
* per accelerator: cycles per class, utilisation (busy / total, i.e. the
  firing rate: a busy cycle is a firing cycle, so there is no separate
  firing count), and beats per port;
* per streamer: cycles per class, its xbar ports, and its FIFO occupancy
  (max, time-weighted mean and histogram per lane, D40);
* per DMA: cycles per class, beats and bytes each way, max buffered beats;
* per L1 bank: reads, writes (L1Memory), grants, conflicts, stalls and
  blocked (Xbar);
* per xbar port: owner, width, grants, stalls, stalls caused by a wider
  grant;
* per L2: reads and writes (beats);
* ``functional_check``: empty until the reference executor exists (E2E1).

``skip_idle`` is not part of the profile on purpose: the profile must be
the same with skipping on and off.

Dataclasses with ``to_dict`` / ``from_dict`` (D26). Maps are keyed by name,
in registration order, so the JSON reads and diffs well.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .accel import Accelerator
from .ctrl import Controller, Wait
from .dma import Dma
from .streamer import Streamer
from .xbar import Xbar

# =============================================================================
# Parts
# =============================================================================


@dataclass
class FifoProfile:
    """Occupancy of one FIFO, per lane; ``hist[lane][c]`` = cycles holding c elements."""

    name: str
    depth: int
    max: list[int]
    mean: list[float]
    hist: list[list[int]]


@dataclass
class StreamerProfile:
    write: bool
    cycles: dict[str, int]
    ports: list[str]
    fifo: FifoProfile

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StreamerProfile:
        return cls(d["write"], dict(d["cycles"]), list(d["ports"]), FifoProfile(**d["fifo"]))


@dataclass
class AccelProfile:
    cycles: dict[str, int]  # "busy" = firings (accel.py cycle classes)
    utilisation: float
    beats: dict[str, int]


@dataclass
class DmaProfile:
    cycles: dict[str, int]
    beats_read: int
    beats_written: int
    bytes_read: int
    bytes_written: int
    max_buffered: int


@dataclass
class WaitProfile:
    """One wait: first and last cycle, and the block's done_cycle at the end (D37)."""

    pc: int
    block: str
    mode: str
    first: int
    done: int | None
    last: int


@dataclass
class ControllerProfile:
    name: str
    cycles: dict[str, int]
    commands: int  # finished csr_write / csr_read commands (waits not included)
    reads: int  # csr_read commands
    polls: int  # poll samples over all waits
    waits: list[WaitProfile]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ControllerProfile:
        return cls(d["name"], dict(d["cycles"]), d["commands"], d["reads"], d["polls"],
                   [WaitProfile(**w) for w in d["waits"]])  # fmt: skip


@dataclass
class BankProfile:
    """Per L1 bank, indexed by bank number."""

    reads: list[int]
    writes: list[int]
    grants: list[int]
    conflicts: list[int]
    stalls: list[int]
    blocked: list[int]


@dataclass
class PortProfile:
    owner: str
    width: int
    grants: int
    stalls: int
    stalls_wider: int


@dataclass
class L2Profile:
    reads: int
    writes: int


# =============================================================================
# Profile
# =============================================================================


@dataclass
class Profile:
    total_cycles: int
    controller: ControllerProfile | None = None
    accelerators: dict[str, AccelProfile] = field(default_factory=dict)
    streamers: dict[str, StreamerProfile] = field(default_factory=dict)
    dmas: dict[str, DmaProfile] = field(default_factory=dict)
    banks: BankProfile | None = None
    ports: dict[str, PortProfile] = field(default_factory=dict)
    l2: dict[str, L2Profile] = field(default_factory=dict)
    functional_check: dict[str, Any] | None = None  # MOD9 / E2E1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Profile:
        ctl = d.get("controller")
        banks = d.get("banks")
        return cls(
            total_cycles=d["total_cycles"],
            controller=None if ctl is None else ControllerProfile.from_dict(ctl),
            accelerators={k: AccelProfile(**v) for k, v in d.get("accelerators", {}).items()},
            streamers={k: StreamerProfile.from_dict(v) for k, v in d.get("streamers", {}).items()},
            dmas={k: DmaProfile(**v) for k, v in d.get("dmas", {}).items()},
            banks=None if banks is None else BankProfile(**banks),
            ports={k: PortProfile(**v) for k, v in d.get("ports", {}).items()},
            l2={k: L2Profile(**v) for k, v in d.get("l2", {}).items()},
            functional_check=d.get("functional_check"),
        )


# =============================================================================
# Building it from a cluster
# =============================================================================


def _ints(a: Any) -> list[int]:
    return [int(x) for x in a]


def _fifo(fifo: Any, total: int) -> FifoProfile:
    occ = fifo.occupancy(total)  # a copy: the FIFO is not changed
    counts = range(occ.shape[1])
    hist = [_ints(row) for row in occ]
    mx = [max((c for c in counts if row[c]), default=0) for row in hist]
    mean = [sum(c * row[c] for c in counts) / total if total else 0.0 for row in hist]
    return FifoProfile(fifo.name, fifo.depth, mx, mean, hist)


def _controller(ctl: Controller) -> ControllerProfile:
    waits = [
        WaitProfile(pc, block, ctl.program[pc].mode, t, d, last)
        for pc, block, t, d, last in ctl.waits
    ]
    commands = sum(1 for pc, _, _ in ctl.spans if not isinstance(ctl.program[pc], Wait))
    return ControllerProfile(ctl.name, dict(ctl.cycles), commands, len(ctl.reads), ctl.polls, waits)


def build_profile(cluster: Any) -> Profile:
    """Collect the counters of every component after a run (reads only)."""
    total = int(cluster.cycle)
    comps = list(cluster)
    prof = Profile(total_cycles=total)

    ctls = [c for c in comps if isinstance(c, Controller)]
    if len(ctls) > 1:
        raise ValueError(f"one controller per cluster, found {[c.name for c in ctls]}")
    if ctls:
        prof.controller = _controller(ctls[0])

    for c in comps:
        if isinstance(c, Accelerator):
            busy = c.cycles["busy"]
            prof.accelerators[c.name] = AccelProfile(
                dict(c.cycles), busy / total if total else 0.0, dict(c.beats)
            )
        elif isinstance(c, Streamer):
            names = [c.xbar.ports[p].name for p in c.ports]
            prof.streamers[c.name] = StreamerProfile(
                bool(c.cfg.write), dict(c.cycles), names, _fifo(c.fifo, total)
            )
        elif isinstance(c, Dma):
            beat_bytes = c.xbar.mem.cfg.wide_bits // 8
            prof.dmas[c.name] = DmaProfile(
                dict(c.cycles), c.beats_read, c.beats_written,
                c.beats_read * beat_bytes, c.beats_written * beat_bytes, c.max_buffered,
            )  # fmt: skip

    xbars = [c for c in comps if isinstance(c, Xbar)]
    if len(xbars) > 1:
        raise ValueError(f"one xbar per cluster, found {[c.name for c in xbars]}")
    if xbars:
        xb = xbars[0]
        xb._freeze()  # sizes the counters if the xbar never ran (no-op otherwise)
        mem = xb.mem
        prof.banks = BankProfile(
            _ints(mem.reads), _ints(mem.writes), _ints(xb.bank_grants),
            _ints(xb.bank_conflicts), _ints(xb.bank_stalls), _ints(xb.bank_blocked),
        )  # fmt: skip
        for p in xb.ports:
            i = p.index
            prof.ports[p.name] = PortProfile(
                p.owner.name, p.width_bits, int(xb.port_grants[i]),
                int(xb.port_stalls[i]), int(xb.port_stalls_wider[i]),
            )  # fmt: skip

    # L2 memories behind the DMAs, once each, in order of first use.
    seen: list[Any] = []
    for c in comps:
        if isinstance(c, Dma) and all(c.l2 is not s for s in seen):
            seen.append(c.l2)
    for i, l2 in enumerate(seen):
        prof.l2["l2" if i == 0 else f"l2.{i}"] = L2Profile(int(l2.reads), int(l2.writes))
    return prof
