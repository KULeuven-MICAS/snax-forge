"""DMA between L2 and L1 for SNAX-MODEL (MOD6, D29, D33, D34).

What this models
----------------
A DMA that moves wide beats between a flat L2 (l2.py) and the L1, so that
reader streamers find their operands in L1 and writer results can go back
to L2. It is a master on the xbar with one wide port of
``L1Config.wide_bits`` (512 bits: one superbank of 8 banks in SNAX), and
the only user of the L2.

It is started by ``start(descriptor, cycle)``, like the streamer and the
accelerator; ctrl.py (MOD7) drives that start from registers, with
``DmaConfig.dims`` loops per side. No descriptor chaining, no data path
extensions, no L1 -> L1.

Descriptor
----------
Both sides are affine, like the streamer: a base plus loops with bounds and
strides (bytes, loop 0 innermost), one step per wide beat. Source and
destination have their own patterns with the same beat count. Every beat is
aligned to the wide beat, so an L1 beat always covers exactly one aligned
bank group. Useful L1 patterns (64-byte beats):

* contiguous: stride 64 B, beats land on banks [0:7], [8:15], ...;
* one superbank, row by row: stride n_banks * 8 B, so the buffer stays in
  one superbank and leaves the others free for compute;
* mixes: e.g. bounds (2, R), strides (64, n_banks * 8): 2 superbanks, R rows.

Timing: per beat (D34)
----------------------
Two independent sides with a beat buffer between them (as the read and
write sides of the Snitch iDMA):

* source: from ``s + startup`` on (start in ``s``), one beat read per
  ``beat_interval`` cycles; its data reaches the buffer after the source
  latency;
* destination: writes the oldest buffered beat, one per ``beat_interval``
  cycles, at the earliest in the cycle after it reached the buffer;
* done: the last write's response comes ``done_latency`` cycles after it;
  ``done_cycle`` is the cycle after that (with ``done_latency = 0``: the
  cycle after the last beat lands).

Source latency ``Ls`` = ``L2Config.read_latency`` for L2 -> L1, and
``L1Config.read_latency + l1_read_extra`` for L1 -> L2. For N beats with no
contention and ``beat_interval = k``:

    done_cycle - s = startup + Ls + k * (N - 1) + 2 + done_latency

(k = 1: startup + Ls + N + 1 + done_latency).

Which RTL it stands for, and what is not copied
-----------------------------------------------
snax_alu uses the Snitch iDMA (``idma_inst64_top``, in the DMA core with
``xdma: true``), not XDMA. What is copied:

* the wide path: the DMA reaches each superbank through
  ``mem_wide_narrow_mux``, which gives it absolute priority and grants it at
  once (xbar.py). Here the DMA is the only 512-bit port, so it is never
  refused; ``stall_l1`` exists for setups with two ports of that width;
* decoupled read and write sides; one beat per cycle at best;
* the L2 of the RTL simulation: fixed latency, always ready (l2.py).

Not copied (open item 7):

* AXI bursts: the iDMA's cost is per burst, with ``NumAxInFlight = 3``
  bursts in flight and bursts split at 256 beats and 4 KiB. Short bursts
  (e.g. the row-by-row pattern) are slower in RTL than here;
* the iDMA's shape: 2D with one inner contiguous length shared by both
  sides. More general patterns need several iDMA descriptors in RTL, each
  with its own startup;
* back-pressure from the iDMA's 3-deep buffer (the buffer here is
  unbounded; its peak is recorded in ``max_buffered``). It only matters
  when the destination can be refused;
* L1 -> L1, unaligned transfers (hardware legalizer), and XDMA;
* where the AXI latencies sit: they are lumped into ``startup``,
  ``L2Config.read_latency``, ``l1_read_extra`` and ``done_latency``, all
  placeholders until ANC1.

Who calls what, per cycle
-------------------------
    CONTROL    controller: dma.start(desc, cycle)                 (ctrl.py)
    REQUEST    DMA:  xbar.request on the wide port: L1 source read or
                     L1 destination write
    MEMORY     DMA:  l2.read (L2 source) or l2.write (L2 destination)
    RESPONSE   DMA:  xbar.granted; source data from xbar.rdata / l2.resp
                     into the buffer

The L2 is accessed in MEMORY, where memories serve; its order relative to
the xbar does not matter because only the DMA touches it. A refused L1
request is held unchanged (D31): its beat index, address and data only
change on a grant.

Waking (D29)
------------
``next_wake`` reads only the DMA's own committed state (beat counters, last
issue cycles, outstanding reads, the buffer, the response cycle). After
cycle t it is the earliest of these, and at least t+1:

* source beats left: ``max(first request, last source beat + k)``;
* buffer not empty: ``max(head visible, last write + k)``;
* the next outstanding read's ready cycle;
* the last write's response cycle;
* None when not started or done.

A refused request keeps its side ready, so the answer is t+1. Nothing
outside the DMA changes these values while it sleeps, except a start,
which lands at the end of its cycle (``_StartReg``) before the next
``next_wake``. So asking again before the returned cycle gives the same
answer.

Cycle classes (for MOD8)
------------------------
Each cycle counts as exactly one of, in this order:

* ``busy``: a beat moved on either side (read issued or write done);
* ``stall_l1``: an L1 request was refused;
* ``idle`` (interval gap): a beat is ready on some side but
  ``beat_interval`` does not allow it yet;
* ``stall_mem``: running, with reads in flight, buffered beats not yet
  visible, or the last response pending;
* ``idle``: anything else: not started, startup, done.

Skipped cycles are classified in ``on_gap`` from committed state; no
recorded gap class is needed (unlike accel.py). The only external input is
the xbar grant, and a DMA with an L1 request pending is ticked every cycle.
Every point where the class can change (end of startup, end of an interval,
a read arriving, a buffered beat becoming visible, the response) is a wake,
so a gap has one class. A start inside a gap does not break this: before a
start the DMA is idle, and after it the startup cycles are idle too.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import prod
from typing import Any

import numpy as np

from .l2 import L2Config, L2Memory
from .mem import BankReq, L1Config
from .sched import ClassLog, Component, Phase, SimulationError
from .streamer import StreamerRegs, address_stream
from .trace import DmaBeat, Done, Start
from .xbar import Xbar

CYCLE_CLASSES = ("busy", "stall_l1", "stall_mem", "idle")
DIRECTIONS = ("l2_to_l1", "l1_to_l2")


# =============================================================================
# Descriptor and configuration
# =============================================================================


@dataclass(frozen=True)
class DmaPattern:
    """Affine beat addresses of one side: base + sum(i_t * stride_t).

    Loop 0 is innermost; one step is one wide beat. Strides are in bytes and
    may be negative. Alignment and range are checked against a memory by
    ``check_descriptor``.
    """

    base: int
    bounds: tuple[int, ...]
    strides: tuple[int, ...]

    def __post_init__(self) -> None:
        for f in ("bounds", "strides"):
            object.__setattr__(self, f, tuple(int(x) for x in getattr(self, f)))
        if len(self.bounds) != len(self.strides):
            raise ValueError(f"{len(self.bounds)} bounds but {len(self.strides)} strides")
        if not self.bounds:
            raise ValueError("need at least one loop")
        if any(b < 0 for b in self.bounds):
            raise ValueError("bounds must be >= 0")

    @property
    def n_beats(self) -> int:
        return prod(self.bounds)

    def addresses(self) -> np.ndarray:
        """Beat addresses in order, shape [n_beats]. Reuses the streamer's AGU."""
        regs = StreamerRegs(self.base, self.bounds, self.strides)
        return address_stream(regs)[:, 0]


@dataclass(frozen=True)
class DmaDescriptor:
    """One transfer: direction, source pattern, destination pattern."""

    direction: str  # "l2_to_l1" or "l1_to_l2"
    src: DmaPattern
    dst: DmaPattern

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {self.direction!r}")
        if self.src.n_beats != self.dst.n_beats:
            raise ValueError(f"source has {self.src.n_beats} beats, destination {self.dst.n_beats}")

    @property
    def n_beats(self) -> int:
        return self.src.n_beats

    @property
    def l1(self) -> DmaPattern:
        """The pattern on the L1 side."""
        return self.dst if self.direction == "l2_to_l1" else self.src

    @property
    def l2(self) -> DmaPattern:
        """The pattern on the L2 side."""
        return self.src if self.direction == "l2_to_l1" else self.dst


def check_descriptor(
    desc: DmaDescriptor, l1: L1Config, l2: L2Config
) -> tuple[np.ndarray, np.ndarray]:
    """Validate ``desc`` against both memories. Returns (source, destination) addresses.

    Every beat must be aligned to the wide beat and lie inside its memory.
    Checked per beat, since strides may be negative. Raises ValueError.
    """
    beat = l1.wide_bits // 8
    sides = (("L1", desc.l1, l1.base_addr, l1.size_bytes), ("L2", desc.l2, l2.base_addr,
                                                            l2.size_bytes))  # fmt: skip
    for mem, p, base, size in sides:
        if (p.base - base) % beat or any(s % beat for s in p.strides):
            raise ValueError(f"{mem} pattern base and strides must be multiples of {beat} B")
        a = p.addresses()
        if len(a) and (a.min() < base or a.max() + beat > base + size):
            raise ValueError(f"{mem} pattern leaves {mem} ({a.min():#x}..{a.max():#x})")
    return desc.src.addresses(), desc.dst.addresses()


@dataclass(frozen=True)
class DmaConfig:
    """Design-time timing of the DMA. All defaults are placeholders until ANC1."""

    startup: int = 2  # start in s -> first source request in s + startup (>= 1)
    beat_interval: int = 1  # bandwidth: min cycles between beats on each side
    l1_read_extra: int = 0  # L1 source: cycles from L1 read data to the buffer
    done_latency: int = 0  # last write -> its response (AXI B)
    # Loops per side that the DMA's registers hold (MOD7, ctrl.py). A pattern
    # may use fewer; unused loops are bound 1. 2 as the iDMA's 2D shape.
    dims: int = 2

    def __post_init__(self) -> None:
        if self.startup < 1 or self.beat_interval < 1 or self.dims < 1:
            raise ValueError("startup, beat_interval and dims must be >= 1")
        if self.l1_read_extra < 0 or self.done_latency < 0:
            raise ValueError("l1_read_extra and done_latency must be >= 0")


# =============================================================================
# DMA
# =============================================================================


class _StartReg:
    """Holds a start written in a cycle until the end of that cycle (as in streamer.py)."""

    def __init__(self, owner: Dma) -> None:
        self.owner = owner
        self.pending: tuple[DmaDescriptor, int] | None = None

    def commit(self) -> None:
        if self.pending is not None:
            desc, cycle = self.pending
            self.pending = None
            self.owner._load(desc, cycle)


@dataclass
class _Wires:
    """This cycle's wires, cleared in commit."""

    cls: str | None = None
    l1_req: bool = False  # drove an xbar request
    refused: bool = False
    src_moved: bool = False
    dst_moved: bool = False
    new_pending: list[tuple[int, int]] = field(default_factory=list)  # (ready, beat)
    # Source data that reached the buffer: (visible cycle, beat, data).
    arrived: list[tuple[int, int, np.ndarray]] = field(default_factory=list)


class Dma(Component):
    """Per-beat DMA between L2 and L1 on one wide xbar port, see the module doc.

    State:

    * committed: ``desc``, ``_src`` / ``_dst`` (addresses), ``_n``,
      ``_first`` (first source request cycle), ``_s`` / ``_d`` (beats read /
      written), ``_last_src`` / ``_last_dst``, ``_pending`` (outstanding reads:
      (ready cycle, beat)), ``_buf`` (buffered beats: (visible cycle, beat,
      data)), ``_resp_at`` (response cycle of the last write), ``done_cycle``;
    * wires: ``_w``;
    * statistics (for MOD8): ``cycles`` per class (a ``ClassLog``),
      ``beats_read``, ``beats_written``, ``max_buffered``. Trace events:
      ``start``, ``done`` (task), ``dma_beat`` per moved beat (beat).
    """

    phases = (Phase.REQUEST, Phase.MEMORY, Phase.RESPONSE)

    def __init__(self, name: str, xbar: Xbar, l2: L2Memory, cfg: DmaConfig | None = None) -> None:
        super().__init__(name)
        self.cfg = cfg or DmaConfig()
        self.xbar, self.l2 = xbar, l2
        l1c, l2c = xbar.mem.cfg, l2.cfg
        if l2c.beat_bits != l1c.wide_bits:
            raise ValueError(f"L2 beat_bits {l2c.beat_bits} != L1 wide_bits {l1c.wide_bits}")
        same = ("width_bits", "dtype", "elems_per_word")
        if any(getattr(l1c, f) != getattr(l2c, f) for f in same):
            raise ValueError(f"L1 and L2 must agree on {same}")
        # The wide port: exactly wide_bits (add_port checks n_banks against it).
        self.port = xbar.add_port(self, f"{name}.wide", width_bits=l1c.wide_bits)
        self._start = _StartReg(self)
        # Committed state.
        self.desc: DmaDescriptor | None = None
        self._src = self._dst = np.zeros(0, dtype=np.int64)
        self._n = 0
        self._first = 0
        self._s = self._d = 0
        self._last_src: int | None = None
        self._last_dst: int | None = None
        self._pending: deque[tuple[int, int]] = deque()
        self._buf: deque[tuple[int, int, np.ndarray]] = deque()
        self._resp_at: int | None = None
        self.done_cycle: int | None = None
        # Statistics.
        self.cycles = ClassLog(CYCLE_CLASSES)
        self.beats_read = 0
        self.beats_written = 0
        self.max_buffered = 0
        self._w = _Wires()

    # -------------------------------------------------------------------------
    # Control side: start and done
    # -------------------------------------------------------------------------

    @property
    def to_l1(self) -> bool:
        """Committed: the current (or last) transfer goes L2 -> L1."""
        return self.desc is not None and self.desc.direction == "l2_to_l1"

    @property
    def busy(self) -> bool:
        """Committed: beats left to write, or the last response still pending."""
        return self._n > 0 and (self._d < self._n or self._resp_at is not None)

    def start(self, desc: DmaDescriptor, cycle: int | None = None) -> None:
        """Start a transfer. ``busy`` is True from ``cycle + 1``.

        During a run, call it in Phase.CONTROL with the current cycle; before
        a run, without ``cycle`` (as if started in cycle -1). A zero-beat
        transfer never becomes busy; its ``done_cycle`` is ``cycle + 1``.
        """
        for side, p in (("source", desc.src), ("destination", desc.dst)):
            if len(p.bounds) > self.cfg.dims:
                raise ValueError(
                    f"{self.name}: {side} has {len(p.bounds)} loops, "
                    f"DmaConfig.dims is {self.cfg.dims}"
                )
        check_descriptor(desc, self.xbar.mem.cfg, self.l2.cfg)
        if self.busy or self._start.pending is not None:
            raise SimulationError(f"{self.name}: start while busy")
        if cycle is None:
            self._load(desc, -1)
        else:
            self._start.pending = (desc, cycle)
            self.xbar.mem.cluster.touch(self._start)

    def _load(self, desc: DmaDescriptor, cycle: int) -> None:
        """Committed effect of a start in ``cycle``."""
        self.desc = desc
        self._src, self._dst = desc.src.addresses(), desc.dst.addresses()
        self._n = desc.n_beats
        self._first = cycle + self.cfg.startup
        self._s = self._d = 0
        self._last_src = self._last_dst = None
        self._pending.clear()
        self._buf.clear()
        self._resp_at = None
        self.done_cycle = None if self._n else cycle + 1
        tr = self._trace
        if tr is not None and tr.task:
            tr.emit(Start(cycle, self.name))
            if not self._n:
                tr.emit(Done(cycle + 1, self.name))

    # -------------------------------------------------------------------------
    # Conditions (committed state)
    # -------------------------------------------------------------------------

    def _src_at(self) -> int | None:
        """Earliest cycle the next source beat may issue, or None if none is left."""
        if not self.busy or self._s >= self._n:
            return None
        if self._last_src is None:
            return self._first
        return max(self._first, self._last_src + self.cfg.beat_interval)

    def _dst_at(self) -> int | None:
        """Earliest cycle the buffer head may be written, or None if the buffer is empty."""
        if not self._buf:
            return None
        if self._last_dst is None:
            return self._buf[0][0]
        return max(self._buf[0][0], self._last_dst + self.cfg.beat_interval)

    def _src_ready(self, cycle: int) -> bool:
        at = self._src_at()
        return at is not None and cycle >= at

    def _dst_ready(self, cycle: int) -> bool:
        at = self._dst_at()
        return at is not None and cycle >= at

    def _sleep_class(self, cycle: int) -> str:
        """Class of a cycle in which no beat moves and nothing is refused."""
        if not self.busy or cycle < self._first:
            return "idle"  # not started, done, or starting up
        k = self.cfg.beat_interval
        src_gap = self._s < self._n and self._last_src is not None and cycle < self._last_src + k
        dst_gap = (
            bool(self._buf)
            and self._buf[0][0] <= cycle
            and self._last_dst is not None
            and cycle < self._last_dst + k
        )
        if src_gap or dst_gap:
            return "idle"  # bandwidth gap, like the accelerator's II gap
        if self._pending or self._buf or self._resp_at is not None:
            return "stall_mem"
        return "idle"

    # -------------------------------------------------------------------------
    # Component interface
    # -------------------------------------------------------------------------

    def tick(self, cycle: int, phase: Phase) -> None:
        w = self._w
        if phase == Phase.REQUEST:
            w.cls = self._sleep_class(cycle)  # before acting: committed state only
            if not self.to_l1 and self._src_ready(cycle):  # L1 source read
                req = BankReq(int(self._src[self._s]), tag=self._s)
                self.xbar.request(cycle, self.port, req)
                w.l1_req = True
            elif self.to_l1 and self._dst_ready(cycle):  # L1 destination write
                _, beat, data = self._buf[0]
                req = BankReq(int(self._dst[beat]), write=True, wdata=data, tag=beat)
                self.xbar.request(cycle, self.port, req)
                w.l1_req = True
        elif phase == Phase.MEMORY:
            if self.to_l1 and self._src_ready(cycle):  # L2 source read
                ready = self.l2.read(cycle, int(self._src[self._s]), tag=self._s)
                w.new_pending.append((ready, self._s))
                w.src_moved = True
            elif not self.to_l1 and self._dst_ready(cycle):  # L2 destination write
                self.l2.write(cycle, int(self._dst[self._d]), self._buf[0][2])
                w.dst_moved = True
        elif phase == Phase.RESPONSE:
            self._response(cycle)

    def _response(self, cycle: int) -> None:
        w = self._w
        if w.l1_req:
            if self.xbar.granted(cycle, self.port):
                if self.to_l1:
                    w.dst_moved = True
                else:
                    w.new_pending.append((cycle + self.xbar.mem.cfg.read_latency, self._s))
                    w.src_moved = True
            else:
                w.refused = True
        # Source data due now, from reads of earlier cycles or (latency 0) this one.
        extra = 0 if self.to_l1 else self.cfg.l1_read_extra
        for ready, beat in [*self._pending, *w.new_pending]:
            if ready != cycle:
                continue
            if self.to_l1:
                r = self.l2.resp(cycle)
            else:
                r = self.xbar.rdata(cycle, self.port)
            if r is None or r.tag != beat:
                raise SimulationError(f"{self.name}: lost read data of beat {beat} in {cycle}")
            w.arrived.append((cycle + 1 + extra, beat, np.asarray(r.data)))
        if w.src_moved or w.dst_moved:
            w.cls = "busy"
        elif w.refused:
            w.cls = "stall_l1"

    def commit(self, cycle: int) -> None:
        w = self._w
        tr = self._trace
        if tr is not None and tr.beat:
            # Before the counters move: _s / _d are the beats of this cycle.
            # L2 -> L1 reads the L2 and writes the L1; L1 -> L2 the other way.
            src_mem, dst_mem = ("l2", "l1") if self.to_l1 else ("l1", "l2")
            if w.src_moved:
                tr.emit(DmaBeat(cycle, self.name, side="src", i=self._s, mem=src_mem,
                                addr=int(self._src[self._s])))  # fmt: skip
            if w.dst_moved:
                tr.emit(DmaBeat(cycle, self.name, side="dst", i=self._d, mem=dst_mem,
                                addr=int(self._dst[self._d])))  # fmt: skip
        if w.src_moved:
            self._s += 1
            self._last_src = cycle
            self.beats_read += 1
        if w.dst_moved:
            self._buf.popleft()
            self._d += 1
            self._last_dst = cycle
            self.beats_written += 1
            if self._d == self._n:  # last beat written: wait for its response
                self._resp_at = cycle + self.cfg.done_latency
        self._pending = deque(p for p in [*self._pending, *w.new_pending] if p[0] > cycle)
        self._buf.extend(w.arrived)
        self.max_buffered = max(self.max_buffered, len(self._buf))
        if self._resp_at is not None and cycle >= self._resp_at:
            self._resp_at = None
            self.done_cycle = cycle + 1
            if tr is not None and tr.task:
                tr.emit(Done(cycle + 1, self.name))
        cls = w.cls if w.cls is not None else self._sleep_class(cycle)
        self.cycles.add(cls, cycle, cycle + 1)
        self._w = _Wires()

    def next_wake(self, cycle: int) -> int | None:
        """See "Waking" in the module doc. Reads committed state only."""
        cands = [self._src_at(), self._dst_at(), self._resp_at]
        if self._pending:
            cands.append(self._pending[0][0])  # issue order = ready order
        cands = [c for c in cands if c is not None]
        return max(cycle + 1, min(cands)) if cands else None

    def on_gap(self, start: int, stop: int) -> None:
        """Skipped cycles: one class, from committed state (module doc)."""
        self.cycles.add(self._sleep_class(start), start, stop)

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Totals for a quick look; MOD8 builds the real profile."""
        return {
            "cycles": dict(self.cycles),
            "beats_read": self.beats_read,
            "beats_written": self.beats_written,
            "max_buffered": self.max_buffered,
        }
