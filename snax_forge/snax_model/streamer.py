"""Streamers and streamer FIFOs for SNAX-MODEL (MOD4, D12, D29, D31).

What this models
----------------
A streamer moves data between the L1 and one accelerator port (D12: one
streamer per accelerator port). It is a master on the interconnect with
``n_ports`` narrow ports, one per lane of a beat:

* a reader requests the addresses of each beat, collects the read data and
  pushes it into its FIFO; the accelerator pops whole beats;
* a writer pops the accelerator's beats from its FIFO and writes them.

Nothing more: no data path extensions, no byte masks, no channel enables.

Address generation
------------------
The registers (``StreamerRegs``) describe temporal loops (one beat per
step) around a parallel spatial loop (one port per element):

    for each temporal index i:           # loop 0 innermost
        parfor each spatial index j:     # lane = j0 + b0*(j1 + b1*(j2 ...))
            addr = base + sum(i_t * s_t) + sum(j_k * sigma_k)

``address_stream`` is a pure function of the registers. It works like the
RTL AddressGenUnit: one counter per temporal loop that adds its stride on
every tick and wraps at its bound, each counter ticking when all inner ones
wrap. The tests check it against a direct NumPy enumeration.

Which RTL it copies (snax_cluster hw/chisel/src/main/scala/snax/readerWriter/)
------------------------------------------------------------------------------
Only what changes cycle counts:

* Loop order: counter 0 is innermost; lane index splits with spatial dim 0
  fastest (AddressGenUnit ``spatialOffsets``).
* Bounds: a loop of bound 1 runs once. If any temporal bound is 0 the AGU
  ignores ``start`` and the streamer never becomes busy (zero beats).
* Ports are independent, not lockstep. The AGU writes a whole beat of
  addresses into a per-port address queue (ComplexQueueConcat, depth
  ``addr_depth``, pipe). Each port then issues on its own. A port can run
  ahead of the others until its address queue or its data credit runs out.
  The AGU pushes the next beat in a cycle when every port's queue has room
  or that port's request is granted in the same cycle (pipe).
* Reader flow control (DataResponser): per port an up/down counter of
  requests issued minus beats popped by the accelerator. A port may issue
  while the counter is below the FIFO depth, or in a cycle in which the
  accelerator pops a beat (combinational). So at most ``fifo_depth`` reads
  per port are in flight or buffered, and the FIFO cannot overflow. The
  counter is never reset, so credit carries over between tasks.
* Writer: a port writes when it has an address and its FIFO lane has data.
  It does not wait for write responses (the model returns none, MOD3).
* FIFO timing (chisel3.util.Queue): a pushed element is visible one cycle
  later (flow = false). The reader's data buffer has pipe = true: in a
  cycle in which the accelerator pops a lane, that lane accepts a push even
  when full. The writer's data buffer has pipe = false: a pop does not free
  a slot for a push in the same cycle.
* Start and done: ``start`` in cycle s makes the AGU busy from s+1, so the
  first address is pushed in s+1 and the first request goes out in s+2.
  ``busy`` is "AGU busy or an address queue not empty", i.e. it drops the
  cycle after the last request is granted. For a reader, read data may still
  be in flight or in the FIFO at that point (same as RTL).

Not copied (they change cycles, flagged for ANC2):

* Dynamic TCDM priority (``dynamicPriority``, default on in snax_alu): the
  RTL raises a reader port's priority when its FIFO lane is nearly empty
  and a writer's when nearly full. Here every port uses the static
  ``StreamerConfig.prio``.
* Reader temporal stride 0 on loop 0: the RTL reads once and repeats the
  beat ``bound0`` times (HandShakeRepeater). ``start`` rejects it for readers
  instead of silently re-reading.
* Spatial bounds are design-time parameters in RTL (only spatial strides
  are CSRs). Here they are in the registers, but their product must equal
  ``n_ports``.

Who calls what, per cycle
-------------------------
    CONTROL    controller:   streamer.start(regs, cycle)        (MOD7)
    COMPUTE    accelerator:  fifo.pop(cycle) / fifo.push(cycle, beat)   (MOD5)
    REQUEST    streamer:     xbar.request(...) for each port that can issue
    ARBITRATE  xbar
    MEMORY     xbar -> L1
    RESPONSE   streamer:     xbar.granted, xbar.rdata;
                             reader: fifo.push_lane; writer: fifo.pop_lane;
                             AGU pushes the next beat of addresses

Hold rule (D31)
---------------
A refused request is driven again unchanged next cycle, with no extra
logic: the port's address and tag (beat index) only change on a grant, the
reader's credit can only grow while it waits (it only drops on a grant),
and the writer's FIFO lane head only leaves on a grant.

Waking (D29)
------------
``next_wake`` reads only committed state: its own counters, the FIFO's
committed contents, the xbar's outstanding reads, and the next_wake of the
FIFO's other side. After cycle t:

* t+1 if any port can issue (address present, plus credit or FIFO data),
  including a port holding a refused request, or if the AGU can push;
* reader with a port blocked on credit (FIFO full): the accelerator's own
  next_wake. A pop can only happen in a cycle where the accelerator is
  ticked, and the pop frees credit in that same cycle, so the reader must
  be ticked in exactly those cycles (the xbar uses the same trick with its
  port owners);
* writer with a port blocked on an empty FIFO lane: nothing. A push only
  becomes visible after the FIFO commits, and next_wake is asked again
  after every cycle, so it then answers t+1;
* the cycle of the next read data for any port (``xbar.next_rdata``);
* None when done.

None of these can change unless something the streamer reads changes, so
asking again before the returned cycle gives the same answer.

Cycle classes (for MOD8)
------------------------
Each cycle counts as exactly one of, in this order:
``busy`` (some port granted), ``stall_xbar`` (some port requested, none
granted), ``stall_fifo`` (some port has an address but no credit or no
data), ``idle`` (anything else: not started, done, waiting for the first
address or for read data). Skipped cycles are classified from the
committed state; the streamer only sleeps in the ``stall_fifo`` or ``idle``
states, and neither can change while it sleeps.

Statistics and trace (MOD8)
---------------------------
``cycles`` is a ``ClassLog``: commit and on_gap add their cycles to it, and
it keeps the class runs when the run is traced. Trace events (task level):
``start`` when a start lands, ``done`` with the done_cycle. The FIFO keeps an
occupancy histogram per lane (always on, see ``Fifo``) and, at beat level,
emits a ``fifo`` event per lane whose count changes. Grants are traced by
the xbar. None of this is read back by the model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

from .config import Config
from .mem import BankReq
from .sched import ClassLog, Component, Phase, SimulationError
from .trace import Done, FifoCount, Start
from .xbar import Xbar

CYCLE_CLASSES = ("busy", "stall_xbar", "stall_fifo", "idle")


# =============================================================================
# Registers and address generation
# =============================================================================


@dataclass(frozen=True)
class StreamerRegs:
    """Streamer register set: base address, temporal and spatial loops.

    Separate from the accelerator CSRs; the CSR addresses are fixed in MOD7.
    Loop 0 is innermost (temporal) or fastest (spatial). Strides are in
    bytes and may be negative. Lists are stored as tuples.
    """

    base: int
    temporal_bounds: tuple[int, ...]
    temporal_strides: tuple[int, ...]
    spatial_bounds: tuple[int, ...] = (1,)
    spatial_strides: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        for f in ("temporal_bounds", "temporal_strides", "spatial_bounds", "spatial_strides"):
            object.__setattr__(self, f, tuple(int(x) for x in getattr(self, f)))
        tb, ts = self.temporal_bounds, self.temporal_strides
        sb, ss = self.spatial_bounds, self.spatial_strides
        if len(tb) != len(ts):
            raise ValueError(f"{len(tb)} temporal bounds but {len(ts)} temporal strides")
        if len(sb) != len(ss):
            raise ValueError(f"{len(sb)} spatial bounds but {len(ss)} spatial strides")
        if not tb or not sb:
            raise ValueError("need at least one temporal and one spatial loop")
        if any(b < 0 for b in tb):
            raise ValueError("temporal bounds must be >= 0")
        if any(b < 1 for b in sb):
            raise ValueError("spatial bounds must be >= 1")

    @property
    def n_beats(self) -> int:
        """Number of beats (temporal steps); 0 if any temporal bound is 0."""
        return prod(self.temporal_bounds)

    @property
    def n_lanes(self) -> int:
        """Elements per beat; equals the streamer's port count."""
        return prod(self.spatial_bounds)


def lane_offsets(regs: StreamerRegs) -> np.ndarray:
    """Spatial offset of each lane, shape [n_lanes]. Spatial dim 0 is fastest."""
    out = np.zeros(regs.n_lanes, dtype=np.int64)
    for lane in range(regs.n_lanes):
        rem, off = lane, 0
        for b, s in zip(regs.spatial_bounds, regs.spatial_strides):
            off += (rem % b) * s
            rem //= b
        out[lane] = off
    return out


def address_stream(regs: StreamerRegs) -> np.ndarray:
    """All addresses in issue order, shape [n_beats, n_lanes].

    Written like the RTL counters (no multiplications in the temporal part)
    so the NumPy enumeration in the tests is an independent check.
    """
    tb, ts = regs.temporal_bounds, regs.temporal_strides
    n = regs.n_beats
    out = np.empty((n, regs.n_lanes), dtype=np.int64)
    if n == 0:
        return out
    lanes = lane_offsets(regs)
    value = [0] * len(tb)  # counter outputs: i_t * s_t, built by adding s_t
    count = [0] * len(tb)  # index i_t, to detect the wrap (the "small counter")
    for beat in range(n):
        out[beat] = regs.base + sum(value) + lanes
        # Tick counter 0; carry into the next counter only when it wraps.
        for t in range(len(tb)):
            if count[t] == tb[t] - 1:
                count[t], value[t] = 0, 0
            else:
                count[t] += 1
                value[t] += ts[t]
                break
    return out


# =============================================================================
# FIFO between streamer and accelerator
# =============================================================================


@dataclass(eq=False)  # compared by identity, like every touched element
class Fifo:
    """Per-lane queues of ``depth`` entries (ComplexQueueConcat in RTL).

    Shared state element: pushes and pops touch it, and the scheduler calls
    ``commit`` at the end of the cycle. Two sides:

    * wide: ``push(cycle, beat)`` / ``pop(cycle)`` move a whole beat; valid
      only when every lane can take / give one;
    * narrow: ``push_lane`` / ``pop_lane`` move one lane's element.

    The reader streamer uses the narrow push and the accelerator the wide
    pop; for a writer it is the other way round.

    Timing (chisel3 Queue, flow = false):

    * reads (``count``, ``can_pop*``, ``peek*``) see committed state; a push
      becomes visible after ``commit``, i.e. in the next cycle;
    * at most one push and one pop per lane per cycle;
    * ``pipe``: a lane popped earlier in this cycle accepts a push even when
      full. The pop must come in an earlier phase than the push (accelerator
      in COMPUTE, reader in RESPONSE).

    ``pusher`` / ``popper`` are the components on each side. The reader uses
    ``popper.next_wake`` while it waits for credit (see streamer waking).

    Occupancy (MOD8, D40): ``occ[lane, c]`` counts the cycles in which the
    lane held ``c`` elements. Counts change only in ``commit`` (a pushed
    element is visible from the next cycle), so ``commit`` in cycle t closes
    the old count's run at t + 1. ``commit`` runs only when the FIFO was
    touched, in both skip modes, so the histogram is identical with
    skipping on and off. ``occupancy(total)`` adds the open run up to
    ``total`` without changing the FIFO.
    """

    cluster: Any  # anything with touch(elem)
    lanes: int
    depth: int
    pipe: bool = False
    name: str = "fifo"

    _trace = None  # set by Trace.bind (not a dataclass field: no annotation)

    def __post_init__(self) -> None:
        if self.lanes < 1 or self.depth < 1:
            raise ValueError("lanes and depth must be >= 1")
        self.pusher: Component | None = None
        self.popper: Component | None = None
        # Committed state.
        self._q: list[deque[Any]] = [deque() for _ in range(self.lanes)]
        self.pushed = np.zeros(self.lanes, dtype=np.int64)  # totals, also used
        self.popped = np.zeros(self.lanes, dtype=np.int64)  # for reader credit
        # This cycle's wires, cleared in commit.
        self._now: int | None = None
        self._pop_now = [False] * self.lanes
        self._push_now: list[list[Any]] = [[] for _ in range(self.lanes)]
        # Statistics (MOD8): cycles per (lane, count), and where the open run began.
        self.occ = np.zeros((self.lanes, self.depth + 1), dtype=np.int64)
        self._occ_since = np.zeros(self.lanes, dtype=np.int64)

    def _enter(self, cycle: int) -> None:
        if self._now is not None and self._now != cycle:
            raise SimulationError(f"{self.name} was not committed between cycles")
        self._now = cycle
        self.cluster.touch(self)

    # -- lane (narrow) side ---------------------------------------------------

    def count(self, lane: int) -> int:
        """Committed number of elements in ``lane``."""
        return len(self._q[lane])

    def can_pop_lane(self, lane: int) -> bool:
        return bool(self._q[lane]) and not self._pop_now[lane]

    def peek_lane(self, lane: int) -> Any:
        """Head of ``lane`` (committed). Only valid if ``can_pop_lane``."""
        return self._q[lane][0]

    def pop_lane(self, cycle: int, lane: int) -> Any:
        if not self.can_pop_lane(lane):
            raise SimulationError(f"{self.name}: pop from empty lane {lane} in cycle {cycle}")
        self._enter(cycle)
        self._pop_now[lane] = True
        return self._q[lane][0]

    def popped_now(self, cycle: int, lane: int) -> bool:
        """Wire: ``lane`` was popped earlier in ``cycle``."""
        return self._now == cycle and self._pop_now[lane]

    def can_push_lane(self, lane: int) -> bool:
        used = len(self._q[lane]) - (1 if self.pipe and self._pop_now[lane] else 0)
        return not self._push_now[lane] and used < self.depth

    def push_lane(self, cycle: int, lane: int, x: Any) -> None:
        if not self.can_push_lane(lane):
            raise SimulationError(f"{self.name}: push to full lane {lane} in cycle {cycle}")
        self._enter(cycle)
        self._push_now[lane].append(x)

    # -- wide side --------------------------------------------------------------

    def can_pop(self) -> bool:
        return all(self.can_pop_lane(j) for j in range(self.lanes))

    def peek(self) -> list[Any]:
        return [q[0] for q in self._q]

    def pop(self, cycle: int) -> list[Any]:
        """Pop one element from every lane (a beat)."""
        if not self.can_pop():
            raise SimulationError(f"{self.name}: pop without a full beat in cycle {cycle}")
        return [self.pop_lane(cycle, j) for j in range(self.lanes)]

    def can_push(self) -> bool:
        return all(self.can_push_lane(j) for j in range(self.lanes))

    def push(self, cycle: int, beat: Any) -> None:
        """Push one element into every lane. ``beat`` has one entry per lane."""
        if len(beat) != self.lanes:
            raise SimulationError(f"{self.name}: beat has {len(beat)} lanes, need {self.lanes}")
        if not self.can_push():
            raise SimulationError(f"{self.name}: push to a full lane in cycle {cycle}")
        for j in range(self.lanes):
            self.push_lane(cycle, j, beat[j])

    @property
    def empty(self) -> bool:
        return not any(self._q)

    def commit(self) -> None:
        """End of cycle: remove popped heads, append pushes, update occupancy."""
        t = self._now
        tr = self._trace
        for j in range(self.lanes):
            old = len(self._q[j])
            if self._pop_now[j]:
                self._q[j].popleft()
                self.popped[j] += 1
            for x in self._push_now[j]:
                self._q[j].append(x)
                self.pushed[j] += 1
            self._pop_now[j] = False
            self._push_now[j] = []
            new = len(self._q[j])
            if new != old and t is not None:
                # The old count held up to cycle t; the new one from t + 1.
                self.occ[j, old] += t + 1 - self._occ_since[j]
                self._occ_since[j] = t + 1
                if tr is not None and tr.beat:
                    tr.emit(FifoCount(t + 1, self.name, lane=j, count=new))
        self._now = None

    def occupancy(self, total: int) -> np.ndarray:
        """Histogram [lanes, depth + 1] of cycles per count over ``[0, total)``."""
        occ = self.occ.copy()
        for j in range(self.lanes):
            occ[j, len(self._q[j])] += max(0, total - int(self._occ_since[j]))
        return occ


# =============================================================================
# Streamer
# =============================================================================


@dataclass(frozen=True)
class StreamerConfig(Config):
    """Design-time parameters of one streamer."""

    write: bool = False  # False: reader (L1 -> FIFO); True: writer (FIFO -> L1)
    n_ports: int = 1  # interconnect ports = lanes per beat (D12)
    temporal_dims: int = 1  # temporal counters in hardware
    fifo_depth: int = 8  # data buffer per lane (snax_alu fifo_depth)
    addr_depth: int = 8  # address buffer per port (snax_alu uses fifo_depth)
    prio: int = 0  # static TCDM priority of every port

    def __post_init__(self) -> None:
        if min(self.n_ports, self.temporal_dims, self.fifo_depth, self.addr_depth) < 1:
            raise ValueError("n_ports, temporal_dims, fifo_depth, addr_depth must be >= 1")


class _StartReg:
    """Holds a start written in a cycle until the end of that cycle.

    A touched element, so the start lands in commit even if the streamer
    itself was not ticked in that cycle.
    """

    def __init__(self, owner: Streamer) -> None:
        self.owner = owner
        self.pending: tuple[StreamerRegs, int] | None = None

    def commit(self) -> None:
        if self.pending is not None:
            regs, cycle = self.pending
            self.pending = None
            self.owner._load(regs, cycle)


class Streamer(Component):
    """Reader or writer between the L1 (through the xbar) and a FIFO.

    Construct streamers in RTL port order: each adds its ``n_ports`` ports to
    the xbar in lane order, and the index decides round-robin tie-breaks.

    State, split the RTL way:

    * committed state:
        ``regs``, ``_addrs``   current task and its address stream
        ``_gen``      beats whose addresses the AGU has pushed
        ``_k``        per port: beats granted in this task
        ``_issued``   per port: reads granted over all tasks (credit)
        ``done_cycle`` first cycle in which ``busy`` is False after a task
    * this cycle's wires: ``_now``, ``_req``, ``_fire``, ``_agu_push``, ``_cls``
    * statistics (for MOD8): ``cycles`` per class (a ``ClassLog``), see the
      module doc
    """

    phases = (Phase.REQUEST, Phase.RESPONSE)

    def __init__(self, name: str, xbar: Xbar, cfg: StreamerConfig | None = None) -> None:
        super().__init__(name)
        self.cfg = cfg = cfg or StreamerConfig()
        self.xbar = xbar
        self._cluster = xbar.mem.cluster
        n = cfg.n_ports
        self.ports = [xbar.add_port(self, f"{name}.{j}") for j in range(n)]
        # Reader data buffer is pipe, writer data buffer is not (see module doc).
        self.fifo = Fifo(self._cluster, n, cfg.fifo_depth, pipe=not cfg.write, name=f"{name}.fifo")
        if cfg.write:
            self.fifo.popper = self
        else:
            self.fifo.pusher = self
        self._start = _StartReg(self)
        # Committed state.
        self.regs: StreamerRegs | None = None
        self._addrs = np.zeros((0, n), dtype=np.int64)
        self._n = 0
        self._gen = 0
        self._k = np.zeros(n, dtype=np.int64)
        self._issued = np.zeros(n, dtype=np.int64)
        self.done_cycle: int | None = None
        self.cycles = ClassLog(CYCLE_CLASSES)
        self._clear_wires()

    def _clear_wires(self) -> None:
        n = self.cfg.n_ports
        self._now: int | None = None
        self._req = np.zeros(n, dtype=bool)
        self._fire = np.zeros(n, dtype=bool)
        self._agu_push = False
        self._cls: str | None = None

    # -------------------------------------------------------------------------
    # Control side: start and done
    # -------------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        """Committed: AGU busy or a port still has an address to issue."""
        return self._n > 0 and bool((self._k < self._n).any())

    def start(self, regs: StreamerRegs, cycle: int | None = None) -> None:
        """Start a task. ``busy`` becomes True from ``cycle + 1``.

        During a run, call it in Phase.CONTROL with the current cycle. Before
        a run, call it without ``cycle``: the task starts as if started in
        cycle -1, so the first address is pushed in cycle 0.
        """
        self._check(regs)
        if self.busy or self._start.pending is not None:
            raise SimulationError(f"{self.name}: start while busy")
        if cycle is None:
            self._load(regs, -1)
        else:
            self._start.pending = (regs, cycle)
            self._cluster.touch(self._start)

    def _check(self, regs: StreamerRegs) -> None:
        c = self.cfg
        if regs.n_lanes != c.n_ports:
            raise ValueError(
                f"{self.name}: spatial bounds give {regs.n_lanes} lanes, streamer has {c.n_ports}"
            )
        # Fewer loops than counters is the same as padding with bound 1.
        if len(regs.temporal_bounds) > c.temporal_dims:
            raise ValueError(
                f"{self.name}: {len(regs.temporal_bounds)} temporal loops, hardware has "
                f"{c.temporal_dims}"
            )
        if not c.write and regs.temporal_strides[0] == 0:
            raise NotImplementedError(
                f"{self.name}: reader with temporal stride 0 on loop 0 (RTL repeats the beat)"
            )

    def _load(self, regs: StreamerRegs, cycle: int) -> None:
        """Committed effect of a start in ``cycle``."""
        self.regs = regs
        self._addrs = address_stream(regs)
        self._n = regs.n_beats
        self._gen = 0
        self._k[:] = 0
        # A zero-beat task never makes the AGU busy (RTL ignores the start).
        self.done_cycle = None if self._n else cycle + 1
        tr = self._trace
        if tr is not None and tr.task:
            tr.emit(Start(cycle, self.name))
            if not self._n:
                tr.emit(Done(cycle + 1, self.name))

    # -------------------------------------------------------------------------
    # Per-port conditions (committed state plus earlier-phase wires)
    # -------------------------------------------------------------------------

    def _has_addr(self, j: int) -> bool:
        """The AGU pushed this port's next address in an earlier cycle."""
        return bool(self._k[j] < self._gen)

    def _credit(self, j: int, cycle: int | None = None) -> bool:
        """Reader: room for one more read on port ``j``.

        With ``cycle``, includes a pop by the accelerator earlier in that
        cycle (combinational in RTL).
        """
        used = int(self._issued[j] - self.fifo.popped[j])
        if used < self.cfg.fifo_depth:
            return True
        return cycle is not None and self.fifo.popped_now(cycle, j)

    # -------------------------------------------------------------------------
    # Component interface
    # -------------------------------------------------------------------------

    def tick(self, cycle: int, phase: Phase) -> None:
        if phase == Phase.REQUEST:
            self._request(cycle)
        elif phase == Phase.RESPONSE:
            self._response(cycle)

    def _request(self, cycle: int) -> None:
        """Drive a request on every port that has an address and credit/data."""
        self._now = cycle
        write = self.cfg.write
        for j in range(self.cfg.n_ports):
            if not self._has_addr(j):
                continue
            k = int(self._k[j])
            addr = int(self._addrs[k, j])
            if write:
                if not self.fifo.can_pop_lane(j):
                    continue  # no data from the accelerator yet
                req = BankReq(addr, write=True, wdata=self.fifo.peek_lane(j), tag=k)
            else:
                if not self._credit(j, cycle):
                    continue  # FIFO lane full, counting reads in flight
                req = BankReq(addr, tag=k)
            self.xbar.request(cycle, self.ports[j], req, self.cfg.prio)
            self._req[j] = True

    def _response(self, cycle: int) -> None:
        """Grants, read data, the AGU's next beat, and the cycle class."""
        n = self.cfg.n_ports
        for j in range(n):
            p = self.ports[j]
            if self._req[j] and self.xbar.granted(cycle, p):
                self._fire[j] = True
                if self.cfg.write:
                    self.fifo.pop_lane(cycle, j)
            if not self.cfg.write:
                r = self.xbar.rdata(cycle, p)
                if r is not None:
                    self.fifo.push_lane(cycle, j, r.data)  # credit guarantees room

        # AGU: push the next beat if every port's address queue has room,
        # counting a slot freed by this cycle's grant (pipe).
        if self._gen < self._n:
            room = (self._gen - self._k < self.cfg.addr_depth) | self._fire
            self._agu_push = bool(room.all())

        if self._fire.any():
            self._cls = "busy"
        elif self._req.any():
            self._cls = "stall_xbar"
        else:
            self._cls = self._sleep_class()

    def _sleep_class(self) -> str:
        """Class of a cycle without requests: FIFO-blocked or idle."""
        return "stall_fifo" if bool((self._k < self._gen).any()) else "idle"

    def commit(self, cycle: int) -> None:
        was_busy = self.busy
        self._k += self._fire
        self._issued += self._fire
        if self._agu_push:
            self._gen += 1
        if was_busy and not self.busy:
            self.done_cycle = cycle + 1
            if self._trace is not None and self._trace.task:
                self._trace.emit(Done(cycle + 1, self.name))
        if self._cls is not None:
            self.cycles.add(self._cls, cycle, cycle + 1)
        self._clear_wires()

    def next_wake(self, cycle: int) -> int | None:
        """See "Waking" in the module doc."""
        n, write = self.cfg.n_ports, self.cfg.write
        credit_blocked = False
        for j in range(n):
            if not self._has_addr(j):
                continue
            if write:
                if self.fifo.count(j) > 0:
                    return cycle + 1
            elif self._credit(j):
                return cycle + 1
            else:
                credit_blocked = True
        if self._gen < self._n and bool((self._gen - self._k < self.cfg.addr_depth).all()):
            return cycle + 1

        cands = []
        if credit_blocked:
            if self.fifo.popper is None:
                raise SimulationError(f"{self.name}: FIFO full and nothing attached to pop it")
            w = self.fifo.popper.next_wake(cycle)
            if w is not None:
                cands.append(w)
        for p in self.ports:
            r = self.xbar.next_rdata(cycle, p)
            if r is not None:
                cands.append(r)
        return min(cands) if cands else None

    def on_gap(self, start: int, stop: int) -> None:
        """Skipped cycles: the streamer slept in a FIFO stall or idle."""
        self.cycles.add(self._sleep_class(), start, stop)
