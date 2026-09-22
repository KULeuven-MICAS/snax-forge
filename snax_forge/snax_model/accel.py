"""Accelerator interface and generic stubs for SNAX-MODEL (MOD5, D5, D25, D29, D32).

What this models
----------------
An accelerator sits between FIFOs: it pops beats from the FIFOs of its
reader streamers, runs a Python function, holds the results in an L-stage
pipeline, and pushes beats into the FIFOs of its writer streamers. It is the
mirror image of the streamer, which sits between the xbar and a FIFO.

It is a model only: no CSRs (MOD7), no BRM structure (M3), no RTL binding.

Interface (``AccelConfig``)
---------------------------
Per port: name, direction, lanes per beat (must equal the attached FIFO's
lanes, i.e. the streamer's ``n_ports``) and an element rate (D25). Plus
latency ``L``, initiation interval ``II`` (both user-supplied, D5) and the
Python function.

A *firing* is one step of the datapath. The rate says how often a port
moves a beat: once every ``rate`` firings. An input port with rate r is
popped at firings 0, r, 2r, ...; an output port with rate r is pushed at
firings r-1, 2r-1, ... A rate is an int or the name of a start parameter
(e.g. ``"T"``), so the trip count of a reduction can be set per task.

* elementwise stub: every port has rate 1 (N inputs -> 1 output per firing);
* reduce stub: input rate 1, output rate ``"T"`` (T beats -> 1 beat).

The function is called once per firing:

    fn(k, ins, state, params) -> outs

``k`` is the firing index in the task, ``ins`` maps each input port due at
k to its beat (np.ndarray, first axis = lanes), ``state`` is a dict that
lives for one task (the reduce's partial sum), ``params`` the start
parameters. ``outs`` must hold exactly the output ports due at k.

Which RTL it copies
-------------------
snax_cluster (streamer + snax_alu) and snax-forge hw/chisel, only where it
changes cycle counts:

* With no data path extensions, the reader FIFO output is wired straight to
  the accelerator input and the accelerator output straight to the writer
  FIFO input (DataPathExtensionHost: ``io.data.out <> io.data.in``). No
  register in between, so the accelerator pops in the same cycle as it
  sees a committed beat, and its push is visible to the writer one cycle
  later (writer Queue, flow = false, pipe = false).
* Join on the inputs: snax_alu_pe drives ``ready = busy && c_ready &&
  a_valid && b_valid``. A beat is only taken when all inputs are valid
  together, and only when the output can take the result. Copied. (The
  snax-forge ElementwiseLoop does not gate each ready on the other's valid
  and would drop a beat if the operands arrived out of step; with the join
  the model never does that.)
* Latency: snax_alu and the snax-forge elementwise modules are
  combinational (L = 0, II = 1): result in the firing cycle. The
  snax-forge Accumulator registers its sum, so the result is valid the
  cycle after the last input (L = 1).
* Start and done: snax_alu sets busy the cycle after the CSR write and
  clears it the cycle after the last output handshake. Same here:
  ``start`` in cycle s gives busy from s+1; ``done_cycle`` is the cycle
  after the last push.

Not copied (flagged for ANC2 / BRM4):

* Per-stage ready. Here the pipeline has a global stall (see below). Many
  SNAX accelerators instead give each stage its own ready, so a stage can
  move into an empty stage ahead during a stall and bubbles get squeezed
  out. That changes stall lengths, not data. The rule lives in
  ``_frozen`` and ``_advance`` only.
* The Accumulator's drain cycle: its ``in.ready`` is low while the result
  waits, so back-to-back reductions take T+1 cycles each in RTL but T here.
  A single reduction (``dot``) is not affected.

Pipeline and global stall
-------------------------
The pipeline is L slots, each empty (None) or holding the outputs of one
firing. Firings with no output due (a reduce's first T-1) leave no entry.
A firing in cycle t enters slot 0 at the end of t and moves one slot per
cycle, so it is pushed in cycle t + L if nothing stalls, and visible to the
writer from t + L + 1. L = 0 means no slots: the result is pushed in the
firing cycle.

Output room is checked at push time. If the last slot's outputs cannot all
be pushed because an output FIFO is full, the whole pipeline freezes for
that cycle: no slot advances, nothing fires, no input is popped. The
reader streamers then run out of credit (their FIFO fills), which is how
back-pressure reaches the xbar. With L = 0, "frozen" means a firing is
ready but its due output FIFO is full.

During a freeze:

* II counts wall-clock cycles: a firing is allowed at t if
  t >= last firing + II. A freeze does not restart that count, so after a
  freeze longer than II the next firing can go at once.
* the function's ``state`` (a reduce's partial sum) changes only when a
  firing happens, so a freeze leaves it untouched.

Who calls what, per cycle (COMPUTE)
-----------------------------------
In one tick, in this order:

1. freeze decision from committed state (``_frozen``);
2. push: if the last slot is full and not frozen, ``fifo.push`` on every
   output port it holds;
3. fire: if not frozen, a task is running, II allows, and every due input
   FIFO can give a beat (committed): ``fifo.pop`` on the due inputs, call
   the function, and for L = 0 push at once;
4. commit: the pipeline advances one slot (unless frozen) and the new
   result enters slot 0.

Why this order matches the FIFO rules:

* Writer FIFOs have pipe = false and the writer pops in RESPONSE, after
  COMPUTE. So the room seen in step 2 is exactly the committed room; a pop
  by the writer in this cycle does not help (as in RTL).
* Reader FIFOs have pipe = true and the reader pushes in RESPONSE. The pop
  in step 3 comes first, so the reader may push into a lane that was full
  and was popped in this cycle, and its credit counter sees the pop in the
  same cycle (``Fifo.popped_now``).
* flow = false: a beat pushed by the reader in cycle t is only visible in
  t+1, so the accelerator never pops a beat in its push cycle.
* The push in step 2 must be decided before the fire in step 3, because
  under a global stall a new result may only enter if the head leaves.

Waking (D29)
------------
``next_wake`` reads only committed state: its own pipeline and counters and
the committed contents of its FIFOs. It never calls a streamer's
next_wake (a reader blocked on credit calls this one, so that would
recurse). After cycle t:

* last slot full: t+1 if every output it needs has room, else None
  (frozen; nothing can change until a writer pops, and that pop becomes
  visible only after the FIFO commits, when next_wake is asked again);
* results in flight in other slots: t+1 (the pipeline moves every cycle);
* task running and II allows a firing only from a cycle > t: that cycle
  (whether or not the inputs are there, because the class changes then);
* task running, II allows, every due input has a beat: t+1; for L = 0
  None if that firing's output FIFO is full;
* task running, II allows, a due input is empty: None. A reader's push
  becomes visible only after the FIFO commits, so next_wake then answers
  t+1;
* no task, or done: None.

Two extra wakes keep the cycle classes exact (see "Cycle classes"): the
first busy cycle after a start, and any cycle whose class differs from
the class recorded for the current gap. Both depend only on committed
state and are answered t+1, so they do not break the rule above.

So no same-cycle wake from a streamer is needed: every event that unblocks
the accelerator (a reader push, a writer pop) is visible only in the next
cycle, and next_wake is asked after every committed cycle. While the
accelerator sleeps, its inputs only grow and its output room only grows;
any change that matters shows up in the next next_wake, and an II wake is
a fixed cycle. So asking again before the returned cycle, with nothing
changed, gives the same answer.

The other direction works too: a reader blocked on credit wakes exactly
when this accelerator is ticked, and the accelerator only pops in cycles
in which it is ticked.

Cycle classes (for MOD8)
------------------------
Each cycle counts as exactly one of, in this order:

* ``busy``: a firing happened;
* ``stall_out``: the pipeline was frozen on a full output FIFO;
* ``idle`` (II gap): a task is running but II does not allow a firing,
  whether or not the inputs are there;
* ``stall_in``: a task is running, II allows, not frozen, and some due
  input FIFO is empty: it could fire now but for a missing beat;
* ``idle``: anything else: no task, done, or results only in flight after
  the last firing (the drain after the last firing is L cycles of
  ``idle``).

So ``busy`` / total is the firing rate. In a tick, the class is computed
from committed state at the start of the tick. Skipped cycles get the
class of the first cycle of the gap, which ``next_wake`` records when it is
first asked after a tick (``_gap_cls``). It cannot be computed in
``on_gap``: that runs when the gap has already ended, i.e. after the push
or pop that woke the accelerator has committed. To keep one class per
gap, ``next_wake`` answers t+1 whenever the class of t+1 differs from the
recorded one, so a class change always ends a gap with a tick. (Example:
L = 0, output full, an input arrives: the class goes from ``stall_in`` to
``stall_out``, but nothing can fire until the writer pops.) The one change
to the accelerator's own state outside a tick is a start
(``_StartReg``), so the accelerator is also always ticked in the first
busy cycle after a start (``_kick``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import reduce as _fold
from typing import Any

import numpy as np

from .sched import ClassLog, Component, Phase, SimulationError
from .streamer import Fifo
from .trace import Done, Fire, Start

CYCLE_CLASSES = ("busy", "stall_out", "stall_in", "idle")

# fn(k, ins, state, params) -> outs, see the module doc.
AccelFn = Callable[
    [int, Mapping[str, np.ndarray], dict[str, Any], Mapping[str, int]],
    Mapping[str, Any],
]


# =============================================================================
# Interface
# =============================================================================


@dataclass(frozen=True)
class AccelPort:
    """One data port of the accelerator.

    ``rate``: the port moves one beat every ``rate`` firings (D25); an int
    or the name of a start parameter.
    """

    name: str
    direction: str  # "in" or "out"
    lanes: int = 1  # elements per beat = the streamer's n_ports
    rate: int | str = 1

    def __post_init__(self) -> None:
        if self.direction not in ("in", "out"):
            raise ValueError(f"port {self.name}: direction must be 'in' or 'out'")
        if self.lanes < 1:
            raise ValueError(f"port {self.name}: lanes must be >= 1")
        if isinstance(self.rate, int) and self.rate < 1:
            raise ValueError(f"port {self.name}: rate must be >= 1")


@dataclass(frozen=True)
class AccelConfig:
    """Accelerator interface: ports, latency L, initiation interval II, function."""

    ports: tuple[AccelPort, ...]
    fn: AccelFn
    latency: int = 0  # L: pipeline stages between firing and push
    ii: int = 1  # II: minimum cycles between firings
    kind: str = "custom"  # for printing only

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", tuple(self.ports))
        names = [p.name for p in self.ports]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate port names: {names}")
        if not self.inputs or not self.outputs:
            raise ValueError("need at least one input and one output port")
        if self.latency < 0:
            raise ValueError("latency must be >= 0")
        if self.ii < 1:
            raise ValueError("ii must be >= 1")

    @property
    def inputs(self) -> tuple[AccelPort, ...]:
        return tuple(p for p in self.ports if p.direction == "in")

    @property
    def outputs(self) -> tuple[AccelPort, ...]:
        return tuple(p for p in self.ports if p.direction == "out")

    def port(self, name: str) -> AccelPort:
        for p in self.ports:
            if p.name == name:
                return p
        raise KeyError(f"no port {name!r}")


# -----------------------------------------------------------------------------
# Stubs: configs, not subclasses
# -----------------------------------------------------------------------------


def elementwise_stub(
    lanes: int = 1,
    n_inputs: int = 2,
    op: Callable[[Any, Any], Any] = np.add,
    latency: int = 0,
    ii: int = 1,
    inputs: Sequence[str] | None = None,
    output: str = "out",
) -> AccelConfig:
    """N inputs -> 1 output, one beat each per firing, lane by lane.

    Defaults follow snax_alu and the snax-forge elementwise modules
    (combinational: L = 0, II = 1). ``op`` folds the inputs left to right,
    e.g. ``a + b`` for vecadd.
    """
    names = list(inputs) if inputs is not None else [chr(ord("a") + i) for i in range(n_inputs)]
    if len(names) != n_inputs:
        raise ValueError(f"{len(names)} input names for {n_inputs} inputs")

    def fn(k, ins, state, params):
        return {output: _fold(op, (ins[n] for n in names))}

    ports = [AccelPort(n, "in", lanes) for n in names] + [AccelPort(output, "out", lanes)]
    return AccelConfig(tuple(ports), fn, latency, ii, kind="elementwise")


def reduce_stub(
    lanes: int = 1,
    lanes_out: int | None = None,
    op: Callable[[Any, Any], Any] = np.add,
    latency: int = 1,
    ii: int = 1,
    input: str = "in",
    output: str = "out",
) -> AccelConfig:
    """T input beats -> 1 output beat; T is the start parameter ``"T"``.

    ``lanes_out = lanes`` (default): each lane is reduced over T beats (the
    snax-forge Accumulator is the case lanes = 1). ``lanes_out = 1``: the
    lanes are reduced too. Default L = 1, II = 1 as in the Accumulator.
    """
    lanes_out = lanes if lanes_out is None else lanes_out
    if lanes_out not in (lanes, 1):
        raise ValueError("lanes_out must equal lanes or be 1")

    def fn(k, ins, state, params):
        x = ins[input]
        if lanes_out == 1:
            x = _fold(op, (x[j : j + 1] for j in range(x.shape[0])))
        # Partial result lives in the task state; it only changes on a firing.
        state["acc"] = x.copy() if "acc" not in state else op(state["acc"], x)
        if (k + 1) % params["T"] == 0:
            return {output: state.pop("acc")}
        return {}

    ports = (AccelPort(input, "in", lanes, 1), AccelPort(output, "out", lanes_out, "T"))
    return AccelConfig(ports, fn, latency, ii, kind="reduce")


# =============================================================================
# Accelerator
# =============================================================================


class _StartReg:
    """Holds a start written in a cycle until the end of that cycle (as in streamer.py)."""

    def __init__(self, owner: Accelerator) -> None:
        self.owner = owner
        self.pending: tuple[dict[str, int], int] | None = None

    def commit(self) -> None:
        if self.pending is not None:
            params, cycle = self.pending
            self.pending = None
            self.owner._load(params, cycle)


@dataclass
class _Wires:
    """This cycle's wires, cleared in commit."""

    now: int | None = None
    frozen: bool = False
    fired: bool = False
    pushed: bool = False
    new: dict[str, Any] | None = None  # enters slot 0 at commit
    cls: str | None = None
    ports: list[str] = field(default_factory=list)  # ports that moved a beat


class Accelerator(Component):
    """Pipeline between input and output FIFOs, see the module doc.

    State:

    * committed: ``params``, ``_n`` (firings in the task), ``_rate`` (per
      port, resolved at start), ``_k`` (firings done), ``_last_fire``,
      ``_slots`` (pipeline, slot L-1 is next to push), ``state`` (the
      function's task state), ``done_cycle``;
    * wires: ``_w``;
    * statistics: ``cycles`` per class (a ``ClassLog``, MOD8), ``beats`` per
      port. Trace events: ``start``, ``done`` (task), ``fire`` (beat), all
      emitted when the state they describe commits.
    """

    phases = (Phase.COMPUTE,)

    def __init__(self, name: str, cluster: Any, cfg: AccelConfig) -> None:
        super().__init__(name)
        self.cfg = cfg
        self._cluster = cluster  # anything with touch(elem)
        self.fifos: dict[str, Fifo] = {}
        self._start = _StartReg(self)
        self.params: dict[str, int] = {}
        self._n = 0
        self._rate: dict[str, int] = {}
        self._k = 0
        self._last_fire: int | None = None
        self._slots: list[dict[str, Any] | None] = [None] * cfg.latency
        self.state: dict[str, Any] = {}
        self.done_cycle: int | None = None
        self._kick: int | None = None  # first busy cycle after a start: always ticked
        self._gap_cls: str | None = None  # class of skipped cycles, see module doc
        self.cycles = ClassLog(CYCLE_CLASSES)
        self.beats = dict.fromkeys((p.name for p in cfg.ports), 0)
        self._w = _Wires()

    # -------------------------------------------------------------------------
    # Wiring
    # -------------------------------------------------------------------------

    def attach(self, port: str, fifo: Fifo) -> None:
        """Connect ``port`` to ``fifo``: popper of an input FIFO, pusher of an output FIFO.

        Use a reader streamer's FIFO for an input and a writer's for an
        output. The other side of a reader FIFO is already taken by the
        reader (pusher), so attaching it as an output fails, and vice versa.
        """
        p = self.cfg.port(port)
        if port in self.fifos:
            raise ValueError(f"{self.name}: port {port} already attached")
        if fifo.lanes != p.lanes:
            raise ValueError(
                f"{self.name}: port {port} has {p.lanes} lanes, {fifo.name} has {fifo.lanes}"
            )
        side = "popper" if p.direction == "in" else "pusher"
        other = getattr(fifo, side)
        if other is not None and other is not self:
            raise ValueError(f"{self.name}: {fifo.name}.{side} is already {other.name}")
        setattr(fifo, side, self)
        self.fifos[port] = fifo

    # -------------------------------------------------------------------------
    # Control side: start and done
    # -------------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        """Committed: firings left, or results still in the pipeline."""
        return self._n > 0 and (self._k < self._n or any(s is not None for s in self._slots))

    def start(self, params: Mapping[str, int], cycle: int | None = None) -> None:
        """Start a task of ``params["n"]`` firings. ``busy`` is True from ``cycle + 1``.

        Other entries name port rates (e.g. ``T``). During a run, call it in
        Phase.CONTROL with the current cycle; before a run, without
        ``cycle`` (as if started in cycle -1). The CSR mapping is MOD7.
        """
        params = {k: int(v) for k, v in params.items()}
        self._check(params)
        if self.busy or self._start.pending is not None:
            raise SimulationError(f"{self.name}: start while busy")
        if cycle is None:
            self._load(params, -1)
        else:
            self._start.pending = (params, cycle)
            self._cluster.touch(self._start)

    def _resolve(self, params: Mapping[str, int]) -> dict[str, int]:
        rates = {}
        for p in self.cfg.ports:
            r = p.rate if isinstance(p.rate, int) else params.get(p.rate)
            if r is None:
                raise ValueError(f"{self.name}: port {p.name} needs parameter {p.rate!r}")
            if r < 1:
                raise ValueError(f"{self.name}: port {p.name} rate {r} must be >= 1")
            rates[p.name] = r
        return rates

    def _check(self, params: Mapping[str, int]) -> None:
        missing = [p.name for p in self.cfg.ports if p.name not in self.fifos]
        if missing:
            raise ValueError(f"{self.name}: ports not attached: {missing}")
        n = params.get("n")
        if n is None or n < 0:
            raise ValueError(f"{self.name}: need params['n'] >= 0 (firings)")
        for name, r in self._resolve(params).items():
            if n % r:
                raise ValueError(f"{self.name}: n = {n} is not a multiple of {name}'s rate {r}")

    def _load(self, params: dict[str, int], cycle: int) -> None:
        """Committed effect of a start in ``cycle``."""
        self.params = params
        self._rate = self._resolve(params)
        self._n = params["n"]
        self._k = 0
        self.state = {}
        self._kick = cycle + 1
        # A zero-firing task never becomes busy.
        self.done_cycle = None if self._n else cycle + 1
        tr = self._trace
        if tr is not None and tr.task:
            tr.emit(Start(cycle, self.name))
            if not self._n:
                tr.emit(Done(cycle + 1, self.name))

    # -------------------------------------------------------------------------
    # Conditions (committed state)
    # -------------------------------------------------------------------------

    def _due_in(self, k: int) -> list[str]:
        return [p.name for p in self.cfg.inputs if k % self._rate[p.name] == 0]

    def _due_out(self, k: int) -> list[str]:
        return [p.name for p in self.cfg.outputs if (k + 1) % self._rate[p.name] == 0]

    def _inputs_ready(self) -> bool:
        """Every input due at the next firing has a committed beat."""
        return all(self.fifos[n].can_pop() for n in self._due_in(self._k))

    def _room(self, ports: Sequence[str]) -> bool:
        return all(self.fifos[n].can_push() for n in ports)

    def _ii_ok(self, cycle: int) -> bool:
        return self._last_fire is None or cycle >= self._last_fire + self.cfg.ii

    def _can_fire(self, cycle: int) -> bool:
        """Firing conditions other than the freeze."""
        return self._k < self._n and self._ii_ok(cycle) and self._inputs_ready()

    # --- Advance rule: global stall. Per-stage ready would change only these two.

    def _frozen(self, cycle: int) -> bool:
        """Global stall in ``cycle``: the next result to leave cannot be pushed."""
        if self.cfg.latency:
            head = self._slots[-1]
            return head is not None and not self._room(list(head))
        # L = 0: the result leaves in its firing cycle.
        return self._can_fire(cycle) and not self._room(self._due_out(self._k))

    def _advance(self, new: dict[str, Any] | None) -> None:
        """Move every slot one stage on (the head has been pushed) and enter ``new``."""
        if self.cfg.latency:
            self._slots = [new] + self._slots[:-1]

    def _sleep_class(self, cycle: int) -> str:
        """Class of a cycle without a firing, from committed state."""
        if self._frozen(cycle):
            return "stall_out"
        if self._k < self._n and self._ii_ok(cycle) and not self._inputs_ready():
            return "stall_in"
        return "idle"  # includes II gaps

    # -------------------------------------------------------------------------
    # Component interface
    # -------------------------------------------------------------------------

    def tick(self, cycle: int, phase: Phase) -> None:
        w = self._w
        w.now = cycle
        w.cls = self._sleep_class(cycle)  # before acting: committed state only
        w.frozen = self._frozen(cycle)
        if w.frozen:
            return  # nothing moves, nothing is popped

        # 1. push the head (L >= 1). Not frozen, so every output it needs has room.
        if self.cfg.latency and self._slots[-1] is not None:
            self._push(cycle, self._slots[-1])

        # 2. fire
        if not self._can_fire(cycle):
            return
        k = self._k
        ins = {n: np.asarray(self.fifos[n].pop(cycle)) for n in self._due_in(k)}
        w.ports += list(ins)
        outs = dict(self.cfg.fn(k, ins, self.state, self.params))
        due = self._due_out(k)
        if set(outs) != set(due):
            raise SimulationError(
                f"{self.name}: firing {k} returned {sorted(outs)}, expected {sorted(due)}"
            )
        for n, v in outs.items():
            v = np.asarray(v)
            if v.ndim == 0 or v.shape[0] != self.cfg.port(n).lanes:
                raise SimulationError(
                    f"{self.name}: port {n} needs {self.cfg.port(n).lanes} lanes, got {v.shape}"
                )
            outs[n] = v
        w.fired = True
        w.cls = "busy"
        if not self.cfg.latency:
            if outs:
                self._push(cycle, outs)  # room checked by _frozen
        else:
            w.new = outs or None

    def _push(self, cycle: int, outs: Mapping[str, np.ndarray]) -> None:
        for n, v in outs.items():
            self.fifos[n].push(cycle, list(v))
        self._w.pushed = True
        self._w.ports += list(outs)

    def commit(self, cycle: int) -> None:
        w = self._w
        tr = self._trace
        was_busy = self.busy
        if w.fired:
            if tr is not None and tr.beat:
                tr.emit(Fire(cycle, self.name, n=self._k))  # index before the increment
            self._k += 1
            self._last_fire = cycle
        if not w.frozen:
            self._advance(w.new)
        if was_busy and not self.busy:
            self.done_cycle = cycle + 1
            if tr is not None and tr.task:
                tr.emit(Done(cycle + 1, self.name))
        for n in w.ports:
            self.beats[n] += 1
        if w.cls is not None:
            self.cycles.add(w.cls, cycle, cycle + 1)
        if self._kick is not None and self._kick <= cycle:
            self._kick = None
        self._gap_cls = None  # recorded again by the next next_wake
        self._w = _Wires()

    def next_wake(self, cycle: int) -> int | None:
        """See "Waking" in the module doc. Reads committed state only."""
        cls = self._sleep_class(cycle + 1)
        if self._gap_cls is None:
            self._gap_cls = cls
        elif cls != self._gap_cls:
            return cycle + 1  # the class changes: end the gap with a tick
        w = self._wake(cycle)
        if self._kick is not None and self._kick > cycle:
            w = self._kick if w is None else min(w, self._kick)
        return w

    def _wake(self, cycle: int) -> int | None:
        if self.cfg.latency:
            head = self._slots[-1]
            if head is not None:
                return cycle + 1 if self._room(list(head)) else None
            if any(s is not None for s in self._slots):
                return cycle + 1
        if self._k >= self._n:
            return None
        if self._last_fire is not None and cycle + 1 <= self._last_fire + self.cfg.ii:
            # II gap, or its last cycle is next: wake exactly when II allows,
            # since the class changes there. Same answer at every cycle before.
            return self._last_fire + self.cfg.ii
        if not self._inputs_ready():
            return None
        if not self.cfg.latency and not self._room(self._due_out(self._k)):
            return None  # L = 0 frozen
        return cycle + 1

    def on_gap(self, start: int, stop: int) -> None:
        """Skipped cycles: the class recorded when the gap began (module doc)."""
        cls = self._gap_cls if self._gap_cls is not None else self._sleep_class(start)
        self.cycles.add(cls, start, stop)
