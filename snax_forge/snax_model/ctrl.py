"""Uniform register interface and controller for SNAX-MODEL (MOD7, D6, D11, D26, D29, D36, D37).

What this models
----------------
There is no CPU (D6). A controller executes a control program of three
commands, ``csr_write``, ``csr_read`` and ``wait``, against a register
interface that is the same for every block: streamers, accelerators, the
DMA and any later block kind. It does not copy the SNAX interfaces
(ReqRspManager CSRs, iDMA instructions); mapping onto them is the job of
SNAX-LOWER's C backend (D18, GEN2), by register *name*.

Register blocks
---------------
Each block owns an aligned window of ``window`` registers (default 32), one
word each. Addresses are register indices, not bytes. Bases are
``i * window`` in list order unless given. Inside a window:

    offset 0   start        write-only: writing 1 launches the block
    offset 1   busy         read-only: the component's committed ``busy``
    offset 2   busy_cycles  read-only: busy cycles of the current or last task
    offset 3.. configuration registers, read/write, listed by the adapter

The status registers sit at the same offset in every block kind; only the
configuration offsets depend on the block's own design-time config.
``RegisterMap.to_dict`` lists every register by name, so nobody has to count.

Configuration registers are buffered: they are shadow values held by the
controller, and a start copies them into the component's start argument.
The next task can be programmed while the block runs. A start while the
block is busy is an error: the program must wait first. Registers reset to 0.

``busy_cycles`` read in cycle r after a start that landed in cycle s is
``min(r, D) - s - 1``, with D the block's ``done_cycle`` (or r while it
still runs): busy is one continuous span from s+1 to D-1. A zero-work task
reads 0. It is computed from s and ``done_cycle``, so the block needs no
extra ticks.

Adapters
--------
Each block kind has an adapter, found by component class
(``register_adapter``). It lists the configuration registers from the
component's config, ``decode``s their values into the component's start
argument, and ``encode``s a start argument back into values (for the
``start_writes`` helper, tests and later SNAX-LOWER). Component classes do
not change for this. First adapters:

* streamer: ``base``, ``tbound[0..D-1]``, ``tstride[0..D-1]``,
  ``sstride[0..S-1]`` with D = ``temporal_dims``. Spatial bounds are
  design-time and given to the adapter (default ``(n_ports,)``), so S is
  their length -> ``StreamerRegs``.
* accel: ``n``, then every string port rate in port order without
  duplicates (``T`` for the reduce stub) -> params dict.
* dma: ``direction`` (0 = l2_to_l1, 1 = l1_to_l2), ``src_base``,
  ``src_bound[..]``, ``src_stride[..]``, ``dst_base``, ``dst_bound[..]``,
  ``dst_stride[..]`` for ``DmaConfig.dims`` loops each -> ``DmaDescriptor``.

``encode`` pads unused loops with bound 1 and stride 0, which the
components treat the same as fewer loops.

Timing (D37)
------------
The program starts in cycle 0 and runs one command at a time. A command
that begins in cycle t with cost c covers cycles [t, t + c - 1]; its effect
happens in its last cycle, in Phase.CONTROL, and the next command begins in
t + c. Costs are per command kind, and write and read costs may be set per
block kind (``ControllerConfig``). All are declared defaults (D51).

* csr_write of a configuration register: the shadow value changes.
* csr_write of 1 to ``start``, landing in w: the controller checks the
  committed ``busy``, decodes the shadow values and calls
  ``comp.start(arg, w)``; the block is busy from w + 1. A read after it
  therefore always sees ``busy = 1`` for a task with work.
* csr_read: samples committed state (D29) in its last cycle; logged as
  (cycle, addr, value) in ``reads``.
* wait, poll: poll i samples ``busy`` in cycle t + i*P + c_r - 1 (P =
  ``poll_interval``, c_r = the block's read cost) and ends on the first
  sample that reads 0, i.e. in a cycle >= D. Samples are counted in
  ``polls``, not logged in ``reads``.
* wait, signal: covers [t, max(t, D) + S - 1], S = ``signal_latency``.

Hence, with i* = max(0, ceil((D - t - c_r + 1) / P)):

    W_poll = i* * P + c_r        W_signal = max(t, D) - t + S

A wait may only follow a start of its block in program order (checked when
the controller is built), so ``done_cycle`` always belongs to the task the
wait is for.

Waking (D29)
------------
The controller is ticked only in the last cycle of each command. After
committed cycle x, ``next_wake`` is:

* write or read: its last cycle t + c - 1;
* poll: the next sample cycle;
* signal, ``done_cycle`` still None: None. The controller needs nothing in
  the same cycle as the block (unlike a reader blocked on credit, which
  must see its consumer's pop combinationally), ``done_cycle`` is committed
  state, and the scheduler asks every component again after every
  simulated cycle. So it does not call the block's ``next_wake``;
* signal, ``done_cycle`` known: max(t, D) + S - 1, which is > x since D <=
  x + 1 when it becomes known;
* program done: None.

Every answer depends only on the controller's own state and the block's
committed ``done_cycle``, so asking again before the returned cycle gives
the same answer. If the run ends with a signal wait pending, the block
never finished: ``finished`` is False.

Cycle classes (for MOD8)
------------------------
Each cycle is exactly one of ``command`` (a write or read runs: control
overhead), ``wait`` (poll or signal) or ``idle`` (program done). A skipped
gap always lies inside one command or in the idle tail, because the
controller ticks at the end of every command, so ``on_gap`` credits it to
that class. The controller also records ``spans`` (pc, first, last) per
command and ``waits`` (pc, block, t, D, last) per wait, for tests and MOD8.

Errors
------
When the controller is built (ValueError): unknown address, write to
``busy`` or ``busy_cycles``, read of ``start``, a start value other than 1,
wait on an unknown block or with an unknown mode, a wait before any start of
its block, ``poll_interval`` smaller than the block's read cost.
During the run (SimulationError, naming block, register and cycle): start
while busy, and shadow values the adapter or the component rejects.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .accel import Accelerator
from .config import Config
from .dma import DIRECTIONS, Dma, DmaDescriptor, DmaPattern
from .sched import ClassLog, Component, Phase, SimulationError
from .streamer import Streamer, StreamerRegs
from .trace import Cmd, Poll

CYCLE_CLASSES = ("command", "wait", "idle")
STATUS = ("start", "busy", "busy_cycles")  # offsets 0, 1, 2 of every block
READ_ONLY = ("busy", "busy_cycles")
WAIT_MODES = ("poll", "signal")


# =============================================================================
# Commands (D11, D26: dataclasses with plain dicts)
# =============================================================================


@dataclass(frozen=True)
class CsrWrite:
    """Write ``value`` to register ``addr``."""

    addr: int
    value: int

    def to_dict(self) -> dict[str, Any]:
        return {"op": "csr_write", "addr": self.addr, "value": self.value}


@dataclass(frozen=True)
class CsrRead:
    """Read register ``addr``; the value goes to ``Controller.reads``."""

    addr: int

    def to_dict(self) -> dict[str, Any]:
        return {"op": "csr_read", "addr": self.addr}


@dataclass(frozen=True)
class Wait:
    """Wait until ``block`` is done, by polling ``busy`` or by its completion signal."""

    block: str
    mode: str = "signal"

    def __post_init__(self) -> None:
        if self.mode not in WAIT_MODES:
            raise ValueError(f"wait mode must be one of {WAIT_MODES}, got {self.mode!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"op": "wait", "block": self.block, "mode": self.mode}


Command = CsrWrite | CsrRead | Wait


def command_from_dict(d: Mapping[str, Any]) -> Command:
    """Inverse of ``to_dict`` for any command."""
    op = d.get("op")
    if op == "csr_write":
        return CsrWrite(int(d["addr"]), int(d["value"]))
    if op == "csr_read":
        return CsrRead(int(d["addr"]))
    if op == "wait":
        return Wait(str(d["block"]), str(d.get("mode", "signal")))
    raise ValueError(f"unknown command op {op!r}")


def program_to_dicts(program: Iterable[Command]) -> list[dict[str, Any]]:
    return [c.to_dict() for c in program]


def program_from_dicts(ds: Iterable[Mapping[str, Any]]) -> list[Command]:
    return [command_from_dict(d) for d in ds]


# =============================================================================
# Adapters: registers <-> start argument, per block kind
# =============================================================================


def _loops(prefix: str, n: int) -> list[str]:
    return [f"{prefix}[{i}]" for i in range(n)]


def _pad(values: Sequence[int], n: int, fill: int, what: str) -> list[int]:
    """Pad loop values to the hardware loop count; more loops than that is an error."""
    if len(values) > n:
        raise ValueError(f"{what}: {len(values)} loops, registers hold {n}")
    return list(values) + [fill] * (n - len(values))


class Adapter:
    """Registers of one block and the conversion to and from its start argument.

    ``kind`` names the block kind (the key of per-kind costs). ``registers``
    lists the configuration registers in offset order. ``decode`` gets their
    values by name and returns the component's start argument; it raises
    ValueError for values that give no valid argument. ``encode`` is its
    inverse up to padding.
    """

    kind: str = "block"

    def __init__(self, comp: Component) -> None:
        self.comp = comp

    @property
    def registers(self) -> list[str]:
        raise NotImplementedError

    def decode(self, v: Mapping[str, int]) -> Any:
        raise NotImplementedError

    def encode(self, arg: Any) -> dict[str, int]:
        raise NotImplementedError


class StreamerAdapter(Adapter):
    """base, temporal bounds and strides, spatial strides -> StreamerRegs."""

    kind = "streamer"

    def __init__(self, comp: Streamer, spatial_bounds: Sequence[int] | None = None) -> None:
        super().__init__(comp)
        sb = tuple(spatial_bounds) if spatial_bounds is not None else (comp.cfg.n_ports,)
        n = 1
        for b in sb:
            n *= b
        if not sb or n != comp.cfg.n_ports:
            raise ValueError(
                f"{comp.name}: spatial bounds {sb} give {n} lanes, streamer has {comp.cfg.n_ports}"
            )
        self.spatial_bounds = sb
        self.d = comp.cfg.temporal_dims

    @property
    def registers(self) -> list[str]:
        s = len(self.spatial_bounds)
        return ["base", *_loops("tbound", self.d), *_loops("tstride", self.d),
                *_loops("sstride", s)]  # fmt: skip

    def decode(self, v: Mapping[str, int]) -> StreamerRegs:
        d, s = self.d, len(self.spatial_bounds)
        return StreamerRegs(
            v["base"],
            tuple(v[f"tbound[{i}]"] for i in range(d)),
            tuple(v[f"tstride[{i}]"] for i in range(d)),
            self.spatial_bounds,
            tuple(v[f"sstride[{i}]"] for i in range(s)),
        )

    def encode(self, arg: StreamerRegs) -> dict[str, int]:
        name = self.comp.name
        if arg.spatial_bounds != self.spatial_bounds:
            raise ValueError(
                f"{name}: spatial bounds {arg.spatial_bounds} are design-time {self.spatial_bounds}"
            )
        tb = _pad(arg.temporal_bounds, self.d, 1, name)
        ts = _pad(arg.temporal_strides, self.d, 0, name)
        out = {"base": arg.base}
        out |= {f"tbound[{i}]": b for i, b in enumerate(tb)}
        out |= {f"tstride[{i}]": x for i, x in enumerate(ts)}
        out |= {f"sstride[{i}]": x for i, x in enumerate(arg.spatial_strides)}
        return out


class AccelAdapter(Adapter):
    """n, then every named port rate -> params dict."""

    kind = "accel"

    def __init__(self, comp: Accelerator) -> None:
        super().__init__(comp)
        names = ["n"]
        for p in comp.cfg.ports:
            if isinstance(p.rate, str) and p.rate not in names:
                names.append(p.rate)
        self._names = names

    @property
    def registers(self) -> list[str]:
        return list(self._names)

    def decode(self, v: Mapping[str, int]) -> dict[str, int]:
        return {k: v[k] for k in self._names}

    def encode(self, arg: Mapping[str, int]) -> dict[str, int]:
        if set(arg) != set(self._names):
            raise ValueError(f"{self.comp.name}: parameters {sorted(arg)}, registers {self._names}")
        return {k: int(arg[k]) for k in self._names}


class DmaAdapter(Adapter):
    """direction, source loops, destination loops -> DmaDescriptor."""

    kind = "dma"

    def __init__(self, comp: Dma) -> None:
        super().__init__(comp)
        self.d = comp.cfg.dims

    @property
    def registers(self) -> list[str]:
        out = ["direction"]
        for side in ("src", "dst"):
            out += [f"{side}_base", *_loops(f"{side}_bound", self.d),
                    *_loops(f"{side}_stride", self.d)]  # fmt: skip
        return out

    def decode(self, v: Mapping[str, int]) -> DmaDescriptor:
        k = v["direction"]
        if not 0 <= k < len(DIRECTIONS):
            raise ValueError(f"direction {k} is not one of 0..{len(DIRECTIONS) - 1}")

        def side(s: str) -> DmaPattern:
            return DmaPattern(
                v[f"{s}_base"],
                tuple(v[f"{s}_bound[{i}]"] for i in range(self.d)),
                tuple(v[f"{s}_stride[{i}]"] for i in range(self.d)),
            )

        return DmaDescriptor(DIRECTIONS[k], side("src"), side("dst"))

    def encode(self, arg: DmaDescriptor) -> dict[str, int]:
        out = {"direction": DIRECTIONS.index(arg.direction)}
        for s, p in (("src", arg.src), ("dst", arg.dst)):
            what = f"{self.comp.name} {s}"
            out[f"{s}_base"] = p.base
            out |= {f"{s}_bound[{i}]": b for i, b in enumerate(_pad(p.bounds, self.d, 1, what))}
            out |= {f"{s}_stride[{i}]": x for i, x in enumerate(_pad(p.strides, self.d, 0, what))}
        return out


# Component class -> adapter factory. Looked up along the class's MRO, so a
# subclass (e.g. a logging streamer in a test) uses its base's adapter.
_ADAPTERS: dict[type, Callable[..., Adapter]] = {
    Streamer: StreamerAdapter,
    Accelerator: AccelAdapter,
    Dma: DmaAdapter,
}


def register_adapter(cls: type, factory: Callable[..., Adapter]) -> None:
    """Add a block kind: ``factory(comp, **opts)`` returns its adapter (principle 6)."""
    _ADAPTERS[cls] = factory


def adapter_for(comp: Component, **opts: Any) -> Adapter:
    for cls in type(comp).__mro__:
        if cls in _ADAPTERS:
            return _ADAPTERS[cls](comp, **opts)
    raise ValueError(f"{comp.name}: no register adapter for {type(comp).__name__}")


# =============================================================================
# Register map
# =============================================================================


@dataclass(frozen=True)
class Register:
    block: str
    name: str
    addr: int
    access: str  # "wo" (start), "ro" (busy, busy_cycles), "rw" (configuration)


@dataclass
class Block:
    name: str
    comp: Component
    adapter: Adapter
    base: int

    @property
    def kind(self) -> str:
        return self.adapter.kind

    @property
    def config_names(self) -> list[str]:
        return self.adapter.registers

    def addr(self, reg: str) -> int:
        names = [*STATUS, *self.config_names]
        if reg not in names:
            raise KeyError(f"{self.name} has no register {reg!r}")
        return self.base + names.index(reg)


class RegisterMap:
    """Blocks with their windows, built from a list of (name, component).

    Built from a cluster config in MOD9/MOD10. ``spatial_bounds`` gives
    design-time spatial bounds per streamer name (default ``(n_ports,)``).
    """

    def __init__(
        self,
        blocks: Sequence[tuple[str, Component]],
        window: int = 32,
        bases: Mapping[str, int] | None = None,
        spatial_bounds: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        if window < 1 or window & (window - 1):
            raise ValueError(f"window {window} must be a power of two")
        bases = dict(bases or {})
        spatial_bounds = dict(spatial_bounds or {})
        names = [n for n, _ in blocks]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate block names: {names}")
        unknown = (set(bases) | set(spatial_bounds)) - set(names)
        if unknown:
            raise ValueError(f"bases or spatial bounds for unknown blocks: {sorted(unknown)}")
        self.window = window
        self.blocks: dict[str, Block] = {}
        self._regs: dict[int, Register] = {}
        for i, (name, comp) in enumerate(blocks):
            opts = {}
            if name in spatial_bounds:
                if not isinstance(comp, Streamer):
                    raise ValueError(f"{name}: spatial bounds given for a non-streamer")
                opts["spatial_bounds"] = spatial_bounds[name]
            ad = adapter_for(comp, **opts)
            base = bases.get(name, i * window)
            if base < 0 or base % window:
                raise ValueError(f"{name}: base {base} is not a multiple of the window {window}")
            blk = Block(name, comp, ad, base)
            regs = [*STATUS, *ad.registers]
            if len(regs) > window:
                raise ValueError(f"{name}: {len(regs)} registers do not fit a window of {window}")
            for off, reg in enumerate(regs):
                access = "wo" if reg == "start" else "ro" if reg in READ_ONLY else "rw"
                self._regs[base + off] = Register(name, reg, base + off, access)
            for other in self.blocks.values():
                if other.base == base:
                    raise ValueError(f"{name} and {other.name} share the window at {base}")
            self.blocks[name] = blk

    # -- lookup ---------------------------------------------------------------

    def __getitem__(self, name: str) -> Block:
        return self.blocks[name]

    def register(self, addr: int) -> Register:
        if addr not in self._regs:
            raise ValueError(f"no register at address {addr}")
        return self._regs[addr]

    def addr(self, name: str) -> int:
        """Address of ``"block.reg"``, e.g. ``"dma.src_base"``."""
        block, _, reg = name.partition(".")
        return self.blocks[block].addr(reg)

    def describe(self, addr: int) -> str:
        r = self._regs.get(addr)
        return f"{r.block}.{r.name}" if r else f"<unmapped {addr}>"

    def to_dict(self) -> dict[str, Any]:
        """Every register by name, per block: the listing people and LLMs read."""
        return {
            "window": self.window,
            "blocks": {
                b.name: {
                    "kind": b.kind,
                    "base": b.base,
                    "registers": {r.name: r.addr for r in self._regs.values() if r.block == b.name},
                }
                for b in self.blocks.values()
            },
        }

    # -- helpers that expand "start block X with arg" into writes ------------

    def config_writes(self, block: str, arg: Any) -> list[CsrWrite]:
        """Writes of every configuration register of ``block`` for ``arg``."""
        b = self.blocks[block]
        values = b.adapter.encode(arg)
        return [CsrWrite(b.addr(r), values[r]) for r in b.config_names]

    def start_write(self, block: str) -> CsrWrite:
        return CsrWrite(self.blocks[block].addr("start"), 1)

    def start_writes(self, block: str, arg: Any) -> list[CsrWrite]:
        return [*self.config_writes(block, arg), self.start_write(block)]


# =============================================================================
# Controller
# =============================================================================


@dataclass(frozen=True)
class ControllerConfig(Config):
    """Command costs in cycles (all >= 1) and wait timing. Declared defaults (D51)."""

    write_cost: int = 1
    read_cost: int = 1
    kind_write_cost: Mapping[str, int] = field(default_factory=dict)  # e.g. {"dma": 3}
    kind_read_cost: Mapping[str, int] = field(default_factory=dict)
    poll_interval: int = 1  # P: cycles from one poll's start to the next
    signal_latency: int = 1  # S: done -> the controller continues

    def __post_init__(self) -> None:
        costs = [self.write_cost, self.read_cost, self.poll_interval, self.signal_latency,
                 *self.kind_write_cost.values(), *self.kind_read_cost.values()]  # fmt: skip
        if min(costs) < 1:
            raise ValueError("costs, poll_interval and signal_latency must be >= 1")

    def write_cost_of(self, kind: str) -> int:
        return self.kind_write_cost.get(kind, self.write_cost)

    def read_cost_of(self, kind: str) -> int:
        return self.kind_read_cost.get(kind, self.read_cost)


class Controller(Component):
    """Executes a control program through the register map, one command at a time.

    State:

    * committed: ``pc``, ``_t`` (first cycle of the current command), the
      shadow configuration values, ``_started`` (per block: cycle its last
      start landed);
    * wires: ``_done`` (the current command ends this cycle), ``_write``,
      ``_read``, ``_cls``;
    * statistics: ``cycles`` per class (a ``ClassLog``, MOD8), ``reads``,
      ``polls``, ``spans``, ``waits``. Trace events: ``cmd`` per finished
      command (task), ``poll`` per poll sample (beat), both from commit.
    """

    phases = (Phase.CONTROL,)

    def __init__(
        self,
        name: str,
        regmap: RegisterMap,
        program: Sequence[Command],
        cfg: ControllerConfig | None = None,
    ) -> None:
        super().__init__(name)
        self.map = regmap
        self.cfg = cfg or ControllerConfig()
        self.program = list(program)
        self._check_program()
        # Committed state.
        self.pc = 0
        self._t = 0
        self._shadow = {b: dict.fromkeys(blk.config_names, 0) for b, blk in regmap.blocks.items()}
        self._started: dict[str, int] = {}
        # Statistics.
        self.cycles = ClassLog(CYCLE_CLASSES)
        self.reads: list[tuple[int, int, int]] = []  # (cycle, addr, value)
        self.polls = 0
        self.spans: list[tuple[int, int, int]] = []  # (pc, first cycle, last cycle)
        self.waits: list[tuple[int, str, int, int, int]] = []  # (pc, block, t, done, last)
        self._clear_wires()

    def _clear_wires(self) -> None:
        self._done = False
        self._write: tuple[str, str, int] | None = None  # (block, reg, value) to the shadow
        self._read: tuple[int, int, int] | None = None
        self._polled = False
        self._cls: str | None = None

    # -------------------------------------------------------------------------
    # Static checks
    # -------------------------------------------------------------------------

    def _check_program(self) -> None:
        started: set[str] = set()
        for pc, cmd in enumerate(self.program):
            where = f"{self.name}: command {pc} ({cmd.to_dict()})"
            if isinstance(cmd, CsrWrite):
                r = self._reg(cmd.addr, where)
                if r.access == "ro":
                    raise ValueError(f"{where}: {r.block}.{r.name} is read-only")
                if r.name == "start":
                    if cmd.value != 1:
                        raise ValueError(f"{where}: start takes 1, got {cmd.value}")
                    started.add(r.block)
            elif isinstance(cmd, CsrRead):
                r = self._reg(cmd.addr, where)
                if r.access == "wo":
                    raise ValueError(f"{where}: {r.block}.{r.name} is write-only")
            elif isinstance(cmd, Wait):
                if cmd.block not in self.map.blocks:
                    raise ValueError(f"{where}: unknown block {cmd.block!r}")
                if cmd.block not in started:
                    raise ValueError(f"{where}: wait on {cmd.block}, which was never started")
                if cmd.mode == "poll":
                    c_r = self.cfg.read_cost_of(self.map[cmd.block].kind)
                    if self.cfg.poll_interval < c_r:
                        raise ValueError(
                            f"{where}: poll_interval {self.cfg.poll_interval} < read cost {c_r}"
                        )
            else:
                raise TypeError(f"{where}: not a command")

    def _reg(self, addr: int, where: str) -> Register:
        try:
            return self.map.register(addr)
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None

    # -------------------------------------------------------------------------
    # Costs and command timing (own state only)
    # -------------------------------------------------------------------------

    @property
    def finished(self) -> bool:
        return self.pc >= len(self.program)

    def _cost(self, cmd: Command) -> int:
        kind = self.map[self.map.register(cmd.addr).block].kind
        return (
            self.cfg.write_cost_of(kind)
            if isinstance(cmd, CsrWrite)
            else self.cfg.read_cost_of(kind)
        )

    def _first_sample(self, cmd: Wait) -> int:
        return self._t + self.cfg.read_cost_of(self.map[cmd.block].kind) - 1

    def _signal_end(self, cmd: Wait) -> int | None:
        """Last cycle of a signal wait, once the block's done_cycle is known."""
        d = self.map[cmd.block].comp.done_cycle
        return None if d is None else max(self._t, d) + self.cfg.signal_latency - 1

    # -------------------------------------------------------------------------
    # Register reads (committed state, D29)
    # -------------------------------------------------------------------------

    def _read_value(self, addr: int, cycle: int) -> int:
        r = self.map.register(addr)
        blk = self.map[r.block]
        if r.name == "busy":
            return int(blk.comp.busy)
        if r.name == "busy_cycles":
            s = self._started.get(r.block)
            if s is None:
                return 0
            d = blk.comp.done_cycle
            return max(0, (cycle if d is None else min(cycle, d)) - s - 1)
        return self._shadow[r.block][r.name]

    # -------------------------------------------------------------------------
    # Cycle behaviour
    # -------------------------------------------------------------------------

    def tick(self, cycle: int, phase: Phase) -> None:
        if self.finished or cycle < self._t:
            self._cls = "idle"
            return
        cmd = self.program[self.pc]
        if isinstance(cmd, Wait):
            self._cls = "wait"
            self._tick_wait(cmd, cycle)
            return
        self._cls = "command"
        if cycle != self._t + self._cost(cmd) - 1:
            return  # the command's cost is still running
        self._done = True
        if isinstance(cmd, CsrRead):
            self._read = (cycle, cmd.addr, self._read_value(cmd.addr, cycle))
            return
        r = self.map.register(cmd.addr)
        if r.name == "start":
            self._start(r.block, cycle)
        else:
            self._write = (r.block, r.name, int(cmd.value))

    def _tick_wait(self, cmd: Wait, cycle: int) -> None:
        if cmd.mode == "poll":
            s0 = self._first_sample(cmd)
            if cycle >= s0 and (cycle - s0) % self.cfg.poll_interval == 0:
                self._polled = True
                if not self.map[cmd.block].comp.busy:
                    self._done = True
        elif self._signal_end(cmd) == cycle:
            self._done = True

    def _start(self, block: str, cycle: int) -> None:
        """A start write lands: copy the shadow values into the component's start."""
        blk = self.map[block]
        where = f"{self.name}: cycle {cycle}: {block}.start"
        if blk.comp.busy:
            raise SimulationError(f"{where}: start while busy (wait on {block} first)")
        try:
            arg = blk.adapter.decode(self._shadow[block])
            blk.comp.start(arg, cycle)
        except ValueError as e:
            raise SimulationError(f"{where}: invalid configuration: {e}") from e
        self._started[block] = cycle

    def commit(self, cycle: int) -> None:
        tr = self._trace
        if self._write is not None:
            block, reg, value = self._write
            self._shadow[block][reg] = value
        if self._read is not None:
            self.reads.append(self._read)
        if self._polled:
            self.polls += 1
            if tr is not None and tr.beat:
                # A poll ends the wait exactly when it sampled busy = 0.
                cmd = self.program[self.pc]
                tr.emit(Poll(cycle, self.name, block=cmd.block, value=0 if self._done else 1))
        if self._done:
            cmd = self.program[self.pc]
            self.spans.append((self.pc, self._t, cycle))
            if isinstance(cmd, Wait):
                d = self.map[cmd.block].comp.done_cycle
                self.waits.append((self.pc, cmd.block, self._t, d, cycle))
            if tr is not None and tr.task:
                tr.emit(self._cmd_event(cmd, cycle))
            self.pc += 1
            self._t = cycle + 1
        if self._cls is not None:
            self.cycles.add(self._cls, cycle, cycle + 1)
        self._clear_wires()

    def _cmd_event(self, cmd: Command, last: int) -> Cmd:
        """Trace event of the command ending in ``last`` (commit: wires are final)."""
        pc, first = self.pc, self._t
        if isinstance(cmd, Wait):
            d = self.map[cmd.block].comp.done_cycle
            return Cmd(first, self.name, pc=pc, op="wait", last=last, block=cmd.block,
                       mode=cmd.mode, done=d)  # fmt: skip
        r = self.map.register(cmd.addr)
        reg = f"{r.block}.{r.name}"
        if isinstance(cmd, CsrRead):
            value = self._read[2] if self._read is not None else None
            return Cmd(first, self.name, pc=pc, op="csr_read", last=last, reg=reg, value=value)
        return Cmd(first, self.name, pc=pc, op="csr_write", last=last, reg=reg,
                   value=int(cmd.value))  # fmt: skip

    def next_wake(self, cycle: int) -> int | None:
        """See "Waking" in the module doc."""
        if self.finished:
            return None
        cmd = self.program[self.pc]
        if not isinstance(cmd, Wait):
            return self._t + self._cost(cmd) - 1
        if cmd.mode == "signal":
            return self._signal_end(cmd)
        s0, p = self._first_sample(cmd), self.cfg.poll_interval
        if cycle + 1 <= s0:
            return s0
        return s0 + -(-(cycle + 1 - s0) // p) * p

    def on_gap(self, start: int, stop: int) -> None:
        """Skipped cycles belong to the current command, or are idle."""
        if self.finished:
            self.cycles.add("idle", start, stop)
            return
        idle = max(0, min(stop, self._t) - start)  # only before the program starts
        cls = "wait" if isinstance(self.program[self.pc], Wait) else "command"
        self.cycles.add("idle", start, start + idle)
        self.cycles.add(cls, start + idle, stop)
