"""The task list: SNAX-LOWER's ordered list of tasks (D64, D135).

A task list says which component runs which task with which values, where
each task is configured and started, and what it depends on. The C backend
(c.py) turns it into the program; waits.py places its waits.

    {"name": ..., "steps": [step, ...]}

Steps, each with an ``op``:

    configure  task_name, type, component, after, wait_mode, values
               declares a task and writes its component's configuration
               registers at this point in the program
    start      tasks: launches the listed tasks together, after the waits
               they need
    sync       task, mode: waits here for that task's component
    read       reg: a csr_read of "block.register"

``type`` is the component's type (``streamer``, ``accel``, ``dma``) and
decides how ``values`` are read (CONTRACTS.md section 9).
``after`` names earlier tasks whose results this task needs, including a
buffer it overwrites; ``wait_mode`` (``poll`` / ``signal``) is how the
program waits for this task when the lowering inserts a wait for it.

The structural rules are checked here, without a cluster: task names are
unique; ``after``, ``start`` and ``sync`` refer to tasks configured earlier;
a component has at most one configured task that has not started yet (its
registers are buffered, D36), so one start never launches two tasks on one
component; a task starts once, after its ``after`` tasks have started; every
configured task is started.

What needs the design point is ``check_values(tasks, point)``: the
component exists and has the task's type (the DMA when the platform has an
L2, every streamer of the shell, every accelerated instance); the values
have exactly the type's keys (a streamer's ``base``, ``temporal_bounds``,
``temporal_strides``, ``spatial_strides``; a DMA's ``direction``, ``src`` and
``dst``, its L1 side ``base``, ``bounds`` and ``strides``, its L2 side
``symbol`` (a container in L2: its data.h array), ``bounds`` and ``strides``; an accelerator's
runtime params, ``n`` and its named rates); a streamer's loops fit it (as
many temporal strides as bounds, no more than its ``temporal_dims``, one
spatial stride per spatial bound, every bound >= 1, or every one 0: an idle
mover, which never starts, D148); a DMA's direction is
``l2_to_l1`` or ``l1_to_l2`` and its sides have one stride per bound, every
bound >= 1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from snax_forge.common import JsonFile, check_keys, plain
from snax_forge.design.platform import WAIT_MODES
from snax_forge.design.streamers import instances, streamer_name

if TYPE_CHECKING:
    from snax_forge.design.point import DesignPoint


class TaskListError(ValueError):
    """A task list that cannot be lowered."""


_OPTIONAL = {"configure": ("after", "wait_mode"), "sync": ("mode",)}


def _need(d: Mapping[str, Any], keys: tuple[str, ...], what: str) -> None:
    try:
        check_keys(d, ["op", *keys, *_OPTIONAL.get(what, ())], what, keys)
    except ValueError as e:
        raise TaskListError(str(e)) from e


# =============================================================================
# Steps
# =============================================================================


@dataclass
class Configure:
    """Declares task ``task_name`` on ``component`` and writes its registers here."""

    task_name: str
    type: str
    component: str
    values: dict[str, Any]
    after: list[str] = field(default_factory=list)
    wait_mode: str = "poll"
    op: ClassVar[str] = "configure"

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "task_name": self.task_name,
            "type": self.type,
            "component": self.component,
            "after": list(self.after),
            "wait_mode": self.wait_mode,
            "values": plain(self.values),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Configure:
        _need(d, ("task_name", "type", "component", "values"), "configure")
        if not isinstance(d["values"], Mapping):
            raise TaskListError(f"configure {d['task_name']!r}: values must be an object")
        return cls(
            task_name=str(d["task_name"]),
            type=str(d["type"]),
            component=str(d["component"]),
            values=dict(d["values"]),
            after=[str(a) for a in d.get("after", [])],
            wait_mode=str(d.get("wait_mode", "poll")),
        )


@dataclass
class Start:
    """Launches ``tasks`` together, after the waits they need."""

    tasks: list[str]
    op: ClassVar[str] = "start"

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "tasks": list(self.tasks)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Start:
        _need(d, ("tasks",), "start")
        return cls([str(t) for t in d["tasks"]])


@dataclass
class Sync:
    """Waits here for ``task``'s component: a synchronisation point, not a barrier."""

    task: str
    mode: str = "poll"
    op: ClassVar[str] = "sync"

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "task": self.task, "mode": self.mode}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Sync:
        _need(d, ("task",), "sync")
        return cls(str(d["task"]), str(d.get("mode", "poll")))


@dataclass
class Read:
    """A ``csr_read`` of ``reg`` ("block.register")."""

    reg: str
    op: ClassVar[str] = "read"

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "reg": self.reg}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Read:
        _need(d, ("reg",), "read")
        return cls(str(d["reg"]))


Step = Configure | Start | Sync | Read
_STEPS: dict[str, type] = {c.op: c for c in (Configure, Start, Sync, Read)}


def step_from_dict(d: Mapping[str, Any]) -> Step:
    op = d.get("op") if isinstance(d, Mapping) else None
    if op not in _STEPS:
        raise TaskListError(f"step {d!r}: op must be one of {list(_STEPS)}")
    return _STEPS[op].from_dict(d)


# =============================================================================
# Task list
# =============================================================================


@dataclass
class TaskList(JsonFile):
    """An ordered list of steps (module doc); checked when it is made."""

    name: str
    steps: list[Step] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.check()

    def check(self) -> None:
        """The structural rules of the module doc; raises TaskListError."""
        configured: dict[str, Configure] = {}
        started: set[str] = set()
        pending: dict[str, str] = {}  # component -> configured, not yet started task
        for i, s in enumerate(self.steps):
            where = f"{self.name}: step {i} ({s.op})"
            if isinstance(s, Configure):
                if s.task_name in configured:
                    raise TaskListError(f"{where}: task {s.task_name!r} is configured twice")
                if s.wait_mode not in WAIT_MODES:
                    raise TaskListError(f"{where}: wait_mode must be one of {WAIT_MODES}")
                for a in s.after:
                    if a not in configured:
                        raise TaskListError(f"{where}: after {a!r} is not an earlier task")
                if s.component in pending:
                    raise TaskListError(
                        f"{where}: {s.component} is configured again before its task "
                        f"{pending[s.component]!r} has started"
                    )
                configured[s.task_name] = s
                pending[s.component] = s.task_name
            elif isinstance(s, Start):
                if not s.tasks:
                    raise TaskListError(f"{where}: no tasks")
                if len(set(s.tasks)) != len(s.tasks):
                    raise TaskListError(f"{where}: a task is listed twice")
                for t in s.tasks:
                    if t not in configured:
                        raise TaskListError(f"{where}: task {t!r} is not configured before")
                    if t in started:
                        raise TaskListError(f"{where}: task {t!r} is started twice")
                    for a in configured[t].after:
                        if a not in started:
                            raise TaskListError(
                                f"{where}: {t!r} needs {a!r}, which has not started before"
                            )
                for t in s.tasks:
                    started.add(t)
                    del pending[configured[t].component]
            elif isinstance(s, Sync):
                if s.task not in started:
                    raise TaskListError(f"{where}: task {s.task!r} is not started before")
                if s.mode not in WAIT_MODES:
                    raise TaskListError(f"{where}: mode must be one of {WAIT_MODES}")
            elif isinstance(s, Read):
                if "." not in s.reg:
                    raise TaskListError(f"{where}: register {s.reg!r} is not 'block.register'")
            else:
                raise TaskListError(f"{where}: not a step: {s!r}")
        if pending:
            raise TaskListError(
                f"{self.name}: configured but never started: {list(pending.values())}"
            )

    def configured(self) -> dict[str, Configure]:
        """Every task by name."""
        return {s.task_name: s for s in self.steps if isinstance(s, Configure)}

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TaskList:
        try:
            check_keys(d, ["name", "steps"], "task list")
        except ValueError as e:
            raise TaskListError(str(e)) from e
        if "name" not in d:
            raise TaskListError("task list needs a 'name'")
        return cls(str(d["name"]), [step_from_dict(s) for s in d.get("steps", [])])


# =============================================================================
# Values against the design point
# =============================================================================

STREAMER = ("base", "temporal_bounds", "temporal_strides", "spatial_strides")
DMA = ("direction", "src", "dst")
DMA_SIDE = {"l1": ("base", "bounds", "strides"), "l2": ("symbol", "bounds", "strides")}
DIRECTIONS = ("l2_to_l1", "l1_to_l2")


def _exact(v: Any, keys: list[str] | tuple[str, ...], what: str) -> None:
    """``v`` is a mapping with exactly ``keys``."""
    check_keys(v, keys, what, keys)


def _ints(v: Any, what: str) -> list[int]:
    if not isinstance(v, (list, tuple)):
        raise TypeError(f"{what} must be a list, got {v!r}")
    return [int(x) for x in v]


def _fit(n: int, dims: int, what: str) -> None:
    if n > dims:
        raise ValueError(f"{what}: {n} loops, registers hold {dims}")


def components(point: DesignPoint) -> dict[str, str]:
    """The design point's components and their types: the DMA (with an L2), then per instance
    its streamers in port order and the instance."""
    out = {"dma": "dma"} if point.platform.l2 is not None else {}
    for name, inst in instances(point.graph).items():
        for p in inst.brm.interface.ports:
            out[streamer_name(name, p.name)] = "streamer"
        out[name] = "accel"
    return out


def _check(s: Configure, point: DesignPoint) -> None:
    v = s.values
    if s.type == "streamer":
        _exact(v, STREAMER, "streamer values")
        tb, ts = (
            _ints(v["temporal_bounds"], "temporal_bounds"),
            _ints(v["temporal_strides"], "temporal_strides"),
        )
        ss = _ints(v["spatial_strides"], "spatial_strides")
        shell = point.streamers[s.component]
        if len(tb) != len(ts):
            raise ValueError(f"{len(tb)} temporal bounds but {len(ts)} temporal strides")
        if len(shell.spatial_bounds) != len(ss):
            raise ValueError(
                f"{len(shell.spatial_bounds)} spatial bounds but {len(ss)} spatial strides"
            )
        _fit(len(tb), shell.temporal_dims, s.component)
        if any(b < 1 for b in tb) and any(tb):
            raise ValueError("temporal bounds must be >= 1, or all 0 (an idle mover)")
    elif s.type == "dma":
        _exact(v, DMA, "dma values")
        if v["direction"] not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {v['direction']!r}")
        l2 = "src" if v["direction"] == "l2_to_l1" else "dst"
        for side in ("src", "dst"):
            _exact(v[side], DMA_SIDE["l2" if side == l2 else "l1"], side)
            b, st = (
                _ints(v[side]["bounds"], f"{side} bounds"),
                _ints(v[side]["strides"], f"{side} strides"),
            )
            if len(b) != len(st):
                raise ValueError(f"{len(b)} bounds but {len(st)} strides")
            if any(x < 1 for x in b):
                raise ValueError("bounds must be >= 1")
        sym = v[l2]["symbol"]
        if "l2" not in point.memory.layouts.get(sym, {}):
            raise ValueError(f"{l2} symbol {sym!r} is no container in L2")
    else:
        if not isinstance(v, Mapping):
            raise TypeError(f"accel values must be an object, got {v!r}")
        names = instances(point.graph)[s.component].brm.registers
        if set(v) != set(names):
            raise ValueError(f"{s.component}: parameters {sorted(v)}, registers {names}")


def check_values(tasks: TaskList, point: DesignPoint) -> None:
    """Every configure step against the design point (module doc); raises TaskListError."""
    comps = components(point)
    for i, s in enumerate(tasks.steps):
        if not isinstance(s, Configure):
            continue
        where = f"{tasks.name}: step {i} ({s.op})"
        if s.component not in comps:
            raise TaskListError(
                f"{where}: no component {s.component!r} (components: {list(comps)})"
            )
        if s.type != comps[s.component]:
            raise TaskListError(f"{where}: {s.component} is a {comps[s.component]}, not a {s.type}")
        try:
            _check(s, point)
        except (KeyError, TypeError, ValueError) as e:
            raise TaskListError(f"{where}: {e}") from e
