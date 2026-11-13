"""The task list: SNAX-LOWER's ordered list of tasks.

``TaskList`` holds the steps (``Configure``, ``Start``, ``Sync``; ``step_from_dict``) of
CONTRACTS, "Task list": which component runs which task with which values, where each is
configured and started, and what it depends on. c.py turns it into the program; waits.py places
its waits. A ``TaskList`` checks the structural rules when made, without a cluster.

``check_values(tasks, point)`` checks it against a design point (``components`` lists what can
run a task: the DMA when the platform has an L2, every streamer, every accelerated instance on
its own, every linked group), and also that an accelerator task's CSR writes fit: JOBS and every
JOB_BEATS (c.py's ``job_writes``) at most 2^``acc_csr.job_width`` - 1, and its head firings, JOBS
times every T of the group, below 2^32, SNAX-GVSoC's limit. A ``TaskListError`` names what fails.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from snax_forge.common import JsonFile, check_keys, plain
from snax_forge.design.platform import WAIT_MODES
from snax_forge.design.streamers import groups, instance_nodes, instances, streamer_name

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


Step = Configure | Start | Sync
_STEPS: dict[str, type] = {c.op: c for c in (Configure, Start, Sync)}


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
    """An ordered list of steps (CONTRACTS, "Task list"); checked when it is made."""

    name: str
    steps: list[Step] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.check()

    def check(self) -> None:
        """The structural rules of CONTRACTS, "Task list" (its "Checks" part); raises
        TaskListError."""
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
            else:
                raise TaskListError(f"{where}: not a step: {s!r}")
        if pending:
            raise TaskListError(
                f"{self.name}: configured but never started: {list(pending.values())}"
            )

    def configured(self) -> dict[str, Configure]:
        """Every task by name."""
        return {s.task_name: s for s in self.steps if isinstance(s, Configure)}

    def of_type(self, kind: str) -> list[Configure]:
        """The tasks of one type (``dma``, ``streamer``, ``accel``), in order."""
        return [s for s in self.steps if isinstance(s, Configure) and s.type == kind]

    def accel_steps(self) -> list[Configure]:
        """The accelerator tasks, in order."""
        return self.of_type("accel")

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
    its streamers in port order (a port on a link has none) and the instance, or after a
    linked group's last unit the group."""
    out = {"dma": "dma"} if point.platform.l2 is not None else {}
    found = instances(point.graph)
    for group, units in groups(point.graph).items():
        for name in units:
            for p in found[name].ports:
                if streamer_name(name, p.name) in point.streamers:
                    out[streamer_name(name, p.name)] = "streamer"
            if len(units) == 1:
                out[name] = "accel"
        if len(units) > 1:
            out[group] = "accel"
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
        found, units = instances(point.graph), groups(point.graph).get(s.component)
        if units is not None and len(units) > 1:  # a linked group: each unit's, by unit
            if list(v) != units:
                raise ValueError(f"{s.component}: values for {list(v)}, its units are {units}")
            for u in units:
                _check_unit(f"{s.component}.{u}", v[u], found[u], point)
            _check_widths(s, point, units)
            return
        _check_unit(s.component, v, found[s.component], point)
        _check_widths(s, point, [s.component])


def _check_unit(where: str, v: Mapping[str, Any], inst: Any, point: DesignPoint) -> None:
    """A unit's accelerator values: its registers (``n``, the named rates) and each slot it
    reads from a container (``Instance.read``) as ``{"from": <container>}``, a kernel's scalar
    argument (shape [1]) that lives in a memory (the core's)."""
    from snax_forge.dfg.graph import SCALAR

    names = [*inst.brm.registers, *inst.read]
    if set(v) != set(names):
        raise ValueError(f"{where}: parameters {sorted(v)}, registers {names}")
    for k in inst.read:
        src = v[k]
        if not isinstance(src, Mapping) or set(src) != {"from"}:
            raise ValueError(f"{where}: {k} is {src!r}; a slot read from a container is "
                             "{'from': <container>}")  # fmt: skip
        c = point.graph.containers.get(src["from"])
        if c is None or SCALAR not in c.attrs:  # shape [1], read as element 0
            raise ValueError(f"{where}: {k} from {src['from']!r}, which is no kernel's scalar "
                             "argument (sdfg.scalar)")  # fmt: skip
        if not point.memory.layouts.get(src["from"]):
            raise ValueError(f"{where}: {k} from {src['from']!r}, which lives in no memory")


# The head's firings in a run stay below this (SNAX-GVSoC's SnaxGroup counts them in 32 bits)
HEAD_FIRINGS = 1 << 32


def _check_widths(s: Configure, point: DesignPoint, units: list[str]) -> None:
    """An accelerator task's CSR writes against the platform's ``acc_csr.job_width`` and its
    head firings against ``HEAD_FIRINGS`` (module doc)."""
    # gen and c import this module: theirs are imported here, not at the top
    from snax_forge.gen.accelerator import fixed_fold

    from .c import fold_rate, job_writes

    found, nodes = instances(point.graph), instance_nodes(point.graph)
    fixed = {u for u in units if fixed_fold(point.graph, nodes[u], found[u]) is not None}
    writes = job_writes({u: found[u] for u in units}, s.values, fixed)
    width = point.platform.acc_csr.job_width
    for unit, csr, value in writes:
        if value > (1 << width) - 1:
            raise ValueError(f"{s.component}: {csr} {value} (unit {unit}) is past "
                             f"acc_csr.job_width = {width} bits, at most {(1 << width) - 1}")  # fmt: skip
    by_unit = s.values if len(units) > 1 else {units[0]: s.values}
    firings = writes[0][2]
    for u in units:
        rate = fold_rate(found[u])
        firings *= 1 if rate is None else by_unit[u][rate]
    if firings >= HEAD_FIRINGS:
        raise ValueError(f"{s.component}: the head fires JOBS · ∏T = {firings} times, 2^32 or "
                         "more (SNAX-GVSoC counts a run's firings in 32 bits)")  # fmt: skip


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
