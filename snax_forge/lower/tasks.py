"""The task list: SNAX-LOWER's ordered list of tasks (LOW1b, D45, D64).

A task list says which component runs which task with which values, where
each task is configured and started, and what it depends on. ``lower_program``
turns it into the plain command list the model runs (D42); the model never
reads a task list.

    {"name": ..., "steps": [step, ...]}

Steps, each with an ``op``:

    configure  task_name, type, component, after, wait_mode, values
               declares a task and writes its component's configuration
               registers at this point in the program
    start      tasks: launches the listed tasks together, after the waits
               they need
    sync       task, mode: waits here for that task's component
    read       reg: a csr_read of "block.register"

``type`` is the component's type (``streamer``, ``accel``, ``dma``: the kind
of its register adapter) and decides how ``values`` are read (values.py).
``after`` names earlier tasks whose results this task needs, including a
buffer it overwrites; ``wait_mode`` (``poll`` / ``signal``) is how the
program waits for this task when the lowering inserts a wait for it.

The structural rules are checked here, without a cluster: task names are
unique; ``after``, ``start`` and ``sync`` refer to tasks configured earlier;
a component has at most one configured task that has not started yet (its
registers are buffered, D36), so one start never launches two tasks on one
component; a task starts once, after its ``after`` tasks have started; every
configured task is started. What needs the cluster (component, type, values,
register names) is checked by ``lower_program``.

``Tasks`` builds a task list from start arguments, for Python generators such
as scenarios/make.py.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from snax_forge.snax_model.config import check_keys, plain, to_json
from snax_forge.snax_model.ctrl import WAIT_MODES
from snax_forge.snax_model.scenario import ClusterConfig, register_map_of

from .values import values_of


class TaskListError(ValueError):
    """A task list that cannot be lowered."""


_OPTIONAL = {"configure": ("after", "wait_mode"), "sync": ("mode",)}


def _need(d: Mapping[str, Any], keys: tuple[str, ...], what: str) -> None:
    try:
        check_keys(d, ["op", *keys, *_OPTIONAL.get(what, ())], what)
    except ValueError as e:
        raise TaskListError(str(e)) from e
    missing = [k for k in keys if k not in d]
    if missing:
        raise TaskListError(f"{what} {dict(d)}: missing {missing}")


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
class TaskList:
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

    @classmethod
    def load(cls, path: str | Path) -> TaskList:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(to_json(self.to_dict()))


# =============================================================================
# Builder
# =============================================================================


class Tasks:
    """Builds a TaskList from start arguments (``StreamerRegs``, ``DmaDescriptor``, ...).

    ``configure`` takes the component's type from the cluster and checks the
    argument with the component's adapter, so a mistake shows up where the
    generator makes it.
    """

    def __init__(self, name: str, cluster: ClusterConfig) -> None:
        self.name = name
        self.map = register_map_of(cluster)
        self.steps: list[Step] = []

    def configure(
        self,
        task_name: str,
        component: str,
        arg: Any,
        after: tuple[str, ...] | list[str] = (),
        wait_mode: str = "poll",
    ) -> None:
        if component not in self.map.blocks:
            raise TaskListError(f"{task_name}: no component {component!r}")
        block = self.map.blocks[component]
        block.adapter.encode(arg)
        values = values_of(block.kind, arg)
        self.steps.append(
            Configure(task_name, block.kind, component, values, list(after), wait_mode)
        )

    def start(self, *task_names: str) -> None:
        self.steps.append(Start(list(task_names)))

    def sync(self, task: str, mode: str = "poll") -> None:
        self.steps.append(Sync(task, mode))

    def read(self, reg: str) -> None:
        self.steps.append(Read(reg))

    def task_list(self) -> TaskList:
        return TaskList(self.name, list(self.steps))
