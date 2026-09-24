"""Task list -> plain command list (LOW1b, D45, D64).

``lower_program(tasks, cluster)`` walks the steps in order and emits the
commands the model runs (D42), in name form, through the model's register
adapters (D36), the same path the Python generators used before:

    configure  the component's configuration writes, here (config_writes)
    start      the waits its tasks need, then their starts in list order
    sync       one wait on the task's component, always emitted
    read       one csr_read

The waits a start needs: one for every ``after`` task not yet waited for,
and one for every listed component whose latest task has not been waited
for (a start while busy is an error, D36). There is at most one wait per
component, and a wait that another of these waits covers is left out (D66);
waits are ordered by when the task they end on was started, and each uses
that task's ``wait_mode``. A wait is on a component, so it ends on
the component's latest task and covers every earlier one on it.

A wait on a writer streamer also covers the accelerator attached to it and
that accelerator's reader streamers, for the tasks the same start launched:
their data flows into the writer, so the writer finishes last. This comes
from ``attach`` and the streamer ``write`` flag in the cluster file. If the
tasks do not match in size the claim is wrong, and the model raises a start
while busy.

Nothing here decides anything (principle 4): where configures, starts and
syncs go, and every dependency, are the task list's; only the waits
correctness requires are added.
"""

from __future__ import annotations

from snax_forge.snax_model.scenario import ClusterConfig, ScenarioCommand

from .program import Program
from .tasks import Configure, Read, Start, Sync, TaskList, TaskListError
from .values import arg_of


def upstream(cluster: ClusterConfig) -> dict[str, list[str]]:
    """Per component, the components a wait on it also covers (module doc).

    A writer streamer -> the accelerators it is attached to as output; an
    accelerator -> the reader streamers attached to it.
    """
    specs = {c.name: c for c in cluster.components}

    def writes(name: str) -> bool:
        spec = specs.get(name)
        return spec is not None and spec.kind == "streamer" and bool(spec.config.get("write"))

    up: dict[str, list[str]] = {}
    for spec in cluster.components:
        if spec.kind != "accel":
            continue
        attached = list(spec.attach.values())
        up[spec.name] = [s for s in attached if not writes(s)]
        for s in attached:
            if writes(s):
                up.setdefault(s, []).append(spec.name)
    return up


class _Lowering:
    def __init__(self, tasks: TaskList, cluster: ClusterConfig) -> None:
        tasks.check()
        self.tasks = tasks
        self.p = Program(cluster)
        self.up = upstream(cluster)
        self.conf: dict[str, Configure] = tasks.configured()
        self.started_at: dict[str, tuple[int, int]] = {}  # task -> (step, place in the start)
        self.latest: dict[str, str] = {}  # component -> its latest started task
        self.covered: dict[str, str] = {}  # component -> latest task a wait has covered

    def is_covered(self, task: str) -> bool:
        c = self.covered.get(self.conf[task].component)
        return c is not None and self.started_at[c] >= self.started_at[task]

    def covers(self, comp: str) -> dict[str, str]:
        """What one wait on ``comp`` covers: component -> its task (module doc)."""
        step = self.started_at[self.latest[comp]][0]
        out: dict[str, str] = {}
        stack = [comp]
        while stack:
            x = stack.pop()
            t = self.latest.get(x)
            if x in out or t is None or (x != comp and self.started_at[t][0] != step):
                continue
            out[x] = t
            stack += self.up.get(x, [])
        return out

    def wait(self, comp: str, mode: str) -> None:
        """One wait on ``comp``: covers its latest task and, from a writer, its group."""
        self.p.wait(comp, mode)
        self.covered.update(self.covers(comp))

    def configure(self, s: Configure, where: str) -> None:
        blocks = self.p.map.blocks
        if s.component not in blocks:
            raise TaskListError(
                f"{where}: no component {s.component!r} (components: {list(blocks)})"
            )
        kind = blocks[s.component].kind
        if s.type != kind:
            raise TaskListError(f"{where}: {s.component} is a {kind}, not a {s.type}")
        try:
            self.p.config(s.component, arg_of(s.type, s.values, blocks[s.component]))
        except (KeyError, TypeError, ValueError) as e:
            raise TaskListError(f"{where}: {e}") from e

    def start(self, s: Start, i: int) -> None:
        need: dict[str, str] = {}  # component -> the task a wait on it ends on
        for t in s.tasks:
            c = self.conf[t].component
            if c in self.latest and not self.is_covered(self.latest[c]):
                need[c] = self.latest[c]
            for a in self.conf[t].after:
                if not self.is_covered(a):
                    ac = self.conf[a].component
                    need[ac] = self.latest[ac]
        # A wait another needed wait covers is left out (a writer's covers its group).
        covered_by_others = {x for c in need for x in self.covers(c) if x != c}
        for c in sorted(need, key=lambda c: self.started_at[need[c]]):
            if c not in covered_by_others and not self.is_covered(need[c]):
                self.wait(c, self.conf[need[c]].wait_mode)
        for j, t in enumerate(s.tasks):
            c = self.conf[t].component
            self.p.start(c)
            self.latest[c] = t
            self.started_at[t] = (i, j)

    def read(self, s: Read, where: str) -> None:
        try:
            self.p.map.addr(s.reg)
        except (KeyError, ValueError) as e:
            raise TaskListError(f"{where}: no register {s.reg!r}") from e
        self.p.read(s.reg)

    def run(self) -> list[ScenarioCommand]:
        for i, s in enumerate(self.tasks.steps):
            where = f"{self.tasks.name}: step {i} ({s.op})"
            if isinstance(s, Configure):
                self.configure(s, where)
            elif isinstance(s, Start):
                self.start(s, i)
            elif isinstance(s, Sync):
                self.wait(self.conf[s.task].component, s.mode)
            else:
                self.read(s, where)
        return self.p.cmds


def lower_program(tasks: TaskList, cluster: ClusterConfig) -> list[ScenarioCommand]:
    """The command list of ``tasks`` on ``cluster`` (module doc); raises TaskListError."""
    return _Lowering(tasks, cluster).run()
