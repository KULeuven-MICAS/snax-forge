"""SNAX-MODEL's inputs from snax-forge's artefacts: the cluster file and the command list.

What SNAX-LOWER built for SNAX-MODEL before the flow moved to SNAX-GVSoC
(RET1, D116), kept here for the scenarios and SNAX-MODEL's own tests:

    cluster_config(platform, accels) / cluster_of(platform, instances)
                     the cluster file (CONTRACTS.md section 2 at tag pre-ret1):
                     xbar, dma, per accelerator its streamers then itself,
                     ctl; what the platform no longer holds is ``MODEL``,
                     the values small16 held
    Program          a command builder in name form, through the register map
    arg_of           a task's values as an adapter's start argument
    lower_program    a task list -> the command list, with the waits it needs
                     (the same placement as snax_forge/lower/waits.py)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from snax_forge.brm import Instance
from snax_forge.design.platform import Platform
from snax_forge.design.streamers import Streamer, resolve, streamer_name
from snax_forge.lower.tasks import Configure, Read, Start, Sync, TaskList, TaskListError

from .ctrl import Block, ControllerConfig, RegisterMap, Wait
from .dma import DmaConfig, DmaDescriptor, DmaPattern
from .l2 import L2Config
from .mem import L1Config
from .scenario import (
    ClusterConfig,
    ComponentSpec,
    NamedRead,
    RegisterMapSpec,
    ScenarioCommand,
    named,
    register_map_of,
)
from .streamer import StreamerRegs

# =============================================================================
# The cluster file
# =============================================================================

XBAR, DMA, CTL = "xbar", "dma", "ctl"
MODEL = {
    "read_latency": 1,
    "streamer": {"addr_depth": 8, "prio": 0},
    "xbar": {"check_hold": True},
    "dma": DmaConfig(),
    "controller": ControllerConfig(poll_interval=4),
    "register_window": 32,
}


@dataclass(frozen=True)
class Accel:
    """One accelerator of the cluster: its entry and its streamers, in port order."""

    name: str
    accel: str
    params: dict[str, Any]
    streamers: tuple[Streamer, ...]

    def attach(self) -> dict[str, str]:
        return {s.port: s.name for s in self.streamers}


def accel_entry(inst: Instance) -> tuple[str, dict[str, Any]]:
    """``(accel, params)``: the function's params resolved, then ``latency``, ``ii`` and
    ``drain`` when not 0."""
    params = {k: inst.value(v) for k, v in inst.brm.function.params.items()}
    params["latency"] = inst.latency
    params["ii"] = inst.initiation_interval
    if inst.drain:
        params["drain"] = inst.drain
    return inst.brm.function.accel, params


def accel_of(name: str, inst: Instance, streamers: Mapping[str, Streamer]) -> Accel:
    """The accelerator ``name`` built from a BRM instance and its resolved streamers."""
    kind, params = accel_entry(inst)
    ports = [p.name for p in inst.brm.interface.ports]
    return Accel(name, kind, params, tuple(streamers[streamer_name(name, p)] for p in ports))


def stub(
    platform: Platform,
    name: str,
    accel: str,
    params: Mapping[str, Any],
    ports: Sequence[tuple[str, bool, int]],
) -> Accel:
    """An accelerator given by a registered kind and params; ``ports`` are
    ``(port, write, lanes)`` in order, their streamers shaped by the platform's shell."""
    streamers = tuple(
        Streamer(s, name, p, write, (lanes,), platform.options(s))
        for p, write, lanes in ports
        for s in [streamer_name(name, p)]
    )
    return Accel(name, accel, dict(params), streamers)


def cluster_config(platform: Platform, accels: Sequence[Accel]) -> ClusterConfig:
    """The cluster file of ``platform`` with ``accels``; see the module doc."""
    comps = [ComponentSpec(XBAR, "xbar", dict(MODEL["xbar"]))]
    if platform.l2 is not None:
        comps.append(ComponentSpec(DMA, "dma", MODEL["dma"].to_dict()))
    spatial: dict[str, list[int]] = {}
    for a in accels:
        for s in a.streamers:
            config = {"write": s.write, "n_ports": s.n_ports, **s.options.to_dict(),
                      **MODEL["streamer"]}  # fmt: skip
            comps.append(ComponentSpec(s.name, "streamer", config))
            if list(s.spatial_bounds) != [s.n_ports]:
                spatial[s.name] = list(s.spatial_bounds)
        comps.append(
            ComponentSpec(a.name, "accel", accel=a.accel, params=dict(a.params), attach=a.attach())
        )
    comps.append(ComponentSpec(CTL, "controller", MODEL["controller"].to_dict()))
    blocks = [c.name for c in comps if c.kind not in ("xbar", "controller")]
    latency = {"read_latency": MODEL["read_latency"]}
    l2 = platform.l2
    return ClusterConfig(
        l1=L1Config(**platform.l1.to_dict(), **latency),
        l2=None if l2 is None else L2Config(**l2.to_dict(), **latency),
        components=comps,
        register_map=RegisterMapSpec(
            window=MODEL["register_window"], blocks=blocks, spatial_bounds=spatial
        ),
    )


def cluster_of(platform: Platform, instances: Mapping[str, Instance]) -> ClusterConfig:
    """The cluster of BRM instances on ``platform``, the shell resolved here."""
    shell = resolve(platform, dict(instances))
    return cluster_config(platform, [accel_of(k, inst, shell) for k, inst in instances.items()])


# =============================================================================
# Commands
# =============================================================================


class Program:
    """Collects commands in name form for one cluster's register map."""

    def __init__(self, cluster: ClusterConfig) -> None:
        self.map: RegisterMap = register_map_of(cluster)
        self.cmds: list[ScenarioCommand] = []

    def config(self, block: str, arg: Any) -> None:
        self.cmds += [named(c, self.map) for c in self.map.config_writes(block, arg)]

    def start(self, block: str) -> None:
        self.cmds.append(named(self.map.start_write(block), self.map))

    def wait(self, block: str, mode: str) -> None:
        self.cmds.append(Wait(block, mode))

    def read(self, reg: str) -> None:
        self.cmds.append(NamedRead(reg))


# =============================================================================
# Values
# =============================================================================

ToArg = Callable[[Mapping[str, Any], Block], Any]
FromArg = Callable[[Any], dict[str, Any]]


def _exact(v: Any, keys: Sequence[str], what: str) -> None:
    """``v`` is a mapping with exactly ``keys``."""
    if not isinstance(v, Mapping):
        raise TypeError(f"{what} must be an object, got {v!r}")
    missing = [k for k in keys if k not in v]
    unknown = sorted(set(v) - set(keys))
    if missing or unknown:
        raise ValueError(f"{what}: missing {missing}, unknown {unknown} (expected {list(keys)})")


def _ints(v: Any, what: str) -> tuple[int, ...]:
    if not isinstance(v, (list, tuple)):
        raise TypeError(f"{what} must be a list, got {v!r}")
    return tuple(int(x) for x in v)


# -- streamer -----------------------------------------------------------------

_STREAMER = ("base", "temporal_bounds", "temporal_strides", "spatial_strides")


def _streamer_arg(v: Mapping[str, Any], block: Block) -> StreamerRegs:
    _exact(v, _STREAMER, "streamer values")
    return StreamerRegs(
        int(v["base"]),
        _ints(v["temporal_bounds"], "temporal_bounds"),
        _ints(v["temporal_strides"], "temporal_strides"),
        block.adapter.spatial_bounds,
        _ints(v["spatial_strides"], "spatial_strides"),
    )


def _streamer_values(arg: StreamerRegs) -> dict[str, Any]:
    return {
        "base": arg.base,
        "temporal_bounds": list(arg.temporal_bounds),
        "temporal_strides": list(arg.temporal_strides),
        "spatial_strides": list(arg.spatial_strides),
    }


# -- dma ----------------------------------------------------------------------

_DMA = ("direction", "src", "dst")
_PATTERN = ("base", "bounds", "strides")


def _pattern(v: Any, what: str) -> DmaPattern:
    _exact(v, _PATTERN, what)
    return DmaPattern(int(v["base"]), _ints(v["bounds"], f"{what} bounds"),
                      _ints(v["strides"], f"{what} strides"))  # fmt: skip


def _dma_arg(v: Mapping[str, Any], block: Block) -> DmaDescriptor:
    _exact(v, _DMA, "dma values")
    return DmaDescriptor(str(v["direction"]), _pattern(v["src"], "src"), _pattern(v["dst"], "dst"))


def _pattern_values(p: DmaPattern) -> dict[str, Any]:
    return {"base": p.base, "bounds": list(p.bounds), "strides": list(p.strides)}


def _dma_values(arg: DmaDescriptor) -> dict[str, Any]:
    return {"direction": arg.direction, "src": _pattern_values(arg.src),
            "dst": _pattern_values(arg.dst)}  # fmt: skip


# -- accel --------------------------------------------------------------------


def _accel_arg(v: Mapping[str, Any], block: Block) -> dict[str, int]:
    if not isinstance(v, Mapping):
        raise TypeError(f"accel values must be an object, got {v!r}")
    return {k: int(x) for k, x in v.items()}  # names are checked by the adapter


def _accel_values(arg: Mapping[str, int]) -> dict[str, Any]:
    return {k: int(x) for k, x in arg.items()}


# -- registry -----------------------------------------------------------------

_VALUES: dict[str, tuple[ToArg, FromArg]] = {
    "streamer": (_streamer_arg, _streamer_values),
    "dma": (_dma_arg, _dma_values),
    "accel": (_accel_arg, _accel_values),
}


def register_values(kind: str, to_arg: ToArg, from_arg: FromArg) -> None:
    """Add a component type: ``to_arg(values, block)`` and its inverse ``from_arg(arg)``."""
    _VALUES[kind] = (to_arg, from_arg)


def _codec(kind: str) -> tuple[ToArg, FromArg]:
    if kind not in _VALUES:
        raise ValueError(f"no values for component type {kind!r} (types: {list(_VALUES)})")
    return _VALUES[kind]


def arg_of(kind: str, values: Mapping[str, Any], block: Block) -> Any:
    """The start argument of ``block`` for a task's values.

    Raises TypeError for values of the wrong shape, ValueError for values the
    start argument rejects.
    """
    return _codec(kind)[0](values, block)


def values_of(kind: str, arg: Any) -> dict[str, Any]:
    """A start argument as a task's values (the inverse of ``arg_of``)."""
    return _codec(kind)[1](arg)


# =============================================================================
# Task list -> command list
# =============================================================================


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
