"""Scenario runner for SNAX-MODEL (MOD9, D26, D29, D38, D39, D41-D44).

What a scenario is
------------------
A scenario is the input of one model run: a cluster configuration, the
initial memory contents and a control program. It holds inputs only;
expected results live in the tests (D41). Two JSON files:

* the cluster file (``ClusterConfig``): L1, optional L2, the ordered
  component list and the register map. It describes hardware, so many
  scenarios can share one;
* the scenario file (``Scenario``): ``name``, optional ``max_cycles``,
  ``cluster`` (a path relative to the scenario file, or the cluster object
  inline), ``memory`` and ``program``.

Component order
---------------
``components`` is one ordered list of every ticked component, the xbar and
the controller included. The builder adds them to the cluster in exactly
that order, which fixes the scheduler's tick order, the xbar port order (so
the RTL tie-breaks, D31) and the trace's source order (D39). L1 and L2 are
shared elements (D30, D34), not components, so they have their own keys.
There is one L1, one xbar and at most one L2; streamers and the DMA find
them without naming them. The register map is built when the builder
reaches the controller, so every block it lists must come before it.

Program (D42)
-------------
A plain list of ``csr_write``, ``csr_read`` and ``wait`` commands, as the
controller runs them. A register is given by raw address (``"addr": 4``,
ctrl.py's ``CsrWrite`` / ``CsrRead``) or by name (``"reg": "dma.src_base"``,
``NamedWrite`` / ``NamedRead`` here), exactly one of the two. Each command
keeps the form it was written in and ``to_dict`` writes that form back, so
round trips are exact. Names are resolved through the register map at build
time, one write per command: nothing expands (principle 4). Block-level
helpers (``RegisterMap.config_writes`` / ``start_writes``) are for Python
scripts that generate scenario files (scenarios/make.py), never for the
file itself.

Memory
------
``memory`` is a list of fills, applied in order before the run. Each has
``mem`` (``"l1"`` or ``"l2"``), a byte ``addr`` (the L1 one includes
``base_addr``) and exactly one source: ``data`` (inline words), ``npy`` (a
``.npy`` file relative to the scenario file) or ``random`` (``seed``, ``n``,
``low``, ``high``: ``default_rng(seed).integers(low, high, n)``, integers
only as D28 asks). Words are ``[n]`` or ``[n, elems_per_word]``.

Registries (principle 6, D43)
-----------------------------
* component kinds: ``register_component(kind, builder)``; built in:
  ``xbar``, ``streamer``, ``dma``, ``accel``, ``controller``;
* accelerator kinds: ``register_accel(kind, factory)``, the factory takes
  the scenario's ``params`` as keyword arguments and returns an
  ``AccelConfig``; built in: ``elementwise`` and ``reduce`` (the stubs,
  params are the stubs' own keyword arguments with ``op`` by name);
* ops: ``register_op(name, fn)``; built in: add, sub, mul, min, max, and,
  or, xor.

Output (D44)
------------
``run`` returns a ``RunResult``; ``write_outputs`` writes into a directory:

    run.json         scenario, trace_level, total_cycles, register_map, reads
    profile.json     Profile.to_dict()
    trace.jsonl      one event per line (task or beat level only)
    trace_meta.json  Trace.to_dict() without events (task or beat level only)
    l1.npy           final L1, flat words [n_words, elems_per_word], address order
    l2.npy           final L2, same layout (only with an L2)

The files are byte-identical on every run and with skipping on and off
(D29, D38, D40): nothing written depends on the skip mode, JSON keeps
insertion order and ``np.save`` is deterministic. Files of a kind the run
does not produce (a trace at level off, l2.npy without an L2) are removed,
so the directory always matches its run.json.

Contracts are written down in MOD10; the dataclasses here are kept minimal
(D26). None of this changes a component: the builder calls the same
constructors, ``attach``, ``RegisterMap`` and ``Controller`` a hand-written
test would, in the same order.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from .accel import AccelConfig, Accelerator, elementwise_stub, reduce_stub
from .cluster import Cluster
from .ctrl import (
    Command,
    Controller,
    ControllerConfig,
    CsrRead,
    CsrWrite,
    RegisterMap,
    Wait,
    command_from_dict,
)
from .dma import Dma, DmaConfig
from .l2 import L2Config, L2Memory
from .mem import L1Config, L1Memory
from .profile import Profile, build_profile
from .sched import Component
from .streamer import Streamer, StreamerConfig
from .trace import LEVELS, Trace
from .xbar import Xbar

# =============================================================================
# Errors
# =============================================================================


class ScenarioError(ValueError):
    """A scenario that cannot be built. Simulation errors are raised as they are."""


class UnknownComponentKind(ScenarioError):
    """A component ``kind`` that is not registered."""


class UnknownBlockError(ScenarioError):
    """A block name that is not in the register map (or not built before the controller)."""


class UnknownRegisterError(ScenarioError):
    """A register name or address that is not in the register map."""


class UnknownAccelKind(ScenarioError):
    """An accelerator ``accel`` kind that is not registered."""


class UnknownOpError(ScenarioError):
    """An ``op`` name that is not registered."""


class PortAttachError(ScenarioError):
    """An accelerator port attached wrongly: unknown, missing, doubled or mismatched."""


class MemoryInitError(ScenarioError):
    """A memory fill outside L1 / L2, misaligned, or with a bad source."""


# =============================================================================
# Helpers
# =============================================================================


def _check_keys(d: Mapping[str, Any], allowed: Sequence[str], what: str) -> None:
    """Reject unknown keys, so a typo never silently falls back to a default."""
    unknown = sorted(set(d) - set(allowed))
    if unknown:
        raise ScenarioError(f"{what}: unknown keys {unknown} (allowed: {list(allowed)})")


def _plain(v: Any) -> Any:
    """Tuples -> lists, mappings -> dicts, recursively: JSON-ready values."""
    if isinstance(v, Mapping):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Every field of a flat config dataclass (L1Config, StreamerConfig, ...).

    Every field is written, defaults included, so a written file says
    everything. Moves next to each config class in MOD10 / F2.
    """
    return {f.name: _plain(getattr(cfg, f.name)) for f in fields(cfg)}


def config_from_dict(cls: type, d: Mapping[str, Any], what: str) -> Any:
    """Inverse of ``config_to_dict``; missing keys take the class default."""
    _check_keys(d, [f.name for f in fields(cls)], what)
    try:
        return cls(**d)
    except (TypeError, ValueError) as e:
        raise ScenarioError(f"{what}: {e}") from e


JSON_WIDTH = 100  # line length of written JSON, as ruff's for the code


def _scalar(v: Any) -> bool:
    return not isinstance(v, (dict, list, tuple))


def _json(v: Any, indent: int, col: int) -> str:
    """One value, its block indented by ``indent``, starting at column ``col``.

    A container goes on one line if it ends within JSON_WIDTH, and a list of
    scalars always does (a histogram stays one line); otherwise one item per
    line, one space deeper.
    """
    flat = json.dumps(v)
    if _scalar(v) or (isinstance(v, (list, tuple)) and all(map(_scalar, v))):
        return flat
    if col + len(flat) <= JSON_WIDTH or not v:
        return flat
    pad = " " * (indent + 1)
    if isinstance(v, dict):
        items = []
        for k, x in v.items():
            head = f"{pad}{json.dumps(k)}: "
            items.append(head + _json(x, indent + 1, len(head)))
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    items = [pad + _json(x, indent + 1, len(pad)) for x in v]
    return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"


def to_json(obj: Any) -> str:
    """The one JSON format of every written file (D44): insertion order, small
    containers on one line (so one command per line), final newline.
    Deterministic, and ``json.loads`` gives ``obj`` back."""
    return _json(obj, 0, 0) + "\n"


# =============================================================================
# Registries: ops, accelerator kinds, component kinds
# =============================================================================

OPS: dict[str, Callable[[Any, Any], Any]] = {}


def register_op(name: str, fn: Callable[[Any, Any], Any]) -> None:
    """Make ``fn`` usable as ``"op": name`` in accelerator params."""
    if name in OPS:
        raise ValueError(f"op {name!r} already registered")
    OPS[name] = fn


for _name, _fn in (
    ("add", np.add),
    ("sub", np.subtract),  # folds left to right: a - b - c
    ("mul", np.multiply),
    ("min", np.minimum),
    ("max", np.maximum),
    ("and", np.bitwise_and),
    ("or", np.bitwise_or),
    ("xor", np.bitwise_xor),
):
    register_op(_name, _fn)


def _op(name: Any) -> Callable[[Any, Any], Any]:
    if name not in OPS:
        raise UnknownOpError(f"unknown op {name!r} (registered: {sorted(OPS)})")
    return OPS[name]


AccelFactory = Callable[..., AccelConfig]
ACCEL_KINDS: dict[str, AccelFactory] = {}


def register_accel(kind: str, factory: AccelFactory) -> None:
    """Make ``"accel": kind`` buildable; ``factory(**params)`` returns an AccelConfig."""
    if kind in ACCEL_KINDS:
        raise ValueError(f"accelerator kind {kind!r} already registered")
    ACCEL_KINDS[kind] = factory


def _with_op(stub: Callable[..., AccelConfig]) -> AccelFactory:
    """The stub with ``op`` given by name instead of as a function (default: add)."""

    def factory(**params: Any) -> AccelConfig:
        return stub(**{**params, "op": _op(params.get("op", "add"))})

    return factory


register_accel("elementwise", _with_op(elementwise_stub))
register_accel("reduce", _with_op(reduce_stub))


@dataclass
class BuildContext:
    """What component builders see: the cluster, the memories and what is built so far."""

    cluster: Cluster
    l1: L1Memory
    l2: L2Memory | None
    program: list[ScenarioCommand]
    regmap_spec: RegisterMapSpec
    xbar: Xbar | None = None
    comps: dict[str, Component] = field(default_factory=dict)
    regmap: RegisterMap | None = None
    controller: Controller | None = None


ComponentBuilder = Callable[["ComponentSpec", BuildContext], Component]
COMPONENT_KINDS: dict[str, ComponentBuilder] = {}


def register_component(kind: str, builder: ComponentBuilder) -> None:
    """Make ``"kind": kind`` buildable. The builder returns the component; the
    caller adds it to the cluster, so registration order stays the file order."""
    if kind in COMPONENT_KINDS:
        raise ValueError(f"component kind {kind!r} already registered")
    COMPONENT_KINDS[kind] = builder


def _need_xbar(spec: ComponentSpec, ctx: BuildContext) -> Xbar:
    if ctx.xbar is None:
        raise ScenarioError(f"{spec.name}: the xbar must come before it in the component list")
    return ctx.xbar


def _build_xbar(spec: ComponentSpec, ctx: BuildContext) -> Component:
    if ctx.xbar is not None:
        raise ScenarioError(f"{spec.name}: only one xbar per cluster, already have {ctx.xbar.name}")
    _check_keys(spec.config, ["check_hold"], f"{spec.name} config")
    ctx.xbar = Xbar(spec.name, ctx.l1, check_hold=bool(spec.config.get("check_hold", True)))
    return ctx.xbar


def _build_streamer(spec: ComponentSpec, ctx: BuildContext) -> Component:
    cfg = config_from_dict(StreamerConfig, spec.config, f"{spec.name} config")
    return Streamer(spec.name, _need_xbar(spec, ctx), cfg)


def _build_dma(spec: ComponentSpec, ctx: BuildContext) -> Component:
    if ctx.l2 is None:
        raise ScenarioError(f"{spec.name}: a DMA needs an L2 in the cluster config")
    cfg = config_from_dict(DmaConfig, spec.config, f"{spec.name} config")
    try:
        return Dma(spec.name, _need_xbar(spec, ctx), ctx.l2, cfg)
    except ValueError as e:  # L1 / L2 disagree on word or beat shape
        raise ScenarioError(f"{spec.name}: {e}") from e


def _build_accel(spec: ComponentSpec, ctx: BuildContext) -> Component:
    if spec.accel not in ACCEL_KINDS:
        raise UnknownAccelKind(
            f"{spec.name}: unknown accelerator kind {spec.accel!r} "
            f"(registered: {sorted(ACCEL_KINDS)})"
        )
    try:
        cfg = ACCEL_KINDS[spec.accel](**spec.params)
    except ScenarioError:
        raise
    except (TypeError, ValueError) as e:
        raise ScenarioError(f"{spec.name}: params {spec.params}: {e}") from e
    acc = Accelerator(spec.name, ctx.cluster, cfg)
    # Every port attached exactly once, to a streamer built before the accelerator.
    ports = [p.name for p in cfg.ports]
    missing = [p for p in ports if p not in spec.attach]
    unknown = [p for p in spec.attach if p not in ports]
    if missing or unknown:
        raise PortAttachError(
            f"{spec.name}: ports {ports}; not attached {missing}, no such port {unknown}"
        )
    for port in ports:  # port order, so attach errors are reported deterministically
        target = spec.attach[port]
        s = ctx.comps.get(target)
        if not isinstance(s, Streamer):
            raise PortAttachError(
                f"{spec.name}.{port}: {target!r} is not a streamer built before {spec.name}"
            )
        try:
            acc.attach(port, s.fifo)
        except ValueError as e:  # lanes mismatch, reader used as output, FIFO taken
            raise PortAttachError(f"{spec.name}.{port} -> {target}: {e}") from e
    return acc


def _build_controller(spec: ComponentSpec, ctx: BuildContext) -> Component:
    if ctx.controller is not None:
        raise ScenarioError(f"{spec.name}: one controller per cluster, have {ctx.controller.name}")
    rm = ctx.regmap_spec
    names = rm.block_names(ctx.comps)
    for n in [*names, *rm.bases, *rm.spatial_bounds]:
        if n not in ctx.comps:
            raise UnknownBlockError(
                f"register map: block {n!r} is not a component listed before {spec.name}"
            )
    try:
        regmap = RegisterMap(
            [(n, ctx.comps[n]) for n in names],
            window=rm.window,
            bases=rm.bases,
            spatial_bounds={k: tuple(v) for k, v in rm.spatial_bounds.items()},
        )
    except (TypeError, ValueError) as e:
        raise ScenarioError(f"register map: {e}") from e
    program = resolve_program(ctx.program, regmap)
    cfg = config_from_dict(ControllerConfig, spec.config, f"{spec.name} config")
    try:
        ctl = Controller(spec.name, regmap, program, cfg)
    except ValueError as e:  # e.g. a write to a read-only register, start with a value != 1
        raise ScenarioError(f"program: {e}") from e
    ctx.regmap, ctx.controller = regmap, ctl
    return ctl


register_component("xbar", _build_xbar)
register_component("streamer", _build_streamer)
register_component("dma", _build_dma)
register_component("accel", _build_accel)
register_component("controller", _build_controller)


# =============================================================================
# Cluster configuration
# =============================================================================


@dataclass
class ComponentSpec:
    """One entry of ``components``. ``config`` stays a plain dict; the kind's builder
    turns it into its config class. Accelerators use ``accel``, ``params``, ``attach``."""

    name: str
    kind: str
    config: dict[str, Any] = field(default_factory=dict)
    accel: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    attach: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.kind == "accel":
            d |= {"accel": self.accel, "params": _plain(self.params), "attach": dict(self.attach)}
        else:
            d["config"] = _plain(self.config)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ComponentSpec:
        what = f"component {d.get('name', '?')!r}"
        if "name" not in d or "kind" not in d:
            raise ScenarioError(f"{what}: needs 'name' and 'kind'")
        if d["kind"] == "accel":
            _check_keys(d, ["name", "kind", "accel", "params", "attach"], what)
        else:
            _check_keys(d, ["name", "kind", "config"], what)
        return cls(
            name=str(d["name"]),
            kind=str(d["kind"]),
            config=dict(d.get("config", {})),
            accel=d.get("accel"),
            params=dict(d.get("params", {})),
            attach={str(k): str(v) for k, v in d.get("attach", {}).items()},
        )


# Kinds that are never register blocks by default (they have no adapter).
_NOT_BLOCKS = ("xbar", "controller")


@dataclass
class RegisterMapSpec:
    """Arguments of ``RegisterMap``. ``blocks = None``: every component except the
    xbar and the controller, in component order."""

    window: int = 32
    blocks: list[str] | None = None
    bases: dict[str, int] = field(default_factory=dict)
    spatial_bounds: dict[str, list[int]] = field(default_factory=dict)

    def block_names(self, comps: Mapping[str, Component]) -> list[str]:
        if self.blocks is not None:
            return list(self.blocks)
        return [n for n, c in comps.items() if not isinstance(c, (Xbar, Controller))]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"window": self.window}
        if self.blocks is not None:
            d["blocks"] = list(self.blocks)
        d["bases"] = dict(self.bases)
        d["spatial_bounds"] = {k: list(v) for k, v in self.spatial_bounds.items()}
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RegisterMapSpec:
        _check_keys(d, ["window", "blocks", "bases", "spatial_bounds"], "register_map")
        blocks = d.get("blocks")
        return cls(
            window=int(d.get("window", 32)),
            blocks=None if blocks is None else [str(b) for b in blocks],
            bases={str(k): int(v) for k, v in d.get("bases", {}).items()},
            spatial_bounds={
                str(k): [int(x) for x in v] for k, v in d.get("spatial_bounds", {}).items()
            },
        )


@dataclass
class ClusterConfig:
    """The hardware of a scenario: L1, optional L2, ordered components, register map."""

    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config | None = None
    components: list[ComponentSpec] = field(default_factory=list)
    register_map: RegisterMapSpec = field(default_factory=RegisterMapSpec)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"l1": config_to_dict(self.l1)}
        if self.l2 is not None:
            d["l2"] = config_to_dict(self.l2)
        d["components"] = [c.to_dict() for c in self.components]
        d["register_map"] = self.register_map.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ClusterConfig:
        _check_keys(d, ["l1", "l2", "components", "register_map"], "cluster")
        comps = [ComponentSpec.from_dict(c) for c in d.get("components", [])]
        names = [c.name for c in comps]
        if len(set(names)) != len(names):
            raise ScenarioError(f"cluster: duplicate component names {names}")
        return cls(
            l1=config_from_dict(L1Config, d.get("l1", {}), "l1"),
            l2=None if d.get("l2") is None else config_from_dict(L2Config, d["l2"], "l2"),
            components=comps,
            register_map=RegisterMapSpec.from_dict(d.get("register_map", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> ClusterConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(to_json(self.to_dict()))


# =============================================================================
# Memory fills
# =============================================================================

_SOURCES = ("data", "npy", "random")


@dataclass
class MemInit:
    """One fill of L1 or L2 from exactly one source (module doc)."""

    mem: str
    addr: int
    data: list[Any] | None = None
    npy: str | None = None
    random: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.mem not in ("l1", "l2"):
            raise MemoryInitError(f"memory fill: mem must be 'l1' or 'l2', got {self.mem!r}")
        given = [s for s in _SOURCES if getattr(self, s) is not None]
        if len(given) != 1:
            raise MemoryInitError(
                f"memory fill at {self.addr}: exactly one of {_SOURCES}, got {given}"
            )
        if self.random is not None:
            _check_keys(self.random, ["seed", "n", "low", "high"], "random fill")
            if set(self.random) != {"seed", "n", "low", "high"}:
                raise MemoryInitError("random fill needs seed, n, low and high")

    def words(self, dtype: str, epw: int, base_dir: Path | None) -> np.ndarray:
        """The fill as words [n, epw] in the memory's dtype."""
        if self.data is not None:
            arr = np.asarray(self.data)
        elif self.npy is not None:
            path = Path(self.npy) if base_dir is None else base_dir / self.npy
            if not path.is_file():
                raise MemoryInitError(f"memory fill: no .npy file {path}")
            arr = np.load(path, allow_pickle=False)
        else:
            r = self.random or {}
            gen = np.random.default_rng(int(r["seed"]))
            arr = gen.integers(int(r["low"]), int(r["high"]), int(r["n"]))
        if not np.can_cast(arr.dtype, np.dtype(dtype), casting="same_kind"):
            raise MemoryInitError(
                f"memory fill at {self.addr}: {arr.dtype} data for a {dtype} memory"
            )
        if (
            arr.ndim not in (1, 2)
            or (arr.ndim == 2 and arr.shape[1] != epw)
            or (arr.ndim == 1 and epw != 1)
        ):
            raise MemoryInitError(
                f"memory fill at {self.addr}: shape {arr.shape}, need [n] or [n, {epw}]"
            )
        return arr.astype(dtype).reshape(-1, epw)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"mem": self.mem, "addr": self.addr}
        for s in _SOURCES:
            v = getattr(self, s)
            if v is not None:
                d[s] = _plain(v)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> MemInit:
        _check_keys(d, ["mem", "addr", *_SOURCES], "memory fill")
        if "mem" not in d or "addr" not in d:
            raise MemoryInitError(f"memory fill needs 'mem' and 'addr': {dict(d)}")
        return cls(
            mem=str(d["mem"]),
            addr=int(d["addr"]),
            data=None if d.get("data") is None else list(d["data"]),
            npy=d.get("npy"),
            random=None if d.get("random") is None else dict(d["random"]),
        )


def _fill(fill: MemInit, l1: L1Memory, l2: L2Memory | None, base_dir: Path | None) -> None:
    """Apply one fill. Checked here, so the error names the fill, not a bank."""
    if fill.mem == "l2" and l2 is None:
        raise MemoryInitError(f"memory fill at {fill.addr}: the cluster has no L2")
    cfg = l1.cfg if fill.mem == "l1" else l2.cfg  # type: ignore[union-attr]
    words = fill.words(cfg.dtype, cfg.elems_per_word, base_dir)
    size = cfg.size_bytes
    lo, hi = fill.addr - cfg.base_addr, fill.addr - cfg.base_addr + len(words) * cfg.word_bytes
    if lo % cfg.word_bytes or lo < 0 or hi > size:
        raise MemoryInitError(
            f"{fill.mem} fill [{fill.addr}, {fill.addr + hi - lo}) is misaligned or outside "
            f"[{cfg.base_addr}, {cfg.base_addr + size})"
        )
    (l1 if fill.mem == "l1" else l2).load(fill.addr, words)  # type: ignore[union-attr]


# =============================================================================
# Program: raw or named registers, one command each (D42)
# =============================================================================


@dataclass(frozen=True)
class NamedWrite:
    """``csr_write`` by register name, e.g. ``"dma.src_base"``."""

    reg: str
    value: int

    def to_dict(self) -> dict[str, Any]:
        return {"op": "csr_write", "reg": self.reg, "value": self.value}


@dataclass(frozen=True)
class NamedRead:
    """``csr_read`` by register name."""

    reg: str

    def to_dict(self) -> dict[str, Any]:
        return {"op": "csr_read", "reg": self.reg}


ScenarioCommand = CsrWrite | CsrRead | Wait | NamedWrite | NamedRead


def scenario_command_from_dict(d: Mapping[str, Any]) -> ScenarioCommand:
    """A command in either register form; ``to_dict`` gives the same form back."""
    op = d.get("op")
    if op in ("csr_write", "csr_read"):
        allowed = ["op", "addr", "reg", "value"] if op == "csr_write" else ["op", "addr", "reg"]
        _check_keys(d, allowed, f"command {dict(d)}")
        if ("addr" in d) == ("reg" in d):
            raise ScenarioError(f"command {dict(d)}: give exactly one of 'addr' and 'reg'")
        if op == "csr_write" and "value" not in d:
            raise ScenarioError(f"command {dict(d)}: csr_write needs a 'value'")
        if "reg" in d:
            reg = str(d["reg"])
            return NamedWrite(reg, int(d["value"])) if op == "csr_write" else NamedRead(reg)
    elif op == "wait":
        _check_keys(d, ["op", "block", "mode"], f"command {dict(d)}")
    try:
        return command_from_dict(d)
    except (KeyError, ValueError) as e:
        raise ScenarioError(f"command {dict(d)}: {e}") from e


def _addr_of(reg: str, regmap: RegisterMap, where: str) -> int:
    block, dot, _ = reg.partition(".")
    if block not in regmap.blocks:
        raise UnknownBlockError(f"{where}: no block {block!r} (blocks: {list(regmap.blocks)})")
    if not dot:
        raise UnknownRegisterError(f"{where}: register {reg!r} is not 'block.register'")
    try:
        return regmap.addr(reg)
    except KeyError as e:
        raise UnknownRegisterError(f"{where}: {e.args[0]}") from e


def resolve_program(program: Sequence[ScenarioCommand], regmap: RegisterMap) -> list[Command]:
    """Names -> addresses, one command for one command; checks raw addresses and waits."""
    out: list[Command] = []
    for pc, cmd in enumerate(program):
        where = f"command {pc} ({cmd.to_dict()})"
        if isinstance(cmd, NamedWrite):
            out.append(CsrWrite(_addr_of(cmd.reg, regmap, where), cmd.value))
        elif isinstance(cmd, NamedRead):
            out.append(CsrRead(_addr_of(cmd.reg, regmap, where)))
        elif isinstance(cmd, Wait):
            if cmd.block not in regmap.blocks:
                raise UnknownBlockError(f"{where}: no block {cmd.block!r}")
            out.append(cmd)
        else:
            try:
                regmap.register(cmd.addr)
            except ValueError as e:
                raise UnknownRegisterError(f"{where}: {e}") from e
            out.append(cmd)
    return out


def named(cmd: Command, regmap: RegisterMap) -> ScenarioCommand:
    """A raw command in name form, for generators (scenarios/make.py)."""
    if isinstance(cmd, CsrWrite):
        return NamedWrite(regmap.describe(cmd.addr), cmd.value)
    if isinstance(cmd, CsrRead):
        return NamedRead(regmap.describe(cmd.addr))
    return cmd


# =============================================================================
# Scenario
# =============================================================================


@dataclass
class Scenario:
    """Inputs of one run (module doc). ``cluster_ref`` is the cluster path as written
    in the file (None = inline); ``base_dir`` resolves it and ``.npy`` paths."""

    name: str
    cluster: ClusterConfig
    memory: list[MemInit] = field(default_factory=list)
    program: list[ScenarioCommand] = field(default_factory=list)
    max_cycles: int | None = None
    cluster_ref: str | None = None
    base_dir: Path | None = field(default=None, compare=False)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.max_cycles is not None:
            d["max_cycles"] = self.max_cycles
        d["cluster"] = self.cluster_ref if self.cluster_ref is not None else self.cluster.to_dict()
        d["memory"] = [m.to_dict() for m in self.memory]
        d["program"] = [c.to_dict() for c in self.program]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], base_dir: str | Path | None = None) -> Scenario:
        _check_keys(d, ["name", "max_cycles", "cluster", "memory", "program"], "scenario")
        if "name" not in d or "cluster" not in d:
            raise ScenarioError("scenario needs 'name' and 'cluster'")
        base = None if base_dir is None else Path(base_dir)
        ref = d["cluster"] if isinstance(d["cluster"], str) else None
        if ref is not None:
            path = Path(ref) if base is None else base / ref
            if not path.is_file():
                raise ScenarioError(f"scenario {d['name']!r}: no cluster file {path}")
            cluster = ClusterConfig.load(path)
        else:
            cluster = ClusterConfig.from_dict(d["cluster"])
        mc = d.get("max_cycles")
        return cls(
            name=str(d["name"]),
            cluster=cluster,
            memory=[MemInit.from_dict(m) for m in d.get("memory", [])],
            program=[scenario_command_from_dict(c) for c in d.get("program", [])],
            max_cycles=None if mc is None else int(mc),
            cluster_ref=ref,
            base_dir=base,
        )

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()), base_dir=path.parent)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(to_json(self.to_dict()))


# =============================================================================
# Build and run
# =============================================================================


@dataclass
class Built:
    """A built, not yet run, cluster with its parts."""

    cluster: Cluster
    l1: L1Memory
    l2: L2Memory | None
    regmap: RegisterMap
    controller: Controller


def build_cluster(
    cfg: ClusterConfig,
    program: Sequence[ScenarioCommand] = (),
    skip_idle: bool = True,
    trace: Trace | None = None,
) -> Built:
    """Build the cluster: memories, then components in list order (module doc)."""
    cl = Cluster(skip_idle=skip_idle, trace=trace)
    l1 = L1Memory(cl, cfg.l1)
    l2 = None if cfg.l2 is None else L2Memory(cl, cfg.l2)
    ctx = BuildContext(cl, l1, l2, list(program), cfg.register_map)
    for spec in cfg.components:
        if spec.kind not in COMPONENT_KINDS:
            raise UnknownComponentKind(
                f"{spec.name}: unknown component kind {spec.kind!r} "
                f"(registered: {sorted(COMPONENT_KINDS)})"
            )
        comp = COMPONENT_KINDS[spec.kind](spec, ctx)
        ctx.comps[spec.name] = cl.add(comp)
    if ctx.controller is None or ctx.regmap is None:
        raise ScenarioError("cluster: needs one controller (kind 'controller'), listed last")
    return Built(cl, l1, l2, ctx.regmap, ctx.controller)


def build(scenario: Scenario, skip_idle: bool = True, trace: Trace | None = None) -> Built:
    """Build the scenario's cluster and apply its memory fills."""
    b = build_cluster(scenario.cluster, scenario.program, skip_idle, trace)
    for fill in scenario.memory:
        _fill(fill, b.l1, b.l2, scenario.base_dir)
    return b


def register_map_of(cfg: ClusterConfig) -> RegisterMap:
    """The register map of a cluster config, for generators that write programs."""
    return build_cluster(cfg).regmap


@dataclass
class RunResult:
    """Everything a run produces; ``write_outputs`` turns it into files."""

    scenario: str
    trace_level: str
    total_cycles: int
    profile: Profile
    trace: Trace | None
    regmap: RegisterMap
    reads: list[tuple[int, int, int]]  # (cycle, addr, value), as Controller.reads
    l1: np.ndarray  # [n_words, elems_per_word], address order
    l2: np.ndarray | None

    def run_info(self) -> dict[str, Any]:
        """Contents of run.json. No skip mode: output must not depend on it (D44)."""
        return {
            "scenario": self.scenario,
            "trace_level": self.trace_level,
            "total_cycles": self.total_cycles,
            "register_map": self.regmap.to_dict(),
            "reads": [
                {"cycle": c, "addr": a, "reg": self.regmap.describe(a), "value": v}
                for c, a, v in self.reads
            ],
        }


def run(
    scenario: Scenario,
    skip_idle: bool = True,
    trace_level: str = "off",
    max_cycles: int | None = None,
) -> RunResult:
    """Build, run until idle and collect the results.

    ``max_cycles`` overrides the scenario's; with neither, ``Cluster.run``'s
    default. A program that does not finish raises ``SimulationTimeout``.
    """
    if trace_level not in LEVELS:
        raise ScenarioError(f"trace level must be one of {LEVELS}, got {trace_level!r}")
    trace = None if trace_level == "off" else Trace(trace_level)
    b = build(scenario, skip_idle, trace)
    limit = max_cycles if max_cycles is not None else scenario.max_cycles
    total = b.cluster.run() if limit is None else b.cluster.run(max_cycles=limit)
    l1c = b.l1.cfg
    l1 = b.l1.dump(l1c.base_addr, l1c.size_bytes // l1c.word_bytes)
    l2 = None if b.l2 is None else b.l2.dump(b.l2.cfg.base_addr, b.l2.cfg.n_words)
    reads = [(int(c), int(a), int(v)) for c, a, v in b.controller.reads]
    return RunResult(
        scenario.name, trace_level, total, build_profile(b.cluster), trace, b.regmap, reads, l1, l2
    )


# =============================================================================
# Output files (D44)
# =============================================================================

RUN_FILE = "run.json"
PROFILE_FILE = "profile.json"
TRACE_FILE = "trace.jsonl"
TRACE_META_FILE = "trace_meta.json"
L1_FILE = "l1.npy"
L2_FILE = "l2.npy"


def _save_npy(path: Path, arr: np.ndarray) -> None:
    with path.open("wb") as f:
        np.save(f, np.ascontiguousarray(arr), allow_pickle=False)


def write_outputs(result: RunResult, out_dir: str | Path) -> Path:
    """Write the output files (module doc). Returns the directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / RUN_FILE).write_text(to_json(result.run_info()))
    (out / PROFILE_FILE).write_text(to_json(result.profile.to_dict()))
    if result.trace is not None:
        d = result.trace.to_dict()
        events = d.pop("events")
        (out / TRACE_META_FILE).write_text(to_json(d))
        (out / TRACE_FILE).write_text("".join(json.dumps(e) + "\n" for e in events))
    else:  # remove a stale trace, so the directory matches run.json
        for name in (TRACE_FILE, TRACE_META_FILE):
            (out / name).unlink(missing_ok=True)
    _save_npy(out / L1_FILE, result.l1)
    if result.l2 is not None:
        _save_npy(out / L2_FILE, result.l2)
    else:
        (out / L2_FILE).unlink(missing_ok=True)
    return out


@dataclass
class Outputs:
    """The output files read back: what views and tests use."""

    run: dict[str, Any]
    profile: Profile
    trace: Trace | None
    l1: np.ndarray
    l2: np.ndarray | None


def read_outputs(out_dir: str | Path) -> Outputs:
    """Inverse of ``write_outputs``."""
    out = Path(out_dir)
    trace = None
    if (out / TRACE_META_FILE).is_file():
        meta = json.loads((out / TRACE_META_FILE).read_text())
        lines = (out / TRACE_FILE).read_text().splitlines()
        meta["events"] = [json.loads(line) for line in lines]
        trace = Trace.from_dict(meta)
    l2 = np.load(out / L2_FILE) if (out / L2_FILE).is_file() else None
    return Outputs(
        run=json.loads((out / RUN_FILE).read_text()),
        profile=Profile.from_dict(json.loads((out / PROFILE_FILE).read_text())),
        trace=trace,
        l1=np.load(out / L1_FILE),
        l2=l2,
    )


__all__ = [
    "ACCEL_KINDS",
    "COMPONENT_KINDS",
    "OPS",
    "BuildContext",
    "Built",
    "ClusterConfig",
    "ComponentSpec",
    "MemInit",
    "MemoryInitError",
    "NamedRead",
    "NamedWrite",
    "Outputs",
    "PortAttachError",
    "RegisterMapSpec",
    "RunResult",
    "Scenario",
    "ScenarioError",
    "UnknownAccelKind",
    "UnknownBlockError",
    "UnknownComponentKind",
    "UnknownOpError",
    "UnknownRegisterError",
    "build",
    "build_cluster",
    "config_from_dict",
    "config_to_dict",
    "named",
    "read_outputs",
    "register_accel",
    "register_component",
    "register_map_of",
    "register_op",
    "resolve_program",
    "run",
    "scenario_command_from_dict",
    "to_json",
    "write_outputs",
]
