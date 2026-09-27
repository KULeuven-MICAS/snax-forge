"""The platform: the SNAX cluster around the accelerators (DP1a, D84).

A platform file describes everything of the cluster that is not the
accelerator itself: L1, L2, xbar, DMA, controller, the register window, how
the controller waits, and the streamer shell. It never names a recipe, and a
recipe never names it; the design step pairs a bound graph with a platform.

    {
     "name":     "small16",
     "base":     null,              the platform this one was made from
     "changes":  {},                what differs from ``base``, path -> value
     "l1":       {...},             L1Config
     "l2":       {...} or null,     L2Config; no L2, no DMA
     "xbar":     {"check_hold": true},
     "dma":      {...},             DmaConfig, used when there is an L2
     "controller": {...},           ControllerConfig
     "register_window": 32,
     "wait_mode": "poll",           how the controller waits for a task
     "streamers": {
      "default": {"temporal_dims": 1, "fifo_depth": 2, "addr_depth": 8, "prio": 0},
      "acc_out": {"fifo_depth": 4}
     }
    }

**The streamer shell.** SNAX's streamers are a shell shaped to what an
accelerator needs, so the platform does not list them: one streamer is made
for every port of every accelerated node, named ``<instance>_<port>`` (D75).
Its direction and lanes come from the BRM port (``write``, ``n_ports`` and,
unless an entry gives them, ``spatial_bounds``); ``default`` gives the rest,
and an entry named after the streamer overrides any of it for that one
streamer only. Resolving the shell against a graph is streamers.py; an entry
that matches no port is a check (``connect.streamer_key``), not a load error,
because only the graph knows the ports.

**Changes and working copies.** ``with_changes`` applies ``--set`` paths
(``l1.n_banks``, ``streamers.acc_a.temporal_dims``) and returns a working
copy: complete values, ``base`` the platform it all started from, and
``changes`` every path set since, in order. Continuing from a working copy
adds to its changes, so ``changes`` is always the difference from ``base``.
``saved_as`` renames a working copy for ``design save``.

Every field is written; missing keys take their defaults (``l2`` missing is
no L2); unknown keys and values of the wrong type are errors. Loading
collects every problem it finds (``DesignError``) instead of stopping at the
first.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from snax_forge.snax_model import (
    ControllerConfig,
    DmaConfig,
    L1Config,
    L2Config,
    StreamerConfig,
)
from snax_forge.snax_model.config import Config, plain, to_json

from .problems import DesignError, Problem

KEYS = ("name", "base", "changes", "l1", "l2", "xbar", "dma", "controller",
        "register_window", "wait_mode", "streamers")  # fmt: skip
WAIT_MODES = ("poll", "signal")
XBAR_KEYS = ("check_hold",)
# Streamer fields the port decides; an entry may not set them (connect.derived).
DERIVED = ("write", "n_ports")
# Paths the tools write; --set may not change them.
PROVENANCE = ("name", "base", "changes")


@dataclass(frozen=True)
class StreamerOptions(Config):
    """The streamer knobs a platform sets: StreamerConfig without what the port decides."""

    temporal_dims: int = 1
    fifo_depth: int = 8
    addr_depth: int = 8
    prio: int = 0

    def __post_init__(self) -> None:
        self.config(write=False, n_ports=1)  # StreamerConfig's own checks

    def config(self, write: bool, n_ports: int) -> StreamerConfig:
        """The model's StreamerConfig for a port with this direction and lane count."""
        return StreamerConfig(write=write, n_ports=n_ports, **self.to_dict())


OPTIONS = tuple(f.name for f in fields(StreamerOptions))
ENTRY_KEYS = (*OPTIONS, "spatial_bounds")


@dataclass(frozen=True)
class Platform:
    """A parsed platform file; see the module doc."""

    name: str
    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config | None = None
    xbar: dict[str, Any] = field(default_factory=lambda: {"check_hold": True})
    dma: DmaConfig = field(default_factory=DmaConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    register_window: int = 32
    wait_mode: str = "poll"
    default: StreamerOptions = field(default_factory=StreamerOptions)
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)  # per streamer, as written
    base: str | None = None
    changes: dict[str, Any] = field(default_factory=dict)

    # --- the streamer shell ---

    def options(self, streamer: str) -> StreamerOptions:
        """``default`` with ``streamer``'s entry on top (its spatial bounds aside)."""
        entry = {k: v for k, v in self.entries.get(streamer, {}).items() if k in OPTIONS}
        return StreamerOptions.from_dict({**self.default.to_dict(), **entry})

    def spatial_bounds(self, streamer: str) -> list[int] | None:
        """The entry's spatial bounds, or None when the port's nest decides them."""
        sb = self.entries.get(streamer, {}).get("spatial_bounds")
        return None if sb is None else list(sb)

    # --- changes ---

    def with_changes(self, sets: Sequence[tuple[str, Any]] = ()) -> Platform:
        """The working copy with ``sets`` (path, value) applied; see the module doc."""
        d = self.to_dict()
        problems = [p for path, value in sets for p in _apply(d, path, value)]
        if problems:
            raise DesignError(problems, "platform --set")
        new = Platform.from_dict(d)
        changes = dict(self.changes)
        for path, value in sets:
            changes.pop(path, None)
            changes[path] = value
        return replace(new, base=self.base or self.name, changes=changes)

    def saved_as(self, name: str) -> Platform:
        """The same platform under a new name, keeping ``base`` and ``changes`` as its record."""
        _ident(name, "platform name")
        return replace(self, name=name)

    # --- files ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base": self.base,
            "changes": plain(self.changes),
            "l1": self.l1.to_dict(),
            "l2": None if self.l2 is None else self.l2.to_dict(),
            "xbar": dict(self.xbar),
            "dma": self.dma.to_dict(),
            "controller": self.controller.to_dict(),
            "register_window": self.register_window,
            "wait_mode": self.wait_mode,
            "streamers": {"default": self.default.to_dict(), **plain(self.entries)},
        }

    @classmethod
    def from_dict(cls, d: Any) -> Platform:
        """Parse a platform; raises DesignError holding every problem."""
        platform, problems = parse(d)
        if problems:
            raise DesignError(problems, "platform")
        assert platform is not None
        return platform

    def to_json(self) -> str:
        return to_json(self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> Platform:
        try:
            d = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise DesignError([Problem("platform.keys", str(path), str(e))], "platform") from None
        return cls.from_dict(d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())


# =============================================================================
# Parsing
# =============================================================================


def _ident(name: Any, what: str) -> None:
    if not isinstance(name, str) or not name.isidentifier():
        raise DesignError([Problem("platform.keys", what, f"{name!r} is not an identifier")])


def _kind_ok(old: Any, new: Any) -> bool:
    """``new`` has the JSON type of ``old`` (a bool is not an int)."""
    if isinstance(old, bool) or isinstance(new, bool):
        return isinstance(old, bool) and isinstance(new, bool)
    if isinstance(old, Mapping):
        return isinstance(new, Mapping)
    if isinstance(old, (list, tuple)):
        return isinstance(new, (list, tuple))
    return type(old) is type(new)


def _type_name(v: Any) -> str:
    if isinstance(v, bool):
        return "a bool"
    if isinstance(v, Mapping):
        return "an object"
    if isinstance(v, (list, tuple)):
        return "a list"
    return {int: "an int", str: "a string", float: "a number"}.get(type(v), type(v).__name__)


def _section(cls: type, d: Any, sec: str, out: list[Problem]) -> Any:
    """``cls`` from ``d``: unknown keys and wrong types are platform.keys, the class's
    own rejections platform.values. None when anything is wrong."""
    if not isinstance(d, Mapping):
        out.append(Problem("platform.keys", sec, f"must be an object, got {d!r}"))
        return None
    names = [f.name for f in fields(cls)]
    unknown = sorted(set(d) - set(names))
    if unknown:
        out.append(Problem("platform.keys", sec, f"unknown keys {unknown} (allowed: {names})"))
    ref = cls()
    bad = False
    for k in names:
        if k in d and not _kind_ok(getattr(ref, k), d[k]):
            want = _type_name(getattr(ref, k))
            out.append(Problem("platform.keys", f"{sec}.{k}", f"must be {want}, got {d[k]!r}"))
            bad = True
    if unknown or bad:
        return None
    try:
        return cls.from_dict(dict(d))
    except (TypeError, ValueError) as e:
        out.append(Problem("platform.values", sec, str(e)))
        return None


def _entry(name: str, e: Any, default: StreamerOptions | None, out: list[Problem]) -> None:
    """Checks of one streamer entry; ``write`` and ``n_ports`` are left to connect.derived."""
    where = f"streamers.{name}"
    if not name.isidentifier():
        out.append(Problem("platform.keys", where, f"{name!r} is not a streamer name"))
        return
    if not isinstance(e, Mapping):
        out.append(Problem("platform.keys", where, f"must be an object, got {e!r}"))
        return
    unknown = sorted(set(e) - set(ENTRY_KEYS) - set(DERIVED))
    if unknown:
        out.append(
            Problem("platform.keys", where, f"unknown keys {unknown} (allowed: {list(ENTRY_KEYS)})")
        )
    for k in OPTIONS:
        if k in e and (isinstance(e[k], bool) or type(e[k]) is not int):
            out.append(Problem("platform.keys", f"{where}.{k}", f"must be an int, got {e[k]!r}"))
            return
    sb = e.get("spatial_bounds")
    if sb is not None and (
        not isinstance(sb, (list, tuple))
        or not sb
        or any(isinstance(x, bool) or type(x) is not int or x < 1 for x in sb)
    ):
        out.append(
            Problem(
                "platform.streamer", f"{where}.spatial_bounds",
                f"must be a non-empty list of ints >= 1, got {sb!r}",
            )
        )  # fmt: skip
    if default is not None and not unknown:
        opts = {k: v for k, v in e.items() if k in OPTIONS}
        try:
            StreamerOptions.from_dict({**default.to_dict(), **opts})
        except ValueError as err:
            out.append(Problem("platform.streamer", where, str(err)))


def parse(d: Any) -> tuple[Platform | None, list[Problem]]:
    """A platform and every problem found; the platform is None when there is one."""
    out: list[Problem] = []
    if not isinstance(d, Mapping):
        return None, [Problem("platform.keys", "platform", f"must be an object, got {d!r}")]
    unknown = sorted(set(d) - set(KEYS))
    if unknown:
        out.append(
            Problem("platform.keys", "platform", f"unknown keys {unknown} (allowed: {list(KEYS)})")
        )
    name = d.get("name")
    if not isinstance(name, str) or not name.isidentifier():
        out.append(Problem("platform.keys", "name", f"must be an identifier, got {name!r}"))
    base = d.get("base")
    if base is not None and not isinstance(base, str):
        out.append(
            Problem("platform.keys", "base", f"must be a platform name or null, got {base!r}")
        )
    changes = d.get("changes", {})
    if not isinstance(changes, Mapping):
        out.append(Problem("platform.keys", "changes", f"must be an object, got {changes!r}"))

    l1 = _section(L1Config, d.get("l1", {}), "l1", out)
    l2 = None if d.get("l2") is None else _section(L2Config, d["l2"], "l2", out)
    dma = _section(DmaConfig, d.get("dma", {}), "dma", out)
    ctl = _section(ControllerConfig, d.get("controller", {}), "controller", out)

    xbar = d.get("xbar", {"check_hold": True})
    if not isinstance(xbar, Mapping) or set(xbar) - set(XBAR_KEYS):
        out.append(
            Problem("platform.keys", "xbar", f"allowed keys {list(XBAR_KEYS)}, got {xbar!r}")
        )
    elif not isinstance(xbar.get("check_hold", True), bool):
        out.append(Problem("platform.keys", "xbar.check_hold", "must be a bool"))
    window = d.get("register_window", 32)
    if isinstance(window, bool) or type(window) is not int:
        out.append(Problem("platform.keys", "register_window", f"must be an int, got {window!r}"))
    elif window < 1 or window & (window - 1):
        out.append(Problem("platform.values", "register_window", f"{window} is not a power of two"))
    wait_mode = d.get("wait_mode", "poll")
    if wait_mode not in WAIT_MODES:
        out.append(
            Problem(
                "platform.values",
                "wait_mode",
                f"must be one of {list(WAIT_MODES)}, got {wait_mode!r}",
            )
        )

    streamers = d.get("streamers", {})
    default, entries = None, {}
    if not isinstance(streamers, Mapping):
        out.append(Problem("platform.keys", "streamers", f"must be an object, got {streamers!r}"))
    else:
        n = len(out)
        default = _section(StreamerOptions, streamers.get("default", {}), "streamers.default", out)
        for p in out[n:]:  # a rejected default value is a shell problem
            if p.code == "platform.values":
                out[out.index(p)] = replace(p, code="platform.streamer")
        for k, e in streamers.items():
            if k != "default":
                _entry(k, e, default, out)
                entries[k] = plain(e)

    if out:
        return None, out
    return Platform(
        name=name, l1=l1, l2=l2, xbar={"check_hold": bool(xbar.get("check_hold", True))},
        dma=dma, controller=ctl, register_window=window, wait_mode=wait_mode,
        default=default, entries=entries, base=base, changes=dict(changes),
    ), []  # fmt: skip


# =============================================================================
# --set
# =============================================================================


def parse_value(text: str) -> Any:
    """A --set value: JSON (``32``, ``true``, ``[2, 2]``, ``"poll"``) or else the plain text."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _apply(d: dict[str, Any], path: str, value: Any) -> list[Problem]:
    """Set ``path`` in the platform dict ``d`` (a ``to_dict``); the problems if it cannot."""
    where = f"platform.{path}"
    parts = path.split(".")
    if not path or any(not p for p in parts):
        return [Problem("platform.keys", where, "not a path")]
    if parts[0] in PROVENANCE:
        return [Problem("platform.keys", where, "is written by the tools, not by --set")]
    if parts[0] == "streamers":
        if len(parts) != 3:
            return [Problem("platform.keys", where, "give streamers.<streamer>.<option>")]
        _, s, opt = parts
        allowed = OPTIONS if s == "default" else ENTRY_KEYS
        if opt in DERIVED:
            return [
                Problem(
                    "platform.keys", where,
                    f"{opt!r} comes from the accelerator port (direction, lanes), not the platform",
                )
            ]  # fmt: skip
        if opt not in allowed:
            return [Problem("platform.keys", where, f"unknown option (allowed: {list(allowed)})")]
        old = [1] if opt == "spatial_bounds" else d["streamers"]["default"][opt]
        if not _kind_ok(old, value):
            return [Problem("platform.keys", where, f"must be {_type_name(old)}, got {value!r}")]
        if not s.isidentifier():
            return [Problem("platform.keys", where, f"{s!r} is not a streamer name")]
        d["streamers"].setdefault(s, {})[opt] = value
        return []
    node: Any = d
    for i, p in enumerate(parts[:-1]):
        if not isinstance(node, Mapping) or p not in node:
            return [
                Problem("platform.keys", where, f"no {'.'.join(parts[: i + 1])!r} in the platform")
            ]
        node = node[p]
        if node is None:
            return [
                Problem(
                    "platform.keys", where, f"the platform has no {p}",
                    fix=f"start from a platform with an {p}" if p == "l2" else None,
                )
            ]  # fmt: skip
    last = parts[-1]
    if not isinstance(node, Mapping):
        return [
            Problem("platform.keys", where, f"{'.'.join(parts[:-1])!r} is a value, not a section")
        ]
    if last not in node:
        keys = list(node) if isinstance(node, Mapping) else []
        close = difflib.get_close_matches(last, keys, n=1)
        fix = f"did you mean {'.'.join([*parts[:-1], close[0]])}?" if close else None
        return [Problem("platform.keys", where, f"no such field (fields here: {keys})", fix)]
    if not _kind_ok(node[last], value):
        return [Problem("platform.keys", where, f"must be {_type_name(node[last])}, got {value!r}")]
    node[last] = value
    return []
