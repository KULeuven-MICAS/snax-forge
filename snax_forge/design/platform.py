"""The platform: the SNAX cluster around the accelerators (D84).

A platform file describes everything of the cluster that is not the
accelerator itself: L1, L2, how the program waits, and the streamer shell. It
never names a recipe, and a recipe never names it; the design step pairs a
bound graph with a platform.

    {
     "name":     "small16",
     "base":     null,              the platform this one was made from
     "changes":  {},                what differs from ``base``, path -> value
     "hjson":    null,              the source cluster file, a path from
                                    snax-forge's root (``design from-hjson``)
     "accelerator_slot": null,      the generated accelerator's name
     "l1":       {...},             L1Config
     "l2":       {...} or null,     L2Config; no L2, no DMA
     "wait_mode": "poll",           how the program waits for a task
     "streamers": {
      "default": {"temporal_dims": 1, "fifo_depth": 2},
      "acc_out": {"fifo_depth": 4}
     }
    }

``L1Config`` and ``L2Config`` hold only what the design step, SNAX-LOWER and
the generator read (CONTRACTS.md section 13); each checks its own values.

**The streamer shell.** SNAX's streamers are a shell shaped to what an
accelerator needs, so the platform does not list them: one streamer is made
for every port of every accelerated node, named ``<instance>_<port>`` (D75).
Its direction and lanes come from the BRM port (``write``, ``n_ports`` and,
unless an entry gives them, ``spatial_bounds``); ``default`` gives the rest,
and an entry named after the streamer overrides any of it for that one
streamer only. Resolving the shell against a graph is streamers.py; an entry
that matches no port is a check (``connect.streamer_key``), not a load error,
because only the graph knows the ports.

**The source hjson.** A platform read from snax_cluster's cluster file
(hjson_platform.py) keeps the file it came from (``hjson``, a copy in
``platforms/``) and the generated accelerator's name (``accelerator_slot``,
``snax_forge``): the GVSoC generator writes the design's cluster file from
that source, its only slot renamed to it and set (snax_forge/gen). Both are
null for a platform not read from one (small16). ``--set
accelerator_slot=NAME`` changes it as any field.

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

import numpy as np

from snax_forge.common import Config, plain, to_json

from .problems import DesignError, Problem

KEYS = ("name", "base", "changes", "hjson", "accelerator_slot", "l1", "l2", "wait_mode",
        "streamers")  # fmt: skip
WAIT_MODES = ("poll", "signal")
# Streamer fields the port decides; an entry may not set them (connect.derived).
DERIVED = ("write", "n_ports")
# Paths the tools write; --set may not change them.
PROVENANCE = ("name", "base", "changes")
# Top-level fields that are a string or null (the source hjson's)
NULLABLE = ("hjson", "accelerator_slot")


def _fits(dtype: str, elems_per_word: int, width_bits: int) -> None:
    """``elems_per_word`` elements of ``dtype`` fit one word of ``width_bits``."""
    if elems_per_word < 1:
        raise ValueError("elems_per_word must be >= 1")
    if np.dtype(dtype).itemsize * 8 * elems_per_word > width_bits:
        raise ValueError(f"{elems_per_word} x {dtype} does not fit in {width_bits} bits")


@dataclass(frozen=True)
class L1Config(Config):
    """The L1: single-port banks of one word each, word-interleaved (CONTRACTS.md 13)."""

    n_banks: int = 32
    width_bits: int = 64  # bits per bank word
    rows: int = 512  # words per bank; 32 * 512 * 8 B = 128 KiB
    dtype: str = "int64"  # element type, any NumPy dtype name
    elems_per_word: int = 1  # 1 only (open item 21)
    base_addr: int = 0  # byte address of the first L1 word
    wide_bits: int = 512  # widest port, the DMA's: one superbank of 8 banks in SNAX

    def __post_init__(self) -> None:
        if self.n_banks < 1 or self.rows < 1:
            raise ValueError("n_banks and rows must be >= 1")
        if self.width_bits % 8:
            raise ValueError("width_bits must be a multiple of 8")
        _fits(self.dtype, self.elems_per_word, self.width_bits)
        g, rest = divmod(self.wide_bits, self.width_bits)
        if self.wide_bits < 1 or rest or g & (g - 1):
            raise ValueError(
                f"wide_bits {self.wide_bits} must be a power-of-two multiple of the bank width "
                f"{self.width_bits}"
            )


@dataclass(frozen=True)
class L2Config(Config):
    """The L2: flat, moved by the DMA in beats (CONTRACTS.md 13)."""

    size_bytes: int = 1 << 20
    base_addr: int = 0  # byte address of the first L2 word
    beat_bits: int = 512  # one DMA beat; must equal L1Config.wide_bits
    width_bits: int = 64  # word = the L1 bank word (D13)
    dtype: str = "int64"  # element type, as in the L1
    elems_per_word: int = 1  # as in the L1

    def __post_init__(self) -> None:
        if self.width_bits % 8 or self.width_bits < 8:
            raise ValueError("width_bits must be a positive multiple of 8")
        if self.beat_bits < self.width_bits or self.beat_bits % self.width_bits:
            raise ValueError("beat_bits must be a multiple of width_bits")
        if self.size_bytes < 1 or self.size_bytes % self.beat_bytes:
            raise ValueError(f"size_bytes must be a positive multiple of {self.beat_bytes}")
        _fits(self.dtype, self.elems_per_word, self.width_bits)

    @property
    def beat_bytes(self) -> int:
        """Bytes per beat; also the alignment of every DMA access."""
        return self.beat_bits // 8


@dataclass(frozen=True)
class StreamerOptions(Config):
    """The streamer knobs a platform sets; the port decides the rest (``write``, ``n_ports``)."""

    temporal_dims: int = 1
    fifo_depth: int = 8

    def __post_init__(self) -> None:
        if min(self.temporal_dims, self.fifo_depth) < 1:
            raise ValueError("temporal_dims and fifo_depth must be >= 1")


OPTIONS = tuple(f.name for f in fields(StreamerOptions))
ENTRY_KEYS = (*OPTIONS, "spatial_bounds")


@dataclass(frozen=True)
class Platform:
    """A parsed platform file; see the module doc."""

    name: str
    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config | None = None
    wait_mode: str = "poll"
    default: StreamerOptions = field(default_factory=StreamerOptions)
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)  # per streamer, as written
    base: str | None = None
    changes: dict[str, Any] = field(default_factory=dict)
    hjson: str | None = None  # the source cluster file, from snax-forge's root
    accelerator_slot: str | None = None  # the generated accelerator's name

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
            "hjson": self.hjson,
            "accelerator_slot": self.accelerator_slot,
            "l1": self.l1.to_dict(),
            "l2": None if self.l2 is None else self.l2.to_dict(),
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
    source, slot = d.get("hjson"), d.get("accelerator_slot")
    if source is not None and (not isinstance(source, str) or not source):
        out.append(
            Problem(
                "platform.keys", "hjson", f"must be a path from the root or null, got {source!r}"
            )
        )
    if slot is not None and (not isinstance(slot, str) or not slot.isidentifier()):
        out.append(
            Problem(
                "platform.keys", "accelerator_slot", f"must be an identifier or null, got {slot!r}"
            )
        )

    l1 = _section(L1Config, d.get("l1", {}), "l1", out)
    l2 = None if d.get("l2") is None else _section(L2Config, d["l2"], "l2", out)
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
        name=name, l1=l1, l2=l2, wait_mode=wait_mode, default=default, entries=entries, base=base, changes=dict(changes), hjson=source,
        accelerator_slot=slot,
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
    if parts[0] in NULLABLE:
        if len(parts) != 1 or not (value is None or isinstance(value, str)):
            return [Problem("platform.keys", where, f"must be a string or null, got {value!r}")]
        d[path] = value
        return []
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
