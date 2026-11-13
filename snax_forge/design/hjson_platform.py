"""A platform read from snax_cluster's cluster file (CONTRACTS, "Platform").

``platform_from_hjson`` reads the hjson (``{"$ref": "#/<key>"}`` the file's top-level ``<key>``):

    l1.n_banks, l1.width_bits    cluster.tcdm.banks, cluster.data_width
    l1.rows                      cluster.tcdm.size (KiB) * 8192 / (banks * width)
    l1.base_addr, l1.wide_bits   cluster.cluster_base_addr, cluster.dma_data_width
    l2.base_addr, l2.size_bytes  dram.address, dram.length
    l2.beat_bits, l2.width_bits  cluster.dma_data_width, cluster.data_width
    streamers.default.fifo_depth the one accelerator's streamer template, its movers agreeing
    hjson                        the file's ``/``-separated path from ``root``
    accelerator_slot             ``ACCELERATOR_SLOT`` (``snax_forge``), never the source's name

The rest (``wait_mode``) is small16's; the cores, I-cache, interconnect and CSR counts are the
generator's. ``core_count`` reads the core count alone, for ``memory.runtime`` (check.py).
Problems are ``DesignError``s coded ``hjson.parse``, ``hjson.path``, ``hjson.ref``,
``hjson.key``, ``hjson.value``, ``hjson.accelerators`` (not one) and ``hjson.movers``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import hjson

from snax_forge.common import Missing, dig, repo_root

from .platform import L1Config, L2Config, Platform
from .problems import DesignError, Problem

ROOT = repo_root()
SMALL16 = ROOT / "platforms" / "small16.json"
# The generated accelerator's name, the platform's accelerator_slot
ACCELERATOR_SLOT = "snax_forge"
# The movers of a streamer template, in snax_cluster's order.
MOVERS = ("data_reader_params", "data_writer_params", "data_reader_writer_params")


def _fail(code: str, where: str, why: str, fix: str | None = None) -> DesignError:
    return DesignError([Problem(code, where, why, fix)], "hjson platform")


def _resolve(node: Any, top: Mapping[str, Any], path: str, seen: tuple[str, ...] = ()) -> Any:
    """``node`` with every ``{"$ref": "#/<key>"}`` replaced by the top-level ``<key>``."""
    if isinstance(node, Mapping):
        if "$ref" in node:
            ref = node["$ref"]
            if len(node) != 1 or not isinstance(ref, str) or not ref.startswith("#/"):
                raise _fail("hjson.ref", path, f'{dict(node)!r} is not {{$ref: "#/<key>"}}')
            key = ref[2:]
            if key not in top:
                raise _fail("hjson.ref", path, f"{ref!r}: no top-level key {key!r}")
            if key in seen:
                raise _fail("hjson.ref", path, f"{ref!r} refers to itself")
            return _resolve(top[key], top, f"#/{key}", (*seen, key))
        return {k: _resolve(v, top, f"{path}.{k}", seen) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v, top, f"{path}[{i}]", seen) for i, v in enumerate(node)]
    return node


def _get(d: Any, path: str, kind: type = int, at: str = "") -> Any:
    """The value at the dotted ``path`` of ``d`` (itself at ``at``), of type ``kind``."""
    where = (at + "." if at else "") + path
    try:
        node = dig(d, path)
    except Missing as e:
        prefix = ".".join(path.split(".")[: e.at + 1])
        raise _fail("hjson.key", (at + "." if at else "") + prefix, "missing") from None
    if isinstance(node, bool) or not isinstance(node, kind):
        raise _fail("hjson.key", where, f"must be {kind.__name__}, got {node!r}")
    return node


def cores(d: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Every core of the cluster file ``d`` with its path, in hive and core order. A core that is
    ``{"$ref": "#/<key>"}`` is that top-level key's object itself, not a copy, so what is found
    through it can be set in place (gen/gvsoc.py)."""
    out = []
    for h, hive in enumerate(_get(d, "cluster.hives", list)):
        at = f"cluster.hives[{h}]"
        for c, core in enumerate(_get(hive, "cores", list, at)):
            where = f"{at}.cores[{c}]"
            ref = core.get("$ref") if isinstance(core, Mapping) else None
            if ref is not None:
                if not (isinstance(ref, str) and ref.startswith("#/") and ref[2:] in d):
                    raise _fail("hjson.ref", where, f"{ref!r}: no top-level key it names")
                core = d[ref[2:]]
            if not isinstance(core, Mapping):
                raise _fail("hjson.key", where, f"must be an object, got {core!r}")
            out.append((where, core))
    return out


def accelerators(d: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Every ``snax_acc_cfg`` entry of the cluster's cores (``cores``) with its path, in hive and
    core order."""
    out = []
    for where, core in cores(d):
        accs = core.get("snax_acc_cfg", [])
        if not isinstance(accs, list):
            raise _fail("hjson.key", f"{where}.snax_acc_cfg", "must be a list")
        out += [(f"{where}.snax_acc_cfg[{i}]", a) for i, a in enumerate(accs)]
    return out


class CoreCount(ValueError):
    """``dma_and_owner``'s: not exactly one xdma core and one owner of the slot; ``dma`` and
    ``owner`` the cores found. Each caller words its own error."""

    def __init__(self, dma: list[int], owner: list[int]):
        super().__init__(dma, owner)
        self.dma, self.owner = dma, owner


def dma_and_owner(flat: Sequence[Mapping[str, Any]], slot: str) -> tuple[int, int]:
    """The core with ``xdma`` and the core whose ``snax_acc_cfg`` holds the slot ``slot``, of the
    cluster's cores in order (``cores``): a program and a flow run take one of each."""
    dma = [i for i, c in enumerate(flat) if c.get("xdma")]
    owner = [i for i, c in enumerate(flat)
             if any(a.get("snax_acc_name") == slot for a in c.get("snax_acc_cfg", []))]  # fmt: skip
    if len(dma) != 1 or len(owner) != 1:
        raise CoreCount(dma, owner)
    return dma[0], owner[0]


def _load(path: str | Path) -> Any:
    """The cluster file at ``path``; a DesignError (``hjson.parse``) when it cannot be read."""
    try:
        return hjson.loads(Path(path).read_text())
    except (OSError, hjson.HjsonDecodeError) as e:
        raise _fail("hjson.parse", str(path), str(e)) from None


def core_count(path: str | Path) -> int:
    """The cores of every hive of the cluster file at ``path``; a DesignError when it cannot."""
    return len(cores(_load(path)))


def _mover_value(template: Mapping[str, Any], key: str, at: str) -> int:
    """The one value of ``key`` over every mover of the template; movers that differ are named."""
    seen: dict[str, list[Any]] = {}
    for m in MOVERS:
        if m in template:
            seen[m] = _get(template, f"{m}.{key}", list, at)
    if not seen:
        raise _fail("hjson.key", at, f"no movers (any of {list(MOVERS)})")
    flat = {v for values in seen.values() for v in values}
    if len(flat) != 1 or any(isinstance(v, bool) or not isinstance(v, int) for v in flat):
        listed = ", ".join(f"{m} {v}" for m, v in seen.items())
        raise _fail(
            "hjson.movers", f"{at}.{key}",
            f"the movers differ ({listed}); a platform holds one value for every streamer",
            "a frozen streamer file given as input is not supported yet; the GVSoC generator lowers "
            "the streamer per design point",
        )  # fmt: skip
    return flat.pop()


def platform_from_dict(d: Mapping[str, Any], name: str, source: str | None = None) -> Platform:
    """The platform of a parsed cluster file (``$ref`` not yet resolved), from the file at
    ``source`` (a path from the root); see the module doc."""
    if not isinstance(d, Mapping):
        raise _fail("hjson.parse", "file", f"must be an object, got {type(d).__name__}")
    d = _resolve(d, d, "#")
    banks = _get(d, "cluster.tcdm.banks")
    width = _get(d, "cluster.data_width")
    size_kib = _get(d, "cluster.tcdm.size")
    wide = _get(d, "cluster.dma_data_width")
    bits = size_kib * 8192
    if banks < 1 or width < 1 or bits % (banks * width):
        raise _fail(
            "hjson.value", "cluster.tcdm",
            f"{size_kib} KiB is not whole rows of {banks} banks x {width} bits",
        )  # fmt: skip

    accs = accelerators(d)
    if len(accs) != 1:
        names = [a.get("snax_acc_name") if isinstance(a, Mapping) else a for _, a in accs]
        raise _fail(
            "hjson.accelerators", "cluster.hives",
            f"{len(accs)} accelerators {names}; a platform has one accelerator slot",
        )  # fmt: skip
    at, acc = accs[0]
    acc_name = _get(acc, "snax_acc_name", str, at)
    if not acc_name.isidentifier():
        raise _fail("hjson.value", f"{at}.snax_acc_name", f"{acc_name!r} is not an identifier")
    template = _get(acc, "snax_streamer_cfg", Mapping, at)
    at += ".snax_streamer_cfg"
    fifo = _mover_value(template, "fifo_depth", at)

    base = _get(d, "cluster.cluster_base_addr")
    dram, dram_size = _get(d, "dram.address"), _get(d, "dram.length")
    small = Platform.load(SMALL16)
    try:
        l1 = L1Config(
            n_banks=banks, width_bits=width, rows=bits // (banks * width), base_addr=base,
            wide_bits=wide,
        )  # fmt: skip
        l2 = L2Config(
            size_bytes=dram_size, base_addr=dram, beat_bits=wide, width_bits=width,
        )  # fmt: skip
        default = replace(small.default, fifo_depth=fifo)
    except ValueError as e:
        raise _fail("hjson.value", "cluster", str(e)) from None
    return replace(small, name=name, l1=l1, l2=l2, default=default, entries={}, base=None,
                   changes={}, hjson=source, accelerator_slot=ACCELERATOR_SLOT)  # fmt: skip


def source_path(path: str | Path, root: str | Path | None = None) -> str:
    """``path`` relative to ``root`` (snax-forge's root by default), ``/``-separated."""
    root = ROOT if root is None else Path(root)
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        raise _fail(
            "hjson.path", str(path), f"is outside {root}: a platform names its source by its "
            "path from there, the same on every host", "copy the file into platforms/",
        ) from None  # fmt: skip


def platform_from_hjson(path: str | Path, name: str, root: str | Path | None = None) -> Platform:
    """The platform of snax_cluster's cluster file at ``path``, named ``name``; its ``hjson``
    the path from ``root``."""
    if not isinstance(name, str) or not name.isidentifier():
        raise _fail("hjson.value", "name", f"{name!r} is not an identifier")
    source = source_path(path, root)
    return platform_from_dict(_load(path), name, source)
