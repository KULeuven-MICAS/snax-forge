"""A platform read from snax_cluster's cluster file (LNK1, D131).

    platform_from_hjson(path, name, root=None) -> Platform

snax_cluster describes a cluster in an hjson file (``cfg/<cluster>.hjson``,
the copy in snax-gvsoc ``sw/cfg/``, and snax_alu's in ``platforms/``);
``{"$ref": "#/<key>"}`` stands for the file's top-level ``<key>``. The
reader takes from it:

    l1         n_banks       cluster.tcdm.banks
               width_bits    cluster.data_width
               rows          cluster.tcdm.size (KiB) * 8192 / (banks * width)
               base_addr     cluster.cluster_base_addr
               wide_bits     cluster.dma_data_width
    l2         base_addr     dram.address
               size_bytes    dram.length
               beat_bits     cluster.dma_data_width
               width_bits    cluster.data_width
    streamers  default.temporal_dims, default.fifo_depth
                             the one accelerator's streamer template
                             (``snax_acc_cfg`` over every core of every hive,
                             ``snax_streamer_cfg``): every mover's
                             ``temporal_dim`` and ``fifo_depth``, which must
                             agree
    hjson                    the file's path from ``root`` (snax-forge's
                             root by default), ``/``-separated: the
                             platform's source, which the GVSoC generator
                             writes the design's cluster file from
    accelerator_slot         the one accelerator's ``snax_acc_name``

Not in the hjson, so fixed here: the L1's ``read_latency`` 1, both memories'
``dtype`` int64 and ``elems_per_word`` 1, the L2's ``read_latency`` and the
streamers' ``addr_depth`` and ``prio`` small16's. ``spatial_bounds`` and
``num_channel`` are not stored: the shell derives them from the accelerator's
ports (D84): the GVSoC generator lowers the streamer per design point (D134).

The rest of a platform (``xbar``, ``dma``, ``controller``,
``register_window``, ``wait_mode``) is small16's: these are SNAX-MODEL's
fields, unused by SNAX-GVSoC, and go with RET1. The cores, the I-cache, the
interconnect and the CSR counts are not read; they are LNK2a's. The core
count is read on its own (``core_count``) by the design check that keeps the
layouts out of snRuntime's top of L1 (``memory.runtime``, check.py).

Everything wrong is a ``DesignError``, each problem coded ``hjson.<what>``:
``hjson.parse`` (the file), ``hjson.path`` (a file outside ``root``), ``hjson.ref`` (a ``$ref``), ``hjson.key`` (a key
missing or of the wrong type, by its path), ``hjson.value`` (a value no
platform can take), ``hjson.accelerators`` (not exactly one accelerator),
``hjson.movers`` (movers that disagree).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import hjson

from snax_forge.sdfg.paths import _repo_root
from snax_forge.snax_model import L1Config, L2Config

from .platform import Platform
from .problems import DesignError, Problem

ROOT = _repo_root()
SMALL16 = ROOT / "platforms" / "small16.json"
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
    node: Any = d
    parts = path.split(".")
    where = (at + "." if at else "") + path
    for i, p in enumerate(parts):
        if not isinstance(node, Mapping) or p not in node:
            raise _fail("hjson.key", (at + "." if at else "") + ".".join(parts[: i + 1]), "missing")
        node = node[p]
    if isinstance(node, bool) or not isinstance(node, kind):
        raise _fail("hjson.key", where, f"must be {kind.__name__}, got {node!r}")
    return node


def accelerators(d: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Every ``snax_acc_cfg`` entry of the cluster's cores with its path, in hive and core order."""
    out = []
    for h, hive in enumerate(_get(d, "cluster.hives", list)):
        at = f"cluster.hives[{h}]"
        for c, core in enumerate(_get(hive, "cores", list, at)):
            if not isinstance(core, Mapping):
                raise _fail("hjson.key", f"{at}.cores[{c}]", f"must be an object, got {core!r}")
            accs = core.get("snax_acc_cfg", [])
            if not isinstance(accs, list):
                raise _fail("hjson.key", f"{at}.cores[{c}].snax_acc_cfg", "must be a list")
            out += [(f"{at}.cores[{c}].snax_acc_cfg[{i}]", a) for i, a in enumerate(accs)]
    return out


def core_count(path: str | Path) -> int:
    """The cores of every hive of the cluster file at ``path``; a DesignError when it cannot."""
    try:
        d = hjson.loads(Path(path).read_text())
    except (OSError, hjson.HjsonDecodeError) as e:
        raise _fail("hjson.parse", str(path), str(e)) from None
    return sum(len(_get(h, "cores", list)) for h in _get(d, "cluster.hives", list))


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
            "a frozen streamer file given as input is open item 78; the GVSoC generator lowers "
            "the streamer per design point (D134)",
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
    slot = _get(acc, "snax_acc_name", str, at)
    if not slot.isidentifier():
        raise _fail("hjson.value", f"{at}.snax_acc_name", f"{slot!r} is not an identifier")
    template = _get(acc, "snax_streamer_cfg", Mapping, at)
    at += ".snax_streamer_cfg"
    temporal = _mover_value(template, "temporal_dim", at)
    fifo = _mover_value(template, "fifo_depth", at)

    base = _get(d, "cluster.cluster_base_addr")
    dram, dram_size = _get(d, "dram.address"), _get(d, "dram.length")
    small = Platform.load(SMALL16)
    try:
        l1 = L1Config(
            n_banks=banks, width_bits=width, rows=bits // (banks * width), read_latency=1,
            dtype="int64", elems_per_word=1, base_addr=base, wide_bits=wide,
        )  # fmt: skip
        l2 = L2Config(
            size_bytes=dram_size, base_addr=dram, read_latency=small.l2.read_latency,
            beat_bits=wide, width_bits=width, dtype="int64", elems_per_word=1,
        )  # fmt: skip
        default = replace(small.default, temporal_dims=temporal, fifo_depth=fifo)
    except ValueError as e:
        raise _fail("hjson.value", "cluster", str(e)) from None
    return replace(small, name=name, l1=l1, l2=l2, default=default, entries={}, base=None,
                   changes={}, hjson=source, accelerator_slot=slot)  # fmt: skip


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
    try:
        d = hjson.loads(Path(path).read_text())
    except (OSError, hjson.HjsonDecodeError) as e:
        raise _fail("hjson.parse", str(path), str(e)) from None
    return platform_from_dict(d, name, source)
