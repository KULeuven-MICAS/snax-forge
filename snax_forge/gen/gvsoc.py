"""A design point -> its cluster on SNAX-GVSoC: three hjson files and headers (D128, D134).

    generate(design_point, out_dir, root=None) -> Manifest

The hjson is the contract: SNAX-GVSoC's ``snax_hjson`` target reads it at
run time (snax-gvsoc ``snax/snax_cfg.py``), so a design point compiles
nothing and no Python is generated (D128). ``out_dir`` gets three hjson
files, each changing for its own reason, the C headers, the program and its
data, and the manifest:

    <platform>.hjson         the cluster: the platform's source hjson with
                             the platform's values written back and the
                             accelerator slot set
    <slot>_streamer.hjson    the streamer template, as the hjson has it
    <slot>_acc.hjson         the datapath block
    include/                 the C headers (headers.py): the streamer's
                             streamer_csr_addr_map.h and <instance>_csr.h
    data.h                   the program's data (data.py): inputs,
                             outputs and goldens
    main.c                   the program (lower/c.py, D135): the
                             task list on snRuntime, self-checking
    manifest.json            what was written (Manifest)

``<platform>`` is the platform's base (its name when it has none), ``<slot>``
its ``accelerator_slot``.

**The cluster file.** The platform's ``hjson`` (a path from ``root``),
parsed, with the platform's values written back by the reverse of
``design/hjson_platform.py``'s mapping:

    cluster.tcdm.banks         l1.n_banks
    cluster.data_width         l1.width_bits (= l2.width_bits)
    cluster.tcdm.size          l1.n_banks * l1.rows * l1.width_bits, in KiB
    cluster.cluster_base_addr  l1.base_addr
    cluster.dma_data_width     l1.wide_bits (= l2.beat_bits)
    dram.address, dram.length  l2.base_addr, l2.size_bytes

and the slot, the source's one ``snax_acc_cfg`` entry, named
``accelerator_slot``, set: ``snax_tcdm_ports`` the streamer's channels;
``sparse_interconnect_config`` one group of that width (its granularity
kept), or ``snax_narrow_tcdm_ports`` that many and ``snax_wide_tcdm_ports``
0, whichever the source has; ``snax_num_rw_csr`` the shell's fields and
start, ``snax_num_ro_csr`` 2 (busy and the counter, D130);
``snax_streamer_cfg`` ``{"$ref": "<slot>_streamer.hjson"}`` and
``snax_datapath_cfg`` ``{"$ref": "<slot>_acc.hjson"}``, each the whole file.
The source's own template, a top-level key the slot referred to
(``{"$ref": "#/<key>"}``), goes when nothing else refers to it. Everything
else is the source's, its key order kept; its comments are not (the file
is written as JSON, which hjson reads).

**The streamer**, lowered per design point (D134), as the design point's
streamers give it (D87): readers the input ports in the BRM's port order,
writers the output ports, each with ``spatial_bounds``, ``temporal_dim``
(the streamer's ``temporal_dims``), ``num_channel`` (``n_ports * N / 64``
for the port's N-bit dtype) and ``fifo_depth``; ``addr_depth`` and ``prio``
are SNAX-MODEL's and not in it.

**The datapath block** (snax-gvsoc ``snax/snax_cfg.py``): ``unit``
(``SnaxUnit``), then the block's adapter's unit (blocks.py: ``inputs``,
``outputs``, ``lanes``, ``data_width``, ``op``, ``signed``, ``control``,
``config`` when not the default, ``wait``, ``latency``, ``ii``), then
``fields``, the shell's configuration CSRs.

**Refused**, each a ``GenError`` (``gen.<rule>``): a design point outside
``root`` (``gen.point``), more or fewer than one accelerated instance
(``gen.instances``: the platform has one accelerator slot), a null binding
(``gen.binding``), an unregistered block (``gen.block``), what the block's
adapter refuses, a port whose lanes are not whole 64-bit channels
(``gen.dtype``), CSRs past ``SNAX_CSR_LAST`` (``gen.csrs``), a platform
without a source hjson or a source that cannot be read or written into
(``gen.source``), a slot name that is not the source's only slot
(``gen.slot``), and a platform value the hjson cannot express
(``gen.platform``: what the cluster file gives back through
hjson_platform.py must be the platform's L1 and L2). A symbol of the bound
graph still null never gets here: ``DesignPoint.load`` refuses it
(``graph.symbols``).

**The manifest**: ``design``, ``design_point``, ``symbols`` (the bound
graph's, e.g. ``{"N": 64}``: what a program sizes its data by),
``cluster`` (the cluster file, what gvrun's ``--param hjson=`` takes),
``files``, ``program`` (``main.c``), ``trace`` (the unit's trace path), ``csrs`` (the layout) and
``tasks``: per accelerator task of the derived task list
(snax_forge/lower/derive.py, D89) its name, its ``beats`` (the unit's trip
count, the task's ``n``) and its ``csrs`` values.

**Stable paths.** The design point is named once, by its path relative to
``root`` (snax-forge's root by default; ``named_from`` when given, the
flow's folder in a flow), ``/``-separated: the manifest's
``design_point`` and every written file's header comment hold the same
string (``out/design/vecadd/design_point.json``); the platform's source is
read from ``root`` too. ``files`` and ``cluster`` are relative to the
manifest's folder. Nothing written depends on where ``out_dir`` is, the
working directory or the host: one design point written into two folders
gives byte-identical files, which is what lets snax-gvsoc's checked-in
fixture be compared with a fresh run by diff.

Not verified: the order of ``spatial_bounds`` against snax_cluster's
template for more than one spatial dimension (every template so far has one).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hjson

from snax_forge import expr
from snax_forge.brm import Instance
from snax_forge.common import generated_header, plain, repo_root, to_json
from snax_forge.design import DesignError
from snax_forge.design.hjson_platform import accelerators, platform_from_dict
from snax_forge.design.platform import Platform
from snax_forge.design.point import DesignPoint
from snax_forge.design.streamers import streamer_name
from snax_forge.lower.c import MAIN_C, main_c, mover_name
from snax_forge.lower.c import accelerator as c_accelerator
from snax_forge.lower.derive import task_list
from snax_forge.lower.tasks import Configure, TaskList

from .blocks import BLOCKS, GenError, Unit, port_dtype
from .data import DATA_H, arrays, data_h
from .headers import STREAMER_HEADER, accelerator_header, streamer_header

ROOT = repo_root()
OUT = ROOT / "out" / "gvsoc"

# snax-gvsoc snax/snax_acc.py: the first SNAX CSR (CSR_SNAX_BEGIN) and the
# last one the core sends out (SNAX_CSR_FIRST, SNAX_CSR_LAST)
SNAX_CSR_FIRST = 0x3C0
SNAX_CSR_LAST = 0x5FF
# The unit's trace path: the compute core's accelerator, named snax_acc<core>
# by pulp/snitch/snitch_cluster/snitch_cluster.py; the composite
# (snax/snax_unit_acc.py) calls it unit
UNIT_TRACE = "snax_acc0/unit"
# The unit kind of the datapath block (snax-gvsoc snax/snax_unit_acc.py)
UNIT = "SnaxUnit"
# A shell's read-only CSRs: busy and the counter (D130)
SHELL_RO_CSRS = 2


@dataclass(frozen=True)
class CsrLayout:
    """The accelerator's CSRs: the streamer's range, then the shell's (D130)."""

    streamer_first: int
    streamer_last: int
    fields: dict[str, int]
    start: int
    busy: int
    counter: int

    @property
    def shell_first(self) -> int:
        return self.streamer_last + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "streamer": {"first": self.streamer_first, "last": self.streamer_last},
            "fields": dict(self.fields),
            "start": self.start,
            "busy": self.busy,
            "counter": self.counter,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CsrLayout:
        return cls(d["streamer"]["first"], d["streamer"]["last"], dict(d["fields"]), d["start"],
                   d["busy"], d["counter"])  # fmt: skip


@dataclass(frozen=True)
class AccTask:
    """One accelerator task: its name, the unit's trip count and the shell's field values."""

    name: str
    beats: int
    csrs: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "beats": self.beats, "csrs": dict(self.csrs)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AccTask:
        return cls(d["name"], d["beats"], dict(d["csrs"]))


@dataclass(frozen=True)
class Manifest:
    """What ``generate`` wrote; see the module doc."""

    design: str
    design_point: str
    symbols: dict[str, int]
    cluster: str
    files: list[str]
    program: str
    trace: str
    csrs: CsrLayout
    tasks: list[AccTask] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "design": self.design,
            "design_point": self.design_point,
            "symbols": dict(self.symbols),
            "cluster": self.cluster,
            "files": list(self.files),
            "program": self.program,
            "trace": self.trace,
            "csrs": self.csrs.to_dict(),
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Manifest:
        return cls(
            d["design"], d["design_point"], dict(d["symbols"]), d["cluster"], list(d["files"]),
            d["program"], d["trace"], CsrLayout.from_dict(d["csrs"]),
            [AccTask.from_dict(t) for t in d["tasks"]],
        )  # fmt: skip

    def to_json(self) -> str:
        return to_json(self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        return cls.from_dict(json.loads(Path(path).read_text()))


# =============================================================================
# What the design point gives
# =============================================================================


def accelerator(point: DesignPoint) -> tuple[str, Instance, Unit]:
    """The one accelerated instance (lower/c.py's ``accelerator``, its rules ``gen.instances``
    and ``gen.binding``), its name and its unit (the block's adapter)."""
    name, inst = c_accelerator(point, lambda rule, message: GenError(f"gen.{rule}", message))
    b = inst.impl.binding
    if b["block"] not in BLOCKS:
        raise GenError(
            "gen.block",
            f"{inst.brm.name}.{inst.implementation}: block {b['block']!r} not registered "
            f"({sorted(BLOCKS)})",
        )
    return name, inst, BLOCKS[b["block"]](inst)


def streamer_template(point: DesignPoint, name: str, inst: Instance) -> dict[str, Any]:
    """The streamer template, in the hjson's keys; see the module doc."""
    groups: dict[str, list[dict[str, Any]]] = {"data_reader_params": [], "data_writer_params": []}
    for p in inst.brm.interface.ports:
        s = point.streamers[streamer_name(name, p.name)]
        bits, _ = port_dtype(p.dtype, f"port {p.name}")
        if s.n_ports * bits % 64:
            raise GenError(
                "gen.dtype", f"streamer {s.name}: {s.n_ports} x {bits} bits is not whole 64-bit "
                "channels",
            )  # fmt: skip
        mover = {
            "spatial_bounds": list(s.spatial_bounds),
            "temporal_dim": s.options.temporal_dims,
            "num_channel": s.n_ports * bits // 64,
            "fifo_depth": s.options.fifo_depth,
        }
        groups["data_writer_params" if s.write else "data_reader_params"].append(mover)
    return {
        g: {k: [m[k] for m in movers] for k in movers[0]} for g, movers in groups.items() if movers
    }


def movers(template: dict[str, Any]) -> list[dict[str, Any]]:
    """The template's movers, readers first, named as the headers and the program name them
    (lower/c.py's ``mover_name``)."""
    out = []
    for group, write in (("data_reader_params", False), ("data_writer_params", True)):
        params = template.get(group, {})
        for i in range(len(params.get("num_channel", []))):
            out.append({"name": mover_name(write, i), **{k: v[i] for k, v in params.items()}})
    return out


def streamer_csrs(template: dict[str, Any]) -> int:
    """snax_streamer.nb_csrs: per mover the pointer (2), the spatial strides, the temporal
    bounds and strides; then start; then busy and the counter."""
    rw = sum(2 + len(m["spatial_bounds"]) + 2 * m["temporal_dim"] for m in movers(template))
    return rw + 1 + 2


def channels(template: dict[str, Any]) -> int:
    """The streamer's TCDM ports: num_channel over its movers (snax_cfg.streamer_channels)."""
    return sum(m["num_channel"] for m in movers(template))


def csr_layout(template: dict[str, Any], fields: list[str]) -> CsrLayout:
    first = SNAX_CSR_FIRST + streamer_csrs(template)
    nums = {f: first + i for i, f in enumerate(fields)}
    start = first + len(fields)
    layout = CsrLayout(SNAX_CSR_FIRST, first - 1, nums, start, start + 1, start + 2)
    if layout.counter > SNAX_CSR_LAST:
        raise GenError("gen.csrs", f"the shell's last CSR {layout.counter} is past {SNAX_CSR_LAST}")
    return layout


def datapath(unit: Unit) -> dict[str, Any]:
    """The datapath block; see the module doc."""
    return {"unit": UNIT, **unit.properties(), "fields": list(unit.fields)}


def acc_tasks(tasks: TaskList, name: str, inst: Instance) -> list[AccTask]:
    """The accelerator tasks of the derived task list (D89), with the binding's csrs values."""
    out = []
    for step in tasks.steps:
        if isinstance(step, Configure) and step.type == "accel" and step.component == name:
            env = {**inst.params, **step.values}
            csrs = {c["field"]: expr.evaluate(c["value"], env) for c in inst.impl.binding["csrs"]}
            out.append(AccTask(step.task_name, step.values["n"], csrs))
    return out


# =============================================================================
# The cluster file
# =============================================================================


def source(platform: Platform, root: Path) -> dict[str, Any]:
    """The platform's source hjson, parsed (``$ref`` kept)."""
    if platform.hjson is None:
        raise GenError(
            "gen.source", f"platform {platform.name} has no source hjson: read it from one with "
            "`pixi run design from-hjson`",
        )  # fmt: skip
    path = root / platform.hjson
    try:
        d = hjson.loads(path.read_text())
    except (OSError, hjson.HjsonDecodeError) as e:
        raise GenError("gen.source", f"{platform.hjson}: {e}") from None
    if not isinstance(d, Mapping):
        raise GenError("gen.source", f"{platform.hjson}: not an object")
    return plain(d)


def _set(d: dict[str, Any], path: str, value: Any, where: str) -> None:
    """Set the dotted ``path`` of ``d``; every key but the last must be there, an object."""
    *head, last = path.split(".")
    node: Any = d
    for i, k in enumerate(head):
        node = node.get(k) if isinstance(node, Mapping) else None
        if not isinstance(node, Mapping) or "$ref" in node:
            raise GenError(
                "gen.source", f"{where}: {'.'.join(head[: i + 1])} is not an object written out "
                "in the file, so the platform's value cannot be written into it",
            )  # fmt: skip
    node[last] = value


def _local_refs(node: Any) -> list[str]:
    """Every ``#/<key>`` a ``$ref`` of ``node`` names."""
    if isinstance(node, Mapping):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/"):
            return [ref[2:]]
        return [k for v in node.values() for k in _local_refs(v)]
    if isinstance(node, list):
        return [k for v in node for k in _local_refs(v)]
    return []


def write_back(d: dict[str, Any], platform: Platform) -> None:
    """The platform's values into the parsed cluster file ``d``; see the module doc."""
    l1, l2 = platform.l1, platform.l2
    where = platform.hjson or platform.name
    if l2 is None:
        raise GenError(
            "gen.platform", f"platform {platform.name} has no L2; the hjson's dram is one"
        )
    bits = l1.n_banks * l1.rows * l1.width_bits
    if bits % 8192:
        raise GenError(
            "gen.platform", f"l1: {l1.n_banks} banks x {l1.rows} rows x {l1.width_bits} bits is "
            "not whole KiB (cluster.tcdm.size)",
        )  # fmt: skip
    for path, value in (
        ("cluster.tcdm.banks", l1.n_banks), ("cluster.data_width", l1.width_bits),
        ("cluster.tcdm.size", bits // 8192), ("cluster.cluster_base_addr", l1.base_addr),
        ("cluster.dma_data_width", l1.wide_bits), ("dram.address", l2.base_addr),
        ("dram.length", l2.size_bytes),
    ):  # fmt: skip
        _set(d, path, value, where)
    try:
        back = platform_from_dict(d, platform.name)
    except DesignError as e:
        raise GenError("gen.source", f"{where}: {e}") from None
    differ = [
        f"{mem}.{k} {getattr(ours, k)!r} (the hjson gives {getattr(theirs, k)!r})"
        for mem, ours, theirs in (("l1", l1, back.l1), ("l2", l2, back.l2))
        for k in ours.to_dict() if getattr(ours, k) != getattr(theirs, k)
    ]  # fmt: skip
    if differ:
        raise GenError(
            "gen.platform", f"the cluster file cannot express {', '.join(differ)}: SNAX-GVSoC "
            "would run another platform",
        )  # fmt: skip


def find_slot(d: dict[str, Any], platform: Platform) -> tuple[str, dict[str, Any]]:
    """The source's one slot, which must be named ``accelerator_slot``, and its path."""
    where = platform.hjson
    slots: list[tuple[str, dict[str, Any]]] = []
    try:
        found = accelerators(d)
    except DesignError as e:
        raise GenError("gen.source", f"{where}: {e}") from None
    for at, acc in found:
        if not isinstance(acc, dict):
            raise GenError("gen.source", f"{where}: {at} is not an object")
        if all(acc is not s for _, s in slots):
            slots.append((at, acc))
    names = [s.get("snax_acc_name") for _, s in slots]
    if names != [platform.accelerator_slot]:
        raise GenError(
            "gen.slot", f"{where} has the slots {names}; the platform's accelerator_slot "
            f"{platform.accelerator_slot!r} must be its only one",
        )  # fmt: skip
    return slots[0]


def set_slot(d: dict[str, Any], at: str, slot: dict[str, Any], template: dict[str, Any],
             unit: Unit, files: tuple[str, str], where: str) -> None:  # fmt: skip
    """The slot (``find_slot``'s, in ``d``) set for the design; see the module doc."""
    ports = channels(template)
    slot["snax_tcdm_ports"] = ports
    if "sparse_interconnect_config" in slot:
        groups = slot["sparse_interconnect_config"]
        if not (isinstance(groups, list) and len(groups) == 1 and isinstance(groups[0], list)
                and len(groups[0]) == 2):  # fmt: skip
            raise GenError(
                "gen.source", f"{where}: {at}.sparse_interconnect_config {groups!r} is not one "
                "group [width, granularity]",
            )  # fmt: skip
        slot["sparse_interconnect_config"] = [[ports, groups[0][1]]]
    if "snax_narrow_tcdm_ports" in slot or "snax_wide_tcdm_ports" in slot:
        slot["snax_narrow_tcdm_ports"] = ports
        slot["snax_wide_tcdm_ports"] = 0
    slot["snax_num_rw_csr"] = len(unit.fields) + 1
    slot["snax_num_ro_csr"] = SHELL_RO_CSRS
    old = slot.get("snax_streamer_cfg")
    slot["snax_streamer_cfg"] = {"$ref": files[0]}
    slot["snax_datapath_cfg"] = {"$ref": files[1]}
    for key in _local_refs(old):
        if key in d and key not in _local_refs(d):
            del d[key]


# =============================================================================
# The files
# =============================================================================


def point_name(path: Path, root: Path) -> str:
    """The design point's name in everything written: its path relative to ``root``."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        raise GenError(
            "gen.point",
            f"design point {path} is outside {root}: it is named by its path from there, "
            "the same on every host",
        ) from None


def _hjson(point_path: str, what: str, value: Any) -> str:
    """An hjson file: the header comment, then ``value`` as JSON (which hjson reads)."""
    return generated_header(point_path, what) + to_json(value)


# =============================================================================
# generate
# =============================================================================


def generate(
    design_point: str | Path, out_dir: str | Path, root: str | Path | None = None, *,
    named_from: str | Path | None = None, kernel: str | None = None, seed: int = 0,
) -> Manifest:  # fmt: skip
    """Write the three hjson files, the headers, data.h, main.c and the manifest to
    ``out_dir``; see the module doc. ``named_from`` is the folder the design point is named
    from (``root`` by default; the flow's folder for a flow, D136); ``kernel`` and ``seed``
    go to data.h's inputs (data.py's ``arrays``)."""
    point_path, out = Path(design_point), Path(out_dir)
    root = ROOT if root is None else Path(root)
    shown = point_name(point_path, root if named_from is None else Path(named_from))
    point = DesignPoint.load(point_path)
    platform = point.platform
    name, inst, unit = accelerator(point)
    template = streamer_template(point, name, inst)
    layout = csr_layout(template, unit.fields)
    tasks = task_list(point)
    data = arrays(point, seed, kernel)

    cluster = source(platform, root)
    at, found = find_slot(cluster, platform)
    write_back(cluster, platform)
    slot = platform.accelerator_slot
    names = (f"{platform.base or platform.name}.hjson", f"{slot}_streamer.hjson",
             f"{slot}_acc.hjson")  # fmt: skip
    set_slot(cluster, at, found, template, unit, names[1:], platform.hjson)

    accel = (f"instance {name}, BRM {inst.brm.name}, implementation {inst.implementation}, "
             f"block {inst.impl.binding['block']}")  # fmt: skip
    streamer_h, streamer_last = streamer_header(movers(template), SNAX_CSR_FIRST, shown)
    assert streamer_last == layout.streamer_last, (streamer_last, layout)
    files = {
        names[0]: _hjson(shown, f"The cluster: {platform.hjson} with platform {platform.name}'s "
                         f"values and the slot {slot} set for {accel}; its streamer in "
                         f"{names[1]}, its datapath in {names[2]}.", cluster),
        names[1]: _hjson(shown, f"Slot {slot}'s streamer template: readers the input ports in the "
                         "BRM's port order, writers the output ports.", template),
        names[2]: _hjson(shown, f"Slot {slot}'s datapath block: {accel}.", datapath(unit)),
        f"include/{STREAMER_HEADER}": streamer_h,
        f"include/{name}_csr.h": accelerator_header(name, layout, shown),
        DATA_H: data_h(point, shown, data),
        MAIN_C: main_c(point, tasks, cluster, layout, data,
                       [DATA_H, f"{name}_csr.h", STREAMER_HEADER], shown),
    }  # fmt: skip
    (out / "include").mkdir(parents=True, exist_ok=True)
    for f, text in files.items():
        (out / f).write_text(text)
    manifest = Manifest(
        design=point.name,
        design_point=shown,
        symbols=dict(point.graph.symbols),
        cluster=names[0],
        files=[*files, "manifest.json"],
        program=MAIN_C,
        trace=UNIT_TRACE,
        csrs=layout,
        tasks=acc_tasks(tasks, name, inst),
    )
    (out / "manifest.json").write_text(manifest.to_json())
    return manifest
