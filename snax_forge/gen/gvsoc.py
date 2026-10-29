"""A design point -> its cluster on SNAX-GVSoC: three description files and headers (D134).

    generate(design_point, out_dir, root=None) -> Manifest

The hjson is the contract: SNAX-GVSoC's ``snax_hjson`` target reads it at
run time (snax-gvsoc ``snax/snax_cfg.py``), so a design point compiles
nothing and no Python is generated (D134). ``out_dir`` gets three
description files, each changing for its own reason, the C headers, the
program and its data, and the manifest:

    <platform>.hjson         the cluster: the platform's source hjson with
                             the platform's values written back, the
                             cluster renamed and the accelerator slot set
    <slot>_streamer.hjson    the streamer template, as the hjson has it
    accelerator.json         the accelerator's one description
                             (accelerator.py, D145)
    include/                 the C headers (headers.py): the streamer's
                             streamer_csr_addr_map.h and <name>_csr.h,
                             and the runtime's snitch_cluster_cfg.h and
                             snitch_cluster_addrmap.h from the cluster file
    data.h                   the program's data (data.py): inputs,
                             outputs and goldens
    main.c                   the program (lower/c.py, D135): the
                             task list on snRuntime, self-checking
    manifest.json            what was written (Manifest)

``<platform>`` is the platform's base (its name when it has none), ``<slot>``
its ``accelerator_slot``: for ``platforms/snax_forge.json``,
``snax_forge.hjson``, ``snax_forge_streamer.hjson`` and
``accelerator.json``.

**The cluster file.** The platform's ``hjson`` (a path from ``root``),
parsed, with the platform's values written back by the reverse of
``design/hjson_platform.py``'s mapping:

    cluster.tcdm.banks         l1.n_banks
    cluster.data_width         l1.width_bits (= l2.width_bits)
    cluster.tcdm.size          l1.n_banks * l1.rows * l1.width_bits, in KiB
    cluster.cluster_base_addr  l1.base_addr
    cluster.dma_data_width     l1.wide_bits (= l2.beat_bits)
    dram.address, dram.length  l2.base_addr, l2.size_bytes

``cluster.name`` becomes ``CLUSTER`` (``snax_forge_cluster``), and in
``cluster.bender_target`` the entry equal to the source's ``cluster.name``
becomes it too, so the generated cluster takes no name of snax_cluster's own
(D130). The slot, the source's one ``snax_acc_cfg`` entry whatever its
name, is renamed and set, its keys in this order:

    snax_acc_name               the platform's accelerator_slot
    the source's other keys     in their order, ``bender_target`` dropped
                                (the RTL comes from the bundle, bundle.py),
                                with these set (a key the source lacks
                                after the others):
      snax_tcdm_ports           the streamer's channels
      sparse_interconnect_config  one group of that width (its granularity
                                kept), when the source has it
      snax_num_rw_csr           the accelerator's R/W CSRs (accelerator.py)
      snax_num_ro_csr           its R/O CSRs
    snax_disable_csr_manager    true: the wrapper carries its own manager
                                (D130)
    snax_forge_rtl              ``rtl/files.f``, the RTL's file list from
                                the cluster file's folder, which only a
                                bundle has; SNAX-GVSoC ignores it
    snax_streamer_cfg           ``{"$ref": "<slot>_streamer.hjson"}``
    snax_datapath_cfg           ``{"$ref": "accelerator.json"}``

each ``$ref`` the whole file. The source's own template, a top-level key the
slot referred to (``{"$ref": "#/<key>"}``), goes when nothing else refers to
it. Everything else is the source's, its key order kept; its comments are
not (the file is written as JSON, which hjson reads).

**The streamer**, lowered per design point (D134), as the design point's
streamers give it (D87): readers the input ports, writers the output ports,
unit by unit (the accelerator's groups in order, D148) in the BRM's port
order, each with ``spatial_bounds`` in words, fastest
first (``[S_0 / num_slc, S_1, ...]``: the fastest bound over num_slc, the slices
of the port's container dtype a word holds, D142), ``temporal_dim`` (the
streamer's ``temporal_dims``), ``num_channel`` (the streamer's ``num_ch``, the
wire default W / num_slc; snax-gvsoc's streamer requires the bounds' product to
be it)
and ``fifo_depth``.

**The accelerator** is accelerator.py's ``describe``, written as plain JSON
(no header comment: its ``design_point`` names the source). Its CSRs follow
the streamer's.

**Refused**, each a ``GenError`` (``gen.<rule>``): a design point outside
``root`` (``gen.point``), what accelerator.py's ``describe`` refuses
(``gen.instances``: the platform has one accelerator slot, so one
accelerator, its groups put into one by ``encapsulate``; ``gen.binding``,
``gen.datapath``, ``gen.drain``, ``gen.timing`` and the datapath's
adapter's), a dtype that is not intN or uintN (``gen.dtype``), a
streamer whose fastest spatial bound is not whole words (``gen.lanes``,
which connect.pack refuses first), CSRs past ``SNAX_ACC_CSR_LAST`` (``gen.csrs``), a platform
without a source hjson or a source that cannot be read or written into
(``gen.source``, also a ``cluster.bender_target`` without the source's
``cluster.name``), a source with no slot or several (``gen.slot``), and a
platform value the hjson cannot express
(``gen.platform``: what the cluster file gives back through
hjson_platform.py must be the platform's L1 and L2; a
``cluster_base_offset`` the address map cannot take, headers.py). A symbol of the bound
graph still null never gets here: ``DesignPoint.load`` refuses it
(``graph.symbols``).

**The manifest**: ``design``, ``design_point``, ``symbols`` (the bound
graph's, e.g. ``{"N": 64}``: what a program sizes its data by),
``cluster`` (the cluster file, what gvrun's ``--param hjson=`` takes),
``files``, ``program`` (``main.c``), ``trace`` (``trace_path``: the trace path of the
accelerator's one unit, ``snax_acc0/unit``; of its one linked group,
``snax_acc0/group_<group>``; with several groups the accelerator's, ``snax_acc0``), ``csrs``
(the layout: the streamer's first and last CSR, and every accelerator CSR's define -> its
number) and ``tasks``: per accelerator task of the derived task list
(snax_forge/lower/derive.py, D64), in order, its name, its ``beats`` (the unit's firings,
the task's ``n``; a linked group's most firing unit's), its ``csrs`` values (its group's
``JOBS``, and ``JOB_BEATS`` for a fold without ``params.T``, named with its unit in a linked
group, ``MAC_JOB_BEATS``: lower/c.py's ``job_writes``) and, with several groups, its
group's ``trace`` path, ``snax_acc0/unit_<unit>`` or a linked group's
``snax_acc0/group_<group>``, as snax-gvsoc's composite names them then.

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

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hjson

from snax_forge.brm import Instance, port_dtypes
from snax_forge.common import JsonFile, generated_header, plain, repo_root, to_json
from snax_forge.design import DesignError
from snax_forge.design.hjson_platform import accelerators, platform_from_dict
from snax_forge.design.platform import Platform
from snax_forge.design.point import DesignPoint
from snax_forge.design.streamers import instance_nodes, streamer_name
from snax_forge.lower import CError
from snax_forge.lower.c import MAIN_C, job_writes, main_c, task_beats
from snax_forge.lower.derive import task_list
from snax_forge.lower.tasks import Configure, TaskList

from .accelerator import ACCELERATOR_JSON, Accelerator, describe, units
from .blocks import GenError, port_dtype
from .csrs import SNAX_CSR_FIRST, CsrLayout, channels, csr_layout, movers
from .data import DATA_H, arrays, data_h
from .headers import STREAMER_HEADER, accelerator_header, cluster_headers, streamer_header

ROOT = repo_root()
OUT = ROOT / "out" / "gvsoc"

# The trace paths: the compute core's accelerator, named snax_acc<core> by
# pulp/snitch/snitch_cluster/snitch_cluster.py; the composite
# (snax/snax_accelerator.py) calls a unit `unit`, with several groups `unit_<unit>`
ACC_TRACE = "snax_acc0"
UNIT_TRACE = f"{ACC_TRACE}/unit"
# The generated cluster's name (D130)
CLUSTER = "snax_forge_cluster"
# The RTL's file list from the cluster file's folder, written by bundle.py
RTL_FILE_LIST = "rtl/files.f"


@dataclass(frozen=True)
class AccTask:
    """One accelerator task: its name, the unit's firings, its group's CSR values (JOBS,
    JOB_BEATS) and, with several groups, its unit's trace path (written only then)."""

    name: str
    beats: int
    csrs: dict[str, int]
    trace: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"name": self.name, "beats": self.beats, "csrs": dict(self.csrs)}
        return d if self.trace is None else {**d, "trace": self.trace}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AccTask:
        return cls(d["name"], d["beats"], dict(d["csrs"]), d.get("trace"))


@dataclass(frozen=True)
class Manifest(JsonFile):
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


# =============================================================================
# What the design point gives
# =============================================================================


def node_dtypes(point: DesignPoint, name: str) -> dict[str, str]:
    """Instance ``name``'s port dtypes: its first node's containers' (``brm.port_dtypes``)."""
    return port_dtypes(point.graph, instance_nodes(point.graph)[name])


def streamer_template(point: DesignPoint, found: Mapping[str, Instance]) -> dict[str, Any]:
    """The streamer template of the accelerator's units ``found``, in the hjson's keys; see the
    module doc."""
    groups: dict[str, list[dict[str, Any]]] = {"data_reader_params": [], "data_writer_params": []}
    for name, inst in found.items():
        _movers(point, name, inst, groups)
    return {
        g: {k: [m[k] for m in movers] for k in movers[0]} for g, movers in groups.items() if movers
    }


def _movers(point: DesignPoint, name: str, inst: Instance,
            groups: dict[str, list[dict[str, Any]]]) -> None:  # fmt: skip
    """Unit ``name``'s movers onto ``groups`` (readers, writers), in port order."""
    dtypes = node_dtypes(point, name)
    for p in inst.brm.interface.ports:
        s = point.streamers.get(streamer_name(name, p.name))
        if s is None:  # a port on a link: no streamer (D112)
            continue
        port_dtype(dtypes[p.name], f"port {p.name}")  # intN or uintN (gen.dtype)
        s0, *rest = s.spatial_bounds
        if s0 % s.num_slc:
            raise GenError(
                "gen.lanes", f"streamer {s.name}: its fastest spatial bound {s0} is not whole "
                f"words of num_slc = {s.num_slc} {dtypes[p.name]} slices",
            )  # fmt: skip
        mover = {
            "spatial_bounds": [s0 // s.num_slc, *rest],  # in words (D142)
            "temporal_dim": s.temporal_dims,
            "num_channel": s.num_ch,
            "fifo_depth": s.options.fifo_depth,
        }
        groups["data_writer_params" if s.write else "data_reader_params"].append(mover)


def acc_tasks(tasks: TaskList, found: Mapping[str, Instance], acc: Accelerator) -> list[AccTask]:
    """The accelerator tasks of the derived task list (D64), in order, with their group's CSR
    values (lower/c.py's ``job_writes``: a CSR of a unit of a linked group named with the unit,
    ``MAC_JOB_BEATS``) and, with several groups, its group's trace path."""
    several = len(acc.groups) > 1
    fixed = {u.name for g in acc.groups for u in g.units if "T" in u.params}
    out = []
    for s in tasks.steps:
        if not (isinstance(s, Configure) and s.type == "accel"):
            continue
        group = next((g for g in acc.groups if g.name == s.component and len(g.units) > 1), None)
        mine = [u.name for u in group.units] if group else [s.component]
        if not set(mine) <= set(found):
            continue
        writes = job_writes({u: found[u] for u in mine}, s.values, fixed)
        csrs = {(k if group is None or k == "JOBS" else f"{u.upper()}_{k}"): v
                for u, k, v in writes}  # fmt: skip
        trace = None if not several else (f"{ACC_TRACE}/unit_{s.component}" if group is None
                                          else f"{ACC_TRACE}/group_{group.name}")  # fmt: skip
        out.append(AccTask(s.task_name, task_beats(s.values), csrs, trace))
    return out


def trace_path(acc: Accelerator) -> str:
    """The manifest's trace path: one group of one unit ``snax_acc0/unit``; one linked group
    ``snax_acc0/group_<group>``, SNAX-GVSoC's one component for its units; several groups the
    accelerator's, ``snax_acc0``, each task with its own."""
    if len(acc.groups) > 1:
        return ACC_TRACE
    (g,) = acc.groups
    return UNIT_TRACE if len(g.units) == 1 else f"{ACC_TRACE}/group_{g.name}"


def described(name: str, found: Mapping[str, Instance]) -> str:
    """The accelerator as the cluster file's header comment names it."""

    def one(unit: str, inst: Instance) -> str:
        return (f"instance {unit}, BRM {inst.brm.name}, implementation {inst.implementation}, "
                f"datapath {inst.impl.binding['datapath']}")  # fmt: skip

    if len(found) == 1:
        return one(*next(iter(found.items())))
    return f"accelerator {name} of " + "; ".join(f"group {one(u, i)}" for u, i in found.items())


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
    bits = l1.size_bytes * 8
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
    """The source's only slot, whatever its ``snax_acc_name``, and its path; ``gen.slot`` when
    the source has none or several."""
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
    if len(slots) != 1:
        names = [s.get("snax_acc_name") for _, s in slots]
        raise GenError(
            "gen.slot", f"{where} has the slots {names}; the generated accelerator "
            f"{platform.accelerator_slot!r} takes the source's only one",
        )  # fmt: skip
    return slots[0]


def rename_cluster(d: dict[str, Any], where: str) -> None:
    """``cluster.name`` and its entry in ``cluster.bender_target`` become ``CLUSTER``."""
    cluster = d.get("cluster")
    name = cluster.get("name") if isinstance(cluster, Mapping) else None
    targets = cluster.get("bender_target") if isinstance(cluster, Mapping) else None
    if not isinstance(targets, list) or name not in targets:
        raise GenError(
            "gen.source", f"{where}: cluster.bender_target {targets!r} does not hold "
            f"cluster.name {name!r}, which the generated cluster {CLUSTER} takes the place of",
        )  # fmt: skip
    cluster["name"] = CLUSTER
    cluster["bender_target"] = [CLUSTER if t == name else t for t in targets]


def set_slot(d: dict[str, Any], at: str, slot: dict[str, Any], name: str,
             template: dict[str, Any], acc: Accelerator, files: tuple[str, str],
             where: str) -> None:  # fmt: skip
    """The slot (``find_slot``'s, in ``d``) named ``name`` and set for the design; see the
    module doc."""
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
    slot["snax_num_rw_csr"] = acc.rw
    slot["snax_num_ro_csr"] = acc.ro
    old = slot.get("snax_streamer_cfg")
    last = {
        "snax_disable_csr_manager": True,
        "snax_forge_rtl": RTL_FILE_LIST,
        "snax_streamer_cfg": {"$ref": files[0]},
        "snax_datapath_cfg": {"$ref": files[1]},
    }
    rest = {k: v for k, v in slot.items() if k not in ("snax_acc_name", "bender_target", *last)}
    slot.clear()
    slot.update({"snax_acc_name": name, **rest, **last})
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


def load_point(
    design_point: str | Path, root: Path, named_from: str | Path | None = None
) -> tuple[str, DesignPoint, Accelerator]:
    """The design point's name (``point_name`` from ``named_from``, else ``root``), the point
    and its accelerator's description (accelerator.py's ``describe``)."""
    shown = point_name(Path(design_point), root if named_from is None else Path(named_from))
    point = DesignPoint.load(design_point)
    return shown, point, describe(point, shown)


def cli(write: Callable[[], int]) -> int:
    """A generator command: ``write()``, or 1 with the refusal on stderr (its rule named)."""
    try:
        return write()
    except (GenError, CError, DesignError) as e:
        print(e, file=sys.stderr)
        return 1


def out_dir(given: Path | None, under: Path, design_point: Path) -> Path:
    """``given``, else ``under/<design>``, the design point's name."""
    return given or under / DesignPoint.load(design_point).name


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
    """Write the three description files, the headers, data.h, main.c and the manifest to
    ``out_dir``; see the module doc. ``named_from`` is the folder the design point is named
    from (``root`` by default; the flow's folder for a flow, D136); ``kernel`` and ``seed``
    go to data.h's inputs (data.py's ``arrays``)."""
    point_path, out = Path(design_point), Path(out_dir)
    root = ROOT if root is None else Path(root)
    shown, point, acc = load_point(point_path, root, named_from)
    name, members = units(point)
    platform = point.platform
    template = streamer_template(point, members)
    layout = csr_layout(template, acc)
    tasks = task_list(point)
    data = arrays(point, seed, kernel)

    cluster = source(platform, root)
    at, found = find_slot(cluster, platform)
    write_back(cluster, platform)
    rename_cluster(cluster, platform.hjson)
    slot = platform.accelerator_slot
    names = (f"{platform.base or platform.name}.hjson", f"{slot}_streamer.hjson",
             ACCELERATOR_JSON)  # fmt: skip
    set_slot(cluster, at, found, slot, template, acc, names[1:], platform.hjson)

    accel = described(name, members)
    header = f"{acc.name}_csr.h"
    streamer_h, streamer_last = streamer_header(movers(template), SNAX_CSR_FIRST, shown)
    assert streamer_last == layout.streamer_last, (streamer_last, layout)
    files = {
        names[0]: _hjson(shown, f"The cluster {CLUSTER}: {platform.hjson} with platform "
                         f"{platform.name}'s values and the slot {slot} set for {accel}; its "
                         f"streamer in "
                         f"{names[1]}, its accelerator in {names[2]}.", cluster),
        names[1]: _hjson(shown, f"Slot {slot}'s streamer template: readers the input ports in the "
                         "BRM's port order, writers the output ports.", template),
        names[2]: acc.to_json(),
        f"include/{STREAMER_HEADER}": streamer_h,
        f"include/{header}": accelerator_header(acc, layout.first, shown),
        **{f"include/{f}": text for f, text in cluster_headers(cluster, shown).items()},
        DATA_H: data_h(point, shown, data, seed=seed),
        MAIN_C: main_c(point, tasks, cluster, acc, data, [DATA_H, header, STREAMER_HEADER],
                       shown),
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
        trace=trace_path(acc),
        csrs=layout,
        tasks=acc_tasks(tasks, members, acc),
    )
    (out / "manifest.json").write_text(manifest.to_json())
    return manifest
