"""A design point -> its cluster on SNAX-GVSoC: its description files, headers, data and program.

``generate`` writes into ``out_dir`` the files CONTRACTS, "Generated hjson files" lists: the
cluster file, one streamer hjson per set, ``accelerator.json`` (accelerator.py), the C headers
(headers.py), ``data.h`` (data.py), ``main.c`` and ``obs_marks.json`` (lower/c.py) and
``manifest.json`` (``Manifest``). SNAX-GVSoC reads the hjson at run time (snax-gvsoc
``snax/snax_cfg.py``), so nothing is compiled and no Python is generated. The cluster file gets
the platform's values by the reverse of design/hjson_platform.py's mapping.

A file is written only when its text changes (``write_changed``), and a file an earlier run into
``out_dir`` wrote (by its manifest) and this one does not is removed, so the same design point
again changes no file. The design point is named once, by its ``/``-separated path from ``root``
(snax-forge's root, or ``named_from``: the flow's folder), so one design point gives the same
bytes wherever written. Refusals are ``GenError``s; ``gen.point`` is a design point outside
``root``. Not verified: the order of ``spatial_bounds`` against snax_cluster's template for more
than one spatial dimension (every template so far has one).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hjson

from snax_forge.brm import Instance, port_dtypes
from snax_forge.common import (
    JsonFile,
    Missing,
    dig,
    generated_header,
    plain,
    repo_root,
    to_json,
    write_changed,
)
from snax_forge.design import DesignError
from snax_forge.design.hjson_platform import accelerators, platform_from_dict
from snax_forge.design.platform import Platform
from snax_forge.design.point import DesignPoint
from snax_forge.design.streamers import instance_nodes, shared_movers, streamer_name
from snax_forge.dfg.reference import Goldens
from snax_forge.estimate.cycles import EstimateError
from snax_forge.lower import CError
from snax_forge.lower.c import MAIN_C, OBS_MARKS, job_writes, program, task_beats
from snax_forge.lower.derive import task_list
from snax_forge.lower.tasks import TaskList

from .accelerator import ACCELERATOR_JSON, Description, describe, units
from .blocks import GenError, port_dtype
from .csrs import ACC_TRACE, CsrLayout, channels, csr_layout, movers
from .data import DATA_H, arrays, data_h
from .headers import (
    accelerator_header,
    accelerator_header_name,
    cluster_headers,
    streamer_header,
    streamer_header_name,
)

ROOT = repo_root()
OUT = ROOT / "out" / "gvsoc"

# The trace paths (csrs.py's ACC_TRACE the compute core's accelerator): the composite
# (snax/snax_accelerator.py) calls a unit `unit`, with several groups `unit_<unit>`
UNIT_TRACE = f"{ACC_TRACE}/unit"
# The generated cluster's name
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


def streamer_template(point: DesignPoint, found: Mapping[str, Instance],
                      acc: str | None = None) -> dict[str, Any]:  # fmt: skip
    """The streamer template of the accelerator's units ``found`` (``acc`` its name), in the
    hjson's keys; see the module doc. A shared accelerator's are its movers'."""
    groups: dict[str, list[dict[str, Any]]] = {"data_reader_params": [], "data_writer_params": []}
    for name, inst in found.items():
        _movers(point, name, inst, groups)
    shared = shared_movers(point.graph, point.streamers).get(acc) if acc is not None else None
    if shared:
        groups = {"data_reader_params": [], "data_writer_params": []}
        for m in shared:
            groups["data_writer_params" if m.write else "data_reader_params"].append({
                "spatial_bounds": list(m.spatial_bounds), "temporal_dim": m.temporal_dims,
                "num_channel": m.num_ch, "fifo_depth": m.fifo_depth, "configurable_channel": m.masked,
            })  # fmt: skip
        for movers in groups.values():
            if movers and not any(m["configurable_channel"] for m in movers):
                for m in movers:
                    del m["configurable_channel"]
    return {
        g: {k: [m[k] for m in movers] for k in movers[0]} for g, movers in groups.items() if movers
    }


def _movers(point: DesignPoint, name: str, inst: Instance,
            groups: dict[str, list[dict[str, Any]]]) -> None:  # fmt: skip
    """Unit ``name``'s movers onto ``groups`` (readers, writers), in port order."""
    dtypes = node_dtypes(point, name)
    for p in inst.ports:
        s = point.streamers.get(streamer_name(name, p.name))
        if s is None:  # a port on a link: no streamer
            continue
        port_dtype(dtypes[p.name], f"port {p.name}")  # intN or uintN (gen.dtype)
        s0, *rest = s.spatial_bounds
        if s0 % s.num_slc:
            raise GenError(
                "gen.lanes", f"streamer {s.name}: its fastest spatial bound {s0} is not whole "
                f"words of num_slc = {s.num_slc} {dtypes[p.name]} slices",
            )  # fmt: skip
        mover = {
            "spatial_bounds": [s0 // s.num_slc, *rest],  # in words
            "temporal_dim": s.temporal_dims,
            "num_channel": s.num_ch,
            "fifo_depth": s.options.fifo_depth,
        }
        groups["data_writer_params" if s.write else "data_reader_params"].append(mover)


def acc_tasks(tasks: TaskList, found: Mapping[str, Instance], acc: Description) -> list[AccTask]:
    """The accelerator tasks of the derived task list, in order, with their group's CSR
    values (lower/c.py's ``job_writes``: a CSR of a unit of a linked group named with the unit,
    ``MAC_JOB_BEATS``) and, with several groups, its group's trace path."""
    several = len(acc.groups) > 1
    fixed = acc.fixed_folds
    out = []
    for s in tasks.accel_steps():
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


def trace_path(acc: Description) -> str:
    """The manifest's trace path: one group of one unit ``snax_acc0/unit``; one linked group
    ``snax_acc0/group_<group>``, SNAX-GVSoC's one component for its units; several groups the
    accelerator's, ``snax_acc0``, each task with its own (over the whole design)."""
    if len(acc.groups) > 1:
        return ACC_TRACE
    (g,) = acc.groups
    return UNIT_TRACE if len(g.units) == 1 else f"{ACC_TRACE}/group_{g.name}"


def described(found: Mapping[str, Mapping[str, Instance]]) -> str:
    """The accelerators as the cluster file's header comment names them, set by set."""

    def one(unit: str, inst: Instance) -> str:
        return (f"instance {unit}, BRM {inst.brm.name}, implementation {inst.implementation}, "
                f"datapath {inst.impl.binding['datapath']}")  # fmt: skip

    def acc(name: str, units: Mapping[str, Instance]) -> str:
        if len(units) == 1:
            return f"accelerator {name}: " + one(*next(iter(units.items())))
        return f"accelerator {name} of " + "; ".join(f"group {one(u, i)}" for u, i in units.items())

    return "; ".join(acc(n, u) for n, u in found.items())


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

    def refuse(n: int) -> GenError:
        return GenError(
            "gen.source", f"{where}: {'.'.join(head[:n])} is not an object written out in the "
            "file, so the platform's value cannot be written into it",
        )  # fmt: skip

    node: Any = d
    for i in range(len(head)):  # each object on the way, as it is written (no $ref)
        try:
            node = dig(node, head[i])
        except Missing:
            raise refuse(i + 1) from None
        if not isinstance(node, Mapping) or "$ref" in node:
            raise refuse(i + 1)
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
             templates: Mapping[str, dict[str, Any]], acc: Description, files: Mapping[str, str],
             where: str) -> None:  # fmt: skip
    """The slot (``find_slot``'s, in ``d``) named ``name`` (the tag) and set for the design,
    ``files`` each set's streamer file by its name and ``accelerator.json``'s; see the module
    doc."""
    widths = [channels(templates[a.name]) for a in acc.accelerators]
    slot["snax_tcdm_ports"] = sum(widths)
    if "sparse_interconnect_config" in slot:
        groups = slot["sparse_interconnect_config"]
        if not (isinstance(groups, list) and len(groups) == 1 and isinstance(groups[0], list)
                and len(groups[0]) == 2):  # fmt: skip
            raise GenError(
                "gen.source", f"{where}: {at}.sparse_interconnect_config {groups!r} is not one "
                "group [width, granularity]",
            )  # fmt: skip
        slot["sparse_interconnect_config"] = [[w, groups[0][1]] for w in widths]
    old = slot.get("snax_streamer_cfg")
    sets = [{"name": a.name, "snax_streamer_cfg": {"$ref": files[a.name]},
             "snax_num_rw_csr": a.rw, "snax_num_ro_csr": a.ro} for a in acc.accelerators]  # fmt: skip
    last = {
        "snax_disable_csr_manager": True,
        "snax_forge_rtl": RTL_FILE_LIST,
        "snax_streamer_sets": sets,
        "snax_datapath_cfg": {"$ref": files[ACCELERATOR_JSON]},
    }
    dropped = ("snax_acc_name", "bender_target", "snax_streamer_cfg", "snax_num_rw_csr",
               "snax_num_ro_csr", *last)  # fmt: skip
    rest = {k: v for k, v in slot.items() if k not in dropped}
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
) -> tuple[str, DesignPoint, Description]:
    """The design point's name (``point_name`` from ``named_from``, else ``root``), the point
    and its accelerators' description (accelerator.py's ``describe``)."""
    shown = point_name(Path(design_point), root if named_from is None else Path(named_from))
    point = DesignPoint.load(design_point)
    return shown, point, describe(point, shown)


def cli(write: Callable[[], int]) -> int:
    """A generator command: ``write()``, or 1 with the refusal on stderr (its rule named)."""
    try:
        return write()
    except (GenError, CError, DesignError, EstimateError) as e:
        print(e, file=sys.stderr)
        return 1


# A flow's design step writes its design point to <flow folder>/design/ (flow/run.py)
FLOW_DESIGN = "design"


def out_dir(given: Path | None, under: Path, design_point: Path) -> Path:
    """``given``, else ``under/<folder>``, the design point's folder: the design step's
    ``out/design/<folder>/``, or a flow's ``out/flow/<folder>/`` for the flow's
    ``<folder>/design/design_point.json``; either keeps the ``--set`` suffix."""
    if given:
        return given
    folder = Path(design_point).resolve().parent
    return under / (folder.parent.name if folder.name == FLOW_DESIGN else folder.name)


def _hjson(point_path: str, what: str, value: Any) -> str:
    """An hjson file: the header comment, then ``value`` as JSON (which hjson reads)."""
    return generated_header(point_path, what) + to_json(value)


# =============================================================================
# generate
# =============================================================================


# NB (D134): snax-gvsoc checks this output in as fixtures, compared byte for byte; a change to
# any generated text needs them written again.
def generate(
    design_point: str | Path, out_dir: str | Path, root: str | Path | None = None, *,
    named_from: str | Path | None = None, kernel: str | None = None, seed: int = 0,
    tasks: TaskList | None = None, goldens: Goldens | None = None,
) -> Manifest:  # fmt: skip
    """Write the three description files, the headers, data.h, main.c and the manifest to
    ``out_dir``; see the module doc. ``named_from`` is the folder the design point is named
    from (``root`` by default; the flow's folder for a flow); ``kernel`` and ``seed``
    go to data.h's inputs (data.py's ``arrays``). ``tasks`` (the design point's task list) and
    ``goldens`` (data.h's inputs and the reference executor's outputs) are what the caller has
    made already: the flow, which needs both before."""
    point_path, out = Path(design_point), Path(out_dir)
    root = ROOT if root is None else Path(root)
    shown = point_name(point_path, root if named_from is None else Path(named_from))
    point = DesignPoint.load(point_path)
    platform = point.platform
    cluster = source(platform, root)  # first: a platform without one is gen.source (RUN.md §6)
    acc = describe(point, shown)
    found = units(point)
    members = {u: inst for mine in found.values() for u, inst in mine.items()}
    templates = {a: streamer_template(point, mine, a) for a, mine in found.items()}
    layout = csr_layout(templates, acc)
    tasks = task_list(point) if tasks is None else tasks
    data = arrays(point, seed, kernel, goldens)

    at, slot = find_slot(cluster, platform)
    write_back(cluster, platform)
    rename_cluster(cluster, platform.hjson)
    cluster_file = f"{platform.base or platform.name}.hjson"
    streamer_files = {a: f"{acc.tag}_streamer_{a}.hjson" for a in found}
    set_slot(cluster, at, slot, acc.tag, templates, acc,
             {**streamer_files, ACCELERATOR_JSON: ACCELERATOR_JSON}, platform.hjson)  # fmt: skip

    files = {
        cluster_file: _hjson(shown, f"The cluster {CLUSTER}: {platform.hjson} with platform "
                             f"{platform.name}'s values and its slot the entry {acc.tag}, one "
                             f"streamer set per accelerator: {described(found)}; each set's "
                             f"streamer in {acc.tag}_streamer_<set>.hjson, the accelerators in "
                             f"{ACCELERATOR_JSON}.", cluster),
        **{f: _hjson(shown, f"Streamer set {a}'s streamer template: " + (
                     "readers its input ports in the BRM's port order, writers its output ports."
                     if acc.port_mux_sel(a) is None else "its movers, which its groups share "
                     ", readers then writers, as accelerator.json's movers."), templates[a])
           for a, f in streamer_files.items()},
        ACCELERATOR_JSON: acc.to_json(),
    }  # fmt: skip
    includes = [DATA_H]
    for a in acc.accelerators:
        r = layout.streamer(a.name)
        text, last = streamer_header(movers(templates[a.name]), r.first, f"{a.name.upper()}_",
                                     shown)  # fmt: skip
        assert last == r.last, (last, r)
        files[f"include/{streamer_header_name(a.name)}"] = text
        files[f"include/{accelerator_header_name(a.name)}"] = accelerator_header(
            a, layout.accelerator, shown
        )
        includes += [accelerator_header_name(a.name), streamer_header_name(a.name)]
    files |= {f"include/{f}": text for f, text in cluster_headers(cluster, shown).items()}
    files[DATA_H] = data_h(point, shown, data, seed=seed)
    files[MAIN_C], marks = program(point, tasks, cluster, acc, data, includes, shown)
    files[OBS_MARKS] = json.dumps(marks, indent=1) + "\n"
    before = _written(out)
    (out / "include").mkdir(parents=True, exist_ok=True)
    for f, text in files.items():
        write_changed(out / f, text)
    manifest = Manifest(
        design=point.name,
        design_point=shown,
        symbols=dict(point.graph.symbols),
        cluster=cluster_file,
        files=[*files, "manifest.json"],
        program=MAIN_C,
        trace=trace_path(acc),
        csrs=layout,
        tasks=acc_tasks(tasks, members, acc),
    )
    write_changed(out / "manifest.json", manifest.to_json())
    for f in before - set(manifest.files):
        (out / f).unlink(missing_ok=True)
    return manifest


def _written(out: Path) -> set[str]:
    """The files an earlier generation into ``out`` wrote, by its manifest (none without one)."""
    try:
        return set(Manifest.load(out / "manifest.json").files)
    except (OSError, ValueError, KeyError, TypeError):
        return set()
