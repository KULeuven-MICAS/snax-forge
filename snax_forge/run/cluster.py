"""A flow run's cluster, as the converter, the reports and the viewer name it (C§7, D137).

``load_cluster(cluster_hjson, design_point)`` reads the generator's files (the cluster
file, and what its slot refers to: per streamer set its streamer file, ``snax_streamer_sets``,
and ``accelerator.json``, D134, D157; and ``manifest.json`` beside them, for each set's
streamer trace path, ``csrs.streamers[].trace``, C§16) and the design point, and gives what the events need to
be named and read: the accelerator, the generated entry as one shell (accelerator.json's one
accelerator's ``name``, or with several the entry's ``tag``), on the core whose
``snax_acc_cfg`` holds the entry, ``<accelerator_slot>_<name>`` (the design point's
``name``); its units (``Unit``, one per group, D148: for a group of one unit the
instance, ``acc``, its group, its SNAX-GVSoC path, ``unit`` with one group and
``unit_<unit>`` with several; for a linked group, SNAX-GVSoC's one component for its units,
the group's name, ``group_<group>``; and its group's START, STATUS and BUSY_COUNTER); its movers, set by set, every unit's, readers first as its
set's streamer orders them (``reader<i>``, ``writer<j>`` counted within the set), each named
by its port as the design point names its streamers (``acc_a``, ``acc_out``), with its set,
its lanes (the crossbar ports ``acc_a.lane<l>``), FIFO depth, spatial bounds and temporal
loops (its CSRs); its CSRs (gen/csrs.py's layout: per set its streamer's range with its start,
busy and counter and its SNAX-GVSoC trace path, ``snax_acc<i>/streamer_<set>``, as the
manifest names it, ``sets``; the accelerators' R/W CSRs but the STARTs as ``fields``, and the
first group's START, STATUS and BUSY_COUNTER as start, busy and counter), from 0x3C0; the cores (``core<i>``) and the DMA's; the
L1 (the cluster's base and TCDM size: the iDMA's local area) and its banks, word-interleaved
(``bank_row``: bank = (a / w) mod B, row = a / (w B) for the offset a, w the bank's bytes, B
the banks); the DMA's beat (the cluster file's ``dma_data_width``). Nothing is inferred from
a run.

``Cluster`` round-trips through ``to_dict`` / ``from_dict``. A file that does not read as
expected is a ``RunError`` (``cluster.<rule>``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import hjson

from snax_forge.common import Missing, dig
from snax_forge.design.hjson_platform import CoreCount, dma_and_owner
from snax_forge.design.hjson_platform import cores as hjson_cores
from snax_forge.design.problems import DesignError
from snax_forge.gen.accelerator import Description
from snax_forge.gen.blocks import GenError
from snax_forge.gen.csrs import csr_layout

from .events import RunError

DMA = "dma"


def core_name(index: int) -> str:
    return f"core{index}"


@dataclass(frozen=True)
class Mover:
    name: str  # C§7: the instance and its port, acc_a
    gvsoc: str  # SNAX-GVSoC's: reader<i>, writer<j>
    port: str
    write: bool
    lanes: int
    fifo_depth: int
    spatial_bounds: tuple[int, ...]
    temporal_dim: int
    instance: str = ""  # the unit whose port it is (the design point's streamer's instance)
    set: str = ""  # its streamer set, accelerator.json's accelerator (D157)

    @property
    def ports(self) -> list[str]:
        """Its crossbar inputs, one per lane."""
        return [f"{self.name}.lane{lane}" for lane in range(self.lanes)]

    @property
    def fifo(self) -> str:
        return f"{self.name}.fifo"


@dataclass(frozen=True)
class StreamerCsrs:
    """One streamer set's streamer: its first CSR, then its start, busy and counter, which end
    its range."""

    name: str
    first: int
    start: int
    busy: int
    counter: int
    trace: str = ""  # its SNAX-GVSoC path, the manifest's: snax_acc<i>/streamer_<set>

    @property
    def last(self) -> int:
        return self.counter

    @property
    def xbar(self) -> str:
        """Its crossbar inputs' prefix: ``snax_acc<i>.streamer_<set>``, a mover's lane
        ``<xbar>.<mover>.lane<l>``."""
        return self.trace.replace("/", ".")


@dataclass(frozen=True)
class Csrs:
    """The entry's CSR numbers: per set its streamer's range (``sets``, in set order), then the
    shell's fields, start, busy and counter (the first group's)."""

    sets: list[StreamerCsrs]
    fields: dict[str, int]
    start: int
    busy: int
    counter: int

    def _one(self) -> StreamerCsrs:
        if len(self.sets) != 1:
            raise RunError("cluster.sets", f"{len(self.sets)} streamer sets "
                           f"{[s.name for s in self.sets]}: name the set")  # fmt: skip
        return self.sets[0]

    # The one set's streamer, for an entry of one set
    @property
    def streamer_first(self) -> int:
        return self._one().first

    @property
    def streamer_start(self) -> int:
        return self._one().start

    @property
    def streamer_busy(self) -> int:
        return self._one().busy

    @property
    def streamer_counter(self) -> int:
        return self._one().counter

    @property
    def streamer_last(self) -> int:
        return self._one().last

    def set_of(self, csr: int) -> StreamerCsrs | None:
        """The set whose streamer range holds CSR ``csr``."""
        return next((s for s in self.sets if s.first <= csr <= s.last), None)

    def named(self, name: str) -> StreamerCsrs:
        """The set ``name``'s streamer."""
        for s in self.sets:
            if s.name == name:
                return s
        raise RunError("cluster.sets", f"no streamer set {name!r} ({[s.name for s in self.sets]})")


@dataclass(frozen=True)
class Unit:
    """One unit of the accelerator: its instance, its group, its SNAX-GVSoC component
    below ``snax_acc<i>`` and its group's START, STATUS and BUSY_COUNTER, which bound the group's
    CSRs (C§17)."""

    name: str  # C§7: the instance, acc
    group: str
    path: str  # unit, or unit_<unit> with several groups
    start: int
    busy: int
    counter: int


@dataclass(frozen=True)
class Accelerator:
    name: str  # accelerator.json's one accelerator's (D148), or the entry's tag with several
    core: int
    slot: str
    description: dict[str, Any]  # the slot's accelerator.json, as written
    movers: list[Mover]
    csrs: Csrs
    units: list[Unit] = field(default_factory=list)

    @property
    def streamer(self) -> str:
        """The one set's streamer as a block (its CSRs), ``<set>_streamer``."""
        return f"{self.csrs._one().name}_streamer"

    def groups(self) -> list[dict[str, Any]]:
        """accelerator.json's groups, set after set."""
        return [g for a in self.description.get("accelerators", []) for g in a["groups"]]

    def accelerators(self) -> list[tuple[str, list[str]]]:
        """accelerator.json's accelerators, one per streamer set in CSR order: each its name and
        its groups' names (D157)."""
        return [(a["name"], [g["name"] for g in a["groups"]])
                for a in self.description.get("accelerators", [])]  # fmt: skip

    def accelerator_of(self, group: str) -> str | None:
        """The accelerator (its streamer set) a group is in."""
        return next((a for a, gs in self.accelerators() if group in gs), None)

    def window(self, set_: str) -> tuple[int, int]:
        """A set's CSRs, first and last: its streamer's, then its accelerator's own (C§16)."""
        groups = dict(self.accelerators())[set_]
        return (self.csrs.named(set_).first,
                max(u.counter for u in self.units if u.group in groups))  # fmt: skip

    @property
    def unit_names(self) -> list[str]:
        return [u.name for u in self.units]

    def unit_at(self, path: str) -> Unit | None:
        """The unit of a SNAX-GVSoC component below ``snax_acc<i>``, or None."""
        return next((u for u in self.units if u.path == path), None)

    def block_of(self, csr: int) -> str | None:
        """The block a CSR number belongs to: the streamer's, or the unit's whose group's CSRs
        hold it (its shell's)."""
        s = self.csrs.set_of(csr)
        if s is not None:
            return f"{s.name}_streamer"
        for u in self.units:
            if u.start <= csr <= u.counter:
                return u.name
        return None

    def members(self, unit: str) -> list[str]:
        """The instances of a unit's group: the unit itself, or a linked group's units."""
        for g in self.groups():
            names = [u["name"] for u in g["units"]]
            if unit == g["name"] or unit in names:
                return names if len(names) > 1 else [unit]
        return [unit]

    def movers_of(self, unit: str) -> list[Mover]:
        """A unit's movers, readers first as the streamer orders them: those of its group's
        instances (with one unit and no instance recorded, every mover)."""
        if len(self.units) == 1 and not any(m.instance for m in self.movers):
            return list(self.movers)
        mine = set(self.members(unit))
        return [m for m in self.movers if m.instance in mine]

    def unit_of_mover(self, mover: str) -> str | None:
        """The unit (its C§7 name) a mover belongs to."""
        for u in self.units:
            if any(m.name == mover for m in self.movers_of(u.name)):
                return u.name
        return None

    def mover(self, gvsoc: str, set_: str | None = None) -> Mover:
        """The mover SNAX-GVSoC names ``gvsoc`` (``reader0``) in the set ``set_``'s streamer (with
        one set, its)."""
        if set_ is None:
            set_ = self.csrs._one().name
        for m in self.movers:
            if m.gvsoc == gvsoc and m.set == set_:
                return m
        raise RunError("cluster.mover", f"no mover {gvsoc!r} in set {set_}'s streamer "
                       f"({[m.gvsoc for m in self.movers if m.set == set_]})")  # fmt: skip

    def set_at(self, path: str) -> StreamerCsrs | None:
        """The set whose streamer is at the SNAX-GVSoC path ``path`` (its trace path, or a
        path ending in it), or None."""
        return next(
            (
                s
                for s in self.csrs.sets
                if s.trace and (path == s.trace or path.endswith("/" + s.trace))
            ),
            None,
        )


@dataclass(frozen=True)
class Cluster:
    design: str
    nb_cores: int
    dma_core: int
    accelerator: Accelerator
    l1_base: int
    l1_size: int  # bytes
    banks: int
    bank_bytes: int
    dma_beat_bytes: int
    meta: dict[str, Any] = field(default_factory=dict)  # free: e.g. the files it was read from

    @property
    def cores(self) -> list[str]:
        return [core_name(i) for i in range(self.nb_cores)]

    @property
    def sources(self) -> list[str]:
        """Every C§7 source, in the cluster's order: the accelerator, its movers, the DMA, the
        cores."""
        a = self.accelerator
        return [*a.unit_names, *(m.name for m in a.movers), DMA, *self.cores]

    def is_l1(self, addr: int) -> bool:
        """In the cluster's L1, the iDMA's local area (the rule its summary counts bytes by)."""
        return self.l1_base <= addr < self.l1_base + self.l1_size

    def bank_row(self, addr: int) -> tuple[int, int]:
        """The bank and row of an L1 address, word-interleaved (module doc)."""
        offset = addr - self.l1_base
        return (offset // self.bank_bytes) % self.banks, offset // (self.bank_bytes * self.banks)

    def to_dict(self) -> dict[str, Any]:
        a = self.accelerator
        c = a.csrs
        return {
            "design": self.design, "nb_cores": self.nb_cores, "dma_core": self.dma_core,
            "accelerator": {
                "name": a.name, "core": a.core, "slot": a.slot, "description": dict(a.description),
                "movers": [{"name": m.name, "gvsoc": m.gvsoc, "port": m.port, "write": m.write,
                            "lanes": m.lanes, "fifo_depth": m.fifo_depth,
                            "spatial_bounds": list(m.spatial_bounds),
                            "temporal_dim": m.temporal_dim, "instance": m.instance,
                            "set": m.set}
                           for m in a.movers],
                "csrs": {"sets": [{"name": x.name, "first": x.first, "start": x.start,
                                   "busy": x.busy, "counter": x.counter, "trace": x.trace}
                                  for x in c.sets],
                         "fields": dict(c.fields), "start": c.start, "busy": c.busy,
                         "counter": c.counter},
                "units": [{"name": u.name, "group": u.group, "path": u.path, "start": u.start,
                           "busy": u.busy, "counter": u.counter} for u in a.units],
            },
            "l1_base": self.l1_base, "l1_size": self.l1_size, "banks": self.banks,
            "bank_bytes": self.bank_bytes, "dma_beat_bytes": self.dma_beat_bytes,
            "meta": dict(self.meta),
        }  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Cluster:
        a = d["accelerator"]
        acc = Accelerator(a["name"], a["core"], a["slot"], dict(a["description"]),
                          [Mover(**{**m, "spatial_bounds": tuple(m["spatial_bounds"])})
                           for m in a["movers"]],
                          Csrs(**{**a["csrs"], "fields": dict(a["csrs"]["fields"]),
                                  "sets": [StreamerCsrs(**x) for x in a["csrs"]["sets"]]}),
                          [Unit(**u) for u in a.get("units", [])])  # fmt: skip
        return cls(d["design"], d["nb_cores"], d["dma_core"], acc, d["l1_base"], d["l1_size"],
                   d["banks"], d["bank_bytes"], d["dma_beat_bytes"],
                   dict(d.get("meta", {})))  # fmt: skip


def _get(d: Any, path: str, what: str) -> Any:
    try:
        return dig(d, path)
    except Missing:
        raise RunError("cluster.key", f"{what}: no {path}") from None


def _description(description: dict[str, Any]) -> Description:
    try:
        return Description.from_dict(description)
    except (GenError, KeyError, TypeError, ValueError) as e:
        raise RunError("cluster.csrs", f"accelerator.json: {e!r}") from None


def csrs_of(templates: dict[str, dict[str, Any]], description: dict[str, Any],
            instance: str, traces: dict[str, str] | None = None) -> Csrs:  # fmt: skip
    """The CSR numbers of the sets' streamer templates (by set) and the slot's accelerator.json,
    ``instance``'s group's, with each set's streamer trace path from ``traces`` (module doc)."""
    acc = _description(description)
    try:
        c = csr_layout(templates, acc)
        start, busy, counter = (c.accelerator[acc.find(instance, n)]
                                for n in ("START", "STATUS", "BUSY_COUNTER"))  # fmt: skip
    except (GenError, KeyError, ValueError) as e:
        raise RunError("cluster.csrs", f"accelerator.json: {e}") from None
    starts = {c.accelerator[acc.find(u.name, "START")] for u in acc.units}
    fields = {k: v for k, v in c.accelerator.items() if "_RW_" in k and v not in starts}
    sets = [StreamerCsrs(r.name, r.first, r.last - 2, r.last - 1, r.last,
                         (traces or {}).get(r.name, "")) for r in c.streamers]  # fmt: skip
    return Csrs(sets, fields, start, busy, counter)


def traces_of(manifest: dict[str, Any], templates: dict[str, dict[str, Any]],
              description: dict[str, Any]) -> dict[str, str]:  # fmt: skip
    """Each set's streamer trace path, by set, from the manifest's ``csrs.streamers`` (C§16),
    whose sets and CSR ranges must be the layout's."""
    try:
        rows = dig(manifest, "csrs.streamers")
    except (Missing, TypeError):
        rows = None
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise RunError("cluster.manifest", f"the manifest's csrs.streamers: {rows!r}")
    try:
        layout = csr_layout(templates, _description(description)).streamers
    except (GenError, KeyError, ValueError) as e:
        raise RunError("cluster.csrs", f"accelerator.json: {e}") from None
    want = [(r.name, r.first, r.last) for r in layout]
    got = [(r.get("name"), r.get("first"), r.get("last")) for r in rows]
    if got != want:
        raise RunError("cluster.manifest", f"the manifest's streamers {got}, the layout's {want}")
    if not all(isinstance(r.get("trace"), str) and r["trace"] for r in rows):
        raise RunError("cluster.manifest", f"a streamer without its trace path: {rows!r}")
    return {r["name"]: r["trace"] for r in rows}


def _groups(description: dict[str, Any]) -> list[dict[str, Any]]:
    """accelerator.json's groups, set after set."""
    return [g for a in description.get("accelerators", []) for g in a["groups"]]


def units_of(templates: dict[str, dict[str, Any]], description: dict[str, Any]) -> list[Unit]:
    """The accelerators' units, one per group in order, with their group's CSRs: a group of one
    unit by its instance, a linked group by its name at ``group_<group>`` (module doc)."""
    groups = _groups(description)
    out = []
    for g in groups:
        members = g["units"]
        c = csrs_of(templates, description, members[0]["name"])
        if len(members) > 1:
            out.append(Unit(g["name"], g["name"], f"group_{g['name']}", c.start, c.busy,
                            c.counter))  # fmt: skip
            continue
        u = members[0]
        path = "unit" if len(groups) == 1 else f"unit_{u['name']}"
        out.append(Unit(u["name"], g["name"], path, c.start, c.busy, c.counter))
    return out


def linked(description: dict[str, Any]) -> bool:
    """An accelerator has a linked group (several units in one group)."""
    return any(len(g["units"]) > 1 for g in _groups(description))


def _cores(cluster: dict[str, Any]) -> list[dict[str, Any]]:
    """The cluster file's cores, in hive and core order."""
    try:
        return [core for _, core in hjson_cores(cluster)]
    except DesignError as e:
        raise RunError("cluster.cores", f"the cluster file's cores do not read: {e}") from None


def _slots(cores: list[dict[str, Any]], slot: str) -> list[tuple[int, dict[str, Any]]]:
    """Every accelerator slot named ``slot``, with the index of its core."""
    return [(i, a) for i, c in enumerate(cores) for a in c.get("snax_acc_cfg", [])
            if a.get("snax_acc_name") == slot]  # fmt: skip


def entry_name(point: dict[str, Any], what: str) -> str:
    """The generated entry's name, ``<accelerator_slot>_<name>`` (D131), from the design point's
    platform's slot and its name, the design's."""
    return f"{_get(point, 'platform.accelerator_slot', what)}_{_get(point, 'name', what)}"


def cluster_from(cluster: dict[str, Any], templates: dict[str, dict[str, Any]],
                 description: dict[str, Any], point: dict[str, Any],
                 manifest: dict[str, Any]) -> Cluster:  # fmt: skip
    """The description from the files' contents (``templates`` each set's streamer file, by
    set; ``manifest`` the generator's, for the streamers' trace paths) and the design point's
    (module doc)."""
    slot = entry_name(point, "the design point")
    if description.get("tag") != slot:
        raise RunError("cluster.slot", f"accelerator.json's tag {description.get('tag')!r}, the "
                       f"design point's entry {slot!r}")  # fmt: skip
    flat = _cores(cluster)
    try:
        dma, owner = dma_and_owner(flat, slot)
    except CoreCount as e:
        raise RunError("cluster.cores", f"xdma on cores {e.dma}, the slot {slot} on cores "
                       f"{e.owner}; a flow run has one of each") from None  # fmt: skip

    streamers = _get(point, "streamers", "the design point")
    names = list(dict.fromkeys(s["instance"] for s in streamers.values()))
    units = units_of(templates, description)
    members = [u["name"] for g in _groups(description) for u in g["units"]]
    if names != [m for m in members if m in names] or not set(names) <= set(members):
        raise RunError("cluster.instances", f"streamers of instances {names}, accelerator.json's "
                       f"units {members}")  # fmt: skip
    movers = []
    for a in description["accelerators"]:
        mine_units = {u["name"] for g in a["groups"] for u in g["units"]}
        template = templates.get(a["name"])
        if template is None:
            raise RunError("cluster.sets", f"no streamer file for the set {a['name']!r}")
        for write, group, kind in ((False, "data_reader_params", "reader"),
                                   (True, "data_writer_params", "writer")):  # fmt: skip
            mine = [(n, s) for n, s in streamers.items()
                    if bool(s["write"]) == write and s["instance"] in mine_units]  # fmt: skip
            params = template.get(group, {})
            if len(params.get("num_channel", [])) != len(mine):
                raise RunError("cluster.movers", f"set {a['name']}'s streamer file has "
                               f"{len(params.get('num_channel', []))} {kind}s, the design point "
                               f"{len(mine)}")  # fmt: skip
            for i, (n, s) in enumerate(mine):
                movers.append(Mover(n, f"{kind}{i}", s["port"], write, params["num_channel"][i],
                                    params["fifo_depth"][i], tuple(params["spatial_bounds"][i]),
                                    params["temporal_dim"][i], s["instance"], a["name"]))  # fmt: skip

    tcdm = _get(cluster, "cluster.tcdm", "the cluster file")
    accs = description["accelerators"]
    name = accs[0]["name"] if len(accs) == 1 else description["tag"]
    traces = traces_of(manifest, templates, description)
    acc = Accelerator(name, owner, slot, dict(description), movers,
                      csrs_of(templates, description, members[0], traces), units)  # fmt: skip
    return Cluster(
        design=point.get("name", ""), nb_cores=len(flat), dma_core=dma, accelerator=acc,
        l1_base=_get(cluster, "cluster.cluster_base_addr", "the cluster file"),
        l1_size=tcdm["size"] * 1024, banks=tcdm["banks"],
        bank_bytes=_get(cluster, "cluster.data_width", "the cluster file") // 8,
        dma_beat_bytes=_get(cluster, "cluster.dma_data_width", "the cluster file") // 8,
    )  # fmt: skip


def _ref(slot: dict[str, Any], key: str, base: Path) -> dict[str, Any]:
    ref = slot.get(key)
    if not (isinstance(ref, dict) and isinstance(ref.get("$ref"), str) and "#" not in ref["$ref"]):
        raise RunError("cluster.ref", f"the slot's {key} is not a whole file's $ref: {ref!r}")
    path = base / ref["$ref"]
    if not path.is_file():
        raise RunError("cluster.ref", f"{key}: no {path}")
    return hjson.loads(path.read_text())


def load_cluster(cluster_hjson: str | Path, design_point: str | Path) -> Cluster:
    """The description from the generator's cluster file (the files its slot refers to beside
    it: each streamer set's streamer file and accelerator.json; and its manifest.json) and the
    design point's file."""
    cp, pp = Path(cluster_hjson), Path(design_point)
    cluster = hjson.loads(cp.read_text())
    point = json.loads(pp.read_text())
    slot_name = entry_name(point, str(pp))
    slots = [a for _, a in _slots(_cores(cluster), slot_name)]
    if len(slots) != 1:
        raise RunError("cluster.slot", f"{cp}: {len(slots)} slots named {slot_name!r}")
    sets = slots[0].get("snax_streamer_sets")
    if not isinstance(sets, list) or not sets:
        raise RunError("cluster.sets", f"{cp}: the slot {slot_name} has no snax_streamer_sets "
                       f"(the sets form, D157): {sets!r}")  # fmt: skip
    templates = {}
    for k, x in enumerate(sets):
        if not isinstance(x, dict) or not isinstance(x.get("name"), str):
            raise RunError("cluster.sets", f"{cp}: snax_streamer_sets[{k}] has no name: {x!r}")
        templates[x["name"]] = _ref(x, "snax_streamer_cfg", cp.parent)
    description = _ref(slots[0], "snax_datapath_cfg", cp.parent)
    if [a.get("name") for a in description.get("accelerators", [])] != list(templates):
        raise RunError("cluster.sets", f"{cp}: the sets {list(templates)}, accelerator.json's "
                       f"accelerators {[a.get('name') for a in description.get('accelerators', [])]}")  # fmt: skip
    mp = cp.parent / "manifest.json"
    if not mp.is_file():
        raise RunError("cluster.manifest", f"no {mp}: the generator's manifest names each set's "
                       "streamer trace path")  # fmt: skip
    manifest = json.loads(mp.read_text())
    return replace(
        cluster_from(cluster, templates, description, point, manifest), meta={"cluster": cp.name}
    )
