"""A flow run's cluster, as the converter, the reports and the viewer name it (C§7, D137).

``load_cluster(cluster_hjson, design_point)`` reads the generator's three files (the
cluster file, and the streamer file and ``accelerator.json`` its slot refers to, D134) and the design
point, and gives what the events need to be named and read: the accelerator (the design
point's one, accelerator.json's ``name``) on the core whose ``snax_acc_cfg`` holds the
platform's slot; its units (``Unit``, one per group, D148: for a group of one unit the
instance, ``acc``, its group, its SNAX-GVSoC path, ``unit`` with one group and
``unit_<unit>`` with several; for a linked group, SNAX-GVSoC's one component for its units,
the group's name, ``group_<group>``; and its group's START, STATUS and COUNTER); its movers, every unit's, readers first as the streamer
orders them, each named by its port as the design point names its streamers (``acc_a``,
``acc_out``), with its lanes (the crossbar ports ``acc_a.lane<l>``), FIFO depth, spatial
bounds and temporal loops (its CSRs); its CSRs (the streamer's, then the accelerator's from
accelerator.json, its R/W CSRs but START as ``fields``, and the first group's START, STATUS
and COUNTER as start, busy and counter), from 0x3C0; the cores (``core<i>``) and the DMA's; the
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
from snax_forge.gen.accelerator import Accelerator as Description
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

    @property
    def ports(self) -> list[str]:
        """Its crossbar inputs, one per lane."""
        return [f"{self.name}.lane{lane}" for lane in range(self.lanes)]

    @property
    def fifo(self) -> str:
        return f"{self.name}.fifo"


@dataclass(frozen=True)
class Csrs:
    """The accelerator's CSR numbers: the streamer's range with its start, busy and counter,
    then the shell's fields, start, busy and counter."""

    streamer_first: int
    streamer_start: int
    streamer_busy: int
    streamer_counter: int
    fields: dict[str, int]
    start: int
    busy: int
    counter: int

    @property
    def streamer_last(self) -> int:
        return self.streamer_counter


@dataclass(frozen=True)
class Unit:
    """One unit of the accelerator (D148): its instance, its group, its SNAX-GVSoC component
    below ``snax_acc<i>`` and its group's START, STATUS and COUNTER, which bound the group's
    CSRs (C§17)."""

    name: str  # C§7: the instance, acc
    group: str
    path: str  # unit, or unit_<unit> with several groups
    start: int
    busy: int
    counter: int


@dataclass(frozen=True)
class Accelerator:
    name: str  # accelerator.json's: the instance with one group, encapsulate's with several
    core: int
    slot: str
    description: dict[str, Any]  # the slot's accelerator.json, as written
    movers: list[Mover]
    csrs: Csrs
    units: list[Unit] = field(default_factory=list)

    @property
    def streamer(self) -> str:
        """The streamer as a block (its CSRs), ``<accelerator>_streamer``."""
        return f"{self.name}_streamer"

    @property
    def unit_names(self) -> list[str]:
        return [u.name for u in self.units]

    def unit_at(self, path: str) -> Unit | None:
        """The unit of a SNAX-GVSoC component below ``snax_acc<i>``, or None."""
        return next((u for u in self.units if u.path == path), None)

    def block_of(self, csr: int) -> str | None:
        """The block a CSR number belongs to: the streamer's, or the unit's whose group's CSRs
        hold it (its shell's)."""
        c = self.csrs
        if c.streamer_first <= csr <= c.streamer_last:
            return self.streamer
        for u in self.units:
            if u.start <= csr <= u.counter:
                return u.name
        return None

    def mover(self, gvsoc: str) -> Mover:
        for m in self.movers:
            if m.gvsoc == gvsoc:
                return m
        raise RunError("cluster.mover", f"no mover {gvsoc!r} in {self.name}'s streamer "
                       f"({[m.gvsoc for m in self.movers]})")  # fmt: skip


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
                            "temporal_dim": m.temporal_dim} for m in a.movers],
                "csrs": {"streamer_first": c.streamer_first, "streamer_start": c.streamer_start,
                         "streamer_busy": c.streamer_busy,
                         "streamer_counter": c.streamer_counter, "fields": dict(c.fields),
                         "start": c.start, "busy": c.busy, "counter": c.counter},
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
                          Csrs(**{**a["csrs"], "fields": dict(a["csrs"]["fields"])}),
                          [Unit(**u) for u in a.get("units", [])])  # fmt: skip
        return cls(d["design"], d["nb_cores"], d["dma_core"], acc, d["l1_base"], d["l1_size"],
                   d["banks"], d["bank_bytes"], d["dma_beat_bytes"],
                   dict(d.get("meta", {})))  # fmt: skip


def _get(d: Any, path: str, what: str) -> Any:
    try:
        return dig(d, path)
    except Missing:
        raise RunError("cluster.key", f"{what}: no {path}") from None


def csrs_of(streamer: dict[str, Any], description: dict[str, Any], instance: str) -> Csrs:
    """The CSR numbers of a streamer template and the slot's accelerator.json, ``instance``'s
    group's (module doc)."""
    try:
        acc = Description.from_dict(description)
        c = csr_layout(streamer, acc)
        start, busy, counter = (c.accelerator[acc.find(instance, n)]
                                for n in ("START", "STATUS", "COUNTER"))  # fmt: skip
    except (GenError, KeyError, ValueError) as e:
        raise RunError("cluster.csrs", f"accelerator.json: {e}") from None
    starts = {c.accelerator[acc.find(u.name, "START")] for g in acc.groups for u in g.units}
    fields = {k: v for k, v in c.accelerator.items() if "_RW_" in k and v not in starts}
    s_start = c.streamer_last - 2  # the streamer's start, busy and counter end its range
    return Csrs(c.streamer_first, s_start, s_start + 1, s_start + 2, fields, start, busy,
                counter)  # fmt: skip


def units_of(streamer: dict[str, Any], description: dict[str, Any]) -> list[Unit]:
    """The accelerator's units, one per group in order, with their group's CSRs: a group of one
    unit by its instance, a linked group by its name at ``group_<group>`` (module doc)."""
    groups = description.get("groups", [])
    out = []
    for g in groups:
        members = g["units"]
        c = csrs_of(streamer, description, members[0]["name"])
        if len(members) > 1:
            out.append(Unit(g["name"], g["name"], f"group_{g['name']}", c.start, c.busy,
                            c.counter))  # fmt: skip
            continue
        u = members[0]
        path = "unit" if len(groups) == 1 else f"unit_{u['name']}"
        out.append(Unit(u["name"], g["name"], path, c.start, c.busy, c.counter))
    return out


def linked(description: dict[str, Any]) -> bool:
    """The accelerator has a linked group (several units in one group)."""
    return any(len(g["units"]) > 1 for g in description.get("groups", []))


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


def cluster_from(cluster: dict[str, Any], streamer: dict[str, Any], description: dict[str, Any],
                 point: dict[str, Any]) -> Cluster:  # fmt: skip
    """The description from the three files' contents and the design point's (module doc)."""
    slot = _get(point, "platform.accelerator_slot", "the design point")
    flat = _cores(cluster)
    try:
        dma, owner = dma_and_owner(flat, slot)
    except CoreCount as e:
        raise RunError("cluster.cores", f"xdma on cores {e.dma}, the slot {slot} on cores "
                       f"{e.owner}; a flow run has one of each") from None  # fmt: skip

    streamers = _get(point, "streamers", "the design point")
    names = list(dict.fromkeys(s["instance"] for s in streamers.values()))
    units = units_of(streamer, description)
    members = [u["name"] for g in description.get("groups", []) for u in g["units"]]
    if names != [m for m in members if m in names] or not set(names) <= set(members):
        raise RunError("cluster.instances", f"streamers of instances {names}, accelerator.json's "
                       f"units {members}; a flow run has one accelerator")  # fmt: skip
    movers = []
    for write, group, kind in ((False, "data_reader_params", "reader"),
                               (True, "data_writer_params", "writer")):  # fmt: skip
        mine = [(n, s) for n, s in streamers.items() if bool(s["write"]) == write]
        params = streamer.get(group, {})
        if len(params.get("num_channel", [])) != len(mine):
            raise RunError("cluster.movers", f"the streamer file has {len(params.get('num_channel', []))} "
                           f"{kind}s, the design point {len(mine)}")  # fmt: skip
        for i, (n, s) in enumerate(mine):
            movers.append(Mover(n, f"{kind}{i}", s["port"], write, params["num_channel"][i],
                                params["fifo_depth"][i], tuple(params["spatial_bounds"][i]),
                                params["temporal_dim"][i]))  # fmt: skip

    tcdm = _get(cluster, "cluster.tcdm", "the cluster file")
    acc = Accelerator(description["name"], owner, slot, dict(description), movers,
                      csrs_of(streamer, description, members[0]), units)  # fmt: skip
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
    """The description from the generator's cluster file (its slot's two files beside it) and
    the design point's file."""
    cp, pp = Path(cluster_hjson), Path(design_point)
    cluster = hjson.loads(cp.read_text())
    point = json.loads(pp.read_text())
    slot_name = _get(point, "platform.accelerator_slot", str(pp))
    slots = [a for _, a in _slots(_cores(cluster), slot_name)]
    if len(slots) != 1:
        raise RunError("cluster.slot", f"{cp}: {len(slots)} slots named {slot_name!r}")
    streamer = _ref(slots[0], "snax_streamer_cfg", cp.parent)
    description = _ref(slots[0], "snax_datapath_cfg", cp.parent)
    return replace(cluster_from(cluster, streamer, description, point), meta={"cluster": cp.name})
