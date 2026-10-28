"""A design point -> its accelerator's one description, ``accelerator.json`` (D145, C§17).

    describe(point, point_path) -> Accelerator

snax-forge writes it once (gvsoc.py, at the root of every gen-gvsoc folder and
bundle; gen-rtl the same bytes, rtl.py) and SNAX-GVSoC (snax-gvsoc
``snax/snax_cfg.py``) and Emit (``hw/chisel/``) read it. Plain JSON, concrete
values only, in this key order:

    name          the accelerator: the header's <ACC>; ``encapsulate``'s
                  accelerator, else its one unit's instance
    design_point  the design point by its path from the root (gvsoc.py)
    tag           the platform's accelerator_slot
    inputs        the outer ports, "<unit>.<port>", in the streamer's reader
    outputs       and writer order (the BRM's port order)
    groups        in CSR order; each name, csrs, units, links ([] until B3)

A unit: ``name`` (the instance), ``brm``, ``datapath`` (the binding's),
``ports`` (per BRM port in its order: ``port``, ``dtype``, ``lanes``, the
datapath's adapter's, blocks.py), ``opType`` (an elementwise unit only),
``params``, ``timing`` (``latency``, ``ii``) and ``csrs`` (its R/W CSRs:
``JOB_BEATS`` when an output port's rate is a runtime param, a fold).

**The CSRs** (D146), offsets from the shell's first CSR, per group in order:
``START`` and ``JOBS`` (R/W, 32 bits), each unit's R/W CSRs, ``STATUS`` and
``COUNTER`` (R/O). A CSR is ``name``, ``access`` (``rw``, ``ro``),
``offset``, ``width`` (R/W only) and ``pin`` (a scalar's datapath pin, none
yet). ``from_dict`` checks the offsets contiguous from 0 in that order and
never recomputes them. ``define`` names a CSR as the header does:
``<ACC>_<RW|RO>_[<GROUP>_][<UNIT>_]<NAME>``, the group part dropped when the
accelerator has one group, the unit part when the group has one unit.

**The groups** (D146, D148): the design point's one accelerator
(lower/c.py's ``accelerator``), one group per instance, named after it, in
the graph's order (``encapsulate``'s), each of one unit until links come
(M13's B3); the offsets run on from group to group. Several accelerators
are ``gen.instances``.

**Refused**, each a ``GenError``: ``gen.instances`` and ``gen.binding``
(lower/c.py's ``accelerator``), ``gen.datapath`` (a datapath not registered),
``gen.controller`` (a controller other than ``counted``, the only one generated),
what the datapath's adapter refuses (blocks.py), ``gen.drain`` (an
implementation whose drain is not 0: the nest controller has none) and
``gen.timing`` (a latency other than the datapath's, or an II other than 1).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from snax_forge.brm import Instance, port_dtypes
from snax_forge.common import JsonFile
from snax_forge.design.streamers import instance_nodes
from snax_forge.lower.c import accelerator as c_accelerator
from snax_forge.lower.c import fold_rate

from .blocks import DATAPATHS, GenError, Port

if TYPE_CHECKING:
    from snax_forge.design.point import DesignPoint

ACCELERATOR_JSON = "accelerator.json"
# A CSR is 32 bits wide
CSR_BITS = 32
ACCESS = ("rw", "ro")


@dataclass(frozen=True)
class Csr:
    """One CSR; see the module doc."""

    name: str
    access: str
    offset: int
    width: int | None = None
    pin: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "access": self.access, "offset": self.offset}
        if self.width is not None:
            d["width"] = self.width
        if self.pin is not None:
            d["pin"] = self.pin
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Csr:
        return cls(d["name"], d["access"], d["offset"], d.get("width"), d.get("pin"))


@dataclass(frozen=True)
class Unit:
    """One unit of a group; see the module doc."""

    name: str
    brm: str
    datapath: str
    ports: dict[str, Port]
    params: dict[str, Any]
    latency: int
    ii: int
    csrs: list[Csr] = field(default_factory=list)
    op_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "brm": self.brm,
            "datapath": self.datapath,
            "ports": {k: p.to_dict() for k, p in self.ports.items()},
        }
        if self.op_type is not None:
            d["opType"] = self.op_type
        d |= {
            "params": dict(self.params),
            "timing": {"latency": self.latency, "ii": self.ii},
            "csrs": [c.to_dict() for c in self.csrs],
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Unit:
        return cls(d["name"], d["brm"], d["datapath"],
                   {k: Port.from_dict(p) for k, p in d["ports"].items()}, dict(d["params"]),
                   d["timing"]["latency"], d["timing"]["ii"],
                   [Csr.from_dict(c) for c in d["csrs"]], d.get("opType"))  # fmt: skip


@dataclass(frozen=True)
class Group:
    """A connected set of units and links with one JOBS (D146): its own CSRs (START, JOBS,
    then STATUS, COUNTER) and its units'."""

    name: str
    csrs: list[Csr]
    units: list[Unit]
    links: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "csrs": [c.to_dict() for c in self.csrs],
                "units": [u.to_dict() for u in self.units], "links": list(self.links)}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Group:
        return cls(d["name"], [Csr.from_dict(c) for c in d["csrs"]],
                   [Unit.from_dict(u) for u in d["units"]], list(d["links"]))  # fmt: skip

    def in_order(self) -> list[tuple[Unit | None, Csr]]:
        """Its CSRs in CSR order, each with its unit (None for the group's own): START, JOBS,
        the units' R/W, STATUS, COUNTER."""
        own = [(None, c) for c in self.csrs]
        rw = [x for x in own if x[1].access == "rw"]
        ro = [x for x in own if x[1].access == "ro"]
        return [*rw, *((u, c) for u in self.units for c in u.csrs), *ro]


@dataclass(frozen=True)
class Accelerator(JsonFile):
    """``accelerator.json``; see the module doc."""

    name: str
    design_point: str
    tag: str
    inputs: list[str]
    outputs: list[str]
    groups: list[Group]

    def __post_init__(self) -> None:
        offsets = [c.offset for _, _, c in self.csrs()]
        if offsets != list(range(len(offsets))):
            raise ValueError(f"accelerator {self.name}: CSR offsets {offsets} in CSR order are "
                             "not contiguous from 0")  # fmt: skip
        for _, _, c in self.csrs():
            if c.access not in ACCESS or (c.width is None) != (c.access == "ro"):
                raise ValueError(f"accelerator {self.name}: CSR {c.name} is {c.access} with "
                                 f"width {c.width}; an rw CSR has a width, an ro one none")  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "design_point": self.design_point, "tag": self.tag,
                "inputs": list(self.inputs), "outputs": list(self.outputs),
                "groups": [g.to_dict() for g in self.groups]}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Accelerator:
        return cls(d["name"], d["design_point"], d["tag"], list(d["inputs"]), list(d["outputs"]),
                   [Group.from_dict(g) for g in d["groups"]])  # fmt: skip

    def csrs(self) -> list[tuple[Group, Unit | None, Csr]]:
        """Every CSR in CSR order, with its group and its unit (None for a group's own)."""
        return [(g, u, c) for g in self.groups for u, c in g.in_order()]

    @property
    def rw(self) -> int:
        """The R/W CSRs: Σ_g (2 + Σ_u (f_u + k_u)) (C§17)."""
        return sum(c.access == "rw" for _, _, c in self.csrs())

    @property
    def ro(self) -> int:
        """The R/O CSRs: 2G."""
        return sum(c.access == "ro" for _, _, c in self.csrs())

    def group(self, unit: str) -> Group:
        """The group that holds the unit ``unit``."""
        for g in self.groups:
            if any(u.name == unit for u in g.units):
                return g
        raise KeyError(f"accelerator {self.name} has no unit {unit!r}")

    def define(self, group: Group, unit: Unit | None, csr: Csr) -> str:
        """A CSR's name in the header and the program; see the module doc."""
        parts = [self.name.upper(), csr.access.upper()]
        if len(self.groups) > 1:
            parts.append(group.name.upper())
        if unit is not None and len(group.units) > 1:
            parts.append(unit.name.upper())
        return "_".join([*parts, csr.name])

    def defines(self) -> dict[str, int]:
        """Every CSR's ``define`` -> its offset, in CSR order."""
        return {self.define(g, u, c): c.offset for g, u, c in self.csrs()}

    def find(self, unit: str, name: str) -> str:
        """The ``define`` of CSR ``name`` (``START``, ``JOBS``, ``JOB_BEATS``, ...) of unit
        ``unit``'s group, the group's own or the unit's; ``KeyError`` when it has none."""
        g = self.group(unit)
        for u, c in g.in_order():
            if c.name == name and (u is None or u.name == unit):
                return self.define(g, u, c)
        raise KeyError(f"unit {unit}'s group {g.name} has no CSR {name}")


def units(point: DesignPoint) -> tuple[str, dict[str, Instance]]:
    """The one accelerator's name and its units, one per group in order (lower/c.py's
    ``accelerator``, its rules ``gen.instances`` and ``gen.binding``)."""
    return c_accelerator(point, lambda rule, message: GenError(f"gen.{rule}", message))


def describe(point: DesignPoint, point_path: str) -> Accelerator:
    """The design point's ``accelerator.json``, checked; see the module doc. ``point_path``
    names the design point (gvsoc.py's ``point_name``)."""
    name, found = units(point)
    offsets = iter(range(1 << 16))  # contiguous over the groups, in CSR order
    groups, inputs, outputs = [], [], []
    for unit_name, inst in found.items():
        start, jobs = (Csr(n, "rw", next(offsets), CSR_BITS) for n in ("START", "JOBS"))
        unit = _unit(point, unit_name, inst, offsets)
        status, counter = (Csr(n, "ro", next(offsets)) for n in ("STATUS", "COUNTER"))
        groups.append(Group(unit_name, [start, jobs, status, counter], [unit]))
        ports = inst.brm.interface.ports
        inputs += [f"{unit_name}.{p.name}" for p in ports if p.direction == "in"]
        outputs += [f"{unit_name}.{p.name}" for p in ports if p.direction == "out"]
    return Accelerator(name, point_path, point.platform.accelerator_slot, inputs, outputs, groups)


def _unit(point: DesignPoint, name: str, inst: Instance, offsets: Iterator[int]) -> Unit:
    """Instance ``name``'s unit, checked (module doc), its R/W CSRs numbered from ``offsets``."""
    what = f"{inst.brm.name}.{inst.implementation}"
    datapath = inst.impl.binding["datapath"]
    if datapath not in DATAPATHS:
        raise GenError("gen.datapath", f"{what}: datapath {datapath!r} not registered "
                       f"({sorted(DATAPATHS)})")  # fmt: skip
    if inst.impl.binding["controller"] != "counted":
        raise GenError("gen.controller", f"{what}: controller "
                       f"{inst.impl.binding['controller']!r}; the nest controller is counted")  # fmt: skip
    dp = DATAPATHS[datapath]
    fill = dp.fill(inst, port_dtypes(point.graph, instance_nodes(point.graph)[name]))
    if inst.drain != 0:
        raise GenError("gen.drain", f"{what}: drain {inst.drain}; the nest controller has none, "
                       "so a bound implementation's drain is 0 (D133)")  # fmt: skip
    if (inst.latency, inst.initiation_interval) != (dp.latency, 1):
        raise GenError(
            "gen.timing", f"{what}: latency {inst.latency}, II {inst.initiation_interval}; "
            f"{datapath} is latency {dp.latency}, II 1",
        )  # fmt: skip
    unit_rw = ["JOB_BEATS"] if fold_rate(inst) is not None else []
    unit_csrs = [Csr(n, "rw", next(offsets), CSR_BITS) for n in unit_rw]
    return Unit(name, inst.brm.name, datapath, fill.ports, fill.params, inst.latency,
                inst.initiation_interval, unit_csrs, fill.op_type)  # fmt: skip
