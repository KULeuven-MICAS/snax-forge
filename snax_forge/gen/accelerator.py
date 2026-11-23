"""A design point -> the one description of its accelerators, ``accelerator.json``.

``describe`` builds the ``Description`` (accelerators, groups, units, links, movers and the
port mux, and every CSR's offset and width) that CONTRACTS, "Accelerator RTL spec" defines.
snax-forge writes it once (gvsoc.py; rtl.py writes the same bytes), and SNAX-GVSoC
(snax-gvsoc ``snax/snax_cfg.py``) and Emit (``hw/chisel/``) read it. The accelerators come in
the order the graph first uses them (design/streamers.py's ``accelerators``), each with its
groups in the graph's order: a unit alone, named after its instance, or the units ``link``
and ``fork_read`` join, named by their ``group``. A group of several chains (D189) has a
``JOBS_<SINK>`` per chain in place of ``JOBS``, and its forked input is one entry of
``inputs``, a list of the ports it feeds, the reader's first. Group and accelerator names are unique over the design: they name
the control units and the streamer sets. ``Accelerator.define`` names a CSR as the header does.

``from_dict`` and every construction check what a reader needs (links, outer ports, groups,
offsets and widths, SNAX-GVSoC's chain and span limits, shared movers) and never recompute the
offsets. What ``describe`` or a reader refuses is a ``GenError`` (``gen.<rule>``), an offset
or width a ``ValueError``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

from snax_forge import expr
from snax_forge.brm import Instance
from snax_forge.common import JsonFile
from snax_forge.design.streamers import (
    firing_loops,
    forked_ports,
    group_chains,
    instance_nodes,
    is_broadcast,
    is_held,
    jobs_csr,
    linked_ports,
    shared_movers,
    streamer_name,
)
from snax_forge.design.streamers import group_links as shared_links
from snax_forge.design.streamers import groups as graph_groups
from snax_forge.dfg import Graph, Node
from snax_forge.dfg.subset import map_range, parse_dim
from snax_forge.lower.c import accelerators as c_accelerators
from snax_forge.lower.c import bank_rows, fold_rate

from .blocks import DATAPATHS, GenError, Port, fill_node, port_dtype

if TYPE_CHECKING:
    from snax_forge.design.point import DesignPoint

ACCELERATOR_JSON = "accelerator.json"
ACCESS = ("rw", "ro")
# The tag's form: it names modules, files and the cluster file's entry
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# The CSRs whose width the RTL spec fixes: START a strobe (bit 0), STATUS {done, busy}
FIXED_WIDTH = {"START": 1, "STATUS": 2}


@dataclass(frozen=True)
class Csr:
    """One CSR (CONTRACTS, "Accelerator RTL spec")."""

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
    """One unit of a group (CONTRACTS, "Accelerator RTL spec")."""

    name: str
    brm: str
    datapath: str
    ports: dict[str, Port]
    params: dict[str, Any]
    latency: int
    ii: int
    csrs: list[Csr] = field(default_factory=list)
    op_type: str | None = None
    span: int = 1
    scalars: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "brm": self.brm,
            "datapath": self.datapath,
            "ports": {k: p.to_dict() for k, p in self.ports.items()},
        }
        if self.op_type is not None:
            d["opType"] = self.op_type
        d["params"] = dict(self.params)
        if self.scalars:
            d["scalars"] = dict(self.scalars)
        d |= {
            "timing": {"latency": self.latency, "ii": self.ii, "span": self.span},
            "csrs": [c.to_dict() for c in self.csrs],
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Unit:
        return cls(d["name"], d["brm"], d["datapath"],
                   {k: Port.from_dict(p) for k, p in d["ports"].items()}, dict(d["params"]),
                   d["timing"]["latency"], d["timing"]["ii"],
                   [Csr.from_dict(c) for c in d["csrs"]], d.get("opType"),
                   d["timing"]["span"], dict(d.get("scalars", {})))  # fmt: skip


@dataclass(frozen=True)
class Link:
    """One link of a group: ``from_`` an ``out`` port, ``to`` an ``in`` port, each
    ``"<unit>.<port>"``; depth 0 a wire, >= 1 a buffer of that many beats."""

    name: str
    from_: str
    to: str
    depth: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "from": self.from_, "to": self.to, "depth": self.depth}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Link:
        return cls(d["name"], d["from"], d["to"], d["depth"])


@dataclass(frozen=True)
class Group:
    """A connected set of units and links with one JOBS: its own CSRs (START, JOBS,
    then STATUS, BUSY_COUNTER) and its units'."""

    name: str
    csrs: list[Csr]
    units: list[Unit]
    links: list[Link] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "csrs": [c.to_dict() for c in self.csrs],
                "units": [u.to_dict() for u in self.units],
                "links": [k.to_dict() for k in self.links]}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Group:
        return cls(d["name"], [Csr.from_dict(c) for c in d["csrs"]],
                   [Unit.from_dict(u) for u in d["units"]],
                   [Link.from_dict(k) for k in d["links"]])  # fmt: skip

    def chains(self) -> list[list[str]]:
        """Its chains, each its units from head to sink along its links, in the order of their
        sinks among its units (one for a group a fork does not join, D189)."""
        nxt = {k.from_.split(".")[0]: k.to.split(".")[0] for k in self.links}
        prev = {b: a for a, b in nxt.items()}
        out = []
        for u in self.units:
            if u.name in nxt:
                continue
            chain = [u.name]
            while chain[0] in prev:
                chain.insert(0, prev[chain[0]])
            out.append(chain)
        return out

    def in_order(self) -> list[tuple[Unit | None, Csr]]:
        """Its CSRs in CSR order, each with its unit (None for the group's own): START, JOBS,
        the units' R/W, STATUS, BUSY_COUNTER."""
        own = [(None, c) for c in self.csrs]
        rw = [x for x in own if x[1].access == "rw"]
        ro = [x for x in own if x[1].access == "ro"]
        return [*rw, *((u, c) for u in self.units for c in u.csrs), *ro]


@dataclass(frozen=True)
class MoverUser:
    """A group's port on a shared mover: its group, the port (``"<unit>.<port>"``) and its
    channels."""

    group: str
    port: str
    num_channel: int

    def to_dict(self) -> dict[str, Any]:
        return {"group": self.group, "port": self.port, "num_channel": self.num_channel}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MoverUser:
        return cls(d["group"], d["port"], d["num_channel"])


@dataclass(frozen=True)
class Mover:
    """A mover of a shared accelerator (CONTRACTS, "Accelerator RTL spec")."""

    name: str
    role: str
    num_channel: int
    masked: bool
    users: list[MoverUser]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "role": self.role, "num_channel": self.num_channel,
                "masked": self.masked, "users": [u.to_dict() for u in self.users]}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Mover:
        return cls(d["name"], d["role"], d["num_channel"], d["masked"],
                   [MoverUser.from_dict(u) for u in d["users"]])  # fmt: skip


@dataclass(frozen=True)
class PortMux:
    """A shared accelerator's data-side mux: its groups in PORT_MUX_SEL's order and
    PORT_MUX_SEL."""

    groups: list[str]
    csr: Csr

    def to_dict(self) -> dict[str, Any]:
        return {"groups": list(self.groups), "csr": self.csr.to_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PortMux:
        return cls(list(d["groups"]), Csr.from_dict(d["csr"]))


@dataclass(frozen=True)
class Accelerator:
    """One accelerator of the design: one streamer set, its groups sharing its streamer
    and, encapsulated, its start, and with ``share_ports`` its movers too."""

    name: str
    inputs: list[str | list[str]]  # a fork's ports (D189) as one entry, the reader's first
    outputs: list[str]
    groups: list[Group]
    movers: list[Mover] = field(default_factory=list)
    port_mux: PortMux | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {"name": self.name,
             "inputs": [x if isinstance(x, str) else list(x) for x in self.inputs],
             "outputs": list(self.outputs),
             "groups": [g.to_dict() for g in self.groups]}  # fmt: skip
        if self.port_mux is not None:
            d |= {"movers": [m.to_dict() for m in self.movers],
                  "port_mux": self.port_mux.to_dict()}  # fmt: skip
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Accelerator:
        mux = d.get("port_mux")
        return cls(d["name"], [x if isinstance(x, str) else list(x) for x in d["inputs"]],
                   list(d["outputs"]),
                   [Group.from_dict(g) for g in d["groups"]],
                   [Mover.from_dict(m) for m in d.get("movers", [])],
                   None if mux is None else PortMux.from_dict(mux))  # fmt: skip

    def csrs(self) -> list[tuple[Group | None, Unit | None, Csr]]:
        """Its CSRs in CSR order, with their group and unit (None for a group's own; group None
        for PORT_MUX_SEL, last)."""
        own = [] if self.port_mux is None else [(None, None, self.port_mux.csr)]
        return [*((g, u, c) for g in self.groups for u, c in g.in_order()), *own]

    @property
    def rw(self) -> int:
        """Its R/W CSRs: Σ_g (2 + Σ_u (f_u + k_u)), the set's ``snax_num_rw_csr``."""
        return sum(c.access == "rw" for _, _, c in self.csrs())

    @property
    def ro(self) -> int:
        """Its R/O CSRs: 2G, the set's ``snax_num_ro_csr``."""
        return sum(c.access == "ro" for _, _, c in self.csrs())

    @property
    def units(self) -> list[Unit]:
        return [u for g in self.groups for u in g.units]

    @property
    def readers(self) -> list[str]:
        """Its input ports with a reader of their own, a fork's by its first port."""
        return [x if isinstance(x, str) else x[0] for x in self.inputs]

    @property
    def forks(self) -> list[list[str]]:
        """Its forked inputs (D189), each the ports one reader feeds, the reader's first."""
        return [list(x) for x in self.inputs if not isinstance(x, str)]

    def define(self, group: Group | None, unit: Unit | None, csr: Csr) -> str:
        """A CSR's name in the header and the program (CONTRACTS, "Accelerator RTL spec")."""
        parts = [self.name.upper(), csr.access.upper()]
        if group is not None and len(self.groups) > 1:
            parts.append(group.name.upper())
        if unit is not None and len(group.units) > 1:
            parts.append(unit.name.upper())
        return "_".join([*parts, csr.name])

    def defines(self) -> dict[str, int]:
        """Its CSRs' ``define`` -> their offset from the shell's first CSR, in CSR order."""
        return {self.define(g, u, c): c.offset for g, u, c in self.csrs()}


@dataclass(frozen=True)
class Description(JsonFile):
    """``accelerator.json`` (CONTRACTS, "Accelerator RTL spec")."""

    design_point: str
    tag: str
    accelerators: list[Accelerator]

    def __post_init__(self) -> None:
        offsets = [c.offset for _, _, c in self.csrs()]
        if offsets != list(range(len(offsets))):
            raise ValueError(f"{self.tag}: CSR offsets {offsets} in CSR order are not contiguous "
                             "from 0")  # fmt: skip
        for _, _, c in self.csrs():
            if c.access not in ACCESS:
                raise ValueError(f"{self.tag}: CSR {c.name} is {c.access!r}, not one of {ACCESS}")
            if type(c.width) is not int or not 1 <= c.width <= 32:
                raise ValueError(f"{self.tag}: CSR {c.name} has width {c.width!r}; every CSR has a "
                                 "width of 1 to 32 bits")  # fmt: skip
            if c.width != FIXED_WIDTH.get(c.name, c.width):
                raise ValueError(f"{self.tag}: CSR {c.name} has width {c.width}, not "
                                 f"{FIXED_WIDTH[c.name]}")  # fmt: skip
        for names in (("JOBS", "JOB_BEATS"), ("BUSY_COUNTER",)):
            # a chain's JOBS_<SINK> (D189) is a JOBS
            widths = {c.width for _, _, c in self.csrs()
                      if c.name in names or ("JOBS" in names and c.name.startswith("JOBS_"))}  # fmt: skip
            if len(widths) > 1:
                raise ValueError(f"{self.tag}: {' and '.join(names)} have the widths "
                                 f"{sorted(widths)}; they share one")  # fmt: skip
        for what, names in (("accelerator", [a.name for a in self.accelerators]),
                            ("group", [g.name for g in self.groups])):  # fmt: skip
            twice = sorted({n for n in names if names.count(n) > 1})
            if twice:
                raise ValueError(f"{self.tag}: the {what} names {twice} are used twice")
        for u in self.units:
            want = [(c, s) for s, t in u.scalars.items() for c in scalar_csrs(s, t)]
            got = [(c.name, c.pin) for c in u.csrs if c.pin is not None]
            if got != want or any(c.access != "rw" for c in u.csrs if c.pin is not None):
                raise ValueError(f"{self.tag}: unit {u.name}'s scalars {u.scalars} are the R/W "
                                 f"CSRs {want} (name, pin), not {got}")  # fmt: skip
        for a in self.accelerators:
            _check_links(a)
            if a.port_mux is not None or a.movers:
                _check_shared(a)

    def to_dict(self) -> dict[str, Any]:
        return {"design_point": self.design_point, "tag": self.tag,
                "accelerators": [a.to_dict() for a in self.accelerators]}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Description:
        return cls(d["design_point"], d["tag"],
                   [Accelerator.from_dict(a) for a in d["accelerators"]])  # fmt: skip

    @property
    def groups(self) -> list[Group]:
        """Every group, in CSR order over the accelerators."""
        return [g for a in self.accelerators for g in a.groups]

    @property
    def units(self) -> list[Unit]:
        return [u for g in self.groups for u in g.units]

    @property
    def inputs(self) -> list[str | list[str]]:
        """Every outer input, set after set (a fork's ports as one entry, D189): the wrapper's
        ``stream2acc_<k>`` order."""
        return [p for a in self.accelerators for p in a.inputs]

    @property
    def readers(self) -> list[str]:
        """Every outer input with a reader of its own, a fork's by its first port."""
        return [p for a in self.accelerators for p in a.readers]

    @property
    def forks(self) -> list[list[str]]:
        """Every forked input (D189), the ports one reader feeds, the reader's first."""
        return [f for a in self.accelerators for f in a.forks]

    @property
    def outputs(self) -> list[str]:
        """Every outer output port, set after set: the wrapper's ``acc2stream_<k>`` order."""
        return [p for a in self.accelerators for p in a.outputs]

    def csrs(self) -> list[tuple[Group, Unit | None, Csr]]:
        """Every CSR in CSR order, with its group and its unit (None for a group's own)."""
        return [x for a in self.accelerators for x in a.csrs()]

    @property
    def fixed_folds(self) -> set[str]:
        """The units whose T is fixed at design time (``params.T``): their JOB_BEATS is no CSR
        ."""
        return {u.name for u in self.units if "T" in u.params}

    @property
    def job_width(self) -> int:
        """Every JOBS's (a chain's JOBS_<SINK> too) and JOB_BEATS's width (the platform's
        ``acc_csr.job_width``)."""
        return next(c.width for _, _, c in self.csrs() if c.name.startswith("JOBS"))

    def accelerator(self, unit: str) -> Accelerator:
        """The accelerator that holds the unit ``unit``."""
        for a in self.accelerators:
            if any(u.name == unit for u in a.units):
                return a
        raise KeyError(f"{self.tag} has no unit {unit!r}")

    def group(self, unit: str) -> Group:
        """The group that holds the unit ``unit``."""
        for g in self.groups:
            if any(u.name == unit for u in g.units):
                return g
        raise KeyError(f"{self.tag} has no unit {unit!r}")

    def define(self, group: Group | None, unit: Unit | None, csr: Csr) -> str:
        """A CSR's name in the header and the program: its accelerator's ``define``."""
        (a,) = (a for a in self.accelerators
                if any(g is group for g in a.groups) or (a.port_mux and a.port_mux.csr is csr))  # fmt: skip
        return a.define(group, unit, csr)

    def port_mux_sel(self, acc: str) -> str | None:
        """The define of accelerator ``acc``'s PORT_MUX_SEL, None when it shares no movers."""
        (a,) = (a for a in self.accelerators if a.name == acc)
        return None if a.port_mux is None else a.define(None, None, a.port_mux.csr)

    def defines(self) -> dict[str, int]:
        """Every CSR's ``define`` -> its offset from the shell's first CSR, in CSR order."""
        return {k: v for a in self.accelerators for k, v in a.defines().items()}

    def find(self, unit: str, name: str) -> str:
        """The ``define`` of CSR ``name`` (``START``, ``JOBS``, ``JOB_BEATS``, ...) of unit
        ``unit``'s group, the group's own or the unit's; ``KeyError`` when it has none."""
        a, g = self.accelerator(unit), self.group(unit)
        for u, c in g.in_order():
            if c.name == name and (u is None or u.name == unit):
                return a.define(g, u, c)
        raise KeyError(f"unit {unit}'s group {g.name} has no CSR {name}")


def _check_links(acc: Accelerator) -> None:
    """The links, the outer ports and each group's connectedness; see the module doc."""
    linked: set[str] = set()
    outer: dict[str, list[str]] = {"in": [], "out": []}
    for f in acc.forks:
        if len(f) < 2 or len(set(f)) != len(f):
            raise GenError("gen.fork", f"accelerator {acc.name}: the fork {f}: a fork gives one "
                           "reader to two or more outer input ports of its units")  # fmt: skip
    for g in acc.groups:
        ports = {f"{u.name}.{k}": p for u in g.units for k, p in u.ports.items()}
        parent = {u.name: u.name for u in g.units}
        root = partial(_root, parent)

        for k in g.links:
            what = f"accelerator {acc.name}: group {g.name}'s link {k.name}"
            a, b = ports.get(k.from_), ports.get(k.to)
            if a is None or b is None or a.dir != "out" or b.dir != "in":
                raise GenError("gen.link", f"{what} goes from {k.from_} to {k.to}; a link goes "
                               "from an out port to an in port of the group's units")  # fmt: skip
            if (a.dtype, a.lanes) != (b.dtype, b.lanes):
                raise GenError("gen.link", f"{what}: {k.from_} is {a.lanes} lanes of {a.dtype}, "
                               f"{k.to} {b.lanes} of {b.dtype}; both ends are the same")  # fmt: skip
            if type(k.depth) is not int or k.depth < 0:
                raise GenError("gen.link", f"{what}: depth {k.depth!r}; 0 is a wire, >= 1 a buffer")
            for end, p in ((k.from_, a), (k.to, b)):
                if p.d_in > 1 or p.d_out > 1:
                    raise GenError("gen.link", f"{what}: {end} has d_in {p.d_in} and d_out "
                                   f"{p.d_out}; a linked port has no slicer or packer, so both "
                                   "are 1 (absent)")  # fmt: skip
                if p.broadcast:
                    raise GenError("gen.link", f"{what}: {end} is a broadcast; a linked port "
                                   "takes its lanes from the link, so it is not one "
                                   "(absent)")  # fmt: skip
            for end in (k.from_, k.to):
                if end in linked:
                    raise GenError("gen.link_port", f"{what}: port {end} is on another link too")
                linked.add(end)
            parent[root(k.from_.split(".")[0])] = root(k.to.split(".")[0])
        for f in acc.forks:
            mine = [x.split(".")[0] for x in f if x.split(".")[0] in parent]
            for u in mine[1:]:
                parent[root(u)] = root(mine[0])
        if len({root(u.name) for u in g.units}) > 1:
            raise GenError("gen.group", f"accelerator {acc.name}: group {g.name}'s units "
                           f"{[u.name for u in g.units]} are not connected by its links")  # fmt: skip
        _check_chain(acc, g)
        for name, p in ports.items():
            if name not in linked:
                outer[p.dir].append(name)
    want_in: list[str | list[str]] = list(outer["in"])
    for f in acc.forks:
        _check_fork(acc, f, outer["in"])
        want_in = [list(f) if x == f[0] else x for x in want_in if x not in f[1:]]
    for key, have, want in (
        ("inputs", acc.inputs, want_in),
        ("outputs", acc.outputs, outer["out"]),
    ):
        if list(have) != want:
            raise GenError("gen.outer", f"accelerator {acc.name}: {key} {list(have)}; the "
                           f"unlinked ports are {want}, in group, unit and port order")  # fmt: skip


def _check_fork(acc: Accelerator, f: list[str], outer: list[str]) -> None:
    """``gen.fork``: a forked input's ports (D189), two or more outer inputs of one group's
    chain heads, alike in dtype, lanes, slicing and broadcast."""
    what = f"accelerator {acc.name}: the fork {f}"
    group_of = {u.name: g for g in acc.groups for u in g.units}
    port = {f"{u.name}.{k}": p for u in acc.units for k, p in u.ports.items()}
    if len(f) < 2 or len(set(f)) != len(f) or any(x not in outer for x in f):
        raise GenError("gen.fork", f"{what}: a fork gives one reader to two or more outer "
                       f"input ports of its units ({outer})")  # fmt: skip
    if len({group_of[x.split(".")[0]].name for x in f}) > 1:
        raise GenError("gen.fork", f"{what}: its ports are in several groups; a fork feeds "
                       "the chains of one group")  # fmt: skip
    fed = {k.to.split(".")[0] for g in acc.groups for k in g.links}
    for x in f:
        if x.split(".")[0] in fed:
            raise GenError("gen.fork", f"{what}: {x}'s unit is fed by a link; a fork feeds the "
                           "heads of chains")  # fmt: skip
    kinds = {(port[x].dtype, port[x].lanes, port[x].d_in, port[x].broadcast) for x in f}
    if len(kinds) > 1:
        raise GenError("gen.fork", f"{what}: its ports differ in dtype, lanes, d_in or "
                       "broadcast; one reader's beat is the same for all")  # fmt: skip


def _check_shared(acc: Accelerator) -> None:
    """``gen.share``: a shared accelerator's movers and port mux (module doc)."""
    what = f"accelerator {acc.name}"
    if acc.forks:
        raise GenError("gen.share", f"{what}: a fork and shared movers in one accelerator; "
                       "share_ports or fork_read, not both")  # fmt: skip
    if acc.port_mux is None or not acc.movers:
        raise GenError("gen.share", f"{what}: movers and port_mux come together")
    names = [g.name for g in acc.groups]
    if sorted(acc.port_mux.groups) != sorted(names):
        raise GenError("gen.share", f"{what}: port_mux.groups {acc.port_mux.groups}, not its "
                       f"groups {names}")  # fmt: skip
    group_of = {u.name: g.name for g in acc.groups for u in g.units}
    placed: dict[str, str] = {}
    for m in acc.movers:
        outer = acc.inputs if m.role == "read" else acc.outputs
        groups = [u.group for u in m.users]
        if len(set(groups)) != len(groups):
            raise GenError("gen.share", f"{what}: {m.name} serves two ports of one group")
        for u in m.users:
            if u.port not in outer or group_of[u.port.split(".")[0]] != u.group:
                raise GenError("gen.share", f"{what}: {m.name}'s user {u.port} of {u.group} is not "
                               f"one of its {m.role} ports of that group")  # fmt: skip
            if u.port in placed:
                raise GenError("gen.share", f"{what}: {u.port} on {placed[u.port]} and {m.name}")
            if not 1 <= u.num_channel <= m.num_channel or (u.num_channel < m.num_channel
                                                           and not m.masked):  # fmt: skip
                raise GenError("gen.share", f"{what}: {u.port}'s {u.num_channel} channels on "
                               f"{m.name}'s {m.num_channel} (masked {m.masked})")  # fmt: skip
            placed[u.port] = m.name
    missing = [p for p in [*acc.inputs, *acc.outputs] if p not in placed]
    if missing:
        raise GenError("gen.share", f"{what}: {missing} on no mover")


def _check_chain(acc: Accelerator, g: Group) -> None:
    """``gen.chain`` and ``gen.span``, SNAX-GVSoC's limits (module doc): a chain in unit order
    whose last unit's output alone is outer, or with a fork (D189) chains it joins, each
    going forward in unit order with its sink's output outer."""
    names = [u.name for u in g.units]
    hops = [(k.from_.split(".")[0], k.to.split(".")[0]) for k in g.links]
    outs = {u.name: [f"{u.name}.{p}" for p, x in u.ports.items() if x.dir == "out"]
            for u in g.units}  # fmt: skip
    linked = {k.from_ for k in g.links}
    forked = [{x.split(".")[0] for x in f} & set(names) for f in acc.forks]
    if not any(len(f) > 1 for f in forked):
        chain = sorted(hops) == sorted(pairwise(names))
        for i, u in enumerate(g.units):
            outer = [p for p in outs[u.name] if p not in linked]
            chain = chain and bool(outer) == (i == len(g.units) - 1)
    else:
        at = {u: k for k, u in enumerate(names)}
        froms, tos = [a for a, _ in hops], [b for _, b in hops]
        chain = (len(set(froms)) == len(froms) and len(set(tos)) == len(tos)
                 and all(at[a] < at[b] for a, b in hops))  # fmt: skip
        for u in names:
            outer = [p for p in outs[u] if p not in linked]
            chain = chain and bool(outer) == (u not in froms)
    if not chain:
        raise GenError("gen.chain", f"accelerator {acc.name}: group {g.name}'s links {hops} over "
                       f"its units {names}: a group is a chain, or chains a fork joins "
                       "(fork_read)")  # fmt: skip
    to = {k.to for k in g.links}
    for u in g.units:
        outer_in = [p for p, x in u.ports.items() if x.dir == "in" and f"{u.name}.{p}" not in to]
        if u.span > 1 and outer_in:
            raise GenError("gen.span", f"accelerator {acc.name}: unit {u.name} (span {u.span}) "
                           f"reads {outer_in} from the streamer: a unit with span > 1 reads from a "
                           "link")  # fmt: skip


def _root(parent: dict[str, str], u: str) -> str:
    """``u``'s representative in the union-find ``parent``."""
    while parent[u] != u:
        u = parent[u]
    return u


def fixed_fold(graph: Graph, node: Node, inst: Instance) -> int | None:
    """A fold's T when it is a constant once the design params are in: the firing loops one
    output beat folds over (named_rates') have ranges over no kernel symbol. None for a
    unit without a fold or one whose T depends on a symbol (a JOB_BEATS CSR then)."""
    rate = fold_rate(inst)
    if rate is None:
        return None
    port = next(p.name for p in inst.ports if p.direction == "out" and p.rate == rate)
    used = node.outputs[port].names
    loops = list(next(o for n, o in graph.walk() if n is node))
    k, rows = len(loops), 1
    while k and rows < bank_rows(inst):  # a bank's row loops first, as named_rates (D191)
        k -= 1
        rows *= len(map_range(loops[k].attrs["range"], graph.bound_symbols, loops[k].id))
    while k and loops[k - 1].attrs["var"] not in used:
        k -= 1
    folded = loops[k:]
    names = set().union(*(expr.names(x) for m in folded
                          for x in parse_dim(m.attrs["range"], m.id)), set())  # fmt: skip
    if names & set(graph.symbols):
        return None
    t = 1
    for m in folded:
        t *= len(map_range(m.attrs["range"], {}, m.id))
    return t


def units(point: DesignPoint) -> dict[str, dict[str, Instance]]:
    """The accelerators in CSR order, each its units in its groups' order (lower/c.py's
    ``accelerators``, its rule ``gen.binding``)."""
    return c_accelerators(point, lambda rule, message: GenError(f"gen.{rule}", message))


def tag(point: DesignPoint) -> str:
    """The generated entry's name, ``<accelerator_slot>_<name>``, the design's name the design
    point holds; ``gen.tag`` when it is not an identifier (it names modules, files
    and the cluster file's entry)."""
    slot = point.platform.accelerator_slot
    if slot is None:
        raise GenError("gen.tag", f"platform {point.platform.name} has no accelerator_slot: the "
                       "generated entry is named <accelerator_slot>_<name>")  # fmt: skip
    name = f"{slot}_{point.name}"
    if not IDENTIFIER.fullmatch(name):
        raise GenError(
            "gen.tag", f"the generated entry's name {name!r} (<accelerator_slot>_<name>, the "
            f"design {point.name!r}) is not an identifier: it names the RTL's modules, the "
            "streamer files and the cluster's entry; name the design with letters, digits and "
            "_ only, not starting with a digit",
        )  # fmt: skip
    return name


def describe(point: DesignPoint, point_path: str) -> Description:
    """The design point's ``accelerator.json``, checked; see the module doc. ``point_path``
    names the design point (gvsoc.py's ``point_name``)."""
    entry = tag(point)
    found = units(point)
    g = point.graph
    linked = linked_ports(g)
    forked = forked_ports(g)
    offsets = iter(range(1 << 16))  # contiguous over the accelerators and groups, in CSR order
    widths = point.platform.acc_csr
    accs = []
    shared = shared_movers(g, point.streamers)
    for acc_name, mine in found.items():
        groups, inputs, outputs = [], [], []
        for group_name, members in graph_groups(g).items():
            if members[0] not in mine:
                continue
            start = Csr("START", "rw", next(offsets), FIXED_WIDTH["START"])
            chains = group_chains(g, members)
            jobs = [Csr(jobs_csr(chains, c), "rw", next(offsets), widths.job_width)
                    for c in chains]  # fmt: skip
            group_units = [_unit(point, u, mine[u], offsets) for u in members]
            status = Csr("STATUS", "ro", next(offsets), FIXED_WIDTH["STATUS"])
            counter = Csr("BUSY_COUNTER", "ro", next(offsets), widths.counter)
            group_links = [Link(k.attrs["container"], end_from, end_to, k.attrs["depth"])
                           for k, end_from, end_to in shared_links(g, members)]  # fmt: skip
            groups.append(Group(group_name, [start, *jobs, status, counter], group_units,
                                group_links))  # fmt: skip
            for u in members:
                ports = [p for p in mine[u].ports if f"{u}.{p.name}" not in linked]
                for p in ports:
                    name = f"{u}.{p.name}"
                    if p.direction == "out":
                        outputs.append(name)
                    elif name in forked:  # a fork's port: on its reader's entry
                        continue
                    elif name in forked.values():
                        inputs.append([name, *(x for x, y in forked.items() if y == name)])
                    else:
                        inputs.append(name)
        movers, mux = [], None
        if acc_name in shared:
            movers = [Mover(m.name, "write" if m.write else "read", m.num_ch, m.masked,
                            [MoverUser(u.group, u.port, u.num_ch) for u in m.users])
                      for m in shared[acc_name]]  # fmt: skip
            width = max(1, (len(groups) - 1).bit_length())
            mux = PortMux([x.name for x in groups],
                          Csr("PORT_MUX_SEL", "rw", next(offsets), width))  # fmt: skip
        accs.append(Accelerator(acc_name, inputs, outputs, groups, movers, mux))
    return Description(point_path, entry, accs)


def _unit(point: DesignPoint, name: str, inst: Instance, offsets: Iterator[int]) -> Unit:
    """Instance ``name``'s unit, checked (its refusals: CONTRACTS, "Accelerator RTL spec"), its
    R/W CSRs numbered from ``offsets``."""
    what = f"{inst.brm.name}.{inst.implementation}"
    datapath = inst.impl.binding["datapath"]
    if datapath not in DATAPATHS:
        raise GenError("gen.datapath", f"{what}: datapath {datapath!r} not registered "
                       f"({sorted(DATAPATHS)})")  # fmt: skip
    if inst.impl.binding["controller"] != "counted":
        raise GenError("gen.controller", f"{what}: controller "
                       f"{inst.impl.binding['controller']!r}; the nest controller is counted")  # fmt: skip
    dp = DATAPATHS[datapath]
    fill = fill_node(datapath, inst, point.graph, instance_nodes(point.graph)[name])
    if inst.drain != 0:
        raise GenError("gen.drain", f"{what}: drain {inst.drain}; the nest controller has none, "
                       "so a bound implementation's drain is 0")  # fmt: skip
    latency, span = dp.timing(fill)
    if (inst.latency, inst.initiation_interval, inst.span) != (latency, 1, span):
        raise GenError(
            "gen.timing", f"{what}: latency {inst.latency}, II {inst.initiation_interval}, span "
            f"{inst.span}; {datapath} here is latency {latency}, II 1, span {span}",
        )  # fmt: skip
    ports = dict(fill.ports)
    node = instance_nodes(point.graph)[name]
    first = forked_ports(point.graph)  # a forked port takes its first's reader (D189)
    for p, port in ports.items():  # a packed outer output (D174), a sliced outer input (D180)
        u, q = first.get(f"{name}.{p}", f"{name}.{p}").split(".", 1)
        s = point.streamers.get(streamer_name(u, q))
        if s is not None and max(s.d_out, s.d_in) > 1:
            ports[p] = replace(port, d_out=s.d_out, d_in=s.d_in)
        if s is not None and is_broadcast(point.graph, node, inst, p):  # one lane for all (D186)
            if not DATAPATHS[datapath].lanewise:
                raise GenError("gen.broadcast", f"{what}: port {p} is read as one lane for all, "
                               f"but {datapath} sums its lanes; a broadcast feeds a datapath "
                               "whose lanes compute on their own (SimdLanes, Mac)")  # fmt: skip
            ports[p] = replace(ports[p], broadcast=True)
        if is_held(inst, p, (node.inputs | node.outputs)[p], firing_loops(point.graph, node)):
            if not ports[p].broadcast:  # read once a row, held for the row (D190)
                raise GenError("gen.hold", f"{what}: port {p} is held for a bank row but is not "
                               "a broadcast; a held input is one lane for all for now")  # fmt: skip
            ports[p] = replace(ports[p], hold=True)
    fixed = fixed_fold(point.graph, instance_nodes(point.graph)[name], inst)
    params = dict(fill.params) | ({} if fixed is None else {"T": fixed})
    unit_rw = ["JOB_BEATS"] if fold_rate(inst) is not None and fixed is None else []
    unit_csrs = [Csr(n, "rw", next(offsets), point.platform.acc_csr.job_width) for n in unit_rw]
    if (rows := bank_rows(inst)) > 1:  # the rows a pass uses, at most the bank's (D191)
        unit_csrs.append(Csr("ROWS", "rw", next(offsets), rows.bit_length()))
    for slot, dtype in fill.scalars.items():  # after JOB_BEATS and ROWS, in slot order
        width = min(port_dtype(dtype, f"{what}.{slot}")[0], 32)
        unit_csrs += [Csr(c, "rw", next(offsets), width, slot) for c in scalar_csrs(slot, dtype)]
    return Unit(name, inst.brm.name, datapath, ports, params, inst.latency,
                inst.initiation_interval, unit_csrs, fill.op_type, inst.span,
                dict(fill.scalars))  # fmt: skip


def scalar_csrs(slot: str, dtype: str) -> list[str]:
    """A scalar's CSR names: ``K1``, or ``K1_LO`` and ``K1_HI`` (low word first) at 64 bits."""
    name = slot.upper()
    return [f"{name}_LO", f"{name}_HI"] if np.dtype(dtype).itemsize == 8 else [name]
