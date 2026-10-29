"""A design point -> its accelerator's one description, ``accelerator.json`` (D145, C§17).

    describe(point, point_path) -> Accelerator

snax-forge writes it once (gvsoc.py, at the root of every gen-gvsoc folder and
bundle; gen-rtl the same bytes, rtl.py) and SNAX-GVSoC (snax-gvsoc
``snax/snax_cfg.py``) and Emit (``hw/chisel/``) read it. Plain JSON, concrete
values only, in this key order:

    name          the accelerator: the header's <ACC>; ``encapsulate``'s
                  accelerator, else its one group's name (a one-unit group's
                  is its instance)
    design_point  the design point by its path from the root (gvsoc.py)
    tag           the platform's accelerator_slot
    inputs        the outer ports, "<unit>.<port>", in the streamer's reader
    outputs       and writer order (the BRM's port order)
    groups        in CSR order; each name, csrs, units, links

A unit: ``name`` (the instance), ``brm``, ``datapath`` (the binding's),
``ports`` (per BRM port in its order: ``port``, ``dir``, ``dtype``, ``lanes``,
the datapath's adapter's, blocks.py), ``opType`` (an elementwise or Mac
unit only), ``params`` (the adapter's, and ``T`` for a fold whose rate is a
constant once the recipe's design params are in: it depends on no kernel
symbol), ``timing`` (``latency``, ``ii``, ``span``) and ``csrs`` (its R/W
CSRs: ``JOB_BEATS`` for a fold without ``params.T``). A link is ``name``,
``from`` (``"<unit>.<port>"`` of an ``out`` port), ``to`` (of an ``in``) and
``depth`` (0 a wire, >= 1 a buffer of that many beats).

**The CSRs** (D146), offsets from the accelerator's first CSR, per group in order:
``START`` and ``JOBS`` (R/W, 32 bits), each unit's R/W CSRs, ``STATUS`` and
``COUNTER`` (R/O). A CSR is ``name``, ``access`` (``rw``, ``ro``),
``offset``, ``width`` (R/W only) and ``pin`` (a scalar's datapath pin, none
yet). ``from_dict`` checks the offsets contiguous from 0 in that order and
never recomputes them. ``define`` names a CSR as the header does:
``<ACC>_<RW|RO>_[<GROUP>_][<UNIT>_]<NAME>``, the group part dropped when the
accelerator has one group, the unit part when the group has one unit.

**The groups** (D148): the design point's one accelerator
(lower/c.py's ``accelerator``), its groups in the graph's order
(``encapsulate``'s; design/streamers.py's ``groups``): a unit alone, named
after its instance, or the units ``link`` joined, named by its ``group``, in
the graph's order, with their links (each named after its container, from
the writer's port to the reader's, its depth). ``inputs`` and ``outputs``
are the ports on no link, group by group, unit by unit in port order. The
offsets run on from group to group. Several accelerators are
``gen.instances``.

**Refused**, each a ``GenError``: ``gen.instances`` and ``gen.binding``
(lower/c.py's ``accelerator``), ``gen.datapath`` (a datapath not registered),
``gen.controller`` (a controller other than ``counted``, the only one generated),
what the datapath's adapter refuses (blocks.py), ``gen.drain`` (an
implementation whose drain is not 0: the nest controller has none) and
``gen.timing`` (a latency or span other than the datapath's, or an II other than 1).

**A reader's checks** (``from_dict`` and every construction), each a
``GenError``: ``gen.link`` (a link not from an ``out`` port to an ``in`` port
of the group's units with the same dtype and lanes, or a depth below 0),
``gen.link_port`` (a port on more than one link), ``gen.outer`` (``inputs``
or ``outputs`` not exactly the unlinked ports, group by group, unit by unit
in port order) and ``gen.group`` (a group whose units the links do not
connect), besides the offsets and widths (a ValueError); then SNAX-GVSoC's
limits (snax-gvsoc's SnaxGroup), ``gen.chain`` (a group that is not a chain
in unit order: each unit's output linked to the next unit's input, only the
last unit's output outer; lifted in tier A, fork and join) and ``gen.span``
(a unit of span > 1 with an outer input; lifted in B2).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import partial
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from snax_forge import expr
from snax_forge.brm import Instance, port_dtypes
from snax_forge.common import JsonFile
from snax_forge.design.streamers import groups as graph_groups
from snax_forge.design.streamers import instance_nodes, link_ends, linked_ports, links
from snax_forge.dfg import Graph, Node
from snax_forge.dfg.subset import map_range, parse_dim
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
    span: int = 1

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
                   d["timing"]["span"])  # fmt: skip


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
    then STATUS, COUNTER) and its units'."""

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
        _check_links(self)

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
    def fixed_folds(self) -> set[str]:
        """The units whose T is fixed at design time (``params.T``): their JOB_BEATS is no CSR
        ."""
        return {u.name for g in self.groups for u in g.units if "T" in u.params}

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


def _check_links(acc: Accelerator) -> None:
    """The links, the outer ports and each group's connectedness; see the module doc."""
    linked: set[str] = set()
    outer: dict[str, list[str]] = {"in": [], "out": []}
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
            for end in (k.from_, k.to):
                if end in linked:
                    raise GenError("gen.link_port", f"{what}: port {end} is on another link too")
                linked.add(end)
            parent[root(k.from_.split(".")[0])] = root(k.to.split(".")[0])
        if len({root(u.name) for u in g.units}) > 1:
            raise GenError("gen.group", f"accelerator {acc.name}: group {g.name}'s units "
                           f"{[u.name for u in g.units]} are not connected by its links")  # fmt: skip
        _check_chain(acc, g, ports)
        for name, p in ports.items():
            if name not in linked:
                outer[p.dir].append(name)
    for key, have, want in (
        ("inputs", acc.inputs, outer["in"]),
        ("outputs", acc.outputs, outer["out"]),
    ):
        if list(have) != want:
            raise GenError("gen.outer", f"accelerator {acc.name}: {key} {list(have)}; the "
                           f"unlinked ports are {want}, in group, unit and port order")  # fmt: skip


def _check_chain(acc: Accelerator, g: Group, ports: dict[str, Port]) -> None:
    """``gen.chain`` and ``gen.span``, SNAX-GVSoC's limits (module doc)."""
    names = [u.name for u in g.units]
    hops = [(k.from_.split(".")[0], k.to.split(".")[0]) for k in g.links]
    chain = sorted(hops) == sorted(pairwise(names))
    outs = {u.name: [f"{u.name}.{p}" for p, x in u.ports.items() if x.dir == "out"]
            for u in g.units}  # fmt: skip
    linked = {k.from_ for k in g.links}
    for i, u in enumerate(g.units):
        outer = [p for p in outs[u.name] if p not in linked]
        chain = chain and bool(outer) == (i == len(g.units) - 1)
    if not chain:
        raise GenError("gen.chain", f"accelerator {acc.name}: group {g.name}'s links {hops} over "
                       f"its units {names}: a group is a chain until tier A (fork and join)")  # fmt: skip
    to = {k.to for k in g.links}
    for u in g.units:
        outer_in = [p for p, x in u.ports.items() if x.dir == "in" and f"{u.name}.{p}" not in to]
        if u.span > 1 and outer_in:
            raise GenError("gen.span", f"accelerator {acc.name}: unit {u.name} (span {u.span}) "
                           f"reads {outer_in} from the streamer: a unit with span > 1 reads from a "
                           "link until B2")  # fmt: skip


def _root(parent: dict[str, str], u: str) -> str:
    """``u``'s representative in the union-find ``parent``."""
    while parent[u] != u:
        u = parent[u]
    return u


def fixed_fold(graph: Graph, node: Node, inst: Instance) -> int | None:
    """A fold's T when it is a constant once the design params are in: the firing loops one
    output beat folds over (named_rates', D102) have ranges over no kernel symbol. None for a
    unit without a fold or one whose T depends on a symbol (a JOB_BEATS CSR then)."""
    rate = fold_rate(inst)
    if rate is None:
        return None
    port = next(p.name for p in inst.brm.interface.ports if p.direction == "out" and p.rate == rate)
    used = node.outputs[port].names
    loops = list(next(o for n, o in graph.walk() if n is node))
    k = len(loops)
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


def units(point: DesignPoint) -> tuple[str, dict[str, Instance]]:
    """The one accelerator's name and its units, one per group in order (lower/c.py's
    ``accelerator``, its rules ``gen.instances`` and ``gen.binding``)."""
    return c_accelerator(point, lambda rule, message: GenError(f"gen.{rule}", message))


def describe(point: DesignPoint, point_path: str) -> Accelerator:
    """The design point's ``accelerator.json``, checked; see the module doc. ``point_path``
    names the design point (gvsoc.py's ``point_name``)."""
    name, found = units(point)
    g = point.graph
    linked = linked_ports(g)
    offsets = iter(range(1 << 16))  # contiguous over the groups, in CSR order
    groups, inputs, outputs = [], [], []
    for group_name, members in graph_groups(g).items():
        if members[0] not in found:
            continue
        start, jobs = (Csr(n, "rw", next(offsets), CSR_BITS) for n in ("START", "JOBS"))
        group_units = [_unit(point, u, found[u], offsets) for u in members]
        status, counter = (Csr(n, "ro", next(offsets)) for n in ("STATUS", "COUNTER"))
        group_links = []
        for k in links(g):
            if k.attrs["from"] in members:
                end_from, end_to = link_ends(g, k)
                group_links.append(Link(k.attrs["container"], end_from, end_to, k.attrs["depth"]))
        groups.append(Group(group_name, [start, jobs, status, counter], group_units, group_links))
        for u in members:
            ports = [p for p in found[u].brm.interface.ports if f"{u}.{p.name}" not in linked]
            inputs += [f"{u}.{p.name}" for p in ports if p.direction == "in"]
            outputs += [f"{u}.{p.name}" for p in ports if p.direction == "out"]
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
    latency, span = dp.timing(fill)
    if (inst.latency, inst.initiation_interval, inst.span) != (latency, 1, span):
        raise GenError(
            "gen.timing", f"{what}: latency {inst.latency}, II {inst.initiation_interval}, span "
            f"{inst.span}; {datapath} here is latency {latency}, II 1, span {span}",
        )  # fmt: skip
    fixed = fixed_fold(point.graph, instance_nodes(point.graph)[name], inst)
    params = dict(fill.params) | ({} if fixed is None else {"T": fixed})
    unit_rw = ["JOB_BEATS"] if fold_rate(inst) is not None and fixed is None else []
    unit_csrs = [Csr(n, "rw", next(offsets), CSR_BITS) for n in unit_rw]
    return Unit(name, inst.brm.name, datapath, fill.ports, params, inst.latency,
                inst.initiation_interval, unit_csrs, fill.op_type, inst.span)  # fmt: skip
