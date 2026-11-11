"""The streamer shell against a bound graph (D84).

For every accelerated node's instance, in the order the graph first uses
them, and every port of its BRM in port order, one streamer:

    name            <instance>_<port> (D75)
    write           the port's direction is "out"
    spatial_bounds  the platform entry's, else, fastest first, the port nest's
                    spatial loops (``affine``, what snax_forge/lower/streams.py
                    requires) or the bound memlet's ranges (``stream``, as
                    SNAX-LOWER's ``memlet_values`` reads them)
    n_ports         their product: the port's lanes W
    dtype           the dtype of the container behind the port, through its
                    memlet (``brm.port_dtypes``); every node of the instance
                    puts the same one on it (``connect.dtype``)
    num_slc         slices (elements) of that dtype per bank word, width_bits
                    over its bits (D142)
    num_ch          the TCDM channels the port uses (the streamer hjson's
                    ``num_channel``): the entry's ``num_ch``, else the wire
                    default W / num_slc (D142; ``connect.pack`` checks that
                    num_slc divides the fastest spatial bound,
                    ``connect.num_ch`` an entry's value)
    temporal_dims   from the graph: the firing loops the port's memlet runs
                    over (``port_loops``, as SNAX-LOWER takes them), at least
                    1; with several nodes on one instance, the most
    fifo_depth      the platform's ``default`` with the streamer's entry on top

A port on a link (``link``, D148) has no streamer: streamers sit on the
outer ports only (D112). ``instances`` reads the instances off the graph and
resolves them against the BRM library, ``groups`` puts them into groups (an
instance's ``group``, ``link``'s, else its own), ``accelerators`` into
accelerators (D148: an instance's ``accelerator``, ``encapsulate``'s, else its
group's own); ``links``, ``linked_ports`` and ``unplaced`` read the links
and the containers that get no L1 space (a link's, a fused nest's local
value), ``slot_containers`` those read only through a slot connector (a
CSR's value, in the core's memory); ``resolve`` builds the shell. Neither checks the pairing:
that is check.py, which runs first and reports every problem, so
``resolve`` raises on the first thing that does not fit.

**Shared movers** (D169). A container marked ``hw.share_ports`` (``share_ports``,
sandbox/share_ports.py) has its writer's and its reader's groups on one accelerator, whose
groups then share its streamer's movers: ``shared_accelerators`` names those accelerators,
``shared_movers`` maps their ports. Per role (read, write), each group's ports on a streamer,
unit by unit in port order, sorted by channels, widest first (a stable sort); the k-th of every
group goes on mover k (``reader<k>``, ``writer<k>``, snax-gvsoc's names), which takes the
widest channels and its spatial bounds (in words), the most temporal loops and the deepest
FIFO of its users. A narrower user enables its first channels (snax_cluster's
``configurable_channel``, the mover ``masked``). A user whose spatial bounds (in words) are not
the mover's on every dimension but the slowest, and at most the mover's there, does not fit in
its first channels: ``ShareError`` (``share.shape``). The pool's channels, with
:math:`w_{s,r,k}` group s's k-th widest port of role r (0 if none):
``C = Σ_r Σ_k max_s w_{s,r,k}``; ``set_channels`` gives each accelerator's, the one count
check.py's ``crossbar_inputs`` sums.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

from snax_forge import expr
from snax_forge.brm import Instance, load_brm, memlet_ordered, task_nest
from snax_forge.dfg import Graph, Memlet, Node
from snax_forge.dfg.subset import map_range, parse_dim

from .platform import Platform, StreamerOptions


@dataclass(frozen=True)
class Streamer:
    """One resolved streamer of the shell."""

    name: str
    instance: str
    port: str
    write: bool
    spatial_bounds: tuple[int, ...]
    temporal_dims: int
    options: StreamerOptions
    dtype: str
    num_slc: int
    num_ch: int

    @property
    def n_ports(self) -> int:
        return prod(self.spatial_bounds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance": self.instance,
            "port": self.port,
            "write": self.write,
            "n_ports": self.n_ports,
            "spatial_bounds": list(self.spatial_bounds),
            "temporal_dims": self.temporal_dims,
            "dtype": self.dtype,
            "num_slc": self.num_slc,
            "num_ch": self.num_ch,
            **self.options.to_dict(),
        }


def slices_per_word(dtype: str, width_bits: int) -> int:
    """num_slc: slices (elements) of ``dtype`` in one bank word of ``width_bits``, at
    least 1."""
    return max(1, width_bits // (np.dtype(dtype).itemsize * 8))


def port_dtype(graph: Graph, nodes: list[Node], port: str) -> str:
    """The one dtype ``nodes`` (one instance's) put on ``port``, through their memlets; raises
    ValueError when they differ (``connect.dtype`` reports it first)."""
    found = {graph.containers[(n.inputs | n.outputs)[port].data].dtype for n in nodes}
    if len(found) != 1:
        raise ValueError(f"port {port}: its nodes put dtypes {sorted(found)} on it")
    return found.pop()


def accelerated(graph: Graph) -> list[Node]:
    """The accelerated nodes, in execution order."""
    return [n for n, _ in graph.walk() if n.kind == "accelerated"]


def instance_nodes(graph: Graph) -> dict[str, Node]:
    """The first accelerated node of each instance, in the order the graph uses them."""
    out: dict[str, Node] = {}
    for n in accelerated(graph):
        out.setdefault(n.attrs["instance"], n)
    return out


def group_of(node: Node) -> str:
    """The group a node's unit is in: its ``group`` (``link``'s), else its own, named after the
    instance."""
    return node.attrs.get("group", node.attrs["instance"])


def accelerator_of(node: Node) -> str:
    """The accelerator a node's group is in: its ``accelerator`` (``encapsulate``'s), else
    the group's own, named after it."""
    return node.attrs.get("accelerator", group_of(node))


def accelerators(graph: Graph) -> dict[str, list[str]]:
    """Every accelerator, in the order the graph first uses it, with its units (instances) in
    the graph's order; ``groups`` puts them into groups."""
    out: dict[str, list[str]] = {}
    for name, n in instance_nodes(graph).items():
        out.setdefault(accelerator_of(n), []).append(name)
    return out


def groups(graph: Graph) -> dict[str, list[str]]:
    """Every group, in the order the graph first uses it, with its units in the graph's order."""
    out: dict[str, list[str]] = {}
    for name, n in instance_nodes(graph).items():
        out.setdefault(group_of(n), []).append(name)
    return out


def links(graph: Graph) -> list[Node]:
    """The link nodes, in execution order."""
    return [n for n, _ in graph.walk() if n.kind == "link"]


def link_ends(graph: Graph, link: Node) -> tuple[str, str]:
    """A link's ``from`` and ``to`` as ``"<instance>.<port>"``: the writer's port and the
    reader's whose memlets name its container."""
    a, ends = link.attrs, []
    for inst, side in ((a["from"], "outputs"), (a["to"], "inputs")):
        # the instance's node whose memlet names the container: one of several when nodes share
        # the instance (D165), each with its own link (mvt's tmp0_prod and tmp2_prod)
        (port,) = {p for n in accelerated(graph) if n.attrs["instance"] == inst
                   for p, m in getattr(n, side).items() if m.data == a["container"]}  # fmt: skip
        ends.append(f"{inst}.{port}")
    return ends[0], ends[1]


def group_links(graph: Graph, members: list[str]) -> list[tuple[Node, str, str]]:
    """A group's links as ``(link node, from, to)``, one per pair of ports: a group several nests
    share (D165) has a link node per nest on the same ports (mvt's tmp0_prod and tmp2_prod), one
    piece of hardware; the first in the body's order stands for them."""
    out, depth = [], {}
    for k in links(graph):
        if k.attrs["from"] in members:
            ends = link_ends(graph, k)
            if ends in depth:
                if depth[ends] != k.attrs["depth"]:
                    raise ValueError(f"links {ends[0]} -> {ends[1]}: depths {depth[ends]} and "
                                     f"{k.attrs['depth']}; nests sharing a group link alike")  # fmt: skip
                continue
            depth[ends] = k.attrs["depth"]
            out.append((k, *ends))
    return out


def group_runs(graph: Graph) -> dict[str, list[dict[str, Node]]]:
    """Each linked group's runs, in the body's order: per run its units' nodes (instance ->
    node), the nodes its links join (a writer of a link's container and its reader). A group
    whose units each have one node is one run; a group several nests share (D165: mvt's half 2,
    ``tmp2_mult -> tmp2_prod -> sum_1`` on half 1's ``vmul`` and ``vsum``) runs once per set of
    nodes its links join. A run missing one of the group's units, or holding two nodes of one,
    is a ``ValueError``."""
    grouped = groups(graph)
    linked = {k: units for k, units in grouped.items() if len(units) > 1}
    group_of = {u: k for k, units in linked.items() for u in units}
    mine = [n for n in accelerated(graph) if n.attrs["instance"] in group_of]
    parent = {n.id: n.id for n in mine}

    def root(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    contained = {k.attrs["container"] for k in links(graph)}
    writer = {m.data: n.id for n in mine for m in n.outputs.values() if m.data in contained}
    for n in mine:
        for m in n.inputs.values():
            if m.data in writer:
                parent[root(n.id)] = root(writer[m.data])
    runs: dict[str, dict[str, Node]] = {}
    for n in mine:
        run = runs.setdefault(root(n.id), {})
        u = n.attrs["instance"]
        if u in run:
            raise ValueError(f"unit {u}: nodes {run[u].id} and {n.id} in one run "
                              f"of group {group_of[u]}; a run holds one node per unit")  # fmt: skip
        run[u] = n
    out: dict[str, list[dict[str, Node]]] = {k: [] for k in linked}
    for run in runs.values():
        group = group_of[next(iter(run))]
        missing = [u for u in linked[group] if u not in run]
        if missing:
            raise ValueError(f"group {group}: the run of "
                              f"{', '.join(m.id for m in run.values())} has no node of "
                              f"{', '.join(missing)}")  # fmt: skip
        out[group].append({u: run[u] for u in linked[group]})
    return out


def linked_ports(graph: Graph) -> set[str]:
    """Every port on a link, ``"<instance>.<port>"``: it has no streamer."""
    return {end for k in links(graph) for end in link_ends(graph, k)}


def unplaced(graph: Graph) -> set[str]:
    """The containers that get no memory: a link's and a fused nest's local value
    (``loop.local``)."""
    return {k.attrs["container"] for k in links(graph)} | {
        c for c, cont in graph.containers.items() if "loop.local" in cont.attrs
    }


def slot_connectors(node: Node) -> set[str]:
    """An accelerated node's input connectors that are no port of its BRM: slots read from
    their container (a CSR), which no streamer moves."""
    ports = {p.name for p in load_brm(node.attrs["brm"]).interface.ports}
    return {c for c in node.inputs if c not in ports}


def slot_containers(graph: Graph) -> set[str]:
    """The containers read only through slot connectors (a kernel's scalar argument the
    program writes into a CSR, ``k1: alpha[0]``): no streamer, no L1 space."""
    slot, other = set(), set()
    for n, _ in graph.walk():
        slots = slot_connectors(n) if n.kind == "accelerated" else set()
        for c, m in n.inputs.items():
            (slot if c in slots else other).add(m.data)
        other |= {m.data for m in n.outputs.values()}
    return slot - other


SHARE_PORTS = "hw.share_ports"


class ShareError(ValueError):
    """Ports that cannot share a mover (``share.shape``)."""

    code = "share.shape"


@dataclass(frozen=True)
class MoverUser:
    """A group's port on a shared mover: the group, the port's streamer (``<instance>_<port>``),
    the port as ``"<instance>.<port>"`` and its channels."""

    group: str
    streamer: str
    port: str
    num_ch: int


@dataclass(frozen=True)
class SharedMover:
    """A mover of a shared set (module doc): its name (``reader<k>``, ``writer<k>``), whether
    it writes, its channels, spatial bounds in words (fastest first), temporal loops and FIFO
    depth, and its users, at most one per group, in the groups' order."""

    name: str
    write: bool
    num_ch: int
    spatial_bounds: tuple[int, ...]
    temporal_dims: int
    fifo_depth: int
    users: tuple[MoverUser, ...]

    @property
    def masked(self) -> bool:
        """A user narrower than it: it has ``configurable_channel``."""
        return any(u.num_ch < self.num_ch for u in self.users)

    def user(self, streamer: str) -> MoverUser | None:
        return next((u for u in self.users if u.streamer == streamer), None)


def word_bounds(s: Streamer) -> tuple[int, ...]:
    """A streamer's spatial bounds in words, fastest first: the fastest over num_slc (gen.lanes
    refuses a fastest bound that is not whole words)."""
    s0, *rest = s.spatial_bounds
    return (s0 // s.num_slc, *rest)


def shared_containers(graph: Graph) -> list[str]:
    """The containers marked ``hw.share_ports``, in the graph's order."""
    return [c for c, cont in graph.containers.items() if SHARE_PORTS in cont.attrs]


def shared_accelerators(graph: Graph) -> list[str]:
    """The accelerators whose groups share their movers (module doc), in the graph's order: a
    marked container's writer and reader in one accelerator."""
    marked = set(shared_containers(graph))
    out = []
    for n in accelerated(graph):
        for m in n.outputs.values():
            if m.data not in marked:
                continue
            readers = [r for r in accelerated(graph)
                       if any(x.data == m.data for x in r.inputs.values())]  # fmt: skip
            if any(accelerator_of(r) == accelerator_of(n) for r in readers):
                out.append(accelerator_of(n))
    return list(dict.fromkeys(out))


def shared_movers(graph: Graph, found: dict[str, Streamer]) -> dict[str, list[SharedMover]]:
    """Each shared accelerator's movers, readers first (module doc); ``found`` is ``resolve``'s
    shell. Raises ``ShareError`` on a user that does not fit."""
    accs = set(shared_accelerators(graph))
    if not accs:
        return {}
    by_group: dict[str, dict[str, list[Streamer]]] = {}
    acc_groups: dict[str, list[str]] = {}
    nodes = instance_nodes(graph)
    for unit, n in nodes.items():
        acc = accelerator_of(n)
        if acc not in accs:
            continue
        g = group_of(n)
        if g not in acc_groups.setdefault(acc, []):
            acc_groups[acc].append(g)
        mine = by_group.setdefault(g, {"read": [], "write": []})
        for s in found.values():
            if s.instance == unit:
                mine["write" if s.write else "read"].append(s)
    out: dict[str, list[SharedMover]] = {}
    for acc, names in acc_groups.items():
        movers: list[SharedMover] = []
        for role in ("read", "write"):
            ranked = {g: sorted(by_group[g][role], key=lambda s: -s.num_ch) for g in names}
            for k in range(max((len(v) for v in ranked.values()), default=0)):
                users = [(g, ranked[g][k]) for g in names if k < len(ranked[g])]
                widest = max((s for _, s in users), key=lambda s: s.num_ch)
                bounds = word_bounds(widest)
                name = f"{'writer' if role == 'write' else 'reader'}{k}"
                for g, s in users:
                    b = word_bounds(s)
                    if len(b) != len(bounds) or b[:-1] != bounds[:-1] or b[-1] > bounds[-1]:
                        raise ShareError(
                            f"accelerator {acc}'s {name}: {s.name}'s spatial bounds {list(b)} "
                            f"(in words) do not fit in the first channels of {widest.name}'s "
                            f"{list(bounds)}: a shared mover takes ports of its bounds but the "
                            "slowest, which may be smaller"
                        )
                movers.append(SharedMover(
                    name, role == "write", widest.num_ch, bounds,
                    max(s.temporal_dims for _, s in users),
                    max(s.options.fifo_depth for _, s in users),
                    tuple(MoverUser(g, s.name, f"{s.instance}.{s.port}", s.num_ch)
                          for g, s in users),
                ))  # fmt: skip
        out[acc] = movers
    return out


def set_channels(graph: Graph, found: dict[str, Streamer]) -> dict[str, int]:
    """Each accelerator's streamer channels, the TCDM ports its streamer set takes: its ports'
    num_ch summed, or a shared set's movers' (module doc); raises ``ShareError`` as
    ``shared_movers``."""
    shared = shared_movers(graph, found)
    acc_of = {u: accelerator_of(n) for u, n in instance_nodes(graph).items()}
    out: dict[str, int] = {}
    for s in found.values():
        acc = acc_of[s.instance]
        if acc not in shared:
            out[acc] = out.get(acc, 0) + s.num_ch
    for acc, movers in shared.items():
        out[acc] = sum(m.num_ch for m in movers)
    return out


def instance_of(node: Node) -> Instance:
    """The node's instance, resolved against the BRM library (raises BrmError); an input
    connector named after a slot of the BRM's code, and no port, is the slot read from its
    container (``Instance.read``); any other connector is left to the checks."""
    a = node.attrs
    brm = load_brm(a["brm"])
    ports = {p.name for p in brm.interface.ports}
    read = [c for c in node.inputs if c not in ports and c in brm.all_slots()]
    return brm.resolve(a["implementation"], a["params"], read)


def instances(graph: Graph) -> dict[str, Instance]:
    """Every instance of the graph, resolved; raises BrmError on the first that is not."""
    return {k: instance_of(n) for k, n in instance_nodes(graph).items()}


def streamer_name(instance: str, port: str) -> str:
    return f"{instance}_{port}"


@dataclass(frozen=True)
class Loop:
    """One firing loop of an accelerated node: a temporal map around it."""

    map: str
    var: str
    begin: int
    count: int
    step: int


class LoopError(ValueError):
    """An accelerated node whose enclosing maps are not all its firing loops."""


def firing_loops(graph: Graph, node: Node) -> list[Loop]:
    """The temporal maps around ``node``, outermost first: its firing loop (D70, D79).

    Every map around an accelerated node must be ``temporal`` for now: a tile
    or untagged map would start a new task per iteration, which SNAX-LOWER
    does not lower yet (a tile transform and per-tile DMA come later).
    """
    outer = next(o for n, o in graph.walk() if n is node)
    symbols = graph.bound_symbols
    loops = []
    for m in outer:
        kind = m.attrs.get("loop.kind")
        if m.kind != "map" or kind != "temporal":
            raise LoopError(
                f"map {m.id} around {node.id} is {kind or 'untagged'}; only temporal maps "
                "(one task, one firing loop) are lowered for now"
            )
        r = map_range(m.attrs["range"], symbols, m.id)
        loops.append(Loop(m.id, m.attrs["var"], r.start, len(r), r.step))
    return loops


def port_loops(inst: Instance, port: str, memlet: Memlet, loops: list[Loop]) -> list[Loop]:
    """The firing loops a port's streamer runs over: all of them at rate 1; for a named
    rate, which moves one beat per T firings, the loops its memlet uses (D102)."""
    if inst.brm.port(port).rate == 1:
        return loops
    return [lp for lp in loops if lp.var in memlet.names]


def memlet_spatial_bounds(m: Memlet, symbols: dict[str, int], what: str) -> list[int]:
    """The lanes of a bound memlet: one bound per range dimension, fastest (last) first.

    The same rule as SNAX-LOWER's ``memlet_values``: a range's length over its step.
    """
    out = []
    for dim in m.subset:
        parts = parse_dim(dim, what)
        if len(parts) == 3:
            diff = f"({parts[1]}) - ({parts[0]})"
            loop_vars = sorted(expr.names(diff) - set(symbols))  # cancel out of a range's length
            length = expr.evaluate(expr.linear(diff, loop_vars)[0], symbols)
            out.append(len(range(0, length, parts[2])))
    return out[::-1]


def nest_spatial_bounds(inst: Instance, port: str) -> list[int]:
    """The spatial loops of ``port``'s nest, fastest first (as streams.py maps them).

    Evaluated for one task with every start parameter at its smallest value
    (each named rate 1, ``n`` the port's rate); spatial loops do not depend on it.
    """
    task = {r: 1 for r in inst.brm.registers}
    task["n"] = expr.evaluate(inst.brm.port(port).rate, {**inst.params, **task})
    nest = task_nest(inst, port, task)
    return [b for b, _ in reversed(nest.spatial)]


def resolve(platform: Platform, graph: Graph, insts: dict[str, Instance]) -> dict[str, Streamer]:
    """The shell for ``insts`` of ``graph`` on ``platform``, by streamer name; see the module doc."""
    out: dict[str, Streamer] = {}
    symbols = graph.bound_symbols
    nodes = accelerated(graph)
    linked = linked_ports(graph)
    for name, inst in insts.items():
        mine = [n for n in nodes if n.attrs["instance"] == name]
        loops = {n.id: firing_loops(graph, n) for n in mine}
        for p in inst.ports:
            if f"{name}.{p.name}" in linked:
                continue
            s = streamer_name(name, p.name)
            sb = platform.spatial_bounds(s)
            if sb is None and memlet_ordered(inst.brm) and mine:
                m = (mine[0].inputs | mine[0].outputs)[p.name]
                sb = memlet_spatial_bounds(m, symbols, f"node {mine[0].id}.{p.name}")
            if sb is None:
                sb = nest_spatial_bounds(inst, p.name)
            dims = [
                len(port_loops(inst, p.name, (n.inputs | n.outputs)[p.name], loops[n.id]))
                for n in mine
            ]
            dtype = port_dtype(graph, mine, p.name)
            num_slc = slices_per_word(dtype, platform.l1.width_bits)
            num_ch = platform.num_ch(s)
            if num_ch is None:
                if prod(sb) % num_slc:
                    raise ValueError(
                        f"streamer {s}: {prod(sb)} lanes of {dtype} are not whole words of "
                        f"num_slc = {num_slc} (connect.pack)"
                    )
                num_ch = prod(sb) // num_slc
            out[s] = Streamer(
                s, name, p.name, p.direction == "out", tuple(sb), max([1, *dims]),
                platform.options(s), dtype, num_slc, num_ch,
            )  # fmt: skip
    return out
