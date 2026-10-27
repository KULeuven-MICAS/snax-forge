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
    k               elements of that dtype per bank word, width_bits over its
                    bits (D142)
    channels        the TCDM channels: the entry's ``channels``, else the wire
                    default W / k (D142; ``connect.pack`` checks that k
                    divides the fastest spatial bound, ``connect.channels``
                    an entry's value)
    temporal_dims   from the graph: the firing loops the port's memlet runs
                    over (``port_loops``, as SNAX-LOWER takes them), at least
                    1; with several nodes on one instance, the most (D84)
    fifo_depth      the platform's ``default`` with the streamer's entry on top

``instances`` reads the instances off the graph and resolves them against
the BRM library; ``resolve`` builds the shell. Neither checks the pairing:
that is check.py, which runs first and reports every problem, so
``resolve`` raises on the first thing that does not fit.
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
    k: int
    channels: int

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
            "k": self.k,
            "channels": self.channels,
            **self.options.to_dict(),
        }


def elements_per_word(dtype: str, width_bits: int) -> int:
    """k: elements of ``dtype`` in one bank word of ``width_bits`` (D142), at least 1."""
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


def instance_of(node: Node) -> Instance:
    """The node's instance, resolved against the BRM library (raises BrmError)."""
    a = node.attrs
    return load_brm(a["brm"]).resolve(a["implementation"], a["params"])


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
    for name, inst in insts.items():
        mine = [n for n in nodes if n.attrs["instance"] == name]
        loops = {n.id: firing_loops(graph, n) for n in mine}
        for p in inst.brm.interface.ports:
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
            k = elements_per_word(dtype, platform.l1.width_bits)
            channels = platform.channels(s)
            if channels is None:
                if prod(sb) % k:
                    raise ValueError(
                        f"streamer {s}: {prod(sb)} lanes of {dtype} are not whole words of "
                        f"{k} (connect.pack)"
                    )
                channels = prod(sb) // k
            out[s] = Streamer(
                s, name, p.name, p.direction == "out", tuple(sb), max([1, *dims]),
                platform.options(s), dtype, k, channels,
            )  # fmt: skip
    return out
