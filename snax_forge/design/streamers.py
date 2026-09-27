"""The streamer shell against a bound graph (DP1a, D84).

For every accelerated node's instance, in the order the graph first uses
them, and every port of its BRM in port order, one streamer:

    name            <instance>_<port> (D75)
    write           the port's direction is "out"
    spatial_bounds  the platform entry's, else the port nest's spatial loops,
                    fastest first (what snax_forge/lower/streams.py requires)
    n_ports         their product: the port's lanes
    temporal_dims, fifo_depth, addr_depth, prio
                    the platform's ``default`` with the streamer's entry on top

``instances`` reads the instances off the graph and resolves them against
the BRM library; ``resolve`` builds the shell. Neither checks the pairing:
that is check.py, which runs first and reports every problem, so
``resolve`` raises on the first thing that does not fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

from snax_forge import expr
from snax_forge.brm import Instance, load_brm, task_nest
from snax_forge.dfg import Graph, Node
from snax_forge.dfg.subset import parse_dim
from snax_forge.snax_model import StreamerConfig

from .platform import Platform, StreamerOptions


@dataclass(frozen=True)
class Streamer:
    """One resolved streamer of the shell."""

    name: str
    instance: str
    port: str
    write: bool
    spatial_bounds: tuple[int, ...]
    options: StreamerOptions

    @property
    def n_ports(self) -> int:
        return prod(self.spatial_bounds)

    def config(self) -> StreamerConfig:
        """The streamer's entry in the cluster file (LOW1c)."""
        return self.options.config(write=self.write, n_ports=self.n_ports)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance": self.instance,
            "port": self.port,
            "write": self.write,
            "n_ports": self.n_ports,
            "spatial_bounds": list(self.spatial_bounds),
            **self.options.to_dict(),
        }


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
    or untagged map would start a new task per iteration, which LOW1a does
    not lower yet (a tile transform and per-tile DMA come later).
    """
    outer = next(o for n, o in graph.walk() if n is node)
    symbols = {k: v for k, v in graph.symbols.items() if v is not None}
    loops = []
    for m in outer:
        kind = m.attrs.get("loop.kind")
        if m.kind != "map" or kind != "temporal":
            raise LoopError(
                f"map {m.id} around {node.id} is {kind or 'untagged'}; only temporal maps "
                "(one task, one firing loop) are lowered for now"
            )
        b, e, s = parse_dim(m.attrs["range"], m.id)
        lo, hi = expr.evaluate(b, symbols), expr.evaluate(e, symbols)
        loops.append(Loop(m.id, m.attrs["var"], lo, len(range(lo, hi, s)), s))
    return loops


def nest_spatial_bounds(inst: Instance, port: str) -> list[int]:
    """The spatial loops of ``port``'s nest, fastest first (as streams.py maps them).

    Evaluated for one task with every start parameter at its smallest value
    (each named rate 1, ``n`` the port's rate); spatial loops do not depend on it.
    """
    task = {r: 1 for r in inst.brm.registers}
    task["n"] = expr.evaluate(inst.brm.port(port).rate, {**inst.params, **task})
    nest = task_nest(inst, port, task)
    return [b for b, _ in reversed(nest.spatial)]


def resolve(platform: Platform, insts: dict[str, Instance]) -> dict[str, Streamer]:
    """The shell for ``insts`` on ``platform``, by streamer name; see the module doc."""
    out: dict[str, Streamer] = {}
    for name, inst in insts.items():
        for p in inst.brm.interface.ports:
            s = streamer_name(name, p.name)
            sb = platform.spatial_bounds(s) or nest_spatial_bounds(inst, p.name)
            out[s] = Streamer(s, name, p.name, p.direction == "out", tuple(sb), platform.options(s))
    return out
