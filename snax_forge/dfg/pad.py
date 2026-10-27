"""A loop bound that is not a multiple of the spatial bound (open item 31, D144).

``split_map`` of a map over ``n`` iterations by a factor W that does not
divide n runs its temporal map ceil(n / W) times and records the valid count
in ``loop.split`` (``"valid": n``): only the lanes whose combined index
``W * v_t + v_s`` is below n are the kernel's. The rest are pad lanes. The
container is padded in memory, in the data itself, never by the hardware:

    padded_shape(graph, c)   each dimension of ``c`` a tailed split's loop
                             indexes rounded up to a multiple of the split's
                             factor W; the others as they are. A pad lane's
                             index may run into the padding, never past it
    pad_value(graph, c)      what a pad element holds: the identity of the
                             fold that reads ``c`` (dfg/reduce.py, at its
                             dtype: 0 for add, the dtype's minimum for max,
                             its maximum for min), else 0. Readers that need
                             different pads raise DfgError naming them

The reference executor reads a pad lane as ``pad_value`` and never writes
one (execute.py); the memory plan lays a container out over its padded
shape, and data.h fills the pads (D144).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from snax_forge import expr

from .graph import Graph, Node
from .kinds import DfgError
from .reduce import REDUCE_OPS
from .subset import parse_dim


def tailed_split(node: Node) -> dict[str, Any] | None:
    """A map's ``loop.split`` when split_map left a tail (it has ``valid``), else None."""
    split = node.attrs.get("loop.split") if node.kind == "map" else None
    return split if split is not None and "valid" in split else None


def valid_count(graph: Graph, node: Node) -> int:
    """The valid count of a tailed split map, at the graph's bound symbols."""
    return int(expr.evaluate(tailed_split(node)["valid"], graph.bound_symbols))


def _users(graph: Graph, c: str) -> Iterator[tuple[Node, tuple[Node, ...]]]:
    for node, outer in graph.walk():
        if node.kind in ("tasklet", "accelerated") and any(
            m.data == c for m in (*node.inputs.values(), *node.outputs.values())
        ):
            yield node, outer


def split_factor(node: Node) -> int:
    """W of a tailed split map: its spatial map's length, or once bound, the lanes its
    accelerated node's ranges move per step of the map's variable."""
    var = node.attrs["var"]
    for kid in node.body or []:
        if kid.kind == "map" and kid.attrs.get("loop.kind") == "spatial":
            begin, end, _ = parse_dim(kid.attrs["range"], kid.id)
            return int(end) - int(begin)
        for m in (*kid.inputs.values(), *kid.outputs.values()):
            for dim in m.subset:
                parts = parse_dim(dim, kid.id)
                if len(parts) == 3 and var in expr.names(parts[0]):
                    return expr.linear(parts[0], [var])[1][var] // parts[2]
    raise DfgError(f"map {node.id}: a tailed split without a spatial map or bound lanes")


def padded_shape(graph: Graph, c: str) -> tuple[int, ...]:
    """``c``'s shape with each dimension a tailed split indexes rounded up to a multiple of its
    factor W; see the module doc."""
    symbols = graph.bound_symbols
    shape = [int(expr.evaluate(s, symbols)) for s in graph.containers[c].shape]
    padded = list(shape)
    for node, outer in _users(graph, c):
        factors = {m.attrs["var"]: split_factor(m) for m in outer if tailed_split(m)}
        if not factors:
            continue
        for m in (*node.inputs.values(), *node.outputs.values()):
            if m.data != c:
                continue
            for d, dim in enumerate(m.subset):
                names = set().union(*(expr.names(p) for p in parse_dim(dim, f"node {node.id}")))
                for var, w in factors.items():
                    if var in names:
                        padded[d] = max(padded[d], -(-shape[d] // w) * w)
    return tuple(padded)


def _tasklets(node: Node) -> Iterator[Node]:
    if node.kind == "tasklet":
        yield node
    for n in node.body or []:
        yield from _tasklets(n)


def fold_op(node: Node) -> str | None:
    """The op a node folds its inputs with: a tasklet's ``reduce`` output, an accelerated node's
    through the subtree it replaced; None when it does not fold."""
    if node.kind == "accelerated":
        replaced = node.attrs.get("replaced")
        if not replaced:
            return None
        tasks = list(_tasklets(Node.from_dict(replaced, node.id)))
    else:
        tasks = [node]
    ops = {m.reduce.op for t in tasks for m in t.outputs.values() if m.reduce is not None}
    return ops.pop() if len(ops) == 1 else None


def pad_value(graph: Graph, c: str) -> int:
    """What ``c``'s pad elements hold; see the module doc."""
    dtype = np.dtype(graph.containers[c].dtype)
    need: dict[int, list[str]] = {}
    for node, _ in _users(graph, c):
        if not any(m.data == c for m in node.inputs.values()):
            continue
        op = fold_op(node)
        if op is not None:
            need.setdefault(int(REDUCE_OPS[op].identity(dtype)), []).append(f"{node.id} ({op})")
    if len(need) > 1:
        readers = "; ".join(f"{', '.join(ids)} need {v}" for v, ids in sorted(need.items()))
        raise DfgError(f"container {c}: its readers need different pads: {readers}")
    return next(iter(need), 0)
