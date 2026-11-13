"""Padding: a loop bound that is not a multiple of its split factor.

Splitting a map over n iterations by a W that does not divide n (``split``, ``split_map``) runs
the outer map ceil(n / W) times and records ``valid: n`` in ``loop.split`` (CONTRACTS,
"SNAX-DFG"): a lane whose ``W * v_t + v_s`` is n or more is a pad lane. The container is padded
in memory, in the data, never by the hardware. ``padded_shape`` rounds each dimension a tailed
split's loop indexes up to a multiple of W; ``pad_value`` is what a pad element holds, the
identity of the fold that reads the container (``fold_op``) at its dtype, else 0 (readers that
need different pads raise ``DfgError``); ``code_on_pads`` is what a unit writes into its
output's pad lanes, its code on its inputs' pads, None when it cannot say. ``tailed_split``,
``valid_count`` and ``split_factor`` read a split's record. The reference executor reads a pad
lane as ``pad_value`` and never writes one; the memory plan lays a container out over its
padded shape, and data.h fills the pads.
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
    """A map's ``loop.split`` when its split left a tail (it has ``valid``), else None."""
    split = node.attrs.get("loop.split") if node.kind == "map" else None
    return split if split is not None and "valid" in split else None


def valid_count(graph: Graph, node: Node) -> int:
    """The valid count of a tailed split map, at the graph's bound symbols."""
    return int(expr.evaluate(tailed_split(node)["valid"], graph.bound_symbols))


def _users(graph: Graph, c: str) -> Iterator[tuple[Node, tuple[Node, ...]]]:
    for node, outer in graph.walk():
        if node.kind in ("tasklet", "accelerated") and any(m.data == c for m in node.memlets()):
            yield node, outer


def split_factor(node: Node) -> int:
    """W of a tailed split map: its only child map's length whatever its tag (``split`` leaves
    it untagged), else its spatial map's, or once bound, the lanes its accelerated node's
    ranges move per step of the map's variable."""
    var = node.attrs["var"]
    kids = node.body or []
    only = len(kids) == 1 and kids[0].kind == "map" and node.attrs.get("loop.split")
    for kid in kids:
        if kid.kind == "map" and (only or kid.attrs.get("loop.kind") == "spatial"):
            begin, end, _ = parse_dim(kid.attrs["range"], kid.id)
            return int(end) - int(begin)
        for m in kid.memlets():
            for dim in m.subset:
                parts = parse_dim(dim, kid.id)
                if len(parts) == 3 and var in expr.names(parts[0]):
                    return expr.linear(parts[0], [var])[1][var] // parts[2]
    raise DfgError(f"map {node.id}: a tailed split without a spatial map or bound lanes")


# NB (D144): padding lives in the data; the hardware masks nothing, so every pad lane must hold
# a value its reader ignores (zero, or a max or min fold's identity).
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
        for m in node.memlets():
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


def code_on_pads(graph: Graph, node: Node, pads: dict[str, int]) -> int | None:
    """What ``node`` writes into its one output's pad lanes, given its inputs' pad contents by
    port; see the module doc."""
    code = node.attrs.get("code")
    if code is None or len(node.outputs) != 1:
        return None
    ((port, out),) = node.outputs.items()
    values = {p: np.array(pads[p], dtype=graph.containers[m.data].dtype)
              for p, m in node.inputs.items()}  # fmt: skip
    results: dict[str, Any] = {}
    try:
        for target, rhs in expr.statements(code, f"node {node.id}.attrs.code"):
            results[target] = expr.evaluate(rhs, values | results)
    except (expr.ExprError, KeyError, TypeError, ValueError):
        return None
    if port not in results:
        return None
    return int(np.asarray(results[port]).astype(graph.containers[out.data].dtype))


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
