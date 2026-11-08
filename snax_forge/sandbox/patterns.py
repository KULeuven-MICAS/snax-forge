"""Pattern matchers: what a BRM's ``pattern`` part means in a graph.

A BRM's pattern names a ``family`` and gives ``attrs``. The family is
a matcher registered here; the attrs are its parameters. ``bind`` asks the
matcher whether a tasklet is an instance of the pattern, and if so which
tasklet connector feeds which BRM port:

    matcher(tasklet, attrs, brm, dtypes, scalars) -> {connector: port}

``dtypes`` gives each connector's container dtype (the graph's, D143), ``scalars``
the input connectors that are scalar operands (``bind``'s).

or raises ``SandboxError`` saying why not. A family's matcher is code in the
sandbox, registered like any other extension (principle 6).

``simd`` (the first family, no attrs): the tasklet's code is one statement
with one output, in the expression grammar, over its input connectors, each
array input read once, and int literals (``out = in1 + in2``, ``out = max(in1, 0)``); every
input connector is read. The array inputs map onto the BRM's input ports in the order
they first appear, the output onto its one output port; the scalar operands
(``scalars``, ``bind``'s) map to no port. A fused nest of two plain tasklets
(``fuse_map``, its second no fold) is matched as the one tasklet ``fused_view``
composes. What it computes, and so how many inputs it takes and on which ports,
is ``bind``'s code match: the BRM's code entry whose present input ports take the
array inputs, in some assignment, and whose code is the tasklet's up to its slots and
commuting operands (``simd``'s ``op``, ``streams``, ``k``, ``k1`` and ``k2``,
brm.py's ``Code``).

``reduce`` (D102): the tasklet copies its one input into its one output
(``out = in1``), and the output memlet folds (``reduce``) with the op
``attrs.op`` names, from that op's identity for the output container's
dtype. That is
what the importer makes of DaCe's ``Reduce``, and what a BRM of family ``reduce``
(``accumulate``) does: this matcher is the one place where the SNAX-DFG's
``reduce`` and the BRM's pattern meet, so the BRM names no IR's term.

``lane_fold`` (D149, ``mac``): a fused nest (``fuse_map``), the spatial map
``bind`` is given with its two tasklets: the first a ``simd`` fold with
``attrs.op`` over two inputs, writing a local value the second reads at the
same index; the second a ``reduce`` copy of it with ``attrs.fold``, its
output indexed by the spatial variable (one partial per lane, from
``split_reduce``). The connectors are the nest's own (``fused_view``): the
first tasklet's inputs onto the BRM's inputs in fold order, the second's
output onto its output. A tasklet is refused with "fuse_map first", a fold
into one element with "split_reduce first" (D150).

``lane_fold`` without ``attrs.op`` (``lane_accumulate``, no front op): the node
bound is a spatial map whose body is one tasklet copying its one input into its
one output (``out = in1``), the output folding (``reduce``) with ``attrs.fold``
and indexed by the spatial variable, so each lane folds its own element (rowsum
2's ``y[4 * m_1_t + m_2_s]``). The maps around it are temporal (``bind``'s site), and
those the input names and the output does not are the ones it folds over. Refused,
naming what is missing: the tasklet bound instead of its map, a front op (a fused
nest, or a tasklet that is not a copy: "bind mac"), an output not indexed by the
lane, and a fold over the bound map itself (the reduced loop spatial: "a fold
across the lanes binds accumulate").
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from snax_forge import expr
from snax_forge.brm.brm import SEED
from snax_forge.dfg import REDUCE_OPS, Memlet, Node

from .recipe import SandboxError

Matcher = Callable[
    [Node, Mapping[str, Any], Any, Mapping[str, str], frozenset[str]], dict[str, str]
]
PATTERNS: dict[str, Matcher] = {}

OPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}


def register_pattern(family: str, matcher: Matcher) -> None:
    """Make BRMs with ``pattern.family == family`` bindable (principle 6)."""
    if family in PATTERNS:
        raise ValueError(f"pattern family {family!r} already registered")
    PATTERNS[family] = matcher


def match(
    node: Node, brm: Any, dtypes: Mapping[str, str], scalars: frozenset[str] = frozenset()
) -> dict[str, str]:
    """``{connector: port}`` if ``node`` matches ``brm``'s pattern; else SandboxError.
    ``dtypes`` is each connector's container dtype, ``scalars`` the input connectors that are
    scalar operands (``bind``'s: a kernel's scalar argument, the same element every firing),
    which go to no port."""
    family = brm.pattern.family
    if family not in PATTERNS:
        raise SandboxError(f"brm {brm.name!r}: no matcher for pattern family {family!r}")
    return PATTERNS[family](node, brm.pattern.attrs, brm, dtypes, scalars)


def _fold(node: Node, arity: Any, brm: Any, what: str) -> tuple[str, dict[str, str]]:
    """A tasklet that folds ``arity`` distinct inputs with one operator: the operator and
    ``{connector: port}``, the inputs onto the BRM's inputs in fold order."""
    if node.kind != "tasklet":
        raise SandboxError(f"{what}: a {node.kind!r} is not a tasklet")
    stmts = ast.parse(node.attrs["code"]).body
    if len(stmts) != 1 or len(node.outputs) != 1:
        raise SandboxError(f"{what}: simd is one statement with one output")
    stmt, out = stmts[0], next(iter(node.outputs))
    operands: list[str] = []
    ops: set[str] = set()

    def fold(e: ast.expr) -> None:
        if isinstance(e, ast.Name):
            operands.append(e.id)
        elif isinstance(e, ast.BinOp) and type(e.op) in OPS and isinstance(e.right, ast.Name):
            fold(e.left)
            ops.add(OPS[type(e.op)])
            operands.append(e.right.id)
        else:
            raise SandboxError(f"{what}: {node.attrs['code']!r} is not a fold of inputs")

    fold(stmt.value)
    if len(ops) != 1 or len(set(operands)) != len(operands):
        raise SandboxError(
            f"{what}: {node.attrs['code']!r} is not one operator over distinct inputs"
        )
    (op,) = ops
    if len(operands) != arity:
        raise SandboxError(
            f"{what}: the tasklet is {op} of {len(operands)}, the pattern takes {arity} inputs"
        )
    ins = [p.name for p in brm.interface.ports if p.direction == "in"]
    outs = [p.name for p in brm.interface.ports if p.direction == "out"]
    if len(ins) != len(operands) or len(outs) != 1:
        raise SandboxError(f"{what}: the BRM has ports {ins} -> {outs}")
    return op, {**dict(zip(operands, ins, strict=True)), out: outs[0]}


def _simd(
    node: Node, attrs: Mapping[str, Any], brm: Any, dtypes: Mapping[str, str],
    scalars: frozenset[str],
) -> dict[str, str]:  # fmt: skip
    what = f"node {node.id!r} as {brm.name!r}"
    if node.kind == "map":  # a fused nest of two plain tasklets: the one they compose
        found = fused_view(node)
        if found is None:
            raise SandboxError(f"{what}: not a fused nest of two tasklets (fuse_map)")
        if any(m.reduce is not None for m in found[1].outputs.values()):
            raise SandboxError(f"{what}: {found[1].id} folds its input; simd binds a fused nest "
                               "of two plain tasklets")  # fmt: skip
        node = found[2]
    if node.kind != "tasklet":
        raise SandboxError(f"{what}: a {node.kind!r} is not a tasklet")
    code = node.attrs["code"]
    try:
        lines = expr.statements(code, what)
    except expr.ExprError as e:
        raise SandboxError(str(e)) from None
    if len(lines) != 1 or len(node.outputs) != 1:
        raise SandboxError(f"{what}: simd is one statement with one output")
    ((_, rhs),), out = lines, next(iter(node.outputs))
    read = expr.ordered_names(rhs)
    for n in read:
        if n not in node.inputs:
            raise SandboxError(f"{what}: {code!r} reads {n!r}, not an input connector (simd "
                               "reads its inputs and int literals)")  # fmt: skip
        if n not in scalars and expr.occurrences(rhs, n) > 1:
            raise SandboxError(f"{what}: {code!r} reads {n} more than once; simd reads distinct "
                               "inputs, each once")  # fmt: skip
    if unread := [c for c in node.inputs if c not in read]:
        raise SandboxError(f"{what}: {code!r} does not read the input {unread[0]}")
    arrays = [n for n in read if n not in scalars]
    ins = [p.name for p in brm.interface.ports if p.direction == "in"]
    outs = [p.name for p in brm.interface.ports if p.direction == "out"]
    if len(arrays) > len(ins) or len(outs) != 1:
        raise SandboxError(f"{what}: {len(arrays)} inputs; the BRM has ports {ins} -> {outs}")
    return {**dict(zip(arrays, ins, strict=False)), out: outs[0]}


register_pattern("simd", _simd)


def _reduce(
    node: Node, attrs: Mapping[str, Any], brm: Any, dtypes: Mapping[str, str],
    scalars: frozenset[str],
) -> dict[str, str]:  # fmt: skip
    what = f"node {node.id!r} as {brm.name!r}"
    if node.kind != "tasklet":
        raise SandboxError(f"{what}: a {node.kind!r} is not a reduce tasklet")
    if len(node.inputs) != 1 or len(node.outputs) != 1:
        raise SandboxError(f"{what}: a reduce folds one input into one output")
    (cin,), (cout,) = node.inputs, node.outputs
    if node.attrs["code"] != f"{cout} = {cin}":
        raise SandboxError(
            f"{what}: {node.attrs['code']!r} is not a copy of its input ({cout} = {cin})"
        )
    # the seed's port takes no connector (bind puts a seeded fold's seed there)
    ins = [p for p in brm.interface.ports if p.direction == "in" and p.name != SEED]
    outs = [p for p in brm.interface.ports if p.direction == "out"]
    if len(ins) != 1 or len(outs) != 1:
        raise SandboxError(f"{what}: the BRM has ports {[p.name for p in brm.interface.ports]}")
    red = node.outputs[cout].reduce
    if red is None:
        raise SandboxError(f"{what}: its output {cout!r} does not fold (no reduce on the memlet)")
    op = attrs.get("op")
    if op not in REDUCE_OPS:
        raise SandboxError(f"{what}: the pattern's op {op!r} is not a registered reduce op")
    if red.op != op:
        raise SandboxError(f"{what}: the tasklet folds with {red.op!r}, the pattern with {op!r}")
    identity = REDUCE_OPS[op].identity(np.dtype(dtypes[cout]))
    if red.seed is None and red.identity != identity:
        raise SandboxError(
            f"{what}: the fold starts from {red.identity}, the BRM from {op}'s identity {identity}"
        )
    return {cin: ins[0].name, cout: outs[0].name}


register_pattern("reduce", _reduce)


def fused_view(smap: Node) -> tuple[Node, Node, Node] | None:
    """A fused nest's two tasklets and the one tasklet they stand for to ``bind``: the first's
    inputs and the second's other inputs (one clashing with a first's connector renamed to
    the next free ``in<k>``), the second's output, and the second's code with the first's
    expression in place of the local value (axpy's ``out = in3 + in1 * in2``: ``y`` plus
    ``alpha`` times ``x``; a fold's ``out = in1 * in2``). None when ``smap`` is not a spatial
    map of two tasklets, the first's one output read once by the second at the same index."""
    if smap.kind != "map" or smap.attrs.get("loop.kind") != "spatial":
        return None
    kids = smap.body or []
    if len(kids) != 2 or any(k.kind != "tasklet" for k in kids):
        return None
    first, second = kids
    if len(first.outputs) != 1 or len(second.outputs) != 1:
        return None
    (m1,) = first.outputs.values()
    local = [c for c, m in second.inputs.items() if (m.data, m.subset) == (m1.data, m1.subset)]
    if len(local) != 1 or m1.reduce is not None:
        return None
    (c2,) = second.outputs
    s1, s2 = ast.parse(first.attrs["code"]).body, ast.parse(second.attrs["code"]).body
    if len(s1) != 1 or len(s2) != 1 or not isinstance(s1[0], ast.Assign):
        return None
    if not isinstance(s2[0], ast.Assign):
        return None
    others = [c for c in second.inputs if c != local[0]]
    taken = set(first.inputs) | {c2} | set(others)
    rename: dict[str, str] = {}
    for c in others:
        if c in first.inputs or c == c2:
            k = 1
            while f"in{k}" in taken:
                k += 1
            rename[c] = f"in{k}"
            taken.add(rename[c])
    body = _swap_names(s2[0].value, rename | {local[0]: ""}, s1[0].value)
    code = f"{c2} = {ast.unparse(body)}"
    inputs = dict(first.inputs) | {rename.get(c, c): m for c, m in second.inputs.items()
                                   if c != local[0]}  # fmt: skip
    view = Node(first.id, "tasklet", inputs=inputs,
                outputs={c2: Memlet.from_dict(second.outputs[c2].to_dict(), first.id)},
                attrs={"code": code})  # fmt: skip
    return first, second, view


def _swap_names(tree: ast.expr, rename: Mapping[str, str], value: ast.expr) -> ast.expr:
    """``tree`` with each name in ``rename`` renamed, one renamed to ``""`` replaced by
    ``value``."""

    class Swap(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name) -> ast.expr:
            if n.id not in rename:
                return n
            return copy.deepcopy(value) if rename[n.id] == "" else ast.Name(rename[n.id], n.ctx)

    return Swap().visit(copy.deepcopy(tree))


def _lane_fold(
    node: Node, attrs: Mapping[str, Any], brm: Any, dtypes: Mapping[str, str],
    scalars: frozenset[str],
) -> dict[str, str]:  # fmt: skip
    what = f"node {node.id!r} as {brm.name!r}"
    if "op" not in attrs:  # no front op: lane_accumulate
        return _lane_accumulate(node, attrs, brm, dtypes, what)
    if node.kind == "tasklet":
        if any(m.reduce is not None for m in node.outputs.values()):
            raise SandboxError(f"{what}: a {brm.name} folds one partial per lane, this tasklet "
                               "folds into the output: split_reduce first (D150)")  # fmt: skip
        raise SandboxError(f"{what}: a {brm.name} binds a multiply and the fold of its products "
                           "as one nest: fuse_map first (D150)")  # fmt: skip
    found = fused_view(node)
    if found is None:
        raise SandboxError(f"{what}: not a fused nest of a multiply and its fold (fuse_map first)")
    first, second, _ = found
    if len(second.inputs) != 1:
        raise SandboxError(f"{what}: {second.id} reads {len(second.inputs)} inputs; a "
                           f"{brm.name}'s fold reads the product only")  # fmt: skip
    op, ports = _fold(first, 2, brm, what)
    if op != attrs.get("op"):
        raise SandboxError(f"{what}: {first.id} is {op}, the pattern {attrs.get('op')}")
    (cin,), (cout,) = second.inputs, second.outputs
    if second.attrs["code"] != f"{cout} = {cin}":
        raise SandboxError(f"{what}: {second.id}'s {second.attrs['code']!r} is not a copy of its "
                           f"input ({cout} = {cin})")  # fmt: skip
    red = second.outputs[cout].reduce
    fold = attrs.get("fold")
    if red is None or red.op != fold:
        raise SandboxError(f"{what}: {second.id} does not fold with {fold!r}")
    identity = REDUCE_OPS[fold].identity(np.dtype(dtypes[cout]))
    if red.identity != identity:
        raise SandboxError(f"{what}: {second.id}'s fold starts from {red.identity}, the BRM "
                           f"from {fold}'s identity {identity}")  # fmt: skip
    if node.attrs["var"] not in second.outputs[cout].names:
        raise SandboxError(f"{what}: {second.id} folds every lane into one element; a "
                           f"{brm.name} keeps one partial per lane: split_reduce first (D150)")  # fmt: skip
    out = next(p for c, p in ports.items() if c in first.outputs)
    return {c: p for c, p in ports.items() if c in first.inputs} | {cout: out}


def _lane_accumulate(node: Node, attrs: Mapping[str, Any], brm: Any, dtypes: Mapping[str, str],
                     what: str) -> dict[str, str]:  # fmt: skip
    """``lane_fold`` without a front op (module doc): a spatial map of one tasklet copying its
    input into an output folded with ``attrs.fold`` and indexed by the lane."""
    if node.kind == "tasklet":
        raise SandboxError(f"{what}: a {brm.name} binds the spatial map around a folding "
                           "tasklet, not the tasklet: bind its map")  # fmt: skip
    if fused_view(node) is not None:
        first = node.body[0]
        raise SandboxError(f"{what}: {first.id} computes {_rhs(first)!r} before the fold, a front "
                           f"op; a {brm.name} folds its input as it is: bind mac")  # fmt: skip
    kids = node.body or []
    if node.kind != "map" or len(kids) != 1 or kids[0].kind != "tasklet":
        raise SandboxError(f"{what}: a {brm.name} binds a spatial map of one tasklet")
    t = kids[0]
    if len(t.outputs) != 1:
        raise SandboxError(f"{what}: {t.id} writes {len(t.outputs)} outputs; a {brm.name} one")
    (cout,) = t.outputs
    red, fold = t.outputs[cout].reduce, attrs.get("fold")
    if red is None:
        raise SandboxError(f"{what}: {t.id}'s output {cout!r} does not fold (no reduce on the "
                           "memlet)")  # fmt: skip
    if len(t.inputs) != 1 or t.attrs["code"] != f"{cout} = {next(iter(t.inputs))}":
        raise SandboxError(f"{what}: {t.id} computes {_rhs(t)!r} before the fold, a front op; a "
                           f"{brm.name} folds its one input as it is: bind mac")  # fmt: skip
    (cin,) = t.inputs
    if red.op != fold:
        raise SandboxError(f"{what}: {t.id} folds with {red.op!r}, the pattern with {fold!r}")
    identity = REDUCE_OPS[fold].identity(np.dtype(dtypes[cout]))
    if red.identity != identity:
        raise SandboxError(f"{what}: {t.id}'s fold starts from {red.identity}, the BRM from "
                           f"{fold}'s identity {identity}")  # fmt: skip
    var, m = node.attrs["var"], t.outputs[cout]
    if var not in m.names:
        if var in t.inputs[cin].names:
            raise SandboxError(f"{what}: {t.id} folds over {node.id!r} ({var}), which is spatial; a "
                               f"{brm.name} folds each lane over the temporal maps around it, a "
                               "fold across the lanes binds accumulate")  # fmt: skip
        raise SandboxError(f"{what}: {t.id}'s output {m.data}{m.subset} is not indexed by the "
                           f"lane variable {var!r}; each lane folds its own element")  # fmt: skip
    ins = [p.name for p in brm.interface.ports if p.direction == "in"]
    outs = [p.name for p in brm.interface.ports if p.direction == "out"]
    if len(ins) != 1 or len(outs) != 1:
        raise SandboxError(f"{what}: the BRM has ports {ins} -> {outs}")
    return {cin: ins[0], cout: outs[0]}


def _rhs(t: Node) -> str:
    """A tasklet's code's right-hand side."""
    return t.attrs["code"].split("=", 1)[1].strip()


register_pattern("lane_fold", _lane_fold)
