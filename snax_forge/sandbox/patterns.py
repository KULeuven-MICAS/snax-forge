"""Pattern matchers: what a BRM's ``pattern`` part means in a graph.

A BRM's pattern names a ``family`` and gives ``attrs``. The family is
a matcher registered here; the attrs are its parameters. ``bind`` asks the
matcher whether a tasklet is an instance of the pattern, and if so which
tasklet connector feeds which BRM port:

    matcher(tasklet, attrs, brm, dtypes) -> {connector: port}

``dtypes`` gives each connector's container dtype (the graph's, D143).

or raises ``SandboxError`` saying why not. A family's matcher is code in the
sandbox, registered like any other extension (principle 6).

``elementwise`` (the first family): the tasklet's code is one statement,
``out = in1 <op> in2 <op> ...``, a left fold of distinct input connectors
with one operator; ``attrs.op`` names the operator (``add``, ``sub``,
``mul``) and ``attrs.arity`` the number of inputs. Inputs map onto the
BRM's input ports in fold order, the output onto its one output port.

``reduce`` (D102): the tasklet copies its one input into its one output
(``out = in1``), and the output memlet folds (``reduce``) with the op
``attrs.op`` names, from that op's identity for the output container's
dtype. That is
what the importer makes of DaCe's ``Reduce``, and what a BRM of family ``reduce``
(``accumulate``) does: this matcher is the one place where the SNAX-DFG's
``reduce`` and the BRM's pattern meet, so the BRM names no IR's term.

``lane_fold`` (D149, ``mac``): a fused nest (``fuse_map``), the spatial map
``bind`` is given with its two tasklets: the first ``elementwise`` with
``attrs.op`` over two inputs, writing a local value the second reads at the
same index; the second a ``reduce`` copy of it with ``attrs.fold``, its
output indexed by the spatial variable (one partial per lane, from
``split_reduce``). The connectors are the nest's own (``fused_view``): the
first tasklet's inputs onto the BRM's inputs in fold order, the second's
output onto its output. A tasklet is refused with "fuse_map first", a fold
into one element with "split_reduce first" (D150).
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from snax_forge.dfg import REDUCE_OPS, Memlet, Node

from .recipe import SandboxError

Matcher = Callable[[Node, Mapping[str, Any], Any, Mapping[str, str]], dict[str, str]]
PATTERNS: dict[str, Matcher] = {}

OPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}


def register_pattern(family: str, matcher: Matcher) -> None:
    """Make BRMs with ``pattern.family == family`` bindable (principle 6)."""
    if family in PATTERNS:
        raise ValueError(f"pattern family {family!r} already registered")
    PATTERNS[family] = matcher


def match(node: Node, brm: Any, dtypes: Mapping[str, str]) -> dict[str, str]:
    """``{connector: port}`` if ``node`` matches ``brm``'s pattern; else SandboxError.
    ``dtypes`` is each connector's container dtype."""
    family = brm.pattern.family
    if family not in PATTERNS:
        raise SandboxError(f"brm {brm.name!r}: no matcher for pattern family {family!r}")
    return PATTERNS[family](node, brm.pattern.attrs, brm, dtypes)


def _elementwise(
    node: Node, attrs: Mapping[str, Any], brm: Any, dtypes: Mapping[str, str]
) -> dict[str, str]:
    what = f"node {node.id!r} as {brm.name!r}"
    if node.kind != "tasklet":
        raise SandboxError(f"{what}: a {node.kind!r} is not an elementwise tasklet")
    stmts = ast.parse(node.attrs["code"]).body
    if len(stmts) != 1 or len(node.outputs) != 1:
        raise SandboxError(f"{what}: elementwise is one statement with one output")
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
    if op != attrs.get("op") or len(operands) != attrs.get("arity"):
        raise SandboxError(
            f"{what}: the tasklet is {op} of {len(operands)}, the pattern is "
            f"{attrs.get('op')} of {attrs.get('arity')}"
        )
    ins = [p.name for p in brm.interface.ports if p.direction == "in"]
    outs = [p.name for p in brm.interface.ports if p.direction == "out"]
    if len(ins) != len(operands) or len(outs) != 1:
        raise SandboxError(f"{what}: the BRM has ports {ins} -> {outs}")
    return {**dict(zip(operands, ins, strict=True)), out: outs[0]}


register_pattern("elementwise", _elementwise)


def _reduce(
    node: Node, attrs: Mapping[str, Any], brm: Any, dtypes: Mapping[str, str]
) -> dict[str, str]:
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
    ins = [p for p in brm.interface.ports if p.direction == "in"]
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
    if red.identity != identity:
        raise SandboxError(
            f"{what}: the fold starts from {red.identity}, the BRM from {op}'s identity {identity}"
        )
    return {cin: ins[0].name, cout: outs[0].name}


register_pattern("reduce", _reduce)


def fused_view(smap: Node) -> tuple[Node, Node, Node] | None:
    """A fused nest's two tasklets and the one tasklet they stand for to ``bind``: the first's
    inputs, the second's outputs, the first's code onto the second's output connector. None
    when ``smap`` is not a spatial map of two tasklets, the first's one output read by the
    second at the same index."""
    if smap.kind != "map" or smap.attrs.get("loop.kind") != "spatial":
        return None
    kids = smap.body or []
    if len(kids) != 2 or any(k.kind != "tasklet" for k in kids):
        return None
    first, second = kids
    if len(first.outputs) != 1 or len(second.inputs) != 1 or len(second.outputs) != 1:
        return None
    (m1,), (m2,) = first.outputs.values(), second.inputs.values()
    if (m1.data, m1.subset) != (m2.data, m2.subset) or m1.reduce is not None:
        return None
    (c2,) = second.outputs
    if set(first.inputs) & {c2}:
        return None
    stmts = ast.parse(first.attrs["code"]).body
    if len(stmts) != 1 or not isinstance(stmts[0], ast.Assign):
        return None
    code = f"{c2} = {ast.unparse(stmts[0].value)}"
    view = Node(first.id, "tasklet", inputs=dict(first.inputs),
                outputs={c2: Memlet.from_dict(second.outputs[c2].to_dict(), first.id)},
                attrs={"code": code})  # fmt: skip
    return first, second, view


def _lane_fold(
    node: Node, attrs: Mapping[str, Any], brm: Any, dtypes: Mapping[str, str]
) -> dict[str, str]:
    what = f"node {node.id!r} as {brm.name!r}"
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
    ports = _elementwise(first, {"op": attrs.get("op"), "arity": 2}, brm, dtypes)
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


register_pattern("lane_fold", _lane_fold)
