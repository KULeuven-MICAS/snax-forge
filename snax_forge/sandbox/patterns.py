"""Pattern matchers (SBX1, D80): what a BRM's ``pattern`` part means in a graph.

A BRM's pattern names a ``family`` and gives ``attrs`` (D68). The family is
a matcher registered here; the attrs are its parameters. ``bind`` asks the
matcher whether a tasklet is an instance of the pattern, and if so which
tasklet connector feeds which BRM port:

    matcher(tasklet, attrs, brm) -> {connector: port}

or raises ``SandboxError`` saying why not. The BRM's ``predicate`` field
stays null: a family's matcher is code in the sandbox, registered like any
other extension (principle 6).

``elementwise`` (the first family): the tasklet's code is one statement,
``out = in1 <op> in2 <op> ...``, a left fold of distinct input connectors
with one operator; ``attrs.op`` names the operator (``add``, ``sub``,
``mul``) and ``attrs.arity`` the number of inputs. Inputs map onto the
BRM's input ports in fold order, the output onto its one output port.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from typing import Any

from snax_forge.dfg import Node

from .recipe import SandboxError

Matcher = Callable[[Node, Mapping[str, Any], Any], dict[str, str]]
PATTERNS: dict[str, Matcher] = {}

OPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}


def register_pattern(family: str, matcher: Matcher) -> None:
    """Make BRMs with ``pattern.family == family`` bindable (principle 6)."""
    if family in PATTERNS:
        raise ValueError(f"pattern family {family!r} already registered")
    PATTERNS[family] = matcher


def match(node: Node, brm: Any) -> dict[str, str]:
    """``{connector: port}`` if ``node`` matches ``brm``'s pattern; else SandboxError."""
    family = brm.pattern.family
    if family not in PATTERNS:
        raise SandboxError(f"brm {brm.name!r}: no matcher for pattern family {family!r}")
    return PATTERNS[family](node, brm.pattern.attrs, brm)


def _elementwise(node: Node, attrs: Mapping[str, Any], brm: Any) -> dict[str, str]:
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
