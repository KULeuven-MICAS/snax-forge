"""Values in a BRM: an int, or a small expression over the BRM's params (D68).

A value field of a BRM (port lanes and rate, timing, function params) holds
either a plain int or a string. Every string is an expression: a param name
(``"W"``, ``"op"``) or arithmetic over params and ints with ``+``, ``-``,
``*``, ``//`` and parentheses (``"W*n"``, ``"(n + 1) // 2"``). There are no
string literals; a fixed string such as an op name is a design param with a
single allowed value.

Expressions are parsed with ``ast`` and evaluated by walking the tree; nothing
is passed to ``eval``. A bare name may stand for a param of any type; the
arithmetic works on ints only.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

Value = int | str

_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.FloorDiv: lambda a, b: a // b,
}


class ExprError(ValueError):
    """A value that is not an int or a valid expression."""


def _parse(expr: str) -> ast.expr:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ExprError(f"cannot parse {expr!r}") from e
    for node in ast.walk(tree):
        ok = isinstance(node, (ast.Expression, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp))
        ok = ok or type(node) in _BINOPS or isinstance(node, ast.USub)
        ok = ok or (isinstance(node, ast.Constant) and type(node.value) is int)
        if not ok:
            raise ExprError(f"{expr!r}: {type(node).__name__} is not allowed")
    return tree.body


def check(v: Any, what: str) -> None:
    """``v`` is an int or a string that parses as an expression."""
    if type(v) is int:
        return
    if not isinstance(v, str):
        raise ExprError(f"{what}: must be an int or an expression string, got {v!r}")
    try:
        _parse(v)
    except ExprError as e:
        raise ExprError(f"{what}: {e}") from None


def names(v: Value) -> set[str]:
    """The param names ``v`` refers to (none for an int)."""
    if type(v) is int:
        return set()
    return {n.id for n in ast.walk(_parse(v)) if isinstance(n, ast.Name)}


def is_name(v: Value) -> bool:
    """``v`` is a bare param name."""
    return isinstance(v, str) and isinstance(_parse(v), ast.Name)


def constant(v: Value) -> int | None:
    """The value of ``v`` if it refers to no param, else None."""
    return evaluate(v, {}) if not names(v) else None


def evaluate(v: Value, env: Mapping[str, Any]) -> Any:
    """``v`` with its names taken from ``env``."""
    if type(v) is int:
        return v

    def walk(node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ExprError(f"{v!r}: no value for {node.id!r}")
            return env[node.id]
        if isinstance(node, ast.UnaryOp):
            return -_int(walk(node.operand))
        return _BINOPS[type(node.op)](_int(walk(node.left)), _int(walk(node.right)))

    def _int(x: Any) -> int:
        if type(x) is not int:
            raise ExprError(f"{v!r}: arithmetic on non-int {x!r}")
        return x

    return walk(_parse(v))
