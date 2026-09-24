"""Values: an int, or a small integer expression over names (D68, D77).

Shared by SNAX-BRM and SNAX-DFG. A value field holds either a plain int or a
string. Every string is an expression: a name (``"W"``, ``"N"``, ``"i_t"``)
or arithmetic over names and ints with ``+``, ``-``, ``*``, ``//`` and
parentheses (``"W*n"``, ``"(n + 1) // 2"``, ``"4 * i_t + i_s"``). There are
no string literals; in a BRM a fixed string such as an op name is a design
param with a single allowed value.

Expressions are parsed with ``ast`` and evaluated by walking the tree;
nothing is passed to ``eval``. A bare name may stand for a value of any type;
the arithmetic works on integers only: Python ints, NumPy integer scalars and
NumPy integer arrays. Arrays let the reference executor evaluate a subset for
a whole map nest at once (``i_t`` and ``i_s`` given as broadcasting index
arrays).

What a string means is only fixed by what binds its names: a BRM's params,
or a SNAX-DFG's symbols and the variables of the maps around it. This module
does not know either; the caller passes the names.

Operations
----------
``check`` / ``names`` / ``is_name`` / ``constant`` / ``evaluate``
    as used by the BRM.
``canonical``
    the stored form: a value without names is an int, any other string is
    written as ``ast.unparse`` writes it (``"4 * i_t + i_s"``), so a file a
    transform rewrites and one written by hand diff cleanly.
``substitute``
    replace names by values, canonical result (``split_map``: ``i`` becomes
    ``4 * i_t + i_s``).
``linear``
    split a value that is affine in some variables into its constant part
    and one int coefficient per variable, without evaluating anything
    (``4 * i_t + i_s + N`` in ``i_t``, ``i_s`` is ``N``, ``{i_t: 4, i_s: 1}``).
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

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
    """The names ``v`` refers to (none for an int)."""
    if type(v) is int:
        return set()
    return {n.id for n in ast.walk(_parse(v)) if isinstance(n, ast.Name)}


def is_name(v: Value) -> bool:
    """``v`` is a bare name."""
    return isinstance(v, str) and isinstance(_parse(v), ast.Name)


def constant(v: Value) -> int | None:
    """The value of ``v`` if it refers to no name, else None."""
    return evaluate(v, {}) if not names(v) else None


def _is_int(x: Any) -> bool:
    if type(x) is int or isinstance(x, np.integer):
        return True
    return isinstance(x, np.ndarray) and x.dtype.kind in "iu"


def evaluate(v: Value, env: Mapping[str, Any]) -> Any:
    """``v`` with its names taken from ``env`` (ints or NumPy integer arrays)."""
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

    def _int(x: Any) -> Any:
        if not _is_int(x):
            raise ExprError(f"{v!r}: arithmetic on non-int {x!r}")
        return x

    return walk(_parse(v))


# =============================================================================
# Stored form, substitution, affine parts (D77)
# =============================================================================


def canonical(v: Value) -> Value:
    """The stored form of ``v``: an int if it has no names, else ``ast.unparse``'s text."""
    if type(v) is int:
        return v
    tree = _parse(v)
    if not any(isinstance(n, ast.Name) for n in ast.walk(tree)):
        return evaluate(v, {})
    return ast.unparse(tree)


def substitute(v: Value, values: Mapping[str, Value]) -> Value:
    """``v`` with every name in ``values`` replaced by its value, in canonical form."""
    if type(v) is int:
        return v
    trees = {k: ast.Constant(x) if type(x) is int else _parse(x) for k, x in values.items()}

    class Swap(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            return copy.deepcopy(trees[node.id]) if node.id in trees else node

    return canonical(ast.unparse(Swap().visit(_parse(v))))


def linear(v: Value, variables: Iterable[str]) -> tuple[Value, dict[str, int]]:
    """``v`` as ``const + sum(coeff[x] * x)`` over ``variables``.

    ``const`` is a canonical value that uses none of the variables; every
    coefficient is an int, one per variable in the given order (0 if it does
    not occur). Raises ExprError if ``v`` is not affine in the variables: a
    product of two variables, a variable times a name that is not one of them,
    or a variable under ``//``.
    """
    order = list(variables)
    var = set(order)
    Part = tuple[ast.expr | None, dict[str, int]]

    def join(a: ast.expr | None, b: ast.expr | None, op: ast.operator) -> ast.expr | None:
        if b is None:
            return a
        if a is None:
            return b if isinstance(op, ast.Add) else ast.UnaryOp(ast.USub(), b)
        return ast.BinOp(a, op, b)

    def zero(c: ast.expr | None) -> ast.expr:
        return ast.Constant(0) if c is None else c

    def has_vars(p: Part) -> bool:
        return any(n != 0 for n in p[1].values())

    def scale(p: Part, k: int) -> Part:
        c, co = p
        c = None if c is None else ast.BinOp(c, ast.Mult(), ast.Constant(k))
        return c, {x: k * n for x, n in co.items()}

    def factor(p: Part) -> int:
        """The int value of a part without variables."""
        text = ast.unparse(zero(p[0]))
        if names(text):
            raise ExprError(f"{v!r}: not affine, a variable is multiplied by {text!r}")
        return evaluate(text, {})

    def walk(node: ast.expr) -> Part:
        if isinstance(node, ast.Constant):
            return node, {}
        if isinstance(node, ast.Name):
            return (None, {node.id: 1}) if node.id in var else (node, {})
        if isinstance(node, ast.UnaryOp):
            return scale(walk(node.operand), -1)
        left, right = walk(node.left), walk(node.right)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            sign = 1 if isinstance(node.op, ast.Add) else -1
            co = dict(left[1])
            for x, n in right[1].items():
                co[x] = co.get(x, 0) + sign * n
            return join(left[0], right[0], node.op), co
        if isinstance(node.op, ast.Mult):
            if has_vars(left) and has_vars(right):
                raise ExprError(f"{v!r}: not affine, a product of variables")
            if has_vars(left) or has_vars(right):
                var_side, k_side = (left, right) if has_vars(left) else (right, left)
                return scale(var_side, factor(k_side))
            return ast.BinOp(zero(left[0]), node.op, zero(right[0])), {}
        if has_vars(left) or has_vars(right):  # FloorDiv
            raise ExprError(f"{v!r}: not affine, a variable under //")
        return ast.BinOp(zero(left[0]), node.op, zero(right[0])), {}

    if type(v) is int:
        return v, dict.fromkeys(order, 0)
    const, co = walk(_parse(v))
    coeffs = {x: co.get(x, 0) for x in order}
    return (0 if const is None else canonical(ast.unparse(_fold(const)))), coeffs


def _fold(node: ast.expr) -> ast.expr:
    """Fold the parts of ``linear``'s constant that have no names (``1 * 4`` -> ``4``)."""
    if isinstance(node, ast.UnaryOp):
        inner = _fold(node.operand)
        if isinstance(inner, ast.Constant):
            return ast.Constant(-inner.value)
        return ast.UnaryOp(node.op, inner)
    if not isinstance(node, ast.BinOp):
        return node
    left, right = _fold(node.left), _fold(node.right)
    lc = left.value if isinstance(left, ast.Constant) else None
    rc = right.value if isinstance(right, ast.Constant) else None
    if lc is not None and rc is not None:
        return ast.Constant(_BINOPS[type(node.op)](lc, rc))
    if isinstance(node.op, ast.Mult) and 0 in (lc, rc):
        return ast.Constant(0)
    if isinstance(node.op, ast.Mult) and 1 in (lc, rc):
        return right if lc == 1 else left
    if isinstance(node.op, (ast.Add, ast.Sub)) and rc == 0:
        return left
    if isinstance(node.op, ast.Add) and lc == 0:
        return right
    return ast.BinOp(left, node.op, right)
