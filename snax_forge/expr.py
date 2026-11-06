"""Values: an int, or a small integer expression over names (D71).

Shared by SNAX-BRM and SNAX-DFG. A value field holds either a plain int or a
string. Every string is an expression: a name (``"W"``, ``"N"``, ``"i_t"``)
or arithmetic over names and ints with ``+``, ``-``, ``*``, ``//`` and
parentheses (``"W*n"``, ``"(n + 1) // 2"``, ``"4 * i_t + i_s"``), and
``max(a, b)`` and ``min(a, b)``, exactly two arguments, each any expression
(``"max(in1, 0)"``, relu's tasklet), and ``width(t)``, one argument, a name bound
to a dtype as NumPy names it, its bits (``"lanes * width(outType)"``, the datapath
catalogue's ``flops``); no other call. There are
no string literals; in a BRM a fixed string such as an op name is a design
param with a single allowed value.

Expressions are parsed with ``ast`` and evaluated by walking the tree;
nothing is passed to ``eval``. A bare name may stand for a value of any type;
the arithmetic works on integers only: Python ints, NumPy integer scalars and
NumPy integer arrays (``max`` and ``min`` are ``np.maximum`` and
``np.minimum``, elementwise on arrays). Arrays let the reference executor evaluate a subset for
a whole map nest at once (``i_t`` and ``i_s`` given as broadcasting index
arrays).

What a string means is only fixed by what binds its names: a BRM's params,
or a SNAX-DFG's symbols and the variables of the maps around it. This module
does not know either; the caller passes the names.

Operations
----------
``check`` / ``names`` / ``is_name`` / ``constant`` / ``evaluate``
    on one expression; ``outside``, ``check`` then the names a caller does not allow.
``canonical``
    the stored form: a value without names is an int, any other string is
    written as ``ast.unparse`` writes it (``"4 * i_t + i_s"``), so a file a
    transform rewrites and one written by hand diff cleanly.
``substitute``
    replace names by values, canonical result (``split_map``: ``i`` becomes
    ``4 * i_t + i_s``).
``unify``
    match code against a pattern code whose slot names each stand for an int
    constant (a BRM's code entry ``out = max(a, k)`` against ``out = max(a, 0)``:
    ``{k: 0}``).
``linear``
    split a value that is affine in some variables into its constant part
    and one int coefficient per variable, without evaluating anything
    (``4 * i_t + i_s + N`` in ``i_t``, ``i_s`` is ``N``, ``{i_t: 4, i_s: 1}``);
    ``max`` and ``min`` are not affine and are refused.
"""

from __future__ import annotations

import ast
import copy
import functools
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
# the grammar's calls, two arguments each, elementwise on index arrays
CALLS = {"max": np.maximum, "min": np.minimum}
# and its one-argument call on a dtype's name: its bits
DTYPE_CALLS = {"width"}


class ExprError(ValueError):
    """A value that is not an int or a valid expression."""


def _parse(expr: str) -> ast.expr:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ExprError(f"cannot parse {expr!r}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            _check_call(node, expr)
            continue
        ok = isinstance(node, (ast.Expression, ast.Name, ast.Load, ast.BinOp, ast.UnaryOp))
        ok = ok or type(node) in _BINOPS or isinstance(node, ast.USub)
        ok = ok or (isinstance(node, ast.Constant) and type(node.value) is int)
        if not ok:
            raise ExprError(f"{expr!r}: {type(node).__name__} is not allowed")
    return tree.body


@functools.lru_cache(maxsize=8192)
def _parsed(expr: str) -> ast.expr:
    """``_parse``'s tree, one per string, shared: only for the callers that never change it
    (``check``, ``names``, ``is_name``, ``evaluate``, ``canonical``); a caller that rewrites a
    tree (``substitute``) parses its own."""
    return _parse(expr)


def _check_call(node: ast.Call, expr: str) -> None:
    """``max(a, b)``, ``min(a, b)`` or ``width(name)``; any other call is an error."""
    fn = node.func.id if isinstance(node.func, ast.Name) else None
    if fn in DTYPE_CALLS:
        if len(node.args) != 1 or node.keywords or not isinstance(node.args[0], ast.Name):
            raise ExprError(f"{expr!r}: {fn} takes exactly one argument, a name")
        return
    if fn not in CALLS:
        raise ExprError(f"{expr!r}: call {ast.unparse(node)!r} is not allowed (only max, min "
                        "and width)")  # fmt: skip
    if len(node.args) != 2 or node.keywords:
        raise ExprError(f"{expr!r}: {fn} takes exactly two arguments")


def _names(tree: ast.AST) -> set[str]:
    """The names in ``tree``, not counting a call's function (``max``)."""
    funcs = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in funcs}


def check(v: Any, what: str) -> None:
    """``v`` is an int or a string that parses as an expression."""
    if type(v) is int:
        return
    if not isinstance(v, str):
        raise ExprError(f"{what}: must be an int or an expression string, got {v!r}")
    try:
        _parsed(v)
    except ExprError as e:
        raise ExprError(f"{what}: {e}") from None


def names(v: Value) -> set[str]:
    """The names ``v`` refers to (none for an int)."""
    if type(v) is int:
        return set()
    return _names(_parsed(v))


def ordered_names(v: Value) -> list[str]:
    """The names ``v`` refers to, each once, in the order they first appear in its text."""
    if type(v) is int:
        return []
    out: list[str] = []

    def walk(node: ast.expr) -> None:
        if isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.BinOp):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, ast.UnaryOp):
            walk(node.operand)
        elif isinstance(node, ast.Call):
            for a in node.args:
                walk(a)

    walk(_parsed(v))
    return list(dict.fromkeys(out))


def occurrences(v: Value, name: str) -> int:
    """How many times ``v`` reads ``name``."""
    if type(v) is int:
        return 0
    funcs = {id(n.func) for n in ast.walk(_parsed(v)) if isinstance(n, ast.Call)}
    return sum(1 for n in ast.walk(_parsed(v))
               if isinstance(n, ast.Name) and n.id == name and id(n) not in funcs)  # fmt: skip


def outside(v: Any, allowed: Iterable[str], what: str) -> list[str]:
    """``check`` of ``v``, then the names it uses that are not ``allowed``, sorted."""
    check(v, what)
    return sorted(names(v) - set(allowed))


def is_name(v: Value) -> bool:
    """``v`` is a bare name."""
    return isinstance(v, str) and isinstance(_parsed(v), ast.Name)


def constant(v: Value) -> int | None:
    """The value of ``v`` if it refers to no name, else None."""
    return evaluate(v, {}) if not names(v) else None


def _is_int(x: Any) -> bool:
    if type(x) is int or isinstance(x, np.integer):
        return True
    return isinstance(x, np.ndarray) and x.dtype.kind in "iu"


def _width(x: Any, v: Value) -> int:
    """``width(t)``: the bits of dtype ``t``, a NumPy name (``"int64"``: 64)."""
    if isinstance(x, str):
        try:
            return np.dtype(x).itemsize * 8
        except TypeError:
            pass
    raise ExprError(f"{v!r}: width of {x!r}, which is not a dtype's name")


def evaluate(v: Value, env: Mapping[str, Any]) -> Any:
    """``v`` with its names taken from ``env`` (ints or NumPy integer arrays; a dtype's name for
    ``width``'s argument)."""
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
        if isinstance(node, ast.Call) and node.func.id in DTYPE_CALLS:
            return _width(walk(node.args[0]), v)
        if isinstance(node, ast.Call):
            a, b = (_int(walk(x)) for x in node.args)
            r = CALLS[node.func.id](a, b)
            return int(r) if type(a) is int and type(b) is int else r
        return _BINOPS[type(node.op)](_int(walk(node.left)), _int(walk(node.right)))

    def _int(x: Any) -> Any:
        if not _is_int(x):
            raise ExprError(f"{v!r}: arithmetic on non-int {x!r}")
        return x

    return walk(_parsed(v))


# =============================================================================
# Stored form, substitution, affine parts
# =============================================================================


def canonical(v: Value) -> Value:
    """The stored form of ``v``: an int if it has no names, else ``ast.unparse``'s text."""
    if type(v) is int:
        return v
    tree = _parsed(v)
    if not _names(tree):
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

        def visit_Call(self, node: ast.Call) -> ast.expr:  # the arguments, never ``max``
            node.args = [self.visit(a) for a in node.args]
            return node

    return canonical(ast.unparse(Swap().visit(_parse(v))))


def linear(v: Value, variables: Iterable[str]) -> tuple[Value, dict[str, int]]:
    """``v`` as ``const + sum(coeff[x] * x)`` over ``variables``.

    ``const`` is a canonical value that uses none of the variables; every
    coefficient is an int, one per variable in the given order (0 if it does
    not occur). Raises ExprError if ``v`` is not affine in the variables: a
    product of two variables, a variable times a name that is not one of them,
    or a variable under ``//``; and on any ``max`` or ``min``.
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
    tree = _parse(v)
    if any(isinstance(n, ast.Call) for n in ast.walk(tree)):
        raise ExprError(f"{v!r}: not affine, max and min are not linear")
    const, co = walk(tree)
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


# =============================================================================
# Code: one assignment per line
# =============================================================================


def statements(code: Any, what: str) -> list[tuple[str, Value]]:
    """``code`` as ``(target, expression)`` pairs: one ``name = expression`` per line.

    The expressions are in the grammar above and canonical. Used for a
    tasklet's code, an accelerated node's code and a BRM's function code;
    which names a target and an expression may use is the caller's rule.
    """
    if not isinstance(code, str):
        raise ExprError(f"{what}: must be a string, got {code!r}")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ExprError(f"{what}: cannot parse {code!r}") from e
    out = []
    for st in tree.body:
        ok = isinstance(st, ast.Assign) and len(st.targets) == 1
        if not ok or not isinstance(st.targets[0], ast.Name):
            raise ExprError(f"{what}: {ast.unparse(st)!r} is not 'output = expression'")
        rhs = ast.unparse(st.value)
        try:
            _parse(rhs)
        except ExprError as e:
            raise ExprError(f"{what}: {e}") from None
        out.append((st.targets[0].id, canonical(rhs)))
    if not out:
        raise ExprError(f"{what}: no statement")
    return out


def canonical_code(code: Any, what: str) -> str:
    """``code`` in its stored form: one canonical ``target = expression`` per line."""
    return "\n".join(f"{t} = {e}" for t, e in statements(code, what))


def rename_code(code: str, names: Mapping[str, str]) -> str:
    """``code`` (valid, see ``statements``) with targets and names renamed, canonical."""
    lines = statements(code, "code")
    return "\n".join(f"{names.get(t, t)} = {substitute(e, names)}" for t, e in lines)


def check_code(
    code: Any, inputs: Iterable[str], outputs: Iterable[str], what: str
) -> list[tuple[str, Value]]:
    """``statements`` of ``code``, each output assigned once, only inputs read."""
    lines = statements(code, what)
    outs, ins = list(outputs), set(inputs)
    targets = [t for t, _ in lines]
    for t in targets:
        if t not in outs:
            raise ExprError(f"{what}: assigns {t!r}, which is not an output")
    for o in outs:
        if targets.count(o) != 1:
            raise ExprError(f"{what}: output {o!r} must be assigned exactly once")
    for _, e in lines:
        for name in sorted(names(e)):
            if name not in ins:
                raise ExprError(f"{what}: {name!r} is not an input")
    return lines


def unify(pattern: str, code: str, slots: Iterable[str]) -> dict[str, int] | None:
    """The slot values that make ``code`` equal ``pattern``, or None when none do.

    Both are valid code (``statements``), compared statement by statement and tree by tree.
    A name in ``slots`` matches a sub-expression of ``code`` with no names whose value is an
    int (``0``, ``-3``, ``2 * 4``); a slot used twice must match the same value. Everything
    else must be the same: names, int literals, operators and calls. So ``out = max(a, k)``
    takes ``out = max(a, 0)`` with ``k = 0``, and neither ``out = max(0, a)`` nor
    ``out = max(a, b)``.
    """
    names_ = set(slots)
    p_lines, c_lines = statements(pattern, "pattern"), statements(code, "code")
    if len(p_lines) != len(c_lines):
        return None
    found: dict[str, int] = {}

    def value(c: ast.expr) -> int | None:
        if _names(c):
            return None
        try:
            v = evaluate(ast.unparse(c), {})
        except (ExprError, ZeroDivisionError):
            return None
        return v if type(v) is int else None

    def same(p: ast.expr, c: ast.expr) -> bool:
        if isinstance(p, ast.Name) and p.id in names_:
            v = value(c)
            return v is not None and found.setdefault(p.id, v) == v
        if type(p) is not type(c):
            return False
        if isinstance(p, ast.Name):
            return p.id == c.id
        if isinstance(p, ast.Constant):
            return p.value == c.value
        if isinstance(p, ast.UnaryOp):
            return type(p.op) is type(c.op) and same(p.operand, c.operand)
        if isinstance(p, ast.BinOp):
            return type(p.op) is type(c.op) and same(p.left, c.left) and same(p.right, c.right)
        if isinstance(p, ast.Call):
            return (p.func.id == c.func.id and len(p.args) == len(c.args)
                    and all(same(a, b) for a, b in zip(p.args, c.args, strict=True)))  # fmt: skip
        return False

    for (tp, ep), (tc, ec) in zip(p_lines, c_lines, strict=True):
        if tp != tc or not same(_parse(str(ep)), _parse(str(ec))):
            return None
    return found
