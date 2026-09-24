"""One dimension of a memlet subset or a map range (D77).

A dimension is written as one JSON value:

    i                  an index: an int or an expression (snax_forge/expr.py)
    4 * i_t + i_s      an index
    0:N                a range, begin and end, end exclusive (as in Python)
    0:N:2              a range with a step

A range's step is a positive int; begin and end are ints or expressions.
The parts are expressions of their own, so a range never needs brackets:
``4 * i_t:4 * i_t + 4`` is the four elements from ``4 * i_t`` on. What the
names mean (symbols, map variables) is checked where the dimension is used,
not here.

The stored form is canonical: every part as ``expr.canonical`` writes it, an
index without names as an int, and a step of 1 left out.
"""

from __future__ import annotations

from typing import Any

from snax_forge import expr
from snax_forge.expr import ExprError, Value


def parse_dim(v: Any, what: str) -> tuple[Value, ...]:
    """``(index,)`` or ``(begin, end, step)``, each part canonical."""
    if type(v) is int:
        return (v,)
    if not isinstance(v, str):
        raise ExprError(f"{what}: must be an int or a string, got {v!r}")
    parts = v.split(":")
    if len(parts) > 3:
        raise ExprError(f"{what}: {v!r} has more than three parts")
    out = []
    for p in parts:
        if not p.strip():
            raise ExprError(f"{what}: {v!r} has an empty part")
        expr.check(p, what)
        out.append(expr.canonical(p))
    if len(out) == 1:
        return (out[0],)
    if len(out) == 2:
        out.append(1)
    if type(out[2]) is not int or out[2] < 1:
        raise ExprError(f"{what}: the step of {v!r} must be a positive int")
    return tuple(out)


def format_dim(parts: tuple[Value, ...]) -> Value:
    """The stored form of a parsed dimension."""
    if len(parts) == 1:
        return parts[0]
    begin, end, step = parts
    return f"{begin}:{end}" + ("" if step == 1 else f":{step}")


def canonical_dim(v: Any, what: str) -> Value:
    """``v`` in its stored form."""
    return format_dim(parse_dim(v, what))


def is_range(v: Any) -> bool:
    """``v`` (a valid dimension) is a range, not an index."""
    return isinstance(v, str) and ":" in v


def dim_names(v: Value) -> set[str]:
    """Every name a dimension uses."""
    return set().union(*(expr.names(p) for p in parse_dim(v, "dimension")))
