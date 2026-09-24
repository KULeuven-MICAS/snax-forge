"""BRM value expressions (D68): ints, param names and small integer arithmetic."""

import pytest

from snax_forge.brm import expr
from snax_forge.brm.expr import ExprError


@pytest.mark.parametrize(
    ("v", "env", "want"),
    [
        (4, {}, 4),
        ("W", {"W": 4}, 4),
        ("op", {"op": "add"}, "add"),
        ("W*n", {"W": 4, "n": 16}, 64),
        ("(n + 1) // 2", {"n": 7}, 4),
        ("-W + 10", {"W": 4}, 6),
    ],
)
def test_evaluate(v, env, want):
    assert expr.evaluate(v, env) == want


def test_names_constant_is_name():
    assert expr.names("W*n + W") == {"W", "n"}
    assert expr.names(3) == set()
    assert expr.constant("2*3") == 6
    assert expr.constant("W") is None
    assert expr.is_name("T") and not expr.is_name("T*2") and not expr.is_name(2)


@pytest.mark.parametrize(
    "v", ["__import__('os')", "W.x", "'add'", "W / 2", "W ** 2", "1.5", "[W]", "W if n else 1"]
)
def test_rejected(v):
    with pytest.raises(ExprError):
        expr.check(v, "field")


def test_check_rejects_non_values():
    for v in (1.0, True, None, ["W"]):
        with pytest.raises(ExprError):
            expr.check(v, "field")


def test_evaluate_errors():
    with pytest.raises(ExprError, match="no value for 'W'"):
        expr.evaluate("W", {})
    with pytest.raises(ExprError, match="non-int"):
        expr.evaluate("op * 2", {"op": "add"})
