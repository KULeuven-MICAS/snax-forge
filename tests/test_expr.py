"""Value expressions: ints, names and small integer arithmetic, shared by BRM and DFG."""

import pytest

from snax_forge import expr
from snax_forge.expr import ExprError


def test_check_rejects_non_values():
    for v in (1.0, True, None, ["W"]):
        with pytest.raises(ExprError):
            expr.check(v, "field")


def test_evaluate_errors():
    with pytest.raises(ExprError, match="no value for 'W'"):
        expr.evaluate("W", {})
    with pytest.raises(ExprError, match="non-int"):
        expr.evaluate("op * 2", {"op": "add"})


# =============================================================================
# Shared with the SNAX-DFG
# =============================================================================


@pytest.mark.parametrize(
    ("v", "variables", "const", "coeffs"),
    [
        ("4*i_t + i_s", ["i_t", "i_s"], 0, {"i_t": 4, "i_s": 1}),
        ("i + 2", ["i"], 2, {"i": 1}),
        ("-i + N - 2", ["i"], "N - 2", {"i": -1}),
        ("4*(i_t + 1) + i_s", ["i_t", "i_s"], 4, {"i_t": 4, "i_s": 1}),
        ("64*k + 4*i_t", ["i_t", "i_s"], "64 * k", {"i_t": 4, "i_s": 0}),
        ("N", ["i"], "N", {"i": 0}),
        (7, ["i"], 7, {"i": 0}),
        ("(i - i) * N", ["i"], 0, {"i": 0}),
    ],
)
def test_linear(v, variables, const, coeffs):
    assert expr.linear(v, variables) == (const, coeffs)


@pytest.mark.parametrize(
    ("v", "message"),
    [
        ("i * j", "product of variables"),
        ("N * i", "multiplied by 'N'"),
        ("i // 2", "under //"),
        ("max(i, 0)", "max and min are not linear"),
        ("min(N, 4) + i", "max and min are not linear"),
    ],
)
def test_linear_rejects(v, message):
    with pytest.raises(ExprError, match=message):
        expr.linear(v, ["i", "j"])


# =============================================================================
# max and min
# =============================================================================


@pytest.mark.parametrize(
    ("v", "message"),
    [
        ("max(a)", "max takes exactly two arguments"),
        ("min(a, b, c)", "min takes exactly two arguments"),
        ("max(a=1, b=2)", "max takes exactly two arguments"),
        (
            "abs(a)",
            "call 'abs\\(a\\)' is not allowed \\(only max, min, width and the integer casts\\)",
        ),
        ("dace.int32(0)", "is not allowed"),
        ("max(a, 1.5)", "Constant is not allowed"),
    ],
)
def test_max_min_rejected(v, message):
    with pytest.raises(ExprError, match=message):
        expr.check(v, "field")


# =============================================================================
# width
# =============================================================================


@pytest.mark.parametrize(
    ("v", "env", "message"),
    [
        ("width(t)", {"t": 3}, "width of 3, which is not a dtype's name"),
        ("width(t)", {"t": "int128"}, "which is not a dtype's name"),
        ("width(t)", {}, "no value for 't'"),
    ],
)
def test_width_of_a_non_dtype_is_refused(v, env, message):
    with pytest.raises(ExprError, match=message):
        expr.evaluate(v, env)


@pytest.mark.parametrize(
    ("v", "message"),
    [
        ("width(a, b)", "width takes exactly one argument, a name"),
        ("width(8)", "width takes exactly one argument, a name"),
        ("width(t=1)", "width takes exactly one argument, a name"),
    ],
)
def test_width_rejected(v, message):
    with pytest.raises(ExprError, match=message):
        expr.check(v, "field")


# =============================================================================
# unify: code against a pattern with slots
# =============================================================================


@pytest.mark.parametrize(
    ("pattern", "code"),
    [
        ("out = a - k", "out = 3 - a"),  # - does not commute
        ("out = k1 * a + b", "out = a + alpha * b"),  # alpha on b, not a
        ("out = max(a, k)", "out = max(a, b)"),  # a slot takes no name
        ("out = max(a, k)", "out = min(a, 0)"),
        ("out = a * k + k", "out = a * 5 + 6"),  # a slot used twice, two values
        ("out = a + k", "out = a + 1 // 0"),
        ("out = a + k", "x = a + 1"),
        ("out = a + k", "out = a - 1"),
    ],
)
def test_unify_refuses(pattern, code):
    assert expr.unify(pattern, code, ["k", "k1"], ["alpha"]) is None


def test_unify_tries_every_way_before_refusing():
    """A first way that binds a slot and fails later is backtracked: ``k`` is first tried as 2
    against ``a + 2``'s swap, then as the one value both sides agree on."""
    assert expr.unify("out = (a + k) * (b + k)", "out = (b + 2) * (2 + a)", ["k"]) == {"k": 2}
    assert expr.unify("out = (a + k) * (b + k)", "out = (b + 2) * (3 + a)", ["k"]) is None


# =============================================================================
# Value casts (tasklet code, from a kernel's astype)
# =============================================================================


def test_strip_casts():
    assert expr.strip_casts("out = int32(in1) * int32(in2)", "t") == (
        "out = in1 * in2", {"in1": "int32", "in2": "int32"})  # fmt: skip
    assert expr.strip_casts("out = in1 + 3", "t") == ("out = in1 + 3", {})
    with pytest.raises(ExprError, match="the cast 'int32\\(in1 \\+ in2\\)' is of an expression"):
        expr.strip_casts("out = int32(in1 + in2)", "t")
    with pytest.raises(ExprError, match="in1 is cast to int16 and to int32"):
        expr.strip_casts("out = int16(in1) + int32(in1)", "t")
    with pytest.raises(ExprError, match="the cast int32 takes exactly one argument"):
        expr.check("int32(a, b)", "t")
