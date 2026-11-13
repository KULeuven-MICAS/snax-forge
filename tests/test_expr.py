"""Value expressions: ints, names and small integer arithmetic, shared by BRM and DFG."""

import ast

import numpy as np
import pytest

from snax_forge import expr
from snax_forge.expr import ExprError


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


# =============================================================================
# Shared with the SNAX-DFG
# =============================================================================


def test_evaluate_on_index_arrays():
    """A subset over a whole map nest at once: broadcasting index arrays."""
    i_t, i_s = np.arange(3)[:, None], np.arange(4)[None, :]
    got = expr.evaluate("4 * i_t + i_s", {"i_t": i_t, "i_s": i_s})
    assert got.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    assert expr.evaluate("N // 4", {"N": np.int64(64)}) == 16
    with pytest.raises(ExprError, match="non-int"):
        expr.evaluate("x + 1", {"x": np.ones(2)})


@pytest.mark.parametrize(
    ("v", "want"),
    [(3, 3), ("2*3", 6), ("4*i_t+i_s", "4 * i_t + i_s"), ("N//4", "N // 4"), ("(i)", "i")],
)
def test_canonical(v, want):
    assert expr.canonical(v) == want


def test_substitute():
    assert expr.substitute("2*i + 1", {"i": "4*i_t + i_s"}) == "2 * (4 * i_t + i_s) + 1"
    assert expr.substitute("i + j", {"i": 3, "j": 4}) == 7
    assert expr.substitute("N", {"i": 1}) == "N"
    assert expr.substitute(5, {"i": 1}) == 5


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
    ("v", "env", "want"),
    [
        ("max(a, 0)", {"a": -3}, 0),
        ("min(a, b)", {"a": 4, "b": 9}, 4),
        ("max(2 * a, b + 1)", {"a": 3, "b": 7}, 8),
        ("min(max(a, 0), 5)", {"a": 9}, 5),
    ],
)
def test_max_min_on_ints(v, env, want):
    got = expr.evaluate(v, env)
    assert got == want and type(got) is int


def test_max_min_on_arrays():
    """np.maximum and np.minimum: elementwise, broadcasting like the arithmetic."""
    a = np.array([-3, 5, 0], dtype=np.int64)
    assert expr.evaluate("max(a, 0)", {"a": a}).tolist() == [0, 5, 0]
    assert expr.evaluate("min(a, b)", {"a": a, "b": np.int64(1)}).tolist() == [-3, 1, 0]
    i, j = np.arange(2)[:, None], np.arange(3)[None, :]
    assert expr.evaluate("max(i, j)", {"i": i, "j": j}).tolist() == [[0, 1, 2], [1, 1, 2]]


def test_max_min_kept_in_stored_form():
    assert expr.canonical("max( in1 ,0)") == "max(in1, 0)"
    assert expr.canonical("min(2, 7)") == 2
    assert expr.names("max(a, min(b, 1))") == {"a", "b"}
    assert expr.substitute("max(i, N)", {"i": "4*i_t"}) == "max(4 * i_t, N)"
    assert expr.rename_code("__out = max(__in1, 0)", {"__out": "out", "__in1": "in1"}) == (
        "out = max(in1, 0)"
    )


@pytest.mark.parametrize(
    ("v", "message"),
    [
        ("max(a)", "max takes exactly two arguments"),
        ("min(a, b, c)", "min takes exactly two arguments"),
        ("max(a=1, b=2)", "max takes exactly two arguments"),
        ("abs(a)", "call 'abs\\(a\\)' is not allowed \\(only max, min and width\\)"),
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
    ("v", "env", "want"),
    [
        ("width(t)", {"t": "int64"}, 64),
        ("width(t)", {"t": "uint8"}, 8),
        ("lanes * width(outType) * max(accumulate, regOut)",
         {"lanes": 8, "outType": "int32", "accumulate": 1, "regOut": 0}, 256),
        ("depth * lanes * width(dtype)", {"depth": 3, "lanes": 4, "dtype": "int16"}, 192),
    ],
)  # fmt: skip
def test_width_is_a_dtypes_bits(v, env, want):
    got = expr.evaluate(v, env)
    assert got == want and type(got) is int


def test_width_kept_in_stored_form():
    assert expr.canonical("lanes*width( outType )") == "lanes * width(outType)"
    assert expr.names("depth * width(dtype)") == {"depth", "dtype"}
    assert expr.substitute("width(t) * n", {"n": "W"}) == "width(t) * W"


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


def test_substitute_leaves_the_cached_parse_alone():
    """The read-only callers share one tree per string (expr._parsed); substitute rewrites a tree
    of its own, so the shared one, and every answer read from it, stay as they were."""
    v = "max(a, b) + a * 2"
    before = ast.dump(expr._parsed(v))
    assert expr.substitute(v, {"a": "c - 1", "b": 3}) == "max(c - 1, 3) + (c - 1) * 2"
    assert ast.dump(expr._parsed(v)) == before
    assert expr.canonical(v) == "max(a, b) + a * 2" and expr.names(v) == {"a", "b"}


# =============================================================================
# unify: code against a pattern with slots
# =============================================================================


@pytest.mark.parametrize(
    ("pattern", "code", "slots", "found"),
    [
        ("out = max(a, k)", "out = max(a, 0)", ["k"], {"k": 0}),
        ("out = a + k", "out = a + -3", ["k"], {"k": -3}),
        ("out = a + k", "out = a + 2 * 4", ["k"], {"k": 8}),
        ("out = a * k + k", "out = a * 5 + 5", ["k"], {"k": 5}),
        ("out = a + b", "out = a + b", [], {}),
        ("out = a + b", "out = a + b", ["k"], {}),
    ],
)
def test_unify_finds_the_slots(pattern, code, slots, found):
    assert expr.unify(pattern, code, slots) == found


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


@pytest.mark.parametrize(
    ("pattern", "code", "found"),
    [
        ("out = max(a, k)", "out = max(0, a)", {"k": 0}),
        ("out = a + k", "out = 3 + a", {"k": 3}),
        ("out = min(a, k) * b", "out = b * min(-1, a)", {"k": -1}),  # at any depth
        ("out = k1 * a + b", "out = b + alpha * a", {"k1": "alpha"}),
        ("out = k1 * a + k2 * b", "out = b * beta + alpha * a", {"k1": "alpha", "k2": "beta"}),
        ("out = k1 * a + k2 * b", "out = alpha * a + b * 3", {"k1": "alpha", "k2": 3}),
        ("out = a * k", "out = alpha * a", {"k": "alpha"}),
    ],
)
def test_unify_commutes_and_takes_scalars(pattern, code, found):
    """+, *, max and min commute on two operands; a slot takes a scalar's name as it is."""
    assert expr.unify(pattern, code, ["k", "k1", "k2"], ["alpha", "beta"]) == found


def test_unify_tries_every_way_before_refusing():
    """A first way that binds a slot and fails later is backtracked: ``k`` is first tried as 2
    against ``a + 2``'s swap, then as the one value both sides agree on."""
    assert expr.unify("out = (a + k) * (b + k)", "out = (b + 2) * (2 + a)", ["k"]) == {"k": 2}
    assert expr.unify("out = (a + k) * (b + k)", "out = (b + 2) * (3 + a)", ["k"]) is None


def test_ordered_names_and_occurrences():
    assert expr.ordered_names("max(in2, in1) + in2") == ["in2", "in1"]
    assert expr.occurrences("max(in2, in1) + in2", "in2") == 2
    assert expr.occurrences("max(a, 0)", "max") == 0
