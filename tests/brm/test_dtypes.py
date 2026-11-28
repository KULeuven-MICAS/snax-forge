"""A BRM port's dtypes as a rule: each refusal, the accepted outputs, and a SIMD output
narrower than its input when the node's code bounds its values."""

from __future__ import annotations

import pytest

from snax_forge.brm import INT_DTYPES, check_dtypes, load_brm, output_dtypes

VMUL = load_brm("simd")
VSUM = load_brm("accumulate")


def test_every_port_takes_every_integer_dtype_by_default():
    assert all(p.dtypes == list(INT_DTYPES) for p in VMUL.interface.ports)


REFUSED = [
    (
        "not in the port's dtypes",
        {"a": "float64", "b": "int64", "out": "int64"},
        [f"port a is float64; simd takes {list(INT_DTYPES)} there"],
    ),
    (
        "narrower output",
        {"a": "int64", "b": "int32", "out": "int32"},
        [("output out is int32, narrower than input a (int64): an output is at least as wide as "
          "every input")],
    ),
    (
        "signed with uint64",
        {"a": "int8", "b": "uint64", "out": "uint64"},
        [("inputs a (int8) and b (uint64): a signed input with a uint64 one has no integer result "
          "type (NumPy gives float64)")],
    ),
]  # fmt: skip


@pytest.mark.parametrize(("case", "dtypes", "why"), REFUSED, ids=[c[0] for c in REFUSED])
def test_refused(case, dtypes, why):
    assert check_dtypes(VMUL, dtypes) == why


def test_accepted_outputs():
    assert output_dtypes(VMUL, {"a": "int32", "b": "int32"}) == {
        "out": ["int32", "uint32", "int64", "uint64"]
    }
    assert output_dtypes(VSUM, {"a": "int8"}) == {"out": list(INT_DTYPES)}
    assert output_dtypes(VMUL, {"a": "int64", "b": "uint64"}) == {"out": []}


# a SIMD output narrower than an input passes when every value it can take fits it
BOUNDED = [
    ("requant's clamp", "out = min(max(a // 1024, 0), 127)", "int8", []),
    ("a select by clamps", "out = 255 * min(max(a - k + 1, 0), 1)", "uint8", []),
    (
        "a shift alone",
        "out = a // 1024",
        "int8",
        [("output out is int8, narrower than input a (int32), and its values span [-2097152, "
          "2097151]: a narrower output needs every value to fit it (bound them with min and "
          "max)")],
    ),
    (
        "a scalar left unbounded",
        "out = min(a, k)",
        "int8",
        [("output out is int8, narrower than input a (int32): an output is at least as wide as "
         "every input")],
    ),
]  # fmt: skip


@pytest.mark.parametrize(("case", "code", "out", "why"), BOUNDED, ids=[c[0] for c in BOUNDED])
def test_a_simd_output_narrower_than_its_input_when_its_values_fit(case, code, out, why):
    assert check_dtypes(VMUL, {"a": "int32", "out": out}, code) == why


def test_a_fold_output_is_never_narrower_than_its_input():
    """A fold's code is one element's (``out = a``): its sum is not bounded by it."""
    why = check_dtypes(VSUM, {"a": "int32", "out": "int8"}, "out = min(max(a, 0), 1)")
    rule = "an output is at least as wide as every input"
    assert why == [f"output out is int8, narrower than input a (int32): {rule}"]
