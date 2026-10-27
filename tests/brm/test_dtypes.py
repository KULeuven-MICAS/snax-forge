"""A BRM port's dtypes as a rule (D143): each refusal, and the accepted outputs."""

from __future__ import annotations

import pytest

from snax_forge.brm import INT_DTYPES, Brm, check_dtypes, load_brm, output_dtypes

from .helpers import adder

VMUL = load_brm("elementwise_mul")
VSUM = load_brm("accumulate")


def test_every_port_takes_every_integer_dtype_by_default():
    assert all(p.dtypes == list(INT_DTYPES) for p in VMUL.interface.ports)


@pytest.mark.parametrize(
    "dtypes",
    [
        {"a": "int64", "b": "int64", "out": "int64"},
        {"a": "int32", "b": "int32", "out": "int32"},
        {"a": "int32", "b": "int32", "out": "int64"},
        {"a": "int8", "b": "uint8", "out": "int16"},
        {"a": "uint64", "b": "uint64", "out": "uint64"},
    ],
)
def test_accepted(dtypes):
    assert check_dtypes(VMUL, dtypes) == []


REFUSED = [
    (
        "not in the port's dtypes",
        {"a": "float64", "b": "int64", "out": "int64"},
        [f"port a is float64; elementwise_mul takes {list(INT_DTYPES)} there"],
    ),
    (
        "narrower output",
        {"a": "int64", "b": "int32", "out": "int32"},
        [("output out is int32, narrower than input a (int64): an output is at least as wide as "
          "every input (D143)")],
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


def test_a_port_that_narrows_its_dtypes():
    d = adder()
    d["interface"]["ports"][2]["dtypes"] = ["int64"]
    b = Brm.from_dict(d)
    assert check_dtypes(b, {"a": "int32", "b": "int32", "out": "int32"}) == [
        "port out is int32; test_add takes ['int64'] there"
    ]
    assert output_dtypes(b, {"a": "int32", "b": "int32"}) == {"out": ["int64"]}


def test_accepted_outputs():
    assert output_dtypes(VMUL, {"a": "int32", "b": "int32"}) == {
        "out": ["int32", "uint32", "int64", "uint64"]
    }
    assert output_dtypes(VSUM, {"a": "int8"}) == {"out": list(INT_DTYPES)}
    assert output_dtypes(VMUL, {"a": "int64", "b": "uint64"}) == {"out": []}
