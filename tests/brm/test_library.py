"""The BRM library: simd, accumulate and mac.

Each BRM resolves to the function the reference executor runs (its kind in
brm/function.py, its params resolved), and its ``code`` computes what that
kind computes, lane by lane; accumulate's Chisel implementation declares the
Accumulator's drain, 0.
"""

from __future__ import annotations

import json

import pytest

from snax_forge.brm import (
    Brm,
    BrmError,
    load_brm,
)

from .helpers import IMPL

# =============================================================================
# The library
# =============================================================================


# =============================================================================
# simd against the simd kind: a code table by op and streams, the slot k
# =============================================================================

TWO = {"add": "a + b", "sub": "a - b", "mul": "a * b", "min": "min(a, b)", "max": "max(a, b)"}
ONE = {"add": "a + k", "sub": "a - k", "mul": "a * k", "min": "min(a, k)", "max": "max(a, k)"}


def test_one_stream_without_k_or_two_with_one_is_refused():
    brm = load_brm("simd")
    with pytest.raises(BrmError, match="one stream takes the constant k"):
        brm.resolve(IMPL, {"streams": 1})
    with pytest.raises(BrmError, match="k = 3 at two streams"):
        brm.resolve(IMPL, {"k": 3})


@pytest.mark.parametrize(
    ("params", "read", "message"),
    [
        ({"op": "axpby"}, (), "axpby takes k1, got None"),
        ({"op": "axpby", "k2": 2}, (), "axpby takes k1, got None"),
        ({"op": "max", "streams": 1}, (), "one stream takes the constant k, got None"),
        ({"op": "axpby", "k1": 3}, ("k1",), "k1 = 3 is set, and read from a container too"),
        ({"op": "add"}, ("alpha",), "'alpha' is read from a container, but it is no slot"),
        ({"op": "add"}, ("k1",), "k1 is axpby and axpbypcz's, not add's"),
        ({"op": "axpby", "streams": 1, "k": 2}, (), "axpby at 1 stream\\(s\\); it takes 2"),
        (
            {"op": "requant", "streams": 1, "d": 1000, "lo": 0, "hi": 127},
            (),
            "d = 1000 is not a power of two",
        ),
        ({"op": "axpbypcz", "streams": 3, "d": 4}, ("d",), "d = 4 is set, and read from"),
        (
            {"op": "axpbypcz", "streams": 3, "k1": 1},
            ("d",),
            "d read from a container; axpbypcz bakes it",
        ),
        ({"op": "threshold", "streams": 1, "hi": 255, "lo": 0}, ("k",), "lo not threshold's"),
    ],
)
def test_an_instance_no_entry_fits_is_refused(params, read, message):
    """Refused at resolve, not at a firing: a slot the code needs with no value, one set and
    read, and a read slot the code does not name."""
    with pytest.raises(BrmError, match=message):
        load_brm("simd").resolve(IMPL, params, read)


def test_the_code_check_refuses_what_no_entry_fits():
    """resolve's own check (instance.py ``_check_code``), behind the adapter's: a BRM without
    a binding, whose datapath adapter does not run, gets it alone."""
    d = json.loads(load_brm("simd").to_json())
    for impl in d["implementations"].values():
        impl["binding"] = None
    brm = Brm.from_dict(d)
    with pytest.raises(BrmError, match=r"the code is 'out = max\(a, k\)' with k; the slots set "
                       r"here \(baked, or read from a container\): none"):  # fmt: skip
        brm.resolve(IMPL, {"op": "max", "streams": 1})
    with pytest.raises(BrmError, match="'out = k1 \\* a \\+ k2 \\* b' with k1 and k2 or "
                       "'out = k1 \\* a \\+ b' with k1; the slots set here .*: k2"):  # fmt: skip
        brm.resolve(IMPL, {"op": "axpby", "k2": 2})


# =============================================================================
# accumulate against the reduce kind
# =============================================================================
