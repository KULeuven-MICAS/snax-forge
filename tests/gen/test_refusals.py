"""The gen package's refusals, one table: each case builds its input and makes the call that
is refused, with the error and the message it must give."""

from __future__ import annotations

import pytest

from snax_forge.gen import GenError
from snax_forge.gen import data as data_mod


def _refused_a_float_dtype():
    data_mod.ctype("float64", "container A")


CASES = [
    pytest.param(_refused_a_float_dtype, GenError, '\\[gen.dtype\\] container A', id='refused_a_float_dtype'),
]  # fmt: skip


@pytest.mark.parametrize(("case", "error", "message"), CASES)
def test_refusal(case, error, message):
    with pytest.raises(error, match=message):
        case()
