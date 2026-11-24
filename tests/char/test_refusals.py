"""The char package's refusals, one table: each case builds its input and makes the call that
is refused, with the error and the message it must give."""

from __future__ import annotations

import pytest

from snax_forge.char import CharError, load


def _an_unknown_technology_is_refused():
    load("tsmc7")


CASES = [
    pytest.param(_an_unknown_technology_is_refused, CharError, '\\[char.tech\\]', id='an_unknown_technology_is_refused'),
]  # fmt: skip


@pytest.mark.parametrize(("case", "error", "message"), CASES)
def test_refusal(case, error, message):
    with pytest.raises(error, match=message):
        case()
