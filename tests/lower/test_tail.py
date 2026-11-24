"""A tail's padding in memory and in the task list: the layouts over the padded shape,
the DMA's whole beats, the firing count and the rate; main.c's self-check over valid
elements."""

from __future__ import annotations

from snax_forge.gen.data import Array
from snax_forge.lower.c import _compare


def test_the_self_check_compares_a_padded_matrix_row_by_row():
    a = Array("C", "C", "output", "int64_t", 21, (0,) * 24, 64, (3, 7), (3, 8))
    assert _compare(a) == [
        "for (uint32_t i0 = 0; i0 < 3; i0++) {",
        "    for (uint32_t i1 = 0; i1 < 7; i1++) {",
        "        uint32_t i = i0 * 8 + i1;",
        "        if (((volatile int64_t *)C)[i] != C_golden[i]) {",
        "            data_errors++;",
        "        }",
        "    }",
        "}",
    ]
    vector = Array("C", "C", "output", "int64_t", 7, (0,) * 8, 64, (7,), (8,))
    assert _compare(vector)[0] == "for (uint32_t i = 0; i < 7; i++) {"
