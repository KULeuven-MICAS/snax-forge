"""The reduce kind folds at its output's dtype (D143): each input lane cast to the output's
dtype before the tree and the accumulator, as np.add.reduce(a, dtype=out) and SnaxUnit."""

from __future__ import annotations

import numpy as np

from snax_forge.brm.function import reduce_kind


def fold(lanes, in_dtype: str, out_dtype: str, op=np.add) -> list[int]:
    """One beat of ``lanes`` (of ``in_dtype``) folded over its lanes into ``out_dtype``."""
    kind = reduce_kind(lanes=len(lanes), lanes_out=1, op=op,
                       dtypes={"a": in_dtype, "out": out_dtype})  # fmt: skip
    out = kind.fn(0, {"a": np.array(lanes, dtype=in_dtype)}, {}, {"T": 1})["out"]
    return np.asarray(out).astype(out_dtype).tolist()


def test_int32_lanes_into_int64_do_not_wrap():
    """PACK0 notes, topic 6: (2^31 - 1, 1, 5, -2)."""
    assert fold([2**31 - 1, 1, 5, -2], "int32", "int64") == [2147483651]
    assert fold([2**31 - 1, 1, 5, -2], "int32", "int32") == [-2147483645]  # wraps at int32


def test_int8_lanes_into_int64():
    assert fold([100, 100, 100, 100], "int8", "int64") == [400]


def test_a_max_fold_into_int16_reads_the_input_signedness():
    """The bytes 0x90 and 0x10: -112 and 16 as int8, 144 and 16 as uint8."""
    assert fold([-112, 16], "int8", "int16", np.maximum) == [16]
    assert fold([0x90, 0x10], "uint8", "int16", np.maximum) == [144]


def test_the_accumulator_runs_at_the_output_dtype_too():
    """Two beats of int8 100s into int64, T = 2: 800, the accumulator never at int8."""
    kind = reduce_kind(lanes=4, lanes_out=1, dtypes={"a": "int8", "out": "int64"})
    state: dict = {}
    beat = {"a": np.array([100] * 4, dtype="int8")}
    assert kind.fn(0, beat, state, {"T": 2}) == {}
    assert kind.fn(1, beat, state, {"T": 2})["out"].tolist() == [800]
