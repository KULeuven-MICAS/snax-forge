"""The lane accumulator (rowsum 2): the BRM ``lane_accumulate``, its kind and its binding to
``LaneAccumulate`` (gen/blocks.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from snax_forge.brm import LIBRARY, Brm, BrmError, function_of, load_brm
from snax_forge.gen.blocks import DATAPATHS, GenError

IMPL = "chisel_lane_accumulate"


def test_the_brm_round_trips():
    brm = load_brm("lane_accumulate")
    text = (LIBRARY / "lane_accumulate.json").read_text()
    assert brm.to_json() == text and Brm.from_dict(json.loads(text)) == brm
    assert Brm.from_dict(brm.to_dict()) == brm
    assert brm.pattern.family == "lane_fold" and brm.pattern.attrs == {"fold": "add"}
    assert brm.function.accel == "lane_accumulate" and brm.function.code == "out = a"
    assert [(p.name, p.direction, p.lanes, p.rate) for p in brm.interface.ports] == [
        ("a", "in", "W", 1), ("out", "out", "W", "T")]  # fmt: skip
    assert brm.implementations[IMPL].binding == {
        "datapath": "LaneAccumulate", "args": {"lanes": "W"}, "controller": "counted"}  # fmt: skip


def test_each_lane_folds_its_own_t_beats_cast_to_the_output():
    inst = load_brm("lane_accumulate").resolve(IMPL, {"W": 2})
    assert inst.params == {"W": 2, "op": "add"}
    assert (inst.latency, inst.initiation_interval, inst.drain, inst.span) == (1, 1, 0, 1)
    fn = function_of(inst, {"a": "int32", "out": "int64"})
    assert [(p.name, p.lanes, p.rate) for p in fn.ports] == [("a", 2, 1), ("out", 2, "T")]
    a = np.array([[2**31 - 1, 1], [2**31 - 1, 2], [1, 3]], np.int32)
    state: dict = {}
    outs = [fn.fn(k, {"a": a[k]}, state, {"n": 3, "T": 3}) for k in range(3)]
    assert outs[:2] == [{}, {}]
    assert outs[2]["out"].tolist() == [2 * (2**31 - 1) + 1, 6]  # no wrap at int32
    assert outs[2]["out"].dtype == np.int64


def test_only_add():
    with pytest.raises(BrmError, match="op"):
        load_brm("lane_accumulate").resolve(IMPL, {"op": "max"})


def test_the_datapath_takes_the_instance():
    inst = load_brm("lane_accumulate").resolve(IMPL, {"W": 4})
    fill = DATAPATHS["LaneAccumulate"].fill(inst, {"a": "int64", "out": "int64"})
    assert {k: p.to_dict() for k, p in fill.ports.items()} == {
        "a": {"port": "a", "dir": "in", "dtype": "int64", "lanes": 4},
        "out": {"port": "out", "dir": "out", "dtype": "int64", "lanes": 4},
    }
    assert (fill.op_type, fill.params) == (None, {})
    assert DATAPATHS["LaneAccumulate"].timing(fill) == (1, 1)
    one = load_brm("lane_accumulate").resolve(IMPL, {"W": 1})
    with pytest.raises(GenError, match="1 lanes of 32 bits are not whole 64-bit channels"):
        DATAPATHS["LaneAccumulate"].fill(one, {"a": "int32", "out": "int32"})
