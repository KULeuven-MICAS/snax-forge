"""The lane accumulator (rowsum 2): the BRM ``lane_accumulate``, its kind and its binding to
``LaneAccumulate`` (gen/blocks.py)."""

from __future__ import annotations

import json

from snax_forge.brm import LIBRARY, Brm, load_brm

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
