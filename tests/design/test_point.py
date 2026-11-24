"""The design point: made from a checked design, round trip, checked on load."""

from __future__ import annotations

import copy
import json

import pytest

from snax_forge.design import DesignError, run_checks
from snax_forge.design.memory import MemorySpec
from snax_forge.design.point import DesignPoint

from .helpers import BOUND, VECADD_STREAMERS, design


def point(memory=(), **kw) -> DesignPoint:
    d = design(**kw)
    d.memory_spec = MemorySpec().with_changes(list(memory))
    assert not run_checks(d)
    return DesignPoint.of(d, "vecadd")


def test_vecadd_design_point_holds_every_part():
    p = point([("B.l1.base", 576), ("C.l1.base", 1152)])
    d = p.to_dict()
    assert list(d) == ["name", "graph_from", "graph", "platform", "streamers", "memory",
                       "schedule"]  # fmt: skip
    assert d["schedule"] == "serial"
    assert d["graph"] == json.loads(BOUND.read_text())  # carried unchanged: code, replaced, split
    assert d["platform"]["name"] == "small16" and d["platform"]["base"] == "small16"
    assert d["streamers"] == VECADD_STREAMERS
    assert d["memory"]["changes"] == {"B.l1.base": 576, "C.l1.base": 1152}
    assert d["memory"]["layouts"]["B"] == {
        "l2": {"base": 0, "shape": [64], "strides": [8]},
        "l1": {"base": 576, "shape": [64], "strides": [8]},
    }


def test_round_trip_byte_for_byte(tmp_path):
    p = point()
    p.save(tmp_path / "dp.json")
    q = DesignPoint.load(tmp_path / "dp.json")
    assert q == p and q.to_json() == p.to_json() == (tmp_path / "dp.json").read_text()


def _loaded(edit) -> DesignError:
    d = copy.deepcopy(point().to_dict())
    edit(d)
    with pytest.raises(DesignError) as e:
        DesignPoint.from_dict(d)
    return e.value


def test_an_edited_streamer_is_caught():
    e = _loaded(lambda d: d["streamers"]["acc_a"].update(fifo_depth=4))
    assert e.codes == ["point.streamers"]


@pytest.mark.parametrize(
    "edit, code",
    [
        (lambda d: d.pop("memory"), "point.keys"),
        (lambda d: d.update(recipe={}), "point.keys"),
        (lambda d: d["streamers"]["acc_a"].pop("port"), "point.keys"),
        (lambda d: d["streamers"]["acc_a"].update(n_ports=8), "point.keys"),
        (lambda d: d["graph"].update(body=7), "graph.load"),
    ],
)
def test_a_malformed_point_is_named(edit, code):
    assert _loaded(edit).codes == [code]
