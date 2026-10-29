"""The MAC (D149): the BRM ``mac`` and its kind, and ``accumulate``'s ``chisel_serial``; each
implementation's datapath takes its instance (gen/blocks.py)."""

from __future__ import annotations

import numpy as np

from snax_forge.brm import function_of, load_brm
from snax_forge.gen.blocks import DATAPATHS


def test_the_mac_folds_each_lanes_products_over_t_beats():
    inst = load_brm("mac").resolve("chisel_mac", {"W": 2})
    fn = function_of(inst, {"a": "int32", "b": "int32", "out": "int64"})
    assert [(p.name, p.lanes, p.rate) for p in fn.ports] == [("a", 2, 1), ("b", 2, 1),
                                                             ("out", 2, "T")]  # fmt: skip
    state: dict = {}
    a, b = (
        np.array([[1, 2], [3, 4], [5, 6]], np.int32),
        np.array([[7, 8], [9, 10], [11, 12]], np.int32),
    )
    outs = [fn.fn(k, {"a": a[k], "b": b[k]}, state, {"n": 3, "T": 3}) for k in range(3)]
    assert outs[:2] == [{}, {}]
    assert outs[2]["out"].tolist() == [1 * 7 + 3 * 9 + 5 * 11, 2 * 8 + 4 * 10 + 6 * 12]
    assert outs[2]["out"].dtype == np.int64


def test_the_mac_and_the_serial_sum_are_bound_to_their_datapaths():
    mac = load_brm("mac").resolve("chisel_mac", {"W": 4})
    assert (mac.latency, mac.initiation_interval, mac.drain, mac.span) == (1, 1, 0, 1)
    fill = DATAPATHS["Mac"].fill(mac, {"a": "int64", "b": "int64", "out": "int64"})
    assert (fill.op_type, fill.params) == ("int64", {"accumulate": True, "regOut": False})
    assert DATAPATHS["Mac"].timing(fill) == (1, 1)
    serial = load_brm("accumulate").resolve("chisel_serial", {"W": 4})
    assert (serial.latency, serial.span) == (1, 4)
    fill = DATAPATHS["SerialAccumulate"].fill(serial, {"a": "int64", "out": "int64"})
    assert fill.params == {} and DATAPATHS["SerialAccumulate"].timing(fill) == (1, 4)
