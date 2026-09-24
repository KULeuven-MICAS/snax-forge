"""A BRM port's nest through a layout onto streamer values (BRM2, D70)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from snax_forge.brm import Brm, load_brm, task_nest
from snax_forge.lower import Layout, LayoutError, StreamError, arg_of, streamer_values
from snax_forge.snax_model import ControllerConfig, L1Config, StreamerConfig
from snax_forge.snax_model.scenario import (
    ClusterConfig,
    ComponentSpec,
    RegisterMapSpec,
    register_map_of,
)
from snax_forge.snax_model.streamer import address_stream
from tests.brm.helpers import affine_adder, vector_nest

from .helpers import SCEN, cluster

IMPL = "chisel_tiled_spatial"


def _streamer(name, write, lanes, dims):
    cfg = StreamerConfig(write=write, n_ports=lanes, fifo_depth=2, temporal_dims=dims)
    return ComponentSpec(name, "streamer", cfg.to_dict())


def streams_cluster() -> ClusterConfig:
    """Streamers of every shape the cases need, on a 16-bank L1 of 64 rows."""
    return ClusterConfig(
        l1=L1Config(n_banks=16, rows=64),
        components=[
            ComponentSpec("xbar", "xbar", {}),
            _streamer("r4", False, 4, 3),
            _streamer("r22", False, 4, 3),
            _streamer("r1", False, 1, 3),
            _streamer("w4", True, 4, 3),
            ComponentSpec("ctl", "controller", ControllerConfig().to_dict()),
        ],
        register_map=RegisterMapSpec(
            blocks=["r4", "r22", "r1", "w4"], spatial_bounds={"r22": [2, 2]}
        ),
    )


def instance(nest_a=None, W=4):
    """The affine adder, with ``nest_a`` on port a."""
    d = affine_adder()
    if nest_a is not None:
        d["dataflow"]["ports"]["a"] = nest_a
    return Brm.from_dict(d).resolve(IMPL, {"W": W})


# =============================================================================
# vecadd: the three streamer tasks of scenarios/vecadd/tasks.json
# =============================================================================


def test_vecadd_streamer_values():
    """The library's elementwise_add (BRM3): a, b and c of 64 int64 at L1 bytes 0, 576
    and 1152, n = 16 beats of 4, give vecadd's four task values."""
    tasks = json.loads((SCEN / "vecadd" / "tasks.json").read_text())["steps"]
    want = {s["task_name"]: s["values"] for s in tasks if s.get("type") in ("streamer", "accel")}
    inst, cl = load_brm("elementwise_add").resolve(IMPL), cluster("alu4")
    task = {"n": 16}
    assert want.pop("add_acc") == task and list(task) == inst.brm.registers
    got = {
        "add_ra": streamer_values(inst, "a", task, Layout(0, (64,), (8,)), cl, "ra"),
        "add_rb": streamer_values(inst, "b", task, Layout(576, (64,), (8,)), cl, "rb"),
        "add_wr": streamer_values(inst, "out", task, Layout(1152, (64,), (8,)), cl, "wr"),
    }
    assert got == want


# =============================================================================
# The mapped registers reproduce the enumerated addresses (MOD4 streams)
# =============================================================================

COLUMN_WALK = {
    "shape": [4, 4],
    "loops": [{"bound": 4, "strides": [0, 1]}, {"bound": 4, "strides": [1, 0], "spatial": True}],
}
ROW_WALK = {
    "shape": [4, 4],
    "loops": [
        {"bound": 4, "strides": [1, 0]},
        {"bound": 4, "strides": [0, 1]},
        {"bound": 1, "strides": [0, 0], "spatial": True},
    ],
}
TILE_2X2 = {
    "shape": [2, 4],
    "loops": [
        {"bound": 2, "strides": [0, 2]},
        {"bound": 2, "strides": [1, 0], "spatial": True},
        {"bound": 2, "strides": [0, 1], "spatial": True},
    ],
}
STRIDED = {
    "shape": [16],
    "offset": [1],
    "loops": [{"bound": 2, "strides": [8]}, {"bound": 4, "strides": [2], "spatial": True}],
}
REUSE = {
    "shape": [8],
    "loops": [
        {"bound": 2, "strides": [4]},
        {"bound": 3, "strides": [0]},
        {"bound": 4, "strides": [1], "spatial": True},
    ],
}

# name: (nest, W, n, layout, streamer)
STREAMS = {
    "1d": (vector_nest("n", "W"), 4, 16, Layout(0, (64,), (8,)), "r4"),
    "strided": (STRIDED, 4, 2, Layout(256, (16,), (8,)), "r4"),
    "column_row_major": (COLUMN_WALK, 4, 4, Layout(1024, (4, 4), (32, 8)), "r4"),
    "column_col_major": (COLUMN_WALK, 4, 4, Layout(1024, (4, 4), (8, 32)), "r4"),
    "row_padded": (ROW_WALK, 1, 16, Layout(2048, (4, 4), (40, 8)), "r1"),
    "tile_2x2": (TILE_2X2, 4, 2, Layout.contiguous(512, (2, 4)), "r22"),
    "reuse": (REUSE, 4, 6, Layout(64, (8,), (8,)), "r4"),
    "reversed_layout": (vector_nest("n", "W"), 4, 2, Layout(56, (8,), (-8,)), "r4"),
}


@pytest.mark.parametrize("name", list(STREAMS))
def test_registers_reproduce_the_enumeration(name):
    nest, W, n, layout, s = STREAMS[name]
    inst, cl = instance(nest, W), streams_cluster()
    values = streamer_values(inst, "a", {"n": n}, layout, cl, s)
    regs = arg_of("streamer", values, register_map_of(cl)[s])
    want = layout.address(task_nest(inst, "a", {"n": n}).indices())
    assert np.array_equal(address_stream(regs), want)


def test_column_walk_values_by_hand():
    """Row-major 4x4: a column is 4 lanes 32 bytes apart, the next column 8 bytes on."""
    inst = instance(COLUMN_WALK)
    v = streamer_values(inst, "a", {"n": 4}, Layout(1024, (4, 4), (32, 8)), streams_cluster(), "r4")
    assert v == {
        "base": 1024,
        "temporal_bounds": [4],
        "temporal_strides": [8],
        "spatial_strides": [32],
    }


# =============================================================================
# What does not fit is an error
# =============================================================================


def _values(nest=None, W=4, n=16, layout=None, s="r4", port="a", cl=None):
    return streamer_values(
        instance(nest, W), port, {"n": n}, layout or Layout(0, (64,), (8,)),
        cl or streams_cluster(), s,
    )  # fmt: skip


ERRORS = [
    ("not a streamer", {"s": "acc"}, "not a streamer"),
    ("reader for out", {"port": "out"}, "a reader cannot serve an 'out' port"),
    ("writer for in", {"s": "w4"}, "a writer cannot serve an 'in' port"),
    ("spatial bounds", {"s": "r22"}, r"spatial bounds \[4\] .* design-time \[2, 2\]"),
    ("temporal loops",
     {"nest": REUSE, "n": 6, "layout": Layout(0, (8,), (8,)), "cl": cluster("alu4"), "s": "ra"},
     "2 temporal loops, the streamer has 1"),
    ("shape", {"layout": Layout(0, (32,), (8,))}, r"layout shape \[32\]"),
    ("unaligned base", {"layout": Layout(4, (64,), (8,))}, "multiples of the 8-byte word"),
    ("unaligned stride", {"layout": Layout(0, (64,), (12,))}, "multiples of the 8-byte word"),
    ("outside L1", {"layout": Layout(8000, (64,), (8,))}, r"L1 is \[0, 8192\)"),
    ("below L1", {"layout": Layout(0, (64,), (-8,))}, "spans bytes"),
    ("task", {"nest": REUSE, "layout": Layout(0, (8,), (8,))}, "gives 6 beats, need n / rate = 16"),
]  # fmt: skip


@pytest.mark.parametrize(("case", "kw", "msg"), ERRORS, ids=[c[0] for c in ERRORS])
def test_errors(case, kw, msg):
    with pytest.raises(StreamError, match=msg):
        _values(**kw)


def test_packed_words_are_not_supported():
    with pytest.raises(LayoutError, match="open item 21"):
        Layout(0, (4,), (8,)).check(L1Config(dtype="int32", elems_per_word=2))


def test_layout_round_trip_and_contiguous():
    lay = Layout.contiguous(64, (3, 5))
    assert lay.strides == (40, 8)
    assert Layout.from_dict(lay.to_dict()) == lay
    assert lay.to_dict() == {"base": 64, "shape": [3, 5], "strides": [40, 8]}
    with pytest.raises(LayoutError):
        Layout(0, (4,), (8, 8))
