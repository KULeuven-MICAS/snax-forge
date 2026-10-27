"""A BRM port's nest through a layout onto streamer values (D70)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from snax_forge.brm import Brm, task_nest
from snax_forge.design import Platform, StreamerOptions
from snax_forge.design.platform import L1Config
from snax_forge.design.streamers import Streamer
from snax_forge.lower import Layout, LayoutError, StreamError, streamer_values
from snax_forge.lower.derive import addresses
from tests.brm.helpers import IMPL, affine_adder, affine_elementwise_add, vector_nest
from tests.design.helpers import SNAX_FORGE

from .helpers import VECADD_TASKS, vecadd_point


def _streamer(name, write, spatial_bounds, dims):
    return Streamer(name, "x", "a", write, tuple(spatial_bounds), dims, StreamerOptions(2))


# Streamers of every shape the cases need, on a 16-bank L1 of 64 rows
STREAMERS = {s.name: s for s in (
    _streamer("r4", False, [4], 3), _streamer("r22", False, [2, 2], 3),
    _streamer("r1", False, [1], 3), _streamer("w4", True, [4], 3), _streamer("r4d1", False, [4], 1),
)}  # fmt: skip
L1 = L1Config(n_banks=16, rows=64)


def instance(nest_a=None, W=4):
    """The affine adder, with ``nest_a`` on port a."""
    d = affine_adder()
    if nest_a is not None:
        d["dataflow"]["ports"]["a"] = nest_a
    return Brm.from_dict(d).resolve(IMPL, {"W": W})


# =============================================================================
# vecadd: the three streamer tasks of its task list on snax_forge
# =============================================================================


def test_vecadd_streamer_values():
    """elementwise_add with affine nests: a, b and c of 64 int64 at the first three 512-byte
    blocks of snax_forge's L1, n = 16 beats of 4, give vecadd's four task values."""
    tasks = json.loads(VECADD_TASKS.read_text())["steps"]
    want = {s["task_name"]: s["values"] for s in tasks if s.get("type") in ("streamer", "accel")}
    inst, l1 = affine_elementwise_add().resolve(IMPL), Platform.load(SNAX_FORGE).l1
    shell = vecadd_point().streamers
    task = {"n": 16}
    assert want.pop("add_acc") == task and list(task) == inst.brm.registers
    got = {
        f"add_{s}": streamer_values(inst, p, task, Layout(l1.base_addr + base, (64,), (8,)),
                                    shell[s], l1)
        for p, s, base in (("a", "acc_a", 0), ("b", "acc_b", 512), ("out", "acc_out", 1024))
    }  # fmt: skip
    assert got == want


# =============================================================================
# The mapped values reproduce the enumerated addresses
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
    inst = instance(nest, W)
    values = streamer_values(inst, "a", {"n": n}, layout, STREAMERS[s], L1)
    want = layout.address(task_nest(inst, "a", {"n": n}).indices())
    got = addresses(values, list(STREAMERS[s].spatial_bounds))
    assert np.array_equal(got.reshape(want.shape), want)


def test_column_walk_values_by_hand():
    """Row-major 4x4: a column is 4 lanes 32 bytes apart, the next column 8 bytes on."""
    inst = instance(COLUMN_WALK)
    v = streamer_values(inst, "a", {"n": 4}, Layout(1024, (4, 4), (32, 8)), STREAMERS["r4"], L1)
    assert v == {
        "base": 1024,
        "temporal_bounds": [4],
        "temporal_strides": [8],
        "spatial_strides": [32],
    }


# =============================================================================
# What does not fit is an error
# =============================================================================


def _values(nest=None, W=4, n=16, layout=None, s="r4", port="a"):
    return streamer_values(
        instance(nest, W), port, {"n": n}, layout or Layout(0, (64,), (8,)), STREAMERS[s], L1
    )


ERRORS = [
    ("reader for out", {"port": "out"}, "a reader cannot serve an 'out' port"),
    ("writer for in", {"s": "w4"}, "a writer cannot serve an 'in' port"),
    ("spatial bounds", {"s": "r22"}, r"spatial bounds \[4\] .* design-time \[2, 2\]"),
    ("temporal loops",
     {"nest": REUSE, "n": 6, "layout": Layout(0, (8,), (8,)), "s": "r4d1"},
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


def test_layout_round_trip_and_contiguous():
    lay = Layout.contiguous(64, (3, 5))
    assert lay.strides == (40, 8)
    assert Layout.from_dict(lay.to_dict()) == lay
    assert lay.to_dict() == {"base": 64, "shape": [3, 5], "strides": [40, 8]}
    with pytest.raises(LayoutError):
        Layout(0, (4,), (8, 8))
