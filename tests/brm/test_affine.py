"""The affine dataflow notation (BRM2, D70): structure, lanes, beats and enumeration."""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.brm import Brm, BrmError, resolve_nest, task_nest

from .helpers import affine_adder, affine_reducer, vector_nest

IMPL = "chisel_tiled_spatial"


def _adder(d=None):
    return Brm.from_dict(d or affine_adder())


def _lists(nest, env=None):
    """Indices as plain lists: 1D -> [[i, ...], ...], else [[(i, j), ...], ...]."""
    idx = resolve_nest(nest, env or {}).indices()
    if idx.shape[2] == 1:
        return idx[..., 0].tolist()
    return [[tuple(x) for x in beat] for beat in idx.tolist()]


# =============================================================================
# Enumeration against hand-written lists
# =============================================================================


def test_1d():
    assert _lists(vector_nest("n", "W"), {"n": 3, "W": 2}) == [[0, 1], [2, 3], [4, 5]]


def test_strided_with_offset():
    """Every other element from index 1, 4 lanes per beat."""
    nest = {
        "shape": [16],
        "offset": [1],
        "loops": [{"bound": 2, "strides": [8]}, {"bound": 4, "strides": [2], "spatial": True}],
    }
    assert _lists(nest) == [[1, 3, 5, 7], [9, 11, 13, 15]]


def test_2d_column_walk():
    """A 4x4 operand walked column by column, a column per beat (lane k = row k)."""
    nest = {
        "shape": [4, 4],
        "loops": [
            {"bound": 4, "strides": [0, 1]},
            {"bound": 4, "strides": [1, 0], "spatial": True},
        ],
    }
    assert _lists(nest) == [[(k, t) for k in range(4)] for t in range(4)]


def test_2d_temporal_loops_innermost_last():
    """Two temporal loops: the last listed runs fastest."""
    nest = {
        "shape": [2, 3],
        "loops": [
            {"bound": 2, "strides": [1, 0]},
            {"bound": 3, "strides": [0, 1]},
            {"bound": 1, "strides": [0, 0], "spatial": True},
        ],
    }
    assert _lists(nest) == [[(r, c)] for r in range(2) for c in range(3)]


def test_2d_spatial_lane_order():
    """A 2x2 tile per beat: the last spatial loop is lane dimension 0 (fastest)."""
    nest = {
        "shape": [2, 4],
        "loops": [
            {"bound": 2, "strides": [0, 2]},
            {"bound": 2, "strides": [1, 0], "spatial": True},
            {"bound": 2, "strides": [0, 1], "spatial": True},
        ],
    }
    assert _lists(nest) == [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 2), (0, 3), (1, 2), (1, 3)],
    ]


def test_reuse_on_the_innermost_temporal_loop():
    """Stride 0: the same beat three times in a row (a reader repeats it, D69)."""
    nest = {
        "shape": [8],
        "loops": [
            {"bound": 2, "strides": [4]},
            {"bound": 3, "strides": [0]},
            {"bound": 4, "strides": [1], "spatial": True},
        ],
    }
    assert _lists(nest) == [[0, 1, 2, 3]] * 3 + [[4, 5, 6, 7]] * 3


def test_negative_stride():
    nest = {
        "shape": [4],
        "offset": [3],
        "loops": [{"bound": 4, "strides": [-1]}, {"bound": 1, "strides": [0], "spatial": True}],
    }
    assert _lists(nest) == [[3], [2], [1], [0]]


def test_enumeration_equals_numpy_loops():
    """The vectorised enumeration against plain nested loops, on a 3-loop nest."""
    nest = {
        "shape": [40, 40],
        "offset": [1, 2],
        "loops": [
            {"bound": 3, "strides": [5, 1]},
            {"bound": 2, "strides": [0, 7]},
            {"bound": 2, "strides": [1, 3], "spatial": True},
            {"bound": 3, "strides": [2, 0], "spatial": True},
        ],
    }
    want = [
        [(1 + 5 * a + 0 * b + c + 2 * d, 2 + a + 7 * b + 3 * c) for c in range(2) for d in range(3)]
        for a in range(3)
        for b in range(2)
    ]
    assert _lists(nest) == want


# =============================================================================
# Normalisation, instance and task checks
# =============================================================================


def test_nests_are_written_in_full():
    d = _adder().to_dict()["dataflow"]["ports"]["a"]
    assert d == {
        "shape": ["n * W"],
        "offset": [0],
        "loops": [
            {"bound": "n", "strides": ["W"], "spatial": False},
            {"bound": "W", "strides": [1], "spatial": True},
        ],
    }
    assert Brm.from_dict(_adder().to_dict()) == _adder()


def test_task_nest_of_the_adder():
    inst = _adder().resolve(IMPL, {"W": 4})
    nest = task_nest(inst, "out", {"n": 16})
    assert (nest.n_beats, nest.n_lanes, nest.shape) == (16, 4, (64,))
    idx = nest.indices()[..., 0]
    assert np.array_equal(idx, np.arange(64).reshape(16, 4))
    # the ports agree on element positions (D70): the same index on a, b and out
    for p in ("a", "b"):
        assert np.array_equal(task_nest(inst, p, {"n": 16}).indices(), nest.indices())


def test_reducer_output_beats_follow_the_rate():
    inst = Brm.from_dict(affine_reducer()).resolve("chisel_accumulator", {"W": 4})
    assert task_nest(inst, "in", {"n": 8, "T": 4}).n_beats == 8
    assert task_nest(inst, "out", {"n": 8, "T": 4}).n_beats == 2


def _nest(d, port="a"):
    return d["dataflow"]["ports"][port]


STRUCTURE = [
    ("no shape", lambda d: _nest(d).pop("shape"), "missing key 'shape'"),
    ("unknown key", lambda d: _nest(d).update(order="C"), "unknown keys"),
    ("offset rank", lambda d: _nest(d).update(offset=[0, 0]), "2 entries for 1 dimensions"),
    ("strides rank", lambda d: _nest(d)["loops"][0].update(strides=["W", 0]), "2 entries"),
    ("spatial first", lambda d: _nest(d)["loops"].reverse(), "temporal loop after a spatial"),
    ("no spatial", lambda d: _nest(d)["loops"][1].update(spatial=False), "at least one spatial"),
    ("spatial runtime", lambda d: _nest(d)["loops"][1].update(bound="n"), "allowed here"),
    ("unknown name", lambda d: _nest(d)["loops"][0].update(bound="m"), r"uses \['m'\]"),
    ("not an object", lambda d: d["dataflow"]["ports"].update(a=[0, 1]), "must be an object"),
    ("spatial flag", lambda d: _nest(d)["loops"][1].update(spatial=1), "true or false"),
]


@pytest.mark.parametrize(("case", "edit", "msg"), STRUCTURE, ids=[c[0] for c in STRUCTURE])
def test_structure_errors(case, edit, msg):
    d = affine_adder()
    edit(d)
    with pytest.raises(BrmError, match=msg):
        Brm.from_dict(d)


def test_spatial_lanes_checked_at_resolve():
    d = affine_adder()
    _nest(d, "out")["loops"][1]["bound"] = "W // 2"
    b = Brm.from_dict(d)  # fine until W has a value
    with pytest.raises(BrmError, match="give 2 lanes, the port has 4"):
        b.resolve(IMPL, {"W": 4})


TASKS = [
    ("missing n", {}, "must be exactly"),
    ("extra T", {"n": 16, "T": 4}, "must be exactly"),
]


@pytest.mark.parametrize(("case", "task", "msg"), TASKS, ids=[c[0] for c in TASKS])
def test_task_values_are_the_registers(case, task, msg):
    with pytest.raises(ValueError, match=msg):
        task_nest(_adder().resolve(IMPL), "a", task)


def test_beats_must_be_n_over_rate():
    d = affine_adder()
    _nest(d)["loops"][0]["bound"] = "n + 1"
    _nest(d)["shape"] = ["(n + 1) * W"]
    with pytest.raises(ValueError, match="gives 17 beats, need n / rate = 16"):
        task_nest(Brm.from_dict(d).resolve(IMPL), "a", {"n": 16})
    inst = Brm.from_dict(affine_reducer()).resolve("chisel_accumulator", {"W": 4})
    with pytest.raises(ValueError, match="not a multiple of the rate 4"):
        task_nest(inst, "out", {"n": 6, "T": 4})


def test_indices_must_stay_in_the_shape():
    d = affine_adder()
    _nest(d)["offset"] = [1]
    with pytest.raises(ValueError, match=r"run over \[1, 64\], outside the shape \[64\]"):
        task_nest(Brm.from_dict(d).resolve(IMPL), "a", {"n": 16})
