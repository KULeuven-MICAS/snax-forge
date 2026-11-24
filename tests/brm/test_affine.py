"""The affine dataflow notation: structure, lanes, beats and enumeration."""

from __future__ import annotations

import pytest

from snax_forge.brm import Brm, BrmError, task_nest

from .helpers import IMPL, affine_adder, affine_reducer


def _adder(d=None):
    return Brm.from_dict(d or affine_adder())


# =============================================================================
# Enumeration against hand-written lists
# =============================================================================


# =============================================================================
# Normalisation, instance and task checks
# =============================================================================


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
    inst = Brm.from_dict(affine_reducer()).resolve("chisel_adder_tree", {"W": 4})
    with pytest.raises(ValueError, match="not a multiple of the rate 4"):
        task_nest(inst, "out", {"n": 6, "T": 4})
