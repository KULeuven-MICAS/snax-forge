"""The recipe format (SBX1, D80): fields, params, the checked-in vecadd recipe."""

from __future__ import annotations

import pytest

from snax_forge.sandbox import Recipe, SandboxError

from .helpers import RECIPE, recipe


def test_checked_in_recipe_is_in_written_form():
    assert Recipe.load(RECIPE).to_json() == RECIPE.read_text()


def test_round_trip():
    r = recipe()
    assert Recipe.from_dict(r.to_dict()) == r


def test_defaults_are_written():
    r = Recipe.from_dict({"name": "r", "kernel": "vecadd"})
    assert r.to_dict() == {
        "name": "r",
        "kernel": "vecadd",
        "params": {},
        "symbols": {},
        "steps": [],
    }


def test_params_and_symbols():
    r = Recipe.from_dict(
        {"name": "r", "kernel": "k", "params": {"W": 4}, "symbols": {"N": "16 * W"}}
    )
    assert r.bound_symbols() == {"N": 64}
    assert r.with_params({"W": 8}).bound_symbols() == {"N": 128}
    assert r.params == {"W": 4}  # with_params makes a new recipe
    with pytest.raises(SandboxError, match="no param 'V'"):
        r.with_params({"V": 1})


@pytest.mark.parametrize(
    ("d", "message"),
    [
        ({"name": "r"}, "missing key 'kernel'"),
        ({"name": "r", "kernel": "k", "extra": 1}, "unknown keys \\['extra'\\]"),
        ({"name": "r", "kernel": "k", "params": {"W": 1.5}}, "params.W: must be an int or"),
        ({"name": "r", "kernel": "k", "params": {"2W": 1}}, "'2W' is not a valid name"),
        ({"name": "r", "kernel": "k", "symbols": {"N": "V * 2"}}, "no value for 'V'"),
        ({"name": "r", "kernel": "k", "steps": [{"params": {}}]}, "missing key 'transform'"),
        ({"name": "r", "kernel": "k", "steps": {}}, "steps: must be a list"),
    ],
)
def test_rejected(d, message):
    with pytest.raises(SandboxError, match=message):
        Recipe.from_dict(d)
