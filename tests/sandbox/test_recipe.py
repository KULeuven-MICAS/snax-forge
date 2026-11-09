"""The recipe format: fields, params, the checked-in vecadd recipe."""

from __future__ import annotations

import pytest

from snax_forge.sandbox import Recipe, SandboxError
from snax_forge.sandbox.recipe import RECIPE_ROOT, find_recipe, recipe_paths

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


def test_every_recipe_is_in_a_folder_once_under_its_name():
    """The recipes live one folder deep by category (C§12): none directly in ``recipes/``, each
    name in one folder only, and each recipe's ``name`` its file's stem."""
    assert not list(RECIPE_ROOT.glob("*.json"))
    paths = recipe_paths()
    assert sorted(p.name for p in RECIPE_ROOT.iterdir() if p.is_dir()) == [
        "custom", "dot", "elementwise", "matvec", "mvt", "reduce"]  # fmt: skip
    assert {n: len(ps) for n, ps in paths.items() if len(ps) > 1} == {}
    for name, (path,) in paths.items():
        assert Recipe.load(path).name == name == path.stem, path
        assert find_recipe(name) == path
    assert find_recipe("vecadd") == RECIPE_ROOT / "elementwise" / "vecadd.json"


def test_find_recipe_names_what_is_wrong(tmp_path):
    """A name in no folder or in two, and any .json directly in the root, is an error naming it."""
    for folder in ("dot", "custom"):
        (tmp_path / folder).mkdir()
        recipe().save(tmp_path / folder / "twice.json")
    recipe().save(tmp_path / "dot" / "once.json")
    assert find_recipe("once", tmp_path) == tmp_path / "dot" / "once.json"
    with pytest.raises(SandboxError, match=r"no recipe 'gone' in any folder of "):
        find_recipe("gone", tmp_path)
    with pytest.raises(
        SandboxError,
        match=r"recipe 'twice' is in 2 folders of .*\['custom/twice.json', 'dot/twice.json'\]",
    ):
        find_recipe("twice", tmp_path)
    recipe().save(tmp_path / "flat.json")
    with pytest.raises(SandboxError, match=r"not directly in it: \['flat.json'\]"):
        find_recipe("once", tmp_path)
