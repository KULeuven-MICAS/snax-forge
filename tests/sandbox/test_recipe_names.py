"""The checked-in recipes' names, ids and descriptions (D188), and recipes/README.md's table of
them: every recipe outside ``custom/`` is ``<kernel>_v<N>`` with ``_<k>`` per change from a
parent that exists, its id starts with its kernel, and recipes sharing an id build the same
units."""

from __future__ import annotations

import re
from collections import defaultdict

import pytest

from snax_forge.sandbox import Recipe, apply_recipe, bind_recipe_symbols
from snax_forge.sandbox.recipe import RECIPE_ROOT, recipe_paths
from tests.helpers import imported

README = RECIPE_ROOT / "README.md"


def checked_in() -> dict[str, Recipe]:
    return {
        name: Recipe.load(paths[0])
        for name, paths in recipe_paths().items()
        if paths[0].parent.name != "custom"
    }


def base(r: Recipe) -> str:
    """The kernel a name and an id start with: a type-preset kernel counts as its base."""
    return r.kernel.removesuffix("_types")


def test_every_name_is_its_kernels_version_and_extends_a_recipe_that_exists():
    recipes = checked_in()
    for name, r in recipes.items():
        assert re.fullmatch(rf"{base(r)}_v\d+(_\d+)*", name), name
        assert r.name == name
        parent = name.rsplit("_", 1)[0]
        assert re.fullmatch(rf"{base(r)}_v\d+", name) or parent in recipes, name


def test_every_recipe_says_what_it_builds():
    for name, r in checked_in().items():
        assert r.id and r.id.startswith(base(r)), name
        assert r.description and r.description.endswith("."), name
        assert len(re.findall(r"\.(\s|$)", r.description)) <= 3, name


def test_the_readme_lists_every_recipe_as_its_file_says():
    rows = re.findall(
        r"^\| `(\w+)` \| (\w+) \| `(\w+)` \| (.+) \|$", README.read_text(), re.MULTILINE
    )
    want = sorted(
        (n, recipe_paths()[n][0].parent.name, r.id, r.description) for n, r in checked_in().items()
    )
    assert sorted(rows) == want


def _units(r: Recipe) -> list[tuple]:
    kw = {"types": r.types} if r.types else {}
    g = apply_recipe(r, bind_recipe_symbols(r, imported(r.kernel, **kw)))[-1].graph
    units = [
        (n.attrs["brm"], n.attrs.get("implementation"), n.attrs.get("accelerator") is not None)
        for n, _ in g.walk()
        if n.kind == "accelerated"
    ]
    links = [("link", n.attrs.get("depth")) for n, _ in g.walk() if n.kind == "link"]
    return sorted(map(str, units + links))


TWINS = defaultdict(list)
for _name, _r in checked_in().items():
    TWINS[_r.id].append(_name)


@pytest.mark.slow
@pytest.mark.parametrize("names", [sorted(v) for v in TWINS.values() if len(v) > 1], ids=str)
def test_recipes_sharing_an_id_build_the_same_units(names):
    recipes = checked_in()
    first, *rest = (_units(recipes[n]) for n in names)
    assert all(u == first for u in rest), names
