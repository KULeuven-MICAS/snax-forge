"""The checked-in recipes' names, ids and descriptions, and recipes/README.md's table of
them: every recipe outside ``custom/`` is ``<kernel>_v<N>`` with ``_<k>`` per change from a
parent that exists, its id starts with its kernel, and recipes sharing an id build the same
units."""

from __future__ import annotations

import re
from collections import defaultdict

from snax_forge.sandbox import Recipe
from snax_forge.sandbox.recipe import RECIPE_ROOT, recipe_paths

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


TWINS = defaultdict(list)
for _name, _r in checked_in().items():
    TWINS[_r.id].append(_name)
