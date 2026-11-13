"""The every-recipe sweeps run each recipe's cases on one xdist worker.

``recipe_graphs`` caches a recipe's graphs per worker, so a sweep case spread over the workers
applies the recipe again on each. ``pytest_collection_modifyitems`` puts every case of a sweep in
``SWEEPS`` in the xdist group of its recipe; with ``--dist loadgroup`` (the ``test`` task) a group
runs on one worker and every other test spreads as with ``--dist load``.
"""

from __future__ import annotations

import pytest

# The sweeps over RECIPES, each parametrized by ``name``
SWEEPS = {
    "test_every_recipe_applies_names_its_loops_and_checks",
    "test_the_site_accepts_every_recipe_step",
    "test_a_site_refuses_as_its_transform_on_every_step",
    "test_split_map_is_split_then_two_tags",
    "test_the_estimate_of_every_step_of_every_recipe",
}


@pytest.hookimpl(tryfirst=True)  # before xdist's own hook reads the groups
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if getattr(item, "originalname", None) in SWEEPS:
            item.add_marker(pytest.mark.xdist_group(f"recipe-{item.callspec.params['name']}"))
