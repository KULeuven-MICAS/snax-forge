"""What every test package shares: the repository's paths, the snax-gvsoc submodule and its skip
marker, and the graph fixtures read as dicts or bound by their recipes.

    REPO, FIXTURES (tests/dfg/fixtures), SUB (gvsoc/snax-gvsoc)
    recipe_path(name)       ``recipes/<folder>/<name>.json`` (``find_recipe``)
    requires_submodule      skips where the submodule is not checked out (SUBMODULE_MISSING)
    graph_dict(name)        a fixture's .snaxdfg as a dict
    bound_vsum(w, impl)     vsum bound by its recipe, or at W = ``w`` to ``impl``
    imported(kernel, dtypes)  a kernel's import (DaCe), once per kernel and dtypes
    steps(name, **params)   every graph of the recipe ``name`` at ``params``, the import first
    dot1_split(w_mul, w_sum, n)  ``recipes/dot/dot_1.json`` with a factor param per split, ``W_mul``
                            and ``W_sum`` (the dot 1 dry run's, DRY1-2)
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from snax_forge.dfg import Graph
from snax_forge.sandbox import Recipe, apply_recipe
from snax_forge.sandbox.recipe import find_recipe

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "dfg" / "fixtures"
SUB = REPO / "gvsoc" / "snax-gvsoc"

SUBMODULE_MISSING = (
    "the snax-gvsoc submodule is not checked out (git submodule update --init gvsoc/snax-gvsoc)"
)
requires_submodule = pytest.mark.skipif(not (SUB / "snax").is_dir(), reason=SUBMODULE_MISSING)


def recipe_path(name: str) -> Path:
    """The recipe ``name``'s file, ``recipes/<folder>/<name>.json`` (``find_recipe``)."""
    return find_recipe(name)


def graph_dict(name: str = "vecadd_accelerated") -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.snaxdfg").read_text())


def bound_vsum(w: int = 1, impl: str = "chisel_adder_tree") -> Graph:
    """vsum bound by the checked-in recipe (W = 1, N = 64; accumulate's chisel_adder_tree,
    instance acc), or with W = ``w`` to ``impl``."""
    d = Recipe.load(recipe_path("vsum")).with_params({"W": w}).to_dict()
    d["steps"][-1]["params"]["implementation"] = impl
    return apply_recipe(Recipe.from_dict(d), Graph.load(FIXTURES / "vsum.snaxdfg"))[-1].graph


def imported(
    kernel: str, dtypes: Mapping[str, str] | Iterable[tuple[str, str]] | None = None
) -> Graph:
    """A kernel's import, once per kernel and dtypes (a dict or its items); no caller changes it."""
    return _imported(kernel, tuple(sorted(dict(dtypes or {}).items())))


@cache
def _imported(kernel: str, dtypes: tuple[tuple[str, str], ...]) -> Graph:
    from snax_forge.dfg.import_sdfg import import_kernel

    return import_kernel(kernel, dict(dtypes) or None)


def dot1_split(w_mul: int, w_sum: int, n: int = 2048) -> Recipe:
    """dot 1's recipe with ``mult_map`` split by ``W_mul`` and ``sum_map`` by ``W_sum``."""
    from snax_forge.sandbox.recipe import Step

    d = Recipe.load(recipe_path("dot_1"))
    factor = {"mult_map": "W_mul", "sum_map": "W_sum"}
    return Recipe(d.name, d.kernel, {"N": n, "W_mul": w_mul, "W_sum": w_sum}, d.symbols,
                  [Step(x.transform, {**x.params, "factor": factor[x.params["map"]]})
                   if "factor" in x.params else x for x in d.steps])  # fmt: skip


@cache
def steps(name: str, **params: int) -> tuple[Graph, ...]:
    """Every graph of recipe ``name`` at ``params`` (the recipe's defaults for the rest), the
    import first."""
    r = Recipe.load(recipe_path(name)).with_params(params)
    return tuple(x.graph for x in apply_recipe(r, imported(r.kernel, r.dtypes)))
