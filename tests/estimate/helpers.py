"""Shared helpers for the estimate tests: a recipe's graphs at given params, each kernel
imported once (DaCe), and the estimate of its last step on snax_forge."""

from __future__ import annotations

from functools import cache

from snax_forge.design import Platform
from snax_forge.dfg import Graph
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.estimate import Estimate, estimate
from snax_forge.sandbox import Recipe, apply_recipe
from tests.helpers import REPO

SNAX_FORGE = REPO / "platforms" / "snax_forge.json"


@cache
def imported(kernel: str, dtypes: tuple[tuple[str, str], ...] = ()) -> Graph:
    return import_kernel(kernel, dict(dtypes) or None)


@cache
def steps(name: str, **params: int) -> tuple[Graph, ...]:
    """Every graph of recipe ``name`` at ``params`` (the recipe's defaults for the rest)."""
    r = Recipe.load(REPO / "recipes" / f"{name}.json").with_params(params)
    start = imported(r.kernel, tuple(sorted(r.dtypes.items())))
    return tuple(x.graph for x in apply_recipe(r, start))


def estimated(name: str, **params: int) -> Estimate:
    """The estimate of recipe ``name``'s last graph on snax_forge."""
    return estimate(steps(name, **params)[-1], Platform.load(SNAX_FORGE))
