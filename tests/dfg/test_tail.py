"""The reference executor on a split with a tail: the valid lanes only,
a pad lane read as the container's pad value; the pad helpers (dfg/pad.py)."""

from __future__ import annotations

import sys

from snax_forge.dfg import Graph
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.sandbox import Recipe, apply_recipe
from tests.helpers import FIXTURES, recipe_path

execute_mod = sys.modules["snax_forge.dfg.execute"]  # the package exports the function by name


def steps(name: str, **params) -> list[Graph]:
    recipe = Recipe.load(recipe_path(name)).with_params(params)
    return [s.graph for s in apply_recipe(recipe, import_kernel(recipe.kernel))]


def _vsum(dtype: str, op: str, identity: int) -> Graph:
    d = Graph.load(FIXTURES / "vsum.snaxdfg").to_dict()
    d["symbols"]["N"] = 8
    d["containers"]["A"]["dtype"] = dtype
    d["containers"]["out"]["dtype"] = dtype
    d["body"][0]["body"][0]["outputs"]["out"]["reduce"] = {"op": op, "identity": identity}
    return Graph.from_dict(d)
