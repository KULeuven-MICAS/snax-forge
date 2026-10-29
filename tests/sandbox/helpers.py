"""Shared helpers for the SNAX-SANDBOX tests.

The graphs are the SNAX-DFG fixtures of tests/dfg/fixtures (vecadd as
imported, split, accelerated); the recipe is the checked-in
``recipes/vecadd.json``. ``restricted_brm`` is ``elementwise_add`` with W
limited to 4 and 8, for the "a W the BRM does not allow" case.
"""

from __future__ import annotations

import json

from snax_forge.brm import LIBRARY, Brm
from snax_forge.dfg import Graph
from snax_forge.sandbox import Recipe
from tests.helpers import FIXTURES, REPO

RECIPE = REPO / "recipes" / "vecadd.json"


def graph(name: str) -> Graph:
    return Graph.load(FIXTURES / f"{name}.snaxdfg")


def recipe() -> Recipe:
    return Recipe.load(RECIPE)


def restricted_brm() -> Brm:
    d = json.loads((LIBRARY / "elementwise_add.json").read_text())
    d["interface"]["params"]["W"]["values"] = [4, 8]
    return Brm.from_dict(d)
