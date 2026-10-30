"""Shared helpers for the SNAX-SANDBOX tests.

The graphs are the SNAX-DFG fixtures of tests/dfg/fixtures (vecadd as
imported, split, accelerated); the recipe is the checked-in
``recipes/vecadd.json``. ``restricted_brm`` is ``elementwise_add`` with W
limited to 4 and 8, for the "a W the BRM does not allow" case.

For the sites (D72): ``recipe_graphs(name)``, every recipe's graphs from its
input to its last step (``vecadd_undo`` from vecadd's last, as README.md runs
it), ``SELECTED``, the param each transform's site fills from the selection,
and ``default_calls``, a transform's params with the user's ones at defaults.
"""

from __future__ import annotations

import inspect
import json
from functools import cache
from typing import Any

from snax_forge.brm import LIBRARY, Brm, library_names, load_brm
from snax_forge.dfg import Graph
from snax_forge.sandbox import TRANSFORMS, Recipe, SandboxError, apply_recipe
from snax_forge.sandbox.transforms import _locate
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


RECIPES = sorted(p.stem for p in (REPO / "recipes").glob("*.json"))
START = {"vecadd_undo": "vecadd"}  # starts from that recipe's last graph

# The param a transform's site fills from the selected node or container
SELECTED = {
    "split": "map", "split_map": "map", "tag_map": "map", "untag_map": "map", "join_map": "map",
    "split_reduce": "map", "join_reduce": "map", "fuse_map": "first", "unfuse_map": "map",
    "bind": "node", "unbind": "node", "link": "container", "unlink": "container",
    "encapsulate": "units", "unencapsulate": "accelerator",
}  # fmt: skip
DEFAULTS = {"factor": 4, "kind": "temporal", "depth": 1, "group": "g", "accelerator": "acc"}


@cache
def recipe_graphs(name: str) -> tuple[Recipe, tuple[Graph, ...]]:
    """The recipe and its graphs: the input (symbols bound), then one per step."""
    from snax_forge.dfg.import_sdfg import import_kernel

    r = Recipe.load(REPO / "recipes" / f"{name}.json")
    start = (
        recipe_graphs(START[name])[1][-1] if name in START else import_kernel(r.kernel, r.dtypes)
    )
    return r, tuple(res.graph for res in apply_recipe(r, start))


def selections(g: Graph) -> list[str]:
    """Every live node id and every container name."""
    return [n.id for n, _ in g.walk()] + list(g.containers)


def default_calls(g: Graph, name: str, selected: str) -> list[dict[str, Any]]:
    """``name``'s params for ``selected``, the user's at DEFAULTS; bind once per library BRM
    and implementation; fuse_map's ``second`` the next sibling, if any."""
    params: dict[str, Any] = {}
    for p in list(inspect.signature(TRANSFORMS[name].fn).parameters)[1:]:
        if p == SELECTED[name]:
            params[p] = [selected] if p == "units" else selected
        elif p == "second":
            try:
                body, k, _ = _locate(g, selected)
            except SandboxError:
                body, k = [], 0
            params[p] = body[k + 1].id if k + 1 < len(body) else selected
        elif p in DEFAULTS:
            params[p] = DEFAULTS[p]
    if name != "bind":
        return [params]
    return [params | {"brm": b, "implementation": i, "instance": "u"}
            for b in library_names() for i in load_brm(b).implementations]  # fmt: skip
