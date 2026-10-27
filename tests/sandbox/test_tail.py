"""split_map with a tail (open item 31, D144): ceil(n / W) firings, the valid count recorded,
the lanes past it padding; bind and join_map on such a split."""

from __future__ import annotations

import json

import numpy as np

from snax_forge.dfg import Graph, execute
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.sandbox import Recipe, apply_recipe, join_map, split_map

from .helpers import RECIPE

REPO = RECIPE.parents[1]


def steps(name: str, **params) -> list[Graph]:
    """``recipes/<name>.json`` at ``params``, every step's graph (the import first)."""
    recipe = Recipe.load(REPO / "recipes" / f"{name}.json").with_params(params)
    return [s.graph for s in apply_recipe(recipe, import_kernel(recipe.kernel))]


def _map(g: Graph, map_id: str):
    return next(n for n, _ in g.walk() if n.id == map_id)


def test_vecadd_n7_w4_runs_two_firings_and_records_the_valid_count():
    imported, split, bound = steps("vecadd", W=4, N=7)
    m = _map(split, "add_map")
    assert m.attrs["range"] == "0:(N + 3) // 4"
    assert m.attrs["loop.split"] == {"var": "i", "range": "0:N", "valid": "N"}
    assert len(range(*[int(x) for x in ("0", (7 + 3) // 4)])) == 2  # 2 firings at N = 7
    a = np.array([3, -1, 4, 1, -5, 9, 2])
    b = np.array([10, 20, 30, 40, 50, 60, 70])
    for g in (imported, split, bound):
        out = execute(g, {"A": a, "B": b, "C": np.zeros(7, np.int64)})
        assert out["C"].tolist() == [13, 19, 34, 41, 45, 69, 72]


def test_a_dividing_split_records_no_valid_count():
    _, split, _ = steps("vecadd", W=4, N=8)
    assert _map(split, "add_map").attrs["loop.split"] == {"var": "i", "range": "0:N"}


def test_join_map_undoes_a_tailed_split_exactly():
    imported = steps("vecadd", W=4, N=7)[0]
    joined = join_map(split_map(imported, "add_map", 4), "add_map")
    assert json.dumps(joined.to_dict()) == json.dumps(imported.to_dict())
