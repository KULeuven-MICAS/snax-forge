"""Loop variables named after their dimension (dfg/loops.py) through split, tag and join.

Each map carries its dimension's base (``loop.dim``) and its piece (``loop.piece``, 1 outer to
inner); its variable is the base, ``_<piece>`` when the dimension has two or more pieces, then
the tag's suffix. A split numbers the pieces outer to inner, splitting a middle piece moves
the pieces below it down one, a join moves them back, and one piece left is the base again.
interchange_map and fuse_map keep each map's dimension and piece. Every inverse gives back the
graph exactly.
"""

from __future__ import annotations

import json

import pytest

from snax_forge.dfg import Graph, loops
from snax_forge.sandbox import (
    join_map,
    split,
)
from snax_forge.sandbox.transforms import split_map, tag_map, untag_map
from tests.helpers import FIXTURES, imported

from .helpers import RECIPES, recipe_graphs


def bound(name: str, **symbols: int) -> Graph:
    """A fixture or a kernel's import with its symbols bound."""
    path = FIXTURES / f"{name}.snaxdfg"
    d = json.loads(path.read_text()) if path.exists() else imported(name).to_dict()
    d["symbols"].update(symbols)
    return Graph.from_dict(d)


def vecadd(n: int = 64) -> Graph:
    return bound("vecadd", N=n)


def maps(g: Graph) -> list[tuple[str, str, int]]:
    """Each map's id, variable and piece, outer first."""
    return [(n.id, n.attrs["var"], n.attrs["loop.piece"]) for n, _ in g.walk() if n.kind == "map"]


def index(g: Graph) -> str:
    (d,) = g.node("add").inputs["in1"].subset
    return d


def chain() -> list[Graph]:
    """vecadd n, then split by 4, then its inner piece split by 2."""
    g0 = vecadd()
    g1 = split(g0, "add_map", 4)
    return [g0, g1, split(g1, "add_map_inner", 2)]


def test_splitting_a_middle_piece_renumbers_the_pieces_below_it():
    """n_2 of n_1, n_2, n_3 split by 2: n_1, n_2, n_3 (the new inner piece), n_4 (n_3 was); its
    record keeps the one before (``prior``), so join_map gives the three pieces back exactly."""
    *_, g2 = chain()
    g = split(g2, "add_map_inner", 1)
    assert maps(g) == [("add_map", "n_1", 1), ("add_map_inner", "n_2", 2),
                       ("add_map_inner_inner_1", "n_3", 3), ("add_map_inner_inner", "n_4", 4)]  # fmt: skip
    split_rec = g.node("add_map_inner").attrs["loop.split"]
    assert split_rec["prior"] == g2.node("add_map_inner").attrs["loop.split"]
    assert join_map(g, "add_map_inner").to_json() == g2.to_json()


def test_every_inverse_gives_back_the_graph_exactly():
    g = vecadd()
    assert join_map(split(g, "add_map", 4), "add_map").to_json() == g.to_json()
    assert untag_map(tag_map(g, "add_map", "spatial"), "add_map").to_json() == g.to_json()
    assert join_map(split_map(g, "add_map", 4), "add_map").to_json() == g.to_json()
    tail = vecadd(7)  # a tail: the record's valid count
    assert join_map(split_map(tail, "add_map", 4), "add_map").to_json() == tail.to_json()


@pytest.mark.parametrize("name", RECIPES)
def test_every_recipe_applies_names_its_loops_and_checks(name):
    """Each step is checked against the reference executor (recipe_graphs applies them), and
    every map's variable is its rule's."""
    for g in recipe_graphs(name)[1]:
        again = Graph.from_dict(g.to_dict())
        loops.rename(again)  # a no-op on a graph named by the rule
        assert again.to_json() == g.to_json()
        assert all("loop.dim" in n.attrs for n, _ in g.walk() if n.kind == "map")
