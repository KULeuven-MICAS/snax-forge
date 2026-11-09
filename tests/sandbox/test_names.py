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
    Recipe,
    SandboxError,
    apply_recipe,
    interchange_map,
    join_map,
    split,
)
from snax_forge.sandbox.compose import fuse_map, join_reduce, split_reduce, unfuse_map
from snax_forge.sandbox.transforms import split_map, tag_map, untag_map
from tests.helpers import FIXTURES, imported, recipe_path

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


def test_the_rule():
    assert loops.var_name("n", 1, 1, None) == "n"
    assert loops.var_name("n", 1, 1, "temporal") == "n_t"
    assert loops.var_name("n", 2, 3, "spatial") == "n_2_s"
    assert loops.var_name("n", 3, 3, "tile") == "n_3_tile"


def test_splits_number_the_pieces_outer_to_inner_and_tags_add_a_suffix():
    """n -> n_1, n_2 -> n_1, n_2, n_3 -> n_1_t, n_2_t, n_3_s."""
    g0, g1, g2 = chain()
    assert maps(g0) == [("add_map", "n", 1)]
    assert maps(g1) == [("add_map", "n_1", 1), ("add_map_inner", "n_2", 2)]
    assert maps(g2) == [("add_map", "n_1", 1), ("add_map_inner", "n_2", 2),
                        ("add_map_inner_inner", "n_3", 3)]  # fmt: skip
    assert index(g2) == "4 * n_1 + (2 * n_2 + n_3)"
    g = tag_map(g2, "add_map", "temporal")
    g = tag_map(g, "add_map_inner", "temporal")
    g = tag_map(g, "add_map_inner_inner", "spatial")
    assert [v for _, v, _ in maps(g)] == ["n_1_t", "n_2_t", "n_3_s"]
    assert index(g) == "4 * n_1_t + (2 * n_2_t + n_3_s)"
    assert maps(tag_map(g0, "add_map", "temporal")) == [("add_map", "n_t", 1)]  # unsplit: n_t


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


def test_joins_renumber_back_and_one_piece_is_the_base_again():
    g0, g1, g2 = chain()
    assert join_map(g2, "add_map_inner").to_json() == g1.to_json()
    assert join_map(g1, "add_map").to_json() == g0.to_json()
    t = tag_map(tag_map(g2, "add_map", "temporal"), "add_map_inner", "temporal")
    t = tag_map(t, "add_map_inner_inner", "spatial")
    with pytest.raises(SandboxError, match="its inner piece 'add_map_inner' was split since"):
        join_map(t, "add_map")
    joined = join_map(t, "add_map_inner")  # n_2_t with n_3_s: one untagged piece, n_2
    assert maps(joined) == [("add_map", "n_1_t", 1), ("add_map_inner", "n_2", 2)]
    one = join_map(untag_map(joined, "add_map"), "add_map")
    assert maps(one) == [("add_map", "n", 1)] and one.to_json() == g0.to_json()
    assert maps(tag_map(one, "add_map", "temporal")) == [("add_map", "n_t", 1)]  # with its suffix


def test_every_inverse_gives_back_the_graph_exactly():
    g = vecadd()
    assert join_map(split(g, "add_map", 4), "add_map").to_json() == g.to_json()
    assert untag_map(tag_map(g, "add_map", "spatial"), "add_map").to_json() == g.to_json()
    assert join_map(split_map(g, "add_map", 4), "add_map").to_json() == g.to_json()
    tail = vecadd(7)  # a tail: the record's valid count
    assert join_map(split_map(tail, "add_map", 4), "add_map").to_json() == tail.to_json()


def test_interchange_and_fuse_keep_each_maps_dimension_and_piece():
    """rowsum_2's swap keeps m_2_s and n_t; axpy's fused nest takes the first nest's names, then
    one split_map names both tasklets' indices n_1_t, n_2_s, and the inverses give it back."""
    g = tag_map(split_map(bound("rowsum", M=8, N=64), "sum_map", 4), "sum_map_1",
                "temporal")  # fmt: skip
    swapped = interchange_map(g, "sum_map_s", **{"with": "sum_map_1"})
    assert maps(swapped) == [("sum_map", "m_1_t", 1), ("sum_map_1", "n_t", 1),
                             ("sum_map_s", "m_2_s", 2)]  # fmt: skip
    a = bound("axpy", N=64)
    fused = fuse_map(a, "mult_map", "augassign_19_4_map")
    s = split_map(fused, "mult_map", 4)
    assert maps(s) == [("mult_map", "n_1_t", 1), ("mult_map_s", "n_2_s", 2)]
    assert s.node("augassign_19_4").inputs["in1"].subset == ["4 * n_1_t + n_2_s"]
    assert unfuse_map(join_map(s, "mult_map"), "mult_map").to_json() == a.to_json()


def test_rowsum_2s_steps():
    """m_1_t and m_2_s from the split, n_t from the tag; the bound unit reads
    A[4 * m_1_t:4 * m_1_t + 4, n_t]."""
    r = Recipe.load(recipe_path("rowsum_2"))
    graphs = [x.graph for x in apply_recipe(r, imported("rowsum"))]
    assert [maps(g) for g in graphs[1:4]] == [
        [("sum_map", "m_1_t", 1), ("sum_map_s", "m_2_s", 2), ("sum_map_1", "n", 1)],
        [("sum_map", "m_1_t", 1), ("sum_map_s", "m_2_s", 2), ("sum_map_1", "n_t", 1)],
        [("sum_map", "m_1_t", 1), ("sum_map_1", "n_t", 1), ("sum_map_s", "m_2_s", 2)],
    ]
    unit = graphs[-1].node("sum")
    assert unit.inputs["a"].subset == ["4 * m_1_t:4 * m_1_t + 4", "n_t"]
    assert unit.outputs["out"].subset == ["4 * m_1_t:4 * m_1_t + 4"]


def test_dot_4_drys_split_reduce_names():
    """The sum map split as split_map splits it, n_1_t around n_2_s, the partials out_partial[n_2_s];
    the final sum a dimension of its own, n_f, spatial: n_f_s. The two maps over N are siblings,
    each its own dimension n, so the fuse finds the same names on both; join_reduce gives the sum
    map back."""
    r, graphs = recipe_graphs("dot_4_dry")
    i = [s.transform for s in r.steps].index("split_reduce") + 1
    g = graphs[i]
    assert maps(g) == [("mult_map", "n_1_t", 1), ("mult_map_inner", "n_2_s", 2),
                       ("sum_map", "n_1_t", 1), ("sum_map_s", "n_2_s", 2),
                       ("sum_map_final", "n_f_s", 1)]  # fmt: skip
    assert g.node("sum").outputs["out"].subset == ["n_2_s"]
    assert g.node("sum_final").inputs["in1"].subset == ["n_f_s"]
    assert g.node("sum_map_final").attrs["loop.dim"] == "n_f"
    assert join_reduce(g, "sum_map").to_json() == graphs[i - 1].to_json()
    assert split_reduce(graphs[i - 1], "sum_map", 8).to_json() == g.to_json()


@pytest.mark.parametrize("name", RECIPES)
def test_every_recipe_applies_names_its_loops_and_checks(name):
    """Each step is checked against the reference executor (recipe_graphs applies them), and
    every map's variable is its rule's."""
    for g in recipe_graphs(name)[1]:
        again = Graph.from_dict(g.to_dict())
        loops.rename(again)  # a no-op on a graph named by the rule
        assert again.to_json() == g.to_json()
        assert all("loop.dim" in n.attrs for n, _ in g.walk() if n.kind == "map")
