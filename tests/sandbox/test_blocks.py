"""Blocks (REUSE1, open item 117, blocks.py): a kernel's pattern found in a bigger graph, lifted
as a block, and a recipe applied there. mvt holds a dot in each row of each half and a mat-vec
in each half (half 2 through Aᵀ); matvec_dotK is matvec's variant K written as dot's variant K
on the rows."""

from __future__ import annotations

import json

import pytest

from snax_forge.sandbox import Recipe, SandboxError, apply_recipe
from snax_forge.sandbox.blocks import BLOCK, lift, matches, unlift
from snax_forge.sandbox.blocks import apply_recipe as apply_block
from snax_forge.sandbox.transforms import tag_map
from tests.helpers import imported, recipe_path

from .helpers import mvt


def _paths(g, kinds=("map",)) -> dict[str, list[str]]:
    return {n.id: [b["name"] for b in n.attrs.get(BLOCK, [])] for n, _ in g.walk()
            if n.kind in kinds}  # fmt: skip


def test_dot_matches_each_row_of_both_halves_of_mvt():
    found = matches(mvt(), "dot")
    assert [(m.heads, m.symbols, m.outer) for m in found] == [
        (["tmp0_mult_map_1", "sum_map_1"], {"N": "N"},
         {"tmp0_mult_map_1": ["tmp0_mult_map"], "sum_map_1": ["sum_map"]}),
        (["tmp2_mult_map_1", "sum_map_3"], {"N": "M"},
         {"tmp2_mult_map_1": ["tmp2_mult_map"], "sum_map_3": ["sum_map_2"]}),
    ]  # fmt: skip


def test_matvec_matches_half_2_through_a_permuted_a():
    found = matches(mvt(), "matvec_acc")
    assert [(m.heads[0], m.symbols, m.perms) for m in found] == [
        ("tmp0_mult_map", {"M": "M", "N": "N"}, {}),
        ("tmp2_mult_map", {"M": "N", "N": "M"}, {"A": [1, 0]}),
    ]


def test_a_kernel_not_in_the_graph_is_refused():
    assert matches(mvt(), "vecadd") == []
    with pytest.raises(SandboxError, match="no region of the graph matches vecadd"):
        lift(mvt(), "vecadd")


def test_lift_in_either_order_nests_dot_in_matvec():
    g = mvt()
    down, up = lift(lift(g, "matvec_acc"), "dot"), lift(lift(g, "dot"), "matvec_acc")
    assert _paths(down) == _paths(up)
    p = _paths(down)
    assert p["tmp0_mult_map"] == ["matvec_acc"]
    assert p["tmp0_mult_map_1"] == p["sum_map_1"] == ["matvec_acc", "dot"]
    assert p["tmp2_mult_map_1"] == ["matvec_acc_1", "dot_1"]


def test_unlift_gives_the_graph_back():
    g = mvt()
    lifted = lift(lift(g, "matvec_acc"), "dot")
    assert unlift(unlift(lifted, "dot"), "dot_1") == lift(g, "matvec_acc")
    with pytest.raises(SandboxError, match="no block of that name"):
        unlift(lifted, "mvt")


def test_apply_recipe_refuses_a_param_the_sub_recipe_lacks():
    with pytest.raises(SandboxError, match="dot_3 has no param 'K'"):
        apply_block(mvt(), "dot/dot_3", region=["tmp0_mult_map_1", "sum_map_1"], params={"K": 4})


def test_dot_3_on_a_lifted_block_names_its_group_after_the_block():
    g = lift(mvt(), "dot")
    for m in ("tmp2_mult_map", "sum_map_2"):  # the rows' loops, firing loops
        g = tag_map(g, m, "temporal")
    g = apply_block(g, "dot/dot_3", block="dot_1", params={"W": 4})
    units = {n.id: (n.attrs["instance"], n.attrs["group"]) for n, _ in g.walk()
             if n.kind == "accelerated"}  # fmt: skip
    assert units == {"tmp2_mult": ("mul", "dot_1"), "sum_1": ("sum", "dot_1")}
    rec = next(b for n, _ in g.walk() if n.id == "tmp2_mult" for b in n.attrs[BLOCK])
    assert rec == {"name": "dot_1", "kernel": "dot", "heads": ["tmp2_mult_map_1", "sum_map_3"],
                   "recipe": "dot/dot_3", "params": {"W": 4}}  # fmt: skip
    with pytest.raises(SandboxError, match="block dot_1 has dot/dot_3 already"):
        apply_block(g, "dot/dot_3", block="dot_1", params={"W": 4})


def _matvec_dot(k: int) -> Recipe:
    """matvec_dot3 with dot's variant ``k`` on the rows."""
    d = json.loads(recipe_path("matvec_dot3").read_text())
    d["name"] = f"matvec_dot{k}"
    d["steps"][-1]["params"]["recipe"] = f"dot/dot_{k}"
    return Recipe.from_dict(d)


def _graph(r: Recipe) -> dict:
    """The last graph's dict, the block records left out and the group named G (matvec_K names
    it matvec, matvec_dotK dot after its block)."""
    d = apply_recipe(r, imported(r.kernel, r.dtypes))[-1].graph.to_dict()

    def walk(x):
        if isinstance(x, dict):
            x.pop(BLOCK, None)
            if x.get("group") in ("dot", "matvec"):
                x["group"] = "G"
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(d)
    return d


@pytest.mark.slow
@pytest.mark.parametrize("k", range(2, 7))
def test_matvec_dot_k_is_matvec_k(k):
    """dot_4 to dot_6 fuse their two nests: on the rows, the fusion takes the rows' loop."""
    assert _graph(_matvec_dot(k)) == _graph(Recipe.load(recipe_path(f"matvec_{k}")))


@pytest.mark.slow
def test_mvt_from_matvec_dot3_on_both_halves_nests_the_blocks():
    r = Recipe.from_dict({"name": "mvt_try", "kernel": "mvt", "params": {"W": 4},
                          "symbols": {"M": 8, "N": 16}, "steps": [
        {"transform": "apply_recipe", "params": {"recipe": "matvec/matvec_dot3",
                                                 "region": heads, "params": {"W": "W"}}}
        for heads in (["tmp0_mult_map", "sum_map"], ["tmp2_mult_map", "sum_map_2"])]})  # fmt: skip
    g = apply_recipe(r, imported("mvt"))[-1].graph
    units = {n.id: (n.attrs["instance"], n.attrs["group"]) for n, _ in g.walk()
             if n.kind == "accelerated"}  # fmt: skip
    assert units == {"tmp0_mult": ("mul", "dot"), "sum": ("sum", "dot"),
                     "tmp2_mult": ("mul_1", "dot_1"), "sum_1": ("sum_1", "dot_1")}  # fmt: skip
    p = _paths(g, ("accelerated", "link"))
    assert p["tmp0_prod"] == ["matvec", "dot"] and p["sum_1"] == ["matvec_1", "dot_1"]
