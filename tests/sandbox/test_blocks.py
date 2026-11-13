"""Blocks (blocks.py): a kernel's pattern found in a bigger graph, lifted
as a block, and a recipe applied there. mvt holds a dot in each row of each half and a mat-vec
in each half (half 2 through Aᵀ); matvec_dotK is matvec's variant K written as dot's variant K
on the rows."""

from __future__ import annotations

import json

import pytest

from snax_forge.sandbox import Recipe, SandboxError, apply_recipe
from snax_forge.sandbox.blocks import APPLIED, BLOCK, lift, matches, unlift
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
    with pytest.raises(SandboxError, match="neither a block nor a node in one"):
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
    with pytest.raises(SandboxError, match="block dot_1 has dot/dot_3 applied"):
        apply_block(g, "dot/dot_3", block="dot_1", params={"W": 4})


def _matvec_dot(k: int) -> Recipe:
    """recipes/matvec/matvec_dot<k>.json: dot's variant ``k`` on the rows."""
    return Recipe.load(recipe_path(f"matvec_dot{k}"))


def _graph(r: Recipe) -> dict:
    """The last graph's dict, the block records left out and the group named G (matvec_K names
    it matvec, matvec_dotK dot after its block)."""
    d = apply_recipe(r, imported(r.kernel, r.dtypes))[-1].graph.to_dict()

    def walk(x):
        if isinstance(x, dict):
            x.pop(BLOCK, None)
            x.pop(APPLIED, None)
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


# -- the sites and the editor's forms (round 2) ---------------------------------------------------


def test_lift_offers_the_kernels_matching_at_a_node_and_lifts_that_region_only():
    from snax_forge.sandbox.blocks import lift_choices

    g = mvt()
    assert lift_choices(g, "tmp2_mult") == ["dot", "matvec", "matvec_acc", "vecmul"]
    assert lift_choices(g, "augassign_19_4") == ["matvec_acc"]
    one = lift(g, "dot", at="tmp2_mult")
    assert {b for p in _paths(one).values() for b in p} == {"dot"}
    assert _paths(one)["sum_map_3"] == ["dot"]
    with pytest.raises(SandboxError, match="lift at 'tmp0_mult': vecadd does not match here"):
        lift(g, "vecadd", at="tmp0_mult")


def _matvec_session():
    """A session on matvec, the rows' loops tagged, then dot lifted at mult."""
    from snax_forge.sandbox.session import Session

    s = Session.from_kernel("matvec", {"M": 8, "N": 64})
    for m in ("mult_map", "sum_map"):
        assert s.apply("tag_map", {"map": m, "kind": "temporal"})["ok"]
    site = s.sites(s.cursor, "mult")["lift"]
    assert site["params"] == {"at": "mult", "kernel": "dot"}
    assert [c["value"] for c in site["form"][0]["choices"]] == ["dot", "vecmul"]
    assert s.apply("lift", site["params"])["ok"]
    return s


def test_apply_recipe_from_the_editor_on_a_lifted_block():
    s = _matvec_session()
    a = s.sites(s.cursor, "dot")["apply_recipe"]
    assert a["params"] == {"block": "dot", "recipe": "dot/dot_1"}
    fields = {f["name"]: f for f in a["form"]}
    assert fields["block"]["readonly"] and fields["params.W"]["value"] == "W"
    assert fields["params.W"]["new_value"] == 4 and "dot/dot_3" in [
        c["value"] for c in fields["recipe"]["choices"]
    ]
    r = s.apply("apply_recipe", {"block": "dot", "recipe": "dot/dot_3", "params": {"W": "W"}},
                new_params={"W": 4})  # fmt: skip
    assert r["ok"] and s.recipe.params["W"] == 4
    assert s.set_params({"W": 8})["ok"]  # the session's W drives the sub-recipe's
    units = [n.attrs["instance"] for n, _ in s.graphs[s.cursor].walk() if n.kind == "accelerated"]
    assert units == ["mul", "sum"]
    assert not s.sites(s.cursor, "dot")["apply_recipe"]["ok"]  # applied already
    assert s.sites(s.cursor, "dot")["unlift"]["message"].endswith("applied (unapply)")


def test_unlift_from_a_node_in_the_block():
    s = _matvec_session()
    u = s.sites(s.cursor, "mult")["unlift"]
    assert u["params"] == {"name": "dot"}
    assert s.apply("unlift", u["params"])["ok"]
    assert s.graphs[s.cursor] == s.graphs[2]


# -- round 3: unapply, name_block, blocks a sub-recipe makes, mvt from matvec recipes -----------


def _rows_tagged():
    g = mvt()
    for m in ("tmp0_mult_map", "sum_map"):
        g = tag_map(g, m, "temporal")
    return g


def test_unapply_gives_back_the_graph_before_the_recipe():
    from snax_forge.sandbox.blocks import unapply

    g = _rows_tagged()
    a = apply_block(g, "dot/dot_3", region=["tmp0_mult_map_1", "sum_map_1"], params={"W": 4})
    assert unapply(a, "dot") == g
    lifted = lift(g, "dot", at="tmp0_mult")  # dot_5 fuses the rows' loops: undone too
    fused = apply_block(lifted, "dot/dot_5", block="dot", params={"W": 4})
    back = unapply(fused, "tmp0_mult")  # a node in it names it
    assert back == lifted and list(back.containers) == list(lifted.containers)
    other = tag_map(a, "tmp2_mult_map", "temporal")  # a change elsewhere is kept
    assert unapply(other, "dot") == tag_map(g, "tmp2_mult_map", "temporal")


def test_unapply_refuses_a_block_changed_since_or_inside_another():
    from snax_forge.sandbox.blocks import unapply

    a = apply_block(_rows_tagged(), "dot/dot_3", region=["tmp0_mult_map_1", "sum_map_1"],
                    params={"W": 4})  # fmt: skip
    from snax_forge.sandbox.compose import unlink

    changed = unlink(a, "tmp0_prod")
    with pytest.raises(SandboxError, match="a step after its recipe changed block dot"):
        unapply(changed, "dot")
    m = apply_block(mvt(), "matvec/matvec_dot3", region=["tmp0_mult_map", "sum_map"],
                    params={"W": 4})  # fmt: skip
    with pytest.raises(SandboxError, match="inside matvec, which matvec/matvec_dot3 made"):
        unapply(m, "dot")
    assert unapply(m, "matvec") == mvt()


def test_name_block_names_a_region_by_hand():
    from snax_forge.sandbox.blocks import name_block

    g = name_block(mvt(), ["tmp0_mult_map", "sum_map"], "half1")
    assert _paths(g)["sum_map_1"] == ["half1"] and _paths(g)["tmp2_mult_map"] == []
    with pytest.raises(SandboxError, match="'half1' is taken"):
        name_block(g, ["augassign_18_4_map"], "half1")
    with pytest.raises(SandboxError, match="no kernel to apply a recipe of"):
        apply_block(g, "dot/dot_3", block="half1")
    assert lift(g, "dot", at="tmp0_mult") is not None  # a block inside one named by hand


def test_a_sub_recipe_lifts_and_applies_by_the_blocks_name(tmp_path):
    """matvec written as lift dot, then dot_3 on block dot: on mvt's half 2 the block is
    dot_1 (half 1 has dot), and the step's block follows it."""
    r = tmp_path / "matvec_lift3.json"
    r.write_text(json.dumps({"name": "matvec_lift3", "kernel": "matvec", "params": {"W": 4, "M": 16, "N": 512},
                             "symbols": {"M": "M", "N": "N"}, "steps": [
        {"transform": "tag_map", "params": {"map": "mult_map", "kind": "temporal"}},
        {"transform": "tag_map", "params": {"map": "sum_map", "kind": "temporal"}},
        {"transform": "lift", "params": {"kernel": "dot"}},
        {"transform": "apply_recipe", "params": {"recipe": "dot/dot_3", "block": "dot",
                                                 "params": {"W": "W"}}}]}))  # fmt: skip
    g = lift(mvt(), "dot", at="tmp0_mult")  # dot taken in half 1
    g = apply_block(g, str(r), region=["tmp2_mult_map", "sum_map_2"], params={"W": 4})
    p = _paths(g, ("tasklet", "accelerated"))
    assert p["tmp2_mult"] == ["matvec", "dot_1"] and p["tmp0_mult"] == ["dot"]


@pytest.mark.slow
@pytest.mark.parametrize(("reuse", "hand"), [("mvt_1_reuse", "mvt_1"),
                                             ("mvt_1_dot2_reuse", "mvt_1_dot2"),
                                             ("mvt_2_reuse", "mvt_2")])  # fmt: skip
def test_mvt_from_matvec_recipes_is_the_hand_written_mvt(reuse, hand):
    """Both halves from matvec_dot1 or matvec_dot2, the vadd and the sharing by hand: the
    graph of the hand-written recipe, the units' and groups' names aside."""

    def names(g):
        d = g.to_dict()
        seen: dict[str, str] = {}

        def walk(x):
            if isinstance(x, dict):
                for k in (BLOCK, APPLIED, "sandbox.shared"):
                    x.pop(k, None)
                for k in ("instance", "group", "from", "to", "accelerator"):
                    if "kind" not in x and isinstance(x.get(k), str):
                        x[k] = seen.setdefault(x[k], f"u{len(seen)}")
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)

        walk(d)
        return d

    graphs = [apply_recipe(Recipe.load(recipe_path(x)), imported("mvt"))[-1].graph
              for x in (reuse, hand)]  # fmt: skip
    assert names(graphs[0]) == names(graphs[1])
