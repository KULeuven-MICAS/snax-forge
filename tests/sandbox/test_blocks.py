"""Blocks (blocks.py): a kernel's pattern found in a bigger graph, lifted
as a block, and a recipe applied there. mvt holds a dot in each row of each half and a mat-vec
in each half (half 2 through Aᵀ); matvec_dotK is matvec's variant K written as dot's variant K
on the rows."""

from __future__ import annotations

import json

import pytest

from snax_forge.dfg import Graph
from snax_forge.sandbox import SandboxError
from snax_forge.sandbox.blocks import BLOCK, lift, matches, name_block
from snax_forge.sandbox.blocks import apply_recipe as apply_block
from snax_forge.sandbox.transforms import split_map, tag_map

from .helpers import mvt


def _paths(g, kinds=("map",)) -> dict[str, list[str]]:
    return {n.id: [b["name"] for b in n.attrs.get(BLOCK, [])] for n, _ in g.walk()
            if n.kind in kinds}  # fmt: skip


# -- the sites and the editor's forms (round 2) ---------------------------------------------------


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


def test_the_editor_names_an_unknown_param_inside_apply_recipe_and_lights_the_fused_maps():
    """params.W naming a param the session lacks is refused as a param, its fix said; dot_v4
    fuses the rows' loops outside the block, so its choice lights them too."""
    s = _matvec_session()
    r = s.apply("apply_recipe", {"block": "dot", "recipe": "dot/dot_v3", "params": {"W": "Wq"}})
    assert (r["ok"], r["code"], r["param"]) == (False, "session.param", "Wq")
    assert r["message"] == "apply_recipe: params.W 'Wq' is not a param; give its value"
    a = s.sites(s.cursor, "dot")["apply_recipe"]
    nodes = {c["value"]: c["nodes"] for c in {f["name"]: f for f in a["form"]}["recipe"]["choices"]}
    assert nodes["dot/dot_v4"][-2:] == ["mult_map", "sum_map"]
    assert "mult_map" not in nodes["dot/dot_v3"]


def test_a_lifted_block_s_refusals_say_the_fix():
    g = lift(mvt(), "dot", at="tmp0_mult")
    heads = ["tmp0_mult_map_1", "sum_map_1"]
    with pytest.raises(SandboxError, match="in block dot already: give block 'dot' instead"):
        apply_block(g, "dot/dot_v3", region=heads, params={"W": 4})
    with pytest.raises(SandboxError, match="give region or block, not both"):
        apply_block(g, "dot/dot_v3", block="dot", region=heads, params={"W": 4})
    with pytest.raises(SandboxError, match="already in block dot, a dot"):
        lift(g, "dot", at="tmp0_mult")
    with pytest.raises(SandboxError, match="transformed since its lift: unlift it"):
        apply_block(split_map(g, "tmp0_mult_map_1", 4), "dot/dot_v3", block="dot", params={"W": 4})


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
    a = apply_block(g, "dot/dot_v3", region=["tmp0_mult_map_1", "sum_map_1"], params={"W": 4})
    assert unapply(a, "dot") == g
    lifted = lift(g, "dot", at="tmp0_mult")  # dot_v5 fuses the rows' loops: undone too
    fused = apply_block(lifted, "dot/dot_v5", block="dot", params={"W": 4})
    back = unapply(fused, "tmp0_mult")  # a node in it names it
    assert back == lifted and list(back.containers) == list(lifted.containers)
    other = tag_map(a, "tmp2_mult_map", "temporal")  # a change elsewhere is kept
    assert unapply(other, "dot") == tag_map(g, "tmp2_mult_map", "temporal")


def test_unapply_refuses_a_block_changed_since_or_inside_another():
    from snax_forge.sandbox.blocks import unapply

    a = apply_block(_rows_tagged(), "dot/dot_v3", region=["tmp0_mult_map_1", "sum_map_1"],
                    params={"W": 4})  # fmt: skip
    from snax_forge.sandbox.compose import unlink

    changed = unlink(a, "tmp0_prod")
    with pytest.raises(SandboxError, match="a step after its recipe changed block dot"):
        unapply(changed, "dot")
    m = apply_block(mvt(), "matvec/matvec_v3_1", region=["tmp0_mult_map", "sum_map"],
                    params={"W": 4})  # fmt: skip
    with pytest.raises(SandboxError, match="inside matvec, which matvec/matvec_v3_1 made"):
        unapply(m, "dot")
    assert unapply(m, "matvec") == mvt()


def test_name_block_names_a_region_by_hand():

    g = name_block(mvt(), ["tmp0_mult_map", "sum_map"], "half1")
    assert _paths(g)["sum_map_1"] == ["half1"] and _paths(g)["tmp2_mult_map"] == []
    with pytest.raises(SandboxError, match="'half1' is taken"):
        name_block(g, ["augassign_18_4_map"], "half1")
    with pytest.raises(SandboxError, match="no kernel to apply a recipe of"):
        apply_block(g, "dot/dot_v3", block="half1")
    assert lift(g, "dot", at="tmp0_mult") is not None  # a block inside one named by hand


def test_a_sub_recipe_lifts_and_applies_by_the_blocks_name(tmp_path):
    """matvec written as lift dot, then dot_v3 on block dot: on mvt's half 2 the block is
    dot_1 (half 1 has dot), and the step's block follows it."""
    r = tmp_path / "matvec_lift3.json"
    r.write_text(json.dumps({"name": "matvec_lift3", "kernel": "matvec", "params": {"W": 4, "M": 16, "N": 512},
                             "symbols": {"M": "M", "N": "N"}, "steps": [
        {"transform": "tag_map", "params": {"map": "mult_map", "kind": "temporal"}},
        {"transform": "tag_map", "params": {"map": "sum_map", "kind": "temporal"}},
        {"transform": "lift", "params": {"kernel": "dot"}},
        {"transform": "apply_recipe", "params": {"recipe": "dot/dot_v3", "block": "dot",
                                                 "params": {"W": "W"}}}]}))  # fmt: skip
    g = lift(mvt(), "dot", at="tmp0_mult")  # dot taken in half 1
    g = apply_block(g, str(r), region=["tmp2_mult_map", "sum_map_2"], params={"W": 4})
    p = _paths(g, ("tasklet", "accelerated"))
    assert p["tmp2_mult"] == ["matvec", "dot_1"] and p["tmp0_mult"] == ["dot"]


def test_a_container_used_in_two_orders_is_not_the_pattern():
    """The products written [m, n] and read [n, m]: no mat-vec, nor a dot per row, in half 1."""
    d = mvt().to_dict()

    def walk(x):
        if isinstance(x, dict):
            if x.get("id") == "sum":
                x["inputs"]["in1"]["subset"] = ["n", "m"]
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(d)
    g = Graph.from_dict(d)
    for kernel, head in (("matvec", "tmp2_mult_map"), ("dot", "tmp2_mult_map_1")):
        assert [m.heads[0] for m in matches(g, kernel)] == [head]
