"""One instance shared by several nodes: mvt's variant 1, both halves on one vmul, vsum
and vadd (recipes/mvt/mvt_1.json, at its M = N = 40), the sessions at M = 8, N = 16. bind's form
offers the instances a node
can join (session.py's ``share_choices``), and a joining split's editor-named W takes the
instance's param."""

from __future__ import annotations

import pytest

from snax_forge.sandbox.session import Session, share_choices
from tests.helpers import steps

from .helpers import SIZES

SIMD = {"brm": "simd", "implementation": "chisel_tiled_spatial"}


def half_1_bound(w2: int) -> Session:
    """A session on mvt: half 1's multiply bound as vmul_x1 at W = 4, its sum as vsum_x1, half
    2's multiply tagged and split at ``w2`` (a param of its own, W_2)."""
    s = Session.from_kernel("mvt", dict(SIZES))
    for t, p, new in [
        ("tag_map", {"map": "tmp0_mult_map", "kind": "temporal"}, None),
        ("split_map", {"map": "tmp0_mult_map_1", "factor": "W"}, {"W": 4}),
        ("bind", {"node": "tmp0_mult", **SIMD, "instance": "vmul_x1"}, None),
        ("tag_map", {"map": "sum_map", "kind": "temporal"}, None),
        ("split_map", {"map": "sum_map_1", "factor": "W_vmul_x1"}, None),
        ("bind", {"node": "sum", "brm": "accumulate", "implementation": "chisel_adder_tree",
                  "instance": "vsum_x1"}, None),
        ("tag_map", {"map": "tmp2_mult_map", "kind": "temporal"}, None),
        ("split_map", {"map": "tmp2_mult_map_1", "factor": "W_2"}, {"W_2": w2}),
    ]:  # fmt: skip
        assert s.apply(t, p, new_params=new)["ok"], (t, p)
    return s


def test_a_node_may_share_an_instance_of_its_brm_at_the_same_params():
    s = half_1_bound(4)
    share = share_choices(s.graphs[s.cursor], "tmp2_mult")
    # vsum_x1 (accumulate) cannot take a multiply at all: no choice
    assert [(c["value"], c["ok"]) for c in share] == [("vmul_x1", True)]
    field = next(
        f for f in s.sites(s.cursor, "tmp2_mult")["bind"]["form"] if f["name"] == "instance"
    )
    assert [c["value"] for c in field["share"]] == ["vmul_x1"]


def test_another_w_is_refused_naming_it():
    s = half_1_bound(8)
    (c,) = share_choices(s.graphs[s.cursor], "tmp2_mult")
    assert not c["ok"] and c["message"] == "vmul_x1 has W = 4, tmp2_mult would have 8"
    r = s.apply("bind", {"node": "tmp2_mult", **SIMD, "instance": "vmul_x1"})
    assert not r["ok"] and "vmul_x1" in r["message"]


def test_a_joining_split_takes_the_instances_param():
    s = half_1_bound(4)
    r = s.apply("bind", {"node": "tmp2_mult", **SIMD, "instance": "vmul_x1"})
    assert r["ok"] and r["renamed"] == {"W_2": "W_vmul_x1"}
    assert "W_2" not in s.recipe.params and s.recipe.params["W_vmul_x1"] == 4
    splits = [st.params["factor"] for st in s.recipe.steps if st.transform == "split_map"]
    assert splits == ["W_vmul_x1", "W_vmul_x1", "W_vmul_x1"]
    # one param drives both halves
    assert s.set_params({"W_vmul_x1": 8})["ok"]


@pytest.mark.slow
def test_mvt_1_runs_both_halves_on_three_accelerators():
    g = steps("mvt_1")[-1]
    by: dict[str, list[str]] = {}
    for n, _ in g.walk():
        if n.kind == "accelerated":
            by.setdefault(n.attrs["instance"], []).append(n.id)
    assert by == {"vmul": ["tmp0_mult", "tmp2_mult"], "vsum": ["sum", "sum_1"],
                  "vadd": ["augassign_18_4", "augassign_19_4"]}  # fmt: skip


# -- share and unshare: a linked group run on another's units (share.py) -------------------------


def _linked_halves() -> Session:
    """A session on mvt: each half's multiply and sum bound (vmul_xh, vsum_xh) and linked at
    depth 1 as dot_xh, each half with its own W (the editor's names: W_vmul_x1, W_vmul_x2)."""
    s = Session.from_kernel("mvt", dict(SIZES))
    acc = {"brm": "accumulate", "implementation": "chisel_adder_tree"}
    names = [("tmp0_mult_map", "tmp0_mult_map_1", "tmp0_mult", "sum_map", "sum_map_1", "sum",
              "tmp0_prod"),
             ("tmp2_mult_map", "tmp2_mult_map_1", "tmp2_mult", "sum_map_2", "sum_map_3", "sum_1",
              "tmp2_prod")]  # fmt: skip
    for h, (m1, m2, mul, sm, s1, fold, prod) in enumerate(names, 1):
        w = "W" if h == 1 else "W_2"
        for t, p, new in [
            ("tag_map", {"map": m1, "kind": "temporal"}, None),
            ("split_map", {"map": m2, "factor": w}, {w: 4}),
            ("bind", {"node": mul, **SIMD, "instance": f"vmul_x{h}"}, None),
            ("tag_map", {"map": sm, "kind": "temporal"}, None),
            ("split_map", {"map": s1, "factor": f"W_vmul_x{h}"}, None),
            ("bind", {"node": fold, **acc, "instance": f"vsum_x{h}"}, None),
            ("link", {"container": prod, "depth": 1, "group": f"dot_x{h}"}, None),
        ]:  # fmt: skip
            assert s.apply(t, p, new_params=new)["ok"], (t, p)
    return s


def test_share_runs_half_2s_group_on_half_1s_and_unshare_takes_it_back():
    from snax_forge.design.streamers import group_links, group_runs, groups

    s = _linked_halves()
    before = s.graphs[s.cursor]
    site = s.sites(s.cursor, "dot_x2")["share"]
    assert site["ok"] and site["params"] == {"group": "dot_x2", "with": "dot_x1"}
    r = s.apply("share", {"group": "dot_x2", "with": "dot_x1"})
    assert r["ok"] and r["renamed"] == {"W_vmul_x2": "W_vmul_x1", "D_2": "D"}
    g = s.graphs[s.cursor]
    assert groups(g) == {"dot_x1": ["vmul_x1", "vsum_x1"]}
    runs = group_runs(g)["dot_x1"]
    assert [{u: n.id for u, n in run.items()} for run in runs] == [
        {"vmul_x1": "tmp0_mult", "vsum_x1": "sum"},
        {"vmul_x1": "tmp2_mult", "vsum_x1": "sum_1"},
    ]
    assert [(k.attrs["container"], a, b) for k, a, b in group_links(g, ["vmul_x1", "vsum_x1"])] == [
        ("tmp0_prod", "vmul_x1.out", "vsum_x1.a")
    ]  # one hardware link for both nests
    assert s.set_params({"W_vmul_x1": 8})["ok"]  # one W drives both halves
    assert s.set_params({"W_vmul_x1": 4})["ok"]
    assert s.sites(s.cursor, "tmp2_mult")["unshare"]["params"] == {"group": "dot_x2"}
    assert s.apply("unshare", {"group": "dot_x2"})["ok"]
    back = s.graphs[s.cursor].to_dict()
    want = before.to_dict()
    # the W rename stays in the recipe (one param), the graph is the one before share
    assert back == want


def test_share_refuses_another_shape_naming_it():
    from snax_forge.sandbox.share import share_choices as group_choices

    s = _linked_halves()
    assert s.set_params({"W_vmul_x1": 8})["ok"]  # half 1 at 8 lanes, half 2 at 4
    (c,) = group_choices(s.graphs[s.cursor], "dot_x2")
    assert not c["ok"] and c["message"] == "vmul_x2 and vmul_x1 differ: W = 4 against 8"
    r = s.apply("share", {"group": "dot_x2", "with": "dot_x1"})
    assert not r["ok"] and r["message"].endswith(
        "share 'dot_x2': no group of its shape to share with"
    )


@pytest.mark.slow
def test_mvt_1_dot2_is_one_dot_group_run_twice():
    from tests.estimate.helpers import estimated

    e = estimated("mvt_1_dot2")
    assert [(g.name, g.cycles) for g in e.groups] == [("mvt", 804), ("vadd", 20)]
    two = estimated("mvt_2")
    assert e.accelerator_cycles == two.accelerator_cycles == 824
