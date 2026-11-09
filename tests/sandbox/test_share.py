"""One instance shared by several nodes (D165): mvt's variant 1, both halves on one vmul, vsum
and vadd (recipes/mvt/mvt_1.json), at M = 8, N = 16. bind's form offers the instances a node
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
