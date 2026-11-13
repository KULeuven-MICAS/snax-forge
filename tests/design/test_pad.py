"""connect.pad and connect.pack's padded rows: a tail's padding is in the data."""

from __future__ import annotations

import copy

import pytest

from snax_forge.design import Design, Platform
from snax_forge.design.check import pad_content
from snax_forge.dfg import Graph
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.sandbox import Recipe, apply_recipe
from tests.helpers import dot1_split, imported, recipe_path

from .helpers import SNAX_FORGE, design, problems


def bound(name: str, **params) -> Graph:
    recipe = Recipe.load(recipe_path(name)).with_params(params)
    return apply_recipe(recipe, import_kernel(recipe.kernel))[-1].graph


def test_int32_rowsum_3x7_w4_is_valid_packed():
    """A split by 4 pads each row of 7 to 8: row stride 8, a multiple of k = 2."""
    d = bound("rowsum", W=4, M=3, N=7).to_dict()
    d["containers"]["A"]["dtype"] = "int32"  # y stays int64: one lane per word
    assert problems(design(Graph.from_dict(d))) == []


def _vmul_into_vsum(fold: str = "add") -> Graph:
    """vecmul's C, written by an accelerator, read through a tail by vsum's fold (its op
    ``fold``), both at W = 4 and N = 7."""
    mul = bound("vecmul", W=4, N=7).to_dict()
    vsum = bound("vsum_tree", W=4, N=7).to_dict()
    node = vsum["body"][0]["body"][0]
    node["attrs"]["instance"] = "acc2"
    node["inputs"]["a"]["data"] = "C"
    sub = node["attrs"]["replaced"]["body"][0]
    sub["inputs"]["in1"]["data"] = "C"
    if fold != "add":
        (out,) = sub["outputs"].values()
        out["reduce"] = {"op": fold, "identity": -(2**63)}
    mul["containers"]["out"] = copy.deepcopy(vsum["containers"]["out"])
    mul["body"] += vsum["body"]
    return Graph.from_dict(mul)


def test_vmul_into_vsum_at_n7_passes_as_its_pads_give_0():
    """The multiplier writes its pad lanes from A's and B's pads, 0 × 0 = 0, add's
    identity, so the sum reads them as the reference does."""
    g = _vmul_into_vsum()
    assert pad_content(g, "C") == 0
    assert [p for p in problems(design(g)) if p.code == "connect.pad"] == []


def test_vmul_into_a_max_fold_at_n7_is_refused():
    """The same pads, 0, into a max fold, whose identity is the minimum: refused as before."""
    got = [p for p in problems(design(_vmul_into_vsum("max"))) if p.code == "connect.pad"]
    assert [(p.code, p.where) for p in got] == [("connect.pad", "container C")]
    assert got[0].message == (
        "is written by mult and read through a tail by the fold sum: the producer's pad lanes "
        "would reach the fold (the hardware masks nothing)"
    )
    assert got[0].fix == "a size that divides W, or a split factor that divides n"


@pytest.mark.parametrize(("w_mul", "w_sum"), [(10, 10), (8, 10), (4, 4)])
def test_dot_1_with_a_factor_per_split(w_mul, w_sum):
    """The dot 1 dry run at N = 2048: at W_mul = W_sum = 10 the mul writes tmp0's pads
    (2050 words) from A's and B's, 0 × 0 = 0, so the sum may fold them: every design check
    passes; at 8 and 10 the mul writes 2048 words and never the 2 pads the sum reads, refused,
    the pad words never written; 4 and 4 have no tail."""
    g = apply_recipe(dot1_split(w_mul, w_sum), imported("dot"))[-1].graph
    found = problems(Design(Platform.load(SNAX_FORGE), g))
    pads = [p for p in found if p.code == "connect.pad"]
    if (w_mul, w_sum) != (8, 10):
        assert found == []
        assert pad_content(g, "tmp0") == 0
        return
    assert pad_content(g, "tmp0") == "unwritten"
    assert [(p.where, p.message) for p in pads] == [("container tmp0", (
        "is written by mult and read through a tail by the fold sum: the producer's pad lanes "
        "would reach the fold (the hardware masks nothing; they are never written, so the "
        "program would have to write the fold's identity there before the fold's task)"))]  # fmt: skip


def test_a_memlet_past_the_padding_is_refused_by_the_design_check():
    """vecadd N = 7, W = 4, A padded to 8: a hand-edited memlet reaching index 11 leaves the
    padding (the reference executor's rule, met at design time)."""
    d = bound("vecadd", W=4, N=7).to_dict()
    d["body"][0]["body"][0]["inputs"]["a"]["subset"] = ["4 * n_1_t + 4:4 * n_1_t + 8"]
    got = [p for p in problems(design(Graph.from_dict(d))) if p.code == "connect.pack"]
    assert [p.where for p in got] == ["node add.a"]
    assert got[0].message == (
        "memlet A['4 * n_1_t + 4:4 * n_1_t + 8'] reaches index 11 on dimension 0, past its padded "
        "extent 8 (the container's N, a tail's padding included)"
    )


def test_a_seeded_folds_pads_take_the_ops_identity():
    """matvec_seed at N = 510, W = 4 (a tail): the pads feed the tree, not the seed, so tmp0's
    pad lanes are add's identity, 0 (the multiplier's 0 × 0), and connect.pad passes; the seed
    q is read once per job, through no tail, and the run is q + A p."""
    import numpy as np

    from snax_forge.dfg import execute
    from snax_forge.dfg.pad import pad_value

    g = bound("matvec_seed", W=4, M=8, N=510)
    assert g.node("sum").inputs["seed"].to_dict() == {"data": "q", "subset": ["m_t:m_t + 1"]}
    assert (pad_content(g, "tmp0"), pad_value(g, "tmp0"), pad_value(g, "A")) == (0, 0, 0)
    assert [p for p in problems(design(g, edit_platform=None)) if p.code == "connect.pad"] == []
    rng = np.random.default_rng(1)
    a = rng.integers(-9, 9, (8, 510), dtype=np.int64)
    p, q = rng.integers(-9, 9, 510, dtype=np.int64), rng.integers(-9, 9, 8, dtype=np.int64)
    assert execute(g, {"A": a, "p": p, "q": q.copy()})["q"].tolist() == (q + a @ p).tolist()
