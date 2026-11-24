"""connect.pad and connect.pack's padded rows: a tail's padding is in the data."""

from __future__ import annotations

import copy

from snax_forge.design.check import pad_content
from snax_forge.dfg import Graph
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.sandbox import Recipe, apply_recipe
from tests.helpers import recipe_path

from .helpers import design, problems


def bound(name: str, **params) -> Graph:
    recipe = Recipe.load(recipe_path(name)).with_params(params)
    return apply_recipe(recipe, import_kernel(recipe.kernel))[-1].graph


def _vmul_into_vsum(fold: str = "add") -> Graph:
    """vecmul's C, written by an accelerator, read through a tail by vsum's fold (its op
    ``fold``), both at W = 4 and N = 7."""
    mul = bound("vecmul_v1", W=4, N=7).to_dict()
    vsum = bound("vsum_v1_1", W=4, N=7).to_dict()
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


def test_vmul_into_a_max_fold_at_n7_is_refused():
    """The same pads, 0, into a max fold, whose identity is the minimum: refused as before."""
    got = [p for p in problems(design(_vmul_into_vsum("max"))) if p.code == "connect.pad"]
    assert [(p.code, p.where) for p in got] == [("connect.pad", "container C")]
    assert got[0].message == (
        "is written by mult and read through a tail by the fold sum: the producer's pad lanes "
        "would reach the fold (the hardware masks nothing)"
    )
    assert got[0].fix == "a size that divides W, or a split factor that divides n"


def test_a_memlet_past_the_padding_is_refused_by_the_design_check():
    """vecadd N = 7, W = 4, A padded to 8: a hand-edited memlet reaching index 11 leaves the
    padding (the reference executor's rule, met at design time)."""
    d = bound("vecadd_v1", W=4, N=7).to_dict()
    d["body"][0]["body"][0]["inputs"]["a"]["subset"] = ["4 * n_1_t + 4:4 * n_1_t + 8"]
    got = [p for p in problems(design(Graph.from_dict(d))) if p.code == "connect.pack"]
    assert [p.where for p in got] == ["node add.a"]
    assert got[0].message == (
        "memlet A['4 * n_1_t + 4:4 * n_1_t + 8'] reaches index 11 on dimension 0, past its padded "
        "extent 8 (the container's N, a tail's padding included)"
    )


def test_a_seeded_folds_pads_take_the_ops_identity():
    """matvec_acc_v1 at N = 510, W = 4 (a tail): the pads feed the tree, not the seed, so tmp0's
    pad lanes are add's identity, 0 (the multiplier's 0 × 0), and connect.pad passes; the seed
    q is read once per job, through no tail, and the run is q + A p."""
    import numpy as np

    from snax_forge.dfg import execute
    from snax_forge.dfg.pad import pad_value

    g = bound("matvec_acc_v1", W=4, M=8, N=510)
    assert g.node("sum").inputs["seed"].to_dict() == {"data": "q", "subset": ["m_t:m_t + 1"]}
    assert (pad_content(g, "tmp0"), pad_value(g, "tmp0"), pad_value(g, "A")) == (0, 0, 0)
    assert [p for p in problems(design(g, edit_platform=None)) if p.code == "connect.pad"] == []
    rng = np.random.default_rng(1)
    a = rng.integers(-9, 9, (8, 510), dtype=np.int64)
    p, q = rng.integers(-9, 9, 510, dtype=np.int64), rng.integers(-9, 9, 8, dtype=np.int64)
    assert execute(g, {"A": a, "p": p, "q": q.copy()})["q"].tolist() == (q + a @ p).tolist()
