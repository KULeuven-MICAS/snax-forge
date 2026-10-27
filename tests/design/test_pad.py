"""connect.pad and connect.pack's padded rows (D144): a tail's padding is in the data."""

from __future__ import annotations

import copy

from snax_forge.dfg import Graph
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.sandbox import Recipe, apply_recipe

from .helpers import REPO, design, problems


def bound(name: str, **params) -> Graph:
    recipe = Recipe.load(REPO / "recipes" / f"{name}.json").with_params(params)
    return apply_recipe(recipe, import_kernel(recipe.kernel))[-1].graph


def test_int32_rowsum_3x7_w4_is_valid_packed():
    """A split by 4 pads each row of 7 to 8: row stride 8, a multiple of k = 2."""
    d = bound("rowsum", W=4, M=3, N=7).to_dict()
    d["containers"]["A"]["dtype"] = "int32"  # y stays int64: one lane per word
    assert problems(design(Graph.from_dict(d))) == []


def test_vmul_into_vsum_at_n7_is_refused():
    """vecmul's C, written by an accelerator, read through a tail by vsum's fold: the
    multiplier's pad lanes (whatever it computed there) would be summed."""
    mul = bound("vecmul", W=4, N=7).to_dict()
    vsum = bound("vsum_tree", W=4, N=7).to_dict()
    node = vsum["body"][0]["body"][0]
    node["attrs"]["instance"] = "acc2"
    node["inputs"]["a"]["data"] = "C"
    node["attrs"]["replaced"]["body"][0]["inputs"]["in1"]["data"] = "C"
    mul["containers"]["out"] = copy.deepcopy(vsum["containers"]["out"])
    mul["body"] += vsum["body"]
    got = [p for p in problems(design(Graph.from_dict(mul))) if p.code == "connect.pad"]
    assert [(p.code, p.where) for p in got] == [("connect.pad", "container C")]
    assert got[0].message == (
        "is written by mult and read through a tail by the fold sum: the producer's pad lanes "
        "would reach the fold (the hardware masks nothing, D144)"
    )
    assert got[0].fix == "a size that divides W, or a split factor that divides n"


def test_a_memlet_past_the_padding_is_refused_by_the_design_check():
    """vecadd N = 7, W = 4, A padded to 8: a hand-edited memlet reaching index 11 leaves the
    padding (the reference executor's rule, D144, met at design time)."""
    d = bound("vecadd", W=4, N=7).to_dict()
    d["body"][0]["body"][0]["inputs"]["a"]["subset"] = ["4 * i_t + 4:4 * i_t + 8"]
    got = [p for p in problems(design(Graph.from_dict(d))) if p.code == "connect.pack"]
    assert [p.where for p in got] == ["node add.a"]
    assert got[0].message == (
        "memlet A['4 * i_t + 4:4 * i_t + 8'] reaches index 11 on dimension 0, past its padded "
        "extent 8 (the container's N, a tail's padding included, D144)"
    )
