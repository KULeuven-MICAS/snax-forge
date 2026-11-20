"""mvt's half 2 in the column form: a ``mac`` with ``y2`` broadcast (D186) and a seed (D187).

mvt (``x2 += A.T @ y2``) at M = 8, N = 16 (helpers' SIZES): the multiply and the fold's nests
with the lanes on the output index n (strips of W = 4 columns, m in time between the strip and
the lanes), the fold seeded from x2, fused, bound as ``mac``. ``bind`` reads ``y2[m]``, which
does not use the lane variable, as one lane given to all of them, and the seed as the output's
four lanes; every step equals the input graph on the reference check. The refusals: a broadcast
on a datapath that sums its lanes, and a seed on a BRM with none.
"""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.dfg import execute
from snax_forge.sandbox import Recipe, SandboxError, apply_recipe, bind
from tests.sandbox.helpers import SIZES, mvt, step

MAC = {"brm": "mac", "implementation": "chisel_mac", "instance": "mac_x2"}
COLUMN = [
    step("tag_map", map="tmp2_mult_map_1", kind="temporal"),
    step("split_map", map="tmp2_mult_map", factor=4),
    step("interchange_map", map="tmp2_mult_map_s", **{"with": "tmp2_mult_map_1"}),
    step("tag_map", map="sum_map_3", kind="temporal"),
    step("split_map", map="sum_map_2", factor=4),
    step("interchange_map", map="sum_map_2_s", **{"with": "sum_map_3"}),
]
SEED = step("fuse_seed", first="sum_map_2_s", second="augassign_19_4_map")
FUSE = step("fuse_map", first="tmp2_mult_map", second="sum_map_2")


def graphs(steps: list) -> list:
    r = Recipe.from_dict({"name": "col", "kernel": "mvt", "symbols": SIZES, "steps": steps})
    return [res.graph for res in apply_recipe(r, mvt())]


def inputs() -> dict:
    rng = np.random.default_rng(7)
    m, n = SIZES["M"], SIZES["N"]
    a = rng.integers(-50, 50, (m, n))
    return {"A": a, "y1": rng.integers(-9, 9, n), "y2": rng.integers(-9, 9, m),
            "x1": rng.integers(-99, 99, m), "x2": rng.integers(-99, 99, n)}  # fmt: skip


def test_the_column_form_binds_a_seeded_mac_with_y2_broadcast():
    *_, g = graphs([*COLUMN, SEED, FUSE, step("bind", node="tmp2_mult_map_s", **MAC)])
    (node,) = (n for n, _ in g.walk() if n.kind == "accelerated")
    assert node.attrs["params"] == {"W": 4, "op": "add", "seed": 1}
    assert node.inputs["a"].subset == ["m_t", "4 * n_1_t:4 * n_1_t + 4"]
    assert node.inputs["b"].subset == ["m_t:m_t + 1"]  # one lane, given to all four
    assert node.inputs["seed"].subset == ["4 * n_1_t:4 * n_1_t + 4"]  # the output's lanes
    assert node.outputs["out"].subset == ["4 * n_1_t:4 * n_1_t + 4"]
    x = inputs()
    want = x["x2"] + x["A"].T @ x["y2"]
    assert execute(g, {k: v.copy() for k, v in x.items()})["x2"].tolist() == want.tolist()


def test_without_the_seed_the_mac_binds_too():
    """The broadcast alone (the linked vadd's form): an unseeded mac, x2's add left as it is."""
    *_, g = graphs([*COLUMN, FUSE, step("bind", node="tmp2_mult_map_s", **MAC)])
    (node,) = (n for n, _ in g.walk() if n.kind == "accelerated")
    assert "seed" not in node.inputs and node.inputs["b"].subset == ["m_t:m_t + 1"]


def test_a_broadcast_needs_a_datapath_whose_lanes_compute_on_their_own(monkeypatch):
    """With Mac registered as summing its lanes, y2 is refused, the fix named."""
    from dataclasses import replace

    from snax_forge.gen import blocks

    *_, g = graphs([*COLUMN, SEED, FUSE])
    monkeypatch.setitem(blocks.DATAPATHS, "Mac", replace(blocks.DATAPATHS["Mac"], lanewise=False))
    with pytest.raises(
        SandboxError,
        match=r"in2 y2\['m_t'\] does not use the spatial variable "
        r"'n_2_s'; only an input of a datapath whose lanes compute on their own",
    ):
        bind(g, "tmp2_mult_map_s", **MAC)


def test_a_seed_needs_a_brm_with_a_seed_port():
    *_, g = graphs([*COLUMN, SEED, FUSE])
    with pytest.raises(SandboxError, match="'simd' has no seed port"):
        bind(g, "tmp2_mult_map_s", "simd", "chisel_tiled_spatial", "s")
