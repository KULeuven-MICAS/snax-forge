"""data_transpose's copy in the DFG view (api.py): a block "transpose · dma" between A and A_T,
in half 2's column, half 2's multiply reading A_T; A drawn once above both columns, as
mark_transpose's container is, so half 1 and the copy read the same A."""

from __future__ import annotations

from pathlib import Path

from snax_forge.sandbox import data_transpose
from snax_forge.viz.dfg.api import Entry, graph_view
from tests.sandbox.helpers import mvt

from .test_columns import MVT_HALF_2, drawn, pairs


def test_the_copy_is_a_block_between_a_and_a_t():
    g = data_transpose(mvt(), "tmp2_mult_map")
    v = graph_view(Entry(g.name, Path(g.name), g, None))
    assert v["columns"][1]["tops"] == ["transpose_A", *MVT_HALF_2]
    assert [k for k, b in v["boxes"].items() if b["container"] == "A"] == ["A@0"]
    top, block = drawn(v["rows"])
    assert top == ["A@0"] and v["rows"][0]["column"] is None
    assert block[1][:3] == ["transpose_A", ["A_T@1", "y2@0"], "tmp2_mult_map"]
    (node,) = [r["node"] for c in v["rows"][1]["columns"] for r in c["rows"]
               if r["kind"] == "node" and r["node"]["id"] == "transpose_A"]  # fmt: skip
    assert (node["kind"], node["text"]) == ("transpose", "transpose · dma")
    assert {("A@0", "tmp0_mult.in1"), ("A@0", "transpose_A.src"),
            ("transpose_A.dst", "A_T@1"), ("A_T@1", "tmp2_mult.in1")} <= set(pairs(v))  # fmt: skip
    assert not any(e.get("transpose") for e in v["edges"])  # nothing left to mark Aᵀ
    assert v["details"]["node:transpose_A"] == {
        "kind": "transpose", "perm": [1, 0], "by": "dma",
        "inputs": {"src": "A[0:M, 0:N]"}, "outputs": {"dst": "A_T[0:N, 0:M]"}}  # fmt: skip
