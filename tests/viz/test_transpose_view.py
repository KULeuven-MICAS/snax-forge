"""data_transpose's copy in the DFG view (drawing.py): a box ``transpose · dma`` between A and
A_T, half 2's multiply reading A_T; A one box, read by half 1 and the copy."""

from __future__ import annotations

from pathlib import Path

from snax_forge.sandbox import data_transpose
from snax_forge.viz.dfg.api import Entry, graph_view
from tests.sandbox.helpers import mvt

from .helpers import boxes, containers, ends


def test_the_copy_is_a_box_between_a_and_a_t():
    g = data_transpose(mvt(), "tmp2_mult_map")
    v = graph_view(Entry(g.name, Path(g.name), g, None))
    assert [c for c in containers(v) if c.startswith("A")] == ["A@0", "A_T@1"]
    node = boxes(v)["transpose_A"]
    assert (node["kind"], node["text"], node["ins"], node["outs"]) == (
        "copy",
        "transpose · dma",
        ["src"],
        ["dst"],
    )
    assert {("A@0", "tmp0_mult_map"), ("A@0", "transpose_A.src"), ("transpose_A.dst", "A_T@1"),
            ("A_T@1", "tmp2_mult_map")} <= set(ends(v))  # fmt: skip
    assert not any(e.get("transpose") for e in v["drawing"]["edges"])  # nothing left to mark Aᵀ
    assert v["details"]["node:transpose_A"] == {
        "kind": "transpose", "perm": [1, 0], "by": "dma",
        "inputs": {"src": "A[0:M, 0:N]"}, "outputs": {"dst": "A_T[0:N, 0:M]"}}  # fmt: skip
