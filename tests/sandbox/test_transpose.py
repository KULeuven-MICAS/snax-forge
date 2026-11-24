"""mark_transpose, unmark_transpose and data_transpose (transpose.py) on mvt at M = 8, N = 16:
half 2's multiply reads A[m, n] under the loops (n, m), π = (1, 0); half 1 reads everything in
loop order."""

from __future__ import annotations

import json

import pytest

from snax_forge.dfg import Graph
from snax_forge.dfg.graph import TRANSPOSE
from snax_forge.sandbox import (
    SandboxError,
    data_transpose,
    mark_transpose,
    unmark_transpose,
)
from snax_forge.sandbox.session import form
from snax_forge.sandbox.transforms import TRANSFORMS

from .helpers import mvt

HALF_1 = ["tmp0_mult_map", "tmp0_mult_map_1", "tmp0_mult", "sum_map", "sum_map_1", "sum",
          "augassign_18_4_map", "augassign_18_4"]  # fmt: skip


@pytest.mark.parametrize("site", HALF_1)
def test_every_map_of_half_1_refuses(site):
    g = mvt()
    tasklet = {"tmp0": "tmp0_mult", "sum": "sum", "augassign_18_4": "augassign_18_4"}
    t = next(v for k, v in tasklet.items() if site.startswith(k))
    reads = {"tmp0_mult": "A, y1", "sum": "tmp0_prod", "augassign_18_4": "x1, tmp0"}[t]
    message = f"mark_transpose {site!r}: {t} reads {reads} in loop order"
    with pytest.raises(SandboxError, match=f"^{message}$".replace("(", r"\(")):
        TRANSFORMS["mark_transpose"].site(g, site)
    with pytest.raises(SandboxError, match="in loop order$"):
        mark_transpose(g, site)


def test_mark_then_unmark_gives_the_graph_before():
    g = mvt()
    m = mark_transpose(g, "tmp2_mult_map", "A")
    assert unmark_transpose(m, "tmp2_mult_map_1") == g
    assert unmark_transpose(m, "tmp2_mult", "A").to_json() == g.to_json()
    with pytest.raises(SandboxError, match="tmp2_mult marks A already"):
        mark_transpose(m, "tmp2_mult")
    with pytest.raises(SandboxError, match="tmp2_mult marks no transpose"):
        unmark_transpose(g, "tmp2_mult")
    with pytest.raises(SandboxError, match="y2 is not one of A"):
        mark_transpose(g, "tmp2_mult", "y2")


def test_the_mark_round_trips_through_json(tmp_path):
    """Graph.to_json, then back (``from_dict`` of its JSON, as ``Graph.load`` reads a file)."""
    m = mark_transpose(mvt(), "tmp2_mult")
    assert '"sandbox.transpose": {"in1": [1, 0]}' in m.to_json()
    m.save(tmp_path / "mvt.snaxdfg")
    for back in (Graph.from_dict(json.loads(m.to_json())), Graph.load(tmp_path / "mvt.snaxdfg")):
        assert back == m and back.node("tmp2_mult").attrs[TRANSPOSE] == {"in1": [1, 0]}
        assert back.to_json() == m.to_json()


def test_the_form_names_the_container():
    """One container: its name, read-only; two (a tasklet reading A and B across the loop
    order): a choice, each labelled with its π, and the transform asks which."""
    g = mvt()
    answer = {"params": TRANSFORMS["mark_transpose"].site(g, "tmp2_mult_map")}
    node, container = form("mark_transpose", g, answer, {})
    assert (node["value"], node["readonly"]) == ("tmp2_mult_map", True)
    assert (container["value"], container["readonly"]) == ("A", True)
    d = g.to_dict()
    d["containers"]["B"] = d["containers"]["A"]
    t = d["body"][3]["body"][0]["body"][0]
    t["inputs"]["in2"] = {"data": "B", "subset": ["m", "n"]}
    two = Graph.from_dict(d)
    answer = {"params": TRANSFORMS["mark_transpose"].site(two, "tmp2_mult")}
    assert answer["params"] == {"node": "tmp2_mult"}
    (_, container) = form("mark_transpose", two, answer, {})
    assert container["type"] == "choice" and container["value"] == "A"
    assert [c["label"] for c in container["choices"]] == ["A · transpose(1, 0)",
                                                          "B · transpose(1, 0)"]  # fmt: skip
    with pytest.raises(SandboxError, match="give container, one of A, B"):
        mark_transpose(two, "tmp2_mult")
    assert mark_transpose(two, "tmp2_mult", "B").node("tmp2_mult").attrs[TRANSPOSE] == {
        "in2": [1, 0]
    }


def test_data_transpose_refuses():
    g = mvt()
    with pytest.raises(SandboxError, match="^data_transpose by core is not built yet; use by=dma$"):
        data_transpose(g, "tmp2_mult_map", by="core")
    with pytest.raises(SandboxError, match="by 'cpu' is not one of dma, core"):
        data_transpose(g, "tmp2_mult_map", by="cpu")
    with pytest.raises(SandboxError, match="tmp0_mult reads A, y1 in loop order"):
        data_transpose(g, "tmp0_mult_map")
    d = g.to_dict()
    d["containers"]["A_T"] = {"shape": ["N", "M"], "dtype": "int64", "transient": True}
    with pytest.raises(SandboxError, match="a container A_T exists already; rename it, or undo"):
        data_transpose(Graph.from_dict(d), "tmp2_mult_map")
    marked = mark_transpose(g, "tmp2_mult")
    with pytest.raises(SandboxError, match="data_transpose 'tmp2_mult': tmp2_mult marks A already"):
        data_transpose(marked, "tmp2_mult")


def test_the_form_offers_dma_and_refuses_core():
    g = mvt()
    answer = {"params": TRANSFORMS["data_transpose"].site(g, "tmp2_mult_map")}
    _, container, by = form("data_transpose", g, answer, {})
    assert (container["value"], container["readonly"]) == ("A", True)
    assert by["type"] == "choice" and by["value"] == "dma"
    assert [(c["value"], c["ok"], c.get("message")) for c in by["choices"]] == [
        ("dma", True, None),
        ("core", False, "data_transpose by core is not built yet; use by=dma")]  # fmt: skip


def test_mark_transpose_refuses_an_element_narrower_than_a_word():
    """A packed int8 read cannot cross the loop order; data_transpose can take it."""
    d = mvt().to_dict()
    for c in d["containers"].values():
        c["dtype"] = "int8"
    g = Graph.from_dict(d)
    message = "mark_transpose 'tmp2_mult_map': A is int8: a packed read cannot cross the loop order; use data_transpose"
    with pytest.raises(SandboxError, match=f"^{message}$"):
        mark_transpose(g, "tmp2_mult_map")
    assert data_transpose(g, "tmp2_mult_map").containers["A_T"].dtype == "int8"
