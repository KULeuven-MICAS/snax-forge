"""mark_transpose, unmark_transpose and data_transpose (transpose.py) on mvt at M = 8, N = 16:
half 2's multiply reads A[m, n] under the loops (n, m), π = (1, 0); half 1 reads everything in
loop order."""

from __future__ import annotations

import json

import pytest

from snax_forge.dfg import Graph
from snax_forge.dfg.graph import TRANSPOSE
from snax_forge.sandbox import (
    Recipe,
    SandboxError,
    apply_recipe,
    data_transpose,
    mark_transpose,
    unbind,
    unmark_transpose,
)
from snax_forge.sandbox.session import form
from snax_forge.sandbox.transforms import TRANSFORMS, join_map, split_map, tag_map, untag_map
from snax_forge.sandbox.transpose import loop_orders, transpose_choices
from tests.helpers import imported

from .helpers import SIZES, mvt, step

HALF_1 = ["tmp0_mult_map", "tmp0_mult_map_1", "tmp0_mult", "sum_map", "sum_map_1", "sum",
          "augassign_18_4_map", "augassign_18_4"]  # fmt: skip


def test_pi_follows_the_loop_order():
    g = mvt()
    assert loop_orders(g, "tmp2_mult") == [("in1", "A", [1, 0]), ("in2", "y2", [0])]
    assert loop_orders(g, "tmp0_mult") == [("in1", "A", [0, 1]), ("in2", "y1", [0])]


@pytest.mark.parametrize("site", ["tmp2_mult_map", "tmp2_mult_map_1", "tmp2_mult"])
def test_any_map_of_half_2s_multiply_marks_a(site):
    g = mvt()
    assert TRANSFORMS["mark_transpose"].site(g, site) == {"node": site, "container": "A"}
    m = mark_transpose(g, site)
    assert m.node("tmp2_mult").attrs[TRANSPOSE] == {"in1": [1, 0]}
    assert m.node("tmp2_mult").inputs == g.node("tmp2_mult").inputs  # no memlet changes
    assert {n.id: n.attrs.get(TRANSPOSE) for n, _ in m.walk() if n.id != "tmp2_mult"} == {
        n.id: None for n, _ in g.walk() if n.id != "tmp2_mult"
    }


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


def test_the_mark_survives_tag_split_and_bind():
    """tag_map and split_map keep the tasklet's mark, their inverses too; bind carries it onto
    the unit's port (a, A's), unbind gives back the marked tasklet; every step is checked
    against the reference (the mark changes no value)."""
    marked = mark_transpose(mvt(), "tmp2_mult_map")
    tagged = tag_map(marked, "tmp2_mult_map", "temporal")
    split = split_map(tagged, "tmp2_mult_map_1", 4)
    for g in (tagged, split):
        assert g.node("tmp2_mult").attrs[TRANSPOSE] == {"in1": [1, 0]}
    assert untag_map(tagged, "tmp2_mult_map") == marked
    assert join_map(split, "tmp2_mult_map_1") == tagged
    r = Recipe.from_dict({"name": "t", "kernel": "mvt", "symbols": SIZES, "steps": [
        step("mark_transpose", node="tmp2_mult_map", container="A"),
        step("tag_map", map="tmp2_mult_map", kind="temporal"),
        step("split_map", map="tmp2_mult_map_1", factor=4),
        step("bind", node="tmp2_mult", brm="simd", implementation="chisel_tiled_spatial",
             instance="mul"),
    ]})  # fmt: skip
    bound = apply_recipe(r, imported("mvt"))[-1].graph
    unit = bound.node("tmp2_mult")
    assert unit.kind == "accelerated" and unit.attrs[TRANSPOSE] == {"a": [1, 0]}
    assert unit.inputs["a"].data == "A"
    assert unbind(bound, "tmp2_mult") == split
    with pytest.raises(SandboxError, match="the bound unit tmp2_mult \\(unbind first\\)"):
        unmark_transpose(bound, "tmp2_mult")


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


def test_data_transpose_copies_a_and_half_2_reads_the_copy_in_loop_order():
    """A_T, A's shape permuted, a transient; the transpose node just before half 2's multiply;
    the multiply reads A_T[n, m] under (n, m), so nothing is left to transpose there."""
    g = mvt()
    assert TRANSFORMS["data_transpose"].site(g, "tmp2_mult_map") == {
        "node": "tmp2_mult_map", "container": "A", "by": "dma"}  # fmt: skip
    d = data_transpose(g, "tmp2_mult_map")
    a_t = d.containers["A_T"]
    assert (a_t.shape, a_t.dtype, a_t.transient) == (["N", "M"], "int64", True)
    assert list(d.containers).index("A_T") == list(d.containers).index("A") + 1
    top = [n.id for n in d.body]
    assert top.index("transpose_A") == top.index("tmp2_mult_map") - 1
    assert d.node("transpose_A").to_dict() == {
        "id": "transpose_A", "kind": "transpose",
        "inputs": {"src": {"data": "A", "subset": ["0:M", "0:N"]}},
        "outputs": {"dst": {"data": "A_T", "subset": ["0:N", "0:M"]}},
        "attrs": {"perm": [1, 0], "by": "dma"}}  # fmt: skip
    assert loop_orders(d, "tmp2_mult") == [("in1", "A_T", [0, 1]), ("in2", "y2", [0])]
    assert d.node("tmp0_mult").inputs == g.node("tmp0_mult").inputs  # half 1 still reads A
    assert transpose_choices(d, "tmp2_mult") == []
    with pytest.raises(SandboxError, match="tmp2_mult reads A_T, y2 in loop order"):
        data_transpose(d, "tmp2_mult")


def test_a_split_range_moves_with_its_dimension():
    """After split_map on m, the split index sits in A_T's second dimension; every step is
    checked against the reference executor."""
    r = Recipe.from_dict({"name": "t", "kernel": "mvt", "symbols": SIZES, "steps": [
        step("tag_map", map="tmp2_mult_map", kind="temporal"),
        step("split_map", map="tmp2_mult_map_1", factor=4),
        step("data_transpose", node="tmp2_mult", container="A", by="dma"),
    ]})  # fmt: skip
    d = apply_recipe(r, imported("mvt"))[-1].graph
    assert d.node("tmp2_mult").inputs["in1"].to_dict() == {
        "data": "A_T", "subset": ["n_t", "4 * m_1_t + m_2_s"]}  # fmt: skip


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
