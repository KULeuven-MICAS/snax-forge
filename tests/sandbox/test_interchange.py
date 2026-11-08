"""interchange_map: two maps of one perfect nest trade places, each keeping its id, variable,
range and tag, the maps between them and the inner one's body where they are; the pair again,
in either order, undoes it.

rowsum imports as ``sum_map`` (m) around ``sum_map_1`` (n) around ``sum``. Its lane accumulator
(rowsum_2) splits the row map by W, tags the column map temporal and puts the spatial row map
inside it (``{"map": "sum_map_s", "with": "sum_map_1"}``, the innermost map with its parent):
``sum_map`` (m_1_t) -> ``sum_map_1`` (n_t) -> ``sum_map_s`` (m_2_s) -> ``sum``. Every step is
checked against the reference executor (run.py's ``checked_step``).
"""

from __future__ import annotations

import copy

import pytest

from snax_forge.dfg import REDUCE_OPS, Graph
from snax_forge.dfg.reduce import ReduceOp
from snax_forge.sandbox import Recipe, SandboxError, apply_recipe, interchange_map
from snax_forge.sandbox.transforms import TRANSFORMS, interchange, partners, site_of
from tests.helpers import imported

from .helpers import recipe_graphs

SPLIT = {"transform": "split_map", "params": {"map": "sum_map", "factor": "W"}}
TAG = {"transform": "tag_map", "params": {"map": "sum_map_1", "kind": "temporal"}}
SPLIT_COLS = {"transform": "split_map", "params": {"map": "sum_map_1", "factor": "W"}}


def swap(x: str, y: str) -> dict:
    return {"transform": "interchange_map", "params": {"map": x, "with": y}}


def run(steps: list[dict], **params: int) -> list[Graph]:
    """rowsum's import (M 8, N 64, W 4 unless given) and each step's graph, every step checked
    against the reference executor."""
    r = Recipe.from_dict({"name": "t", "kernel": "rowsum",
                          "params": {"W": 4, "M": 8, "N": 64} | params,
                          "symbols": {"M": "M", "N": "N"}, "steps": steps})  # fmt: skip
    return [res.graph for res in apply_recipe(r, imported("rowsum"))]


def nest(g: Graph) -> list[tuple]:
    return [(n.id, n.attrs.get("var"), n.attrs.get("range"), n.attrs.get("loop.kind"))
            for n, _ in g.walk()]  # fmt: skip


def test_the_innermost_map_with_its_parent():
    """rowsum_2's step: the spatial row map and the innermost column map trade places."""
    *_, before, g = run([SPLIT, TAG, swap("sum_map_s", "sum_map_1")])
    assert nest(g) == [
        ("sum_map", "m_1_t", "0:M // 4", "temporal"),
        ("sum_map_1", "n_t", "0:N", "temporal"),
        ("sum_map_s", "m_2_s", "0:4", "spatial"),
        ("sum", None, None, None),
    ]
    s = g.node("sum")
    assert s.inputs["in1"].subset == ["4 * m_1_t + m_2_s", "n_t"]
    assert s.outputs["out"].subset == ["4 * m_1_t + m_2_s"]
    assert g.node("sum_map").attrs["loop.split"] == {"var": "m", "range": "0:M"}
    for n in ("sum_map_1", "sum_map_s"):  # each keeps its attrs, its tag among them
        assert g.node(n).attrs == before.node(n).attrs
    assert interchange(before, "sum_map_1", "sum_map_s").to_json() == g.to_json()  # either order


def test_twice_gives_the_graph_back_in_either_order():
    graphs = run([SPLIT, TAG, swap("sum_map_s", "sum_map_1"), swap("sum_map_s", "sum_map_1"),
                  swap("sum_map_1", "sum_map_s"), swap("sum_map_1", "sum_map_s")])  # fmt: skip
    before, after = graphs[2], graphs[3]
    assert after.to_json() != before.to_json()
    assert graphs[4].to_json() == before.to_json() == graphs[6].to_json()
    assert graphs[5].to_json() == after.to_json()


def test_a_non_adjacent_pair_outer_with_innermost():
    """sum_map (m_1_t) -> sum_map_s (m_2_s) -> sum_map_1 (n): the outer and the innermost map
    trade places, the middle one stays, each keeps its name, and the pair again gives the graph
    back."""
    _, before, g, back = run([SPLIT, swap("sum_map", "sum_map_1"), swap("sum_map_1", "sum_map")])
    assert nest(before)[:3] == [
        ("sum_map", "m_1_t", "0:M // 4", "temporal"),
        ("sum_map_s", "m_2_s", "0:4", "spatial"),
        ("sum_map_1", "n", "0:N", None),
    ]
    assert nest(g) == [
        ("sum_map_1", "n", "0:N", None),
        ("sum_map_s", "m_2_s", "0:4", "spatial"),
        ("sum_map", "m_1_t", "0:M // 4", "temporal"),
        ("sum", None, None, None),
    ]
    assert g.node("sum_map").attrs == before.node("sum_map").attrs
    assert back.to_json() == before.to_json()


def test_the_site_takes_any_map_with_a_partner():
    _, _, g = run([SPLIT, TAG])
    for m in ("sum_map", "sum_map_s", "sum_map_1"):  # the innermost included
        assert TRANSFORMS["interchange_map"].site(g, m) == {"map": m}
        assert site_of("interchange_map", g, [m]) == {"map": m}
    assert partners(g, "sum_map_1") == [("sum_map", None), ("sum_map_s", None)]


def refused(g: Graph, x: str, y: str, match: str) -> None:
    with pytest.raises(SandboxError, match=match):
        interchange_map(g, x, **{"with": y})


def triangle() -> Graph:
    """rowsum with n running to the row, ``0:m``, and a map ``k_map`` (k) under the column map:
    sum_map (m) -> sum_map_1 (n) -> k_map (k) -> sum."""
    d = copy.deepcopy(imported("rowsum").to_dict())
    col = d["body"][0]["body"][0]
    col["attrs"]["range"] = "0:m"
    col["body"] = [{"id": "k_map", "kind": "map", "attrs": {"var": "k", "range": "0:2"},
                    "body": col["body"]}]  # fmt: skip
    return Graph.from_dict(d)


def test_refused_a_range_naming_a_variable_now_inside_it():
    """With the column map outside the row map, its range would name the row map's variable,
    then inside it; the column map trades with k_map, so the pair's own refusal is given."""
    g = triangle()
    want = ("interchange_map 'sum_map_1' with 'sum_map': after the swap sum_map_1's range 0:m "
            "would name sum_map's variable 'm', a map then inside it; a range names only the "
            "variables of the maps around it")  # fmt: skip
    with pytest.raises(SandboxError) as e:
        interchange(g, "sum_map_1", "sum_map")
    assert str(e.value) == want
    assert partners(g, "sum_map_1") == [("sum_map", want), ("k_map", None)]
    # the row map trades with neither (with k_map, sum_map_1 between them names i): the site says
    with pytest.raises(SandboxError, match="^interchange_map 'sum_map': no map above or below it "
                                           "trades places with it: interchange_map 'sum_map' with "
                                           "'sum_map_1': after the swap"):  # fmt: skip
        site_of("interchange_map", g, "sum_map")


def test_refused_a_pair_not_in_one_perfect_nest():
    """sum_map holds sum_map_1 (-> k_map -> sum) and a second nest ``again``: sum_map_1 and
    again are side by side; k_map and sum_map are not one perfect nest."""
    d = copy.deepcopy(imported("rowsum").to_dict())
    col = d["body"][0]["body"][0]
    again = copy.deepcopy(col) | {"id": "again"}
    again["body"][0]["id"] = "sum_2"
    col["body"] = [{"id": "k_map", "kind": "map", "attrs": {"var": "k", "range": "0:2"},
                    "body": col["body"]}]  # fmt: skip
    d["body"][0]["body"].append(again)
    g = Graph.from_dict(d)
    refused(g, "sum_map_1", "again", "^interchange_map 'sum_map_1' with 'again': not in one nest, "
                                     "neither map is inside the other$")  # fmt: skip
    refused(g, "k_map", "sum_map", "^interchange_map 'k_map' with 'sum_map': not one perfect "
                                   "nest: 'sum_map' holds the map 'sum_map_1', the map 'again'; "
                                   "every map from 'sum_map' down to k_map's parent holds one "
                                   "map and nothing else$")  # fmt: skip
    refused(g, "sum_map_1", "sum_map_1", "not with itself")
    refused(g, "sum", "sum_map", "a 'tasklet' is not a map")
    refused(imported("dot"), "mult_map", "sum_map",
            "^interchange_map 'mult_map': no map above or below it$")  # fmt: skip


def test_refused_a_bound_unit_inside():
    *_, g = run([{"transform": "tag_map", "params": {"map": "sum_map", "kind": "temporal"}},
                 SPLIT_COLS,
                 {"transform": "bind", "params": {"node": "sum", "brm": "accumulate",
                                                  "implementation": "chisel_adder_tree",
                                                  "instance": "acc"}}])  # fmt: skip
    refused(
        g, "sum_map", "sum_map_1", "the bound unit 'sum' is inside 'sum_map_1' \\(unbind first\\)"
    )


def test_refused_inside_an_accelerated_node():
    """An accelerated node with a nested block (C§11) holding a nest."""
    d = copy.deepcopy(imported("rowsum").to_dict())
    rowsum = d["body"][0]
    d["body"][0] = {"id": "blk", "kind": "accelerated",
                    "inputs": {"a": {"data": "A", "subset": ["0", "0:1"]}},
                    "outputs": {"out": {"data": "y", "subset": ["0:1"]}},
                    "attrs": {"instance": "blk", "brm": "accumulate",
                              "implementation": "chisel_adder_tree", "code": "out = a",
                              "params": {"W": 1, "op": "add"}, "replaced": None},
                    "body": [rowsum]}  # fmt: skip
    rowsum["body"][0]["body"] = [{"id": "inner", "kind": "map",
                                  "attrs": {"var": "k", "range": "0:2"},
                                  "body": rowsum["body"][0]["body"]}]  # fmt: skip
    refused(Graph.from_dict(d), "sum_map_1", "inner",
            "inside the accelerated node 'blk' \\(unbind first\\)")  # fmt: skip


def test_refused_a_fused_nests_outer_map():
    """dot 4's recipe fuses the split multiply and partial-sum nests into ``mult_map``
    (``loop.fused``) around ``mult_map_s``; a map put around it by hand has the fused nest as
    its only child."""
    _, graphs = recipe_graphs("dot_4")
    fused = next(g for g in graphs[1:] if any("loop.fused" in n.attrs for n, _ in g.walk()))
    (top,) = [n for n in fused.body if n.attrs.get("loop.fused")]
    (inner,) = top.body
    match = f"'{top.id}' is a fused nest's outer map \\(unfuse_map first\\)"
    refused(fused, top.id, inner.id, match)
    d = copy.deepcopy(fused.to_dict())
    i = next(k for k, n in enumerate(d["body"]) if n["id"] == top.id)
    d["body"][i] = {"id": "around", "kind": "map", "attrs": {"var": "r", "range": "0:1"},
                    "body": [d["body"][i]]}  # fmt: skip
    g = Graph.from_dict(d)
    refused(g, "around", top.id, match)
    refused(g, "around", inner.id, match)  # the fused map between them


def test_refused_an_order_dependent_fold(monkeypatch):
    """A fold over two of the chain's maps, one of them moved, with an op that is not
    order-free; every registered op is order-free (D28), so add is marked as one that is not.
    sum_map (m) -> sum_map_1 (n_1_t) -> sum_map_1_s (n_2_s): y[m] folds over n_1_t and n_2_s."""
    _, g = run([SPLIT_COLS])
    monkeypatch.setitem(REDUCE_OPS, "add", ReduceOp(REDUCE_OPS["add"].ufunc,
                                                    REDUCE_OPS["add"].identity, False))  # fmt: skip
    match = "sum's out folds over 'n_1_t', 'n_2_s' with 'add', whose result depends on the order"
    refused(g, "sum_map_1", "sum_map_1_s", match)
    refused(g, "sum_map", "sum_map_1_s", match)  # n_1_t and n_2_s swap their order too
    # the row map with the next: y[m] moves with m, so each element's writes keep their order
    assert interchange(g, "sum_map", "sum_map_1").body[0].id == "sum_map_1"


def test_the_bind_site_names_a_spatial_tag():
    """A map around the bound one tagged spatial is named so, not untagged (D: bind's site)."""
    *_, g = run([{"transform": "tag_map", "params": {"map": "sum_map", "kind": "spatial"}},
                 {"transform": "tag_map", "params": {"map": "sum_map_1", "kind": "spatial"}}],
                M=4, N=4)  # fmt: skip
    from snax_forge.sandbox import bind

    with pytest.raises(SandboxError, match="map 'sum_map' around it is spatial; the maps around "
                                           "the spatial map are its firing loops, temporal: "
                                           "untag_map it, then tag_map it temporal"):  # fmt: skip
        bind(g, "sum", "accumulate", "chisel_adder_tree", "acc", {"W": 4})


@pytest.mark.slow
def test_the_editors_choices_and_undo(tmp_path, monkeypatch):
    """The editor lists interchange_map on each of rowsum's three maps, the innermost included;
    ``with`` is a choice of every map above and below, the nearest that can trade the default
    (the one below on a tie), one that cannot disabled with its refusal; Undo moves the
    cursor back, and the pair again is the step's own inverse."""
    from snax_forge.sandbox import session
    from snax_forge.sandbox.session import Roots, Session

    monkeypatch.setattr(session, "_import", imported)
    roots = Roots(tmp_path / "sessions", tmp_path / "recipes", tmp_path / "sandbox")
    s = Session.from_kernel("rowsum", {"M": 8, "N": 64, "W": 4}, roots=roots)
    assert s.apply("split_map", {"map": "sum_map", "factor": "W"})["ok"]
    assert s.apply("tag_map", {"map": "sum_map_1", "kind": "temporal"})["ok"]

    def with_field(m: str) -> dict:
        site = s.sites(2, m)["interchange_map"]
        assert site["ok"] and site["params"] == {"map": m}
        (f,) = [f for f in site["form"] if f["name"] == "with"]
        return f

    inner = with_field("sum_map_1")
    assert inner["type"] == "choice" and inner["value"] == "sum_map_s"
    assert [(c["value"], c["label"], c["ok"]) for c in inner["choices"]] == [
        ("sum_map", "sum_map (above)", True), ("sum_map_s", "sum_map_s (above)", True)]  # fmt: skip
    assert with_field("sum_map_s")["value"] == "sum_map_1"
    assert with_field("sum_map")["value"] == "sum_map_s"
    before = s.graphs[2].to_json()
    assert s.apply("interchange_map", {"map": "sum_map_s", "with": "sum_map_1"})["ok"]
    assert s.move(2)["ok"] and s.graphs[s.cursor].to_json() == before
    assert s.move(3)["ok"]
    assert s.apply("interchange_map", {"map": "sum_map_1", "with": "sum_map_s"}, drop=True)["ok"]
    assert s.graphs[4].to_json() == before


def test_a_refused_partner_is_a_disabled_choice():
    from snax_forge.sandbox.session import _partner_field

    f = _partner_field(triangle(), "sum_map_1")
    assert [(c["value"], c["ok"]) for c in f["choices"]] == [("sum_map", False), ("k_map", True)]
    assert "would name sum_map's variable 'm'" in f["choices"][0]["message"]
    assert f["value"] == "k_map"
