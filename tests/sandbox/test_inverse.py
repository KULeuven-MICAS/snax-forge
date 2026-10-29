"""Going back (D82): unbind and join_map undo bind and split_map, from the file alone.

A bound graph says what each accelerator computes (the node's ``code``) and
what it replaced (``replaced``), and a split map what it was split from
(``loop.split``), so vecadd_accelerated.snaxdfg alone gives back
vecadd_split.snaxdfg and the imported vecadd, byte for byte.
"""

from __future__ import annotations

import pytest

from snax_forge.dfg import Graph
from snax_forge.sandbox import (
    Recipe,
    SandboxError,
    apply_recipe,
    bind,
    join_map,
    split_map,
    unbind,
)
from snax_forge.sandbox import transforms as tr
from snax_forge.sandbox.__main__ import main
from tests.helpers import graph_dict

from .helpers import FIXTURES, REPO, graph, restricted_brm

BIND = {
    "node": "add",
    "brm": "elementwise_add",
    "implementation": "chisel_tiled_spatial",
    "instance": "acc",
}


def imported(n: int = 64) -> Graph:
    d = graph_dict("vecadd")
    d["symbols"]["N"] = n
    return Graph.from_dict(d)


def test_the_accelerated_node_says_what_it_computes_and_replaced():
    acc = graph("vecadd_accelerated").node("add")
    assert acc.attrs["code"] == "out = a + b"
    assert acc.attrs["replaced"] == graph("vecadd_split").node("add_map_s").to_dict()
    assert graph("vecadd_split").node("add_map").attrs["loop.split"] == {"var": "i", "range": "0:N"}


def test_back_from_the_accelerated_fixture():
    split = unbind(graph("vecadd_accelerated"), "add")
    assert split.to_json() == (FIXTURES / "vecadd_split.snaxdfg").read_text()
    assert join_map(split, "add_map").to_json() == imported().to_json()


@pytest.mark.parametrize("w", [1, 4, 8, 64])
def test_round_trip_for_every_w(w):
    g = imported()
    acc = bind(split_map(g, "add_map", w), **BIND)
    assert join_map(unbind(acc, "add"), "add_map").to_json() == g.to_json()


def test_round_trip_with_an_offset_range():
    d = graph_dict("vecadd")
    d["symbols"]["N"] = 66
    d["body"][0]["attrs"]["range"] = "2:N"
    t = d["body"][0]["body"][0]
    t["inputs"]["in2"]["subset"] = ["i - 2"]
    t["outputs"]["out"]["subset"] = ["i + 0"]
    g = Graph.from_dict(d)
    back = join_map(split_map(g, "add_map", 8), "add_map")
    node = back.node("add")
    assert back.node("add_map").attrs == g.node("add_map").attrs
    assert [m.subset for m in node.inputs.values()] == [["i"], ["i - 2"]]
    assert node.outputs["out"].subset == ["i"]  # written back in normal form


def test_the_undo_recipe(tmp_path):
    """recipes/vecadd_undo.json on the accelerated fixture gives the imported graph back."""
    r = Recipe.load(REPO / "recipes" / "vecadd_undo.json")
    results = apply_recipe(r, graph("vecadd_accelerated"))
    assert results[-1].graph.to_json() == imported().to_json()
    args = [str(REPO / "recipes" / "vecadd_undo.json"), "--graph",
            str(FIXTURES / "vecadd_accelerated.snaxdfg"), "--out", str(tmp_path)]  # fmt: skip
    assert main(args) == 0
    assert Graph.load(tmp_path / "2_join_map.snaxdfg").to_json() == imported().to_json()


def test_bind_checks_the_code(monkeypatch):
    brm = restricted_brm()
    brm.function.code = "out = a - b"
    monkeypatch.setattr(tr, "load_brm", lambda name: brm)
    with pytest.raises(SandboxError, match="the tasklet computes 'out = a \\+ b'.*'out = a - b'"):
        bind(graph("vecadd_split"), **BIND)
    brm.function.code = None
    with pytest.raises(SandboxError, match="does not say what it computes"):
        bind(graph("vecadd_split"), **BIND)


def test_unbind_rejected():
    with pytest.raises(SandboxError, match="a 'map' is not an accelerated node"):
        unbind(graph("vecadd_accelerated"), "add_map")
    d = graph_dict("vecadd_accelerated")
    d["body"][0]["body"][0]["attrs"]["replaced"] = None
    with pytest.raises(SandboxError, match="records nothing it replaced"):
        unbind(Graph.from_dict(d), "add")


def test_join_rejected():
    with pytest.raises(SandboxError, match="no loop.split"):
        join_map(imported(), "add_map")
    with pytest.raises(SandboxError, match="unbind first"):
        join_map(graph("vecadd_accelerated"), "add_map")
    d = graph_dict("vecadd_split")
    d["body"][0]["body"][0]["body"][0]["inputs"]["in1"]["subset"] = ["i_t + 16 * i_s"]
    with pytest.raises(SandboxError, match="is not an index split_map wrote"):
        join_map(Graph.from_dict(d), "add_map")


# =============================================================================
# The linking transforms (D150): each undone by its inverse, the graph as it started
# =============================================================================


def _dot(n: int = 8) -> Graph:
    d = graph_dict("dot")
    d["symbols"]["N"] = n
    return Graph.from_dict(d)


def test_join_reduce_undoes_split_reduce():
    g = _dot()
    assert (
        tr.TRANSFORMS["join_reduce"]
        .fn(tr.TRANSFORMS["split_reduce"].fn(g, "sum_map", 4), "sum_map")
        .to_json()
        == g.to_json()
    )


def test_unfuse_map_undoes_fuse_map():
    from snax_forge.sandbox import fuse_map, split_reduce, unfuse_map

    g = split_reduce(split_map(_dot(), "mult_map", 4), "sum_map", 4)
    assert unfuse_map(fuse_map(g, "mult_map", "sum_map"), "mult_map").to_json() == g.to_json()


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_unlink_undoes_link(depth):
    from snax_forge.sandbox import link, unlink

    g = bind(split_map(_dot(), "mult_map", 4), "mult", "elementwise_mul", "chisel_tiled_spatial",
             "mul")  # fmt: skip
    g = bind(split_map(g, "sum_map", 4), "sum", "accumulate", "chisel_adder_tree", "sum")
    assert unlink(link(g, "tmp0", depth, "dot"), "tmp0").to_json() == g.to_json()


def test_dot4_goes_back_to_the_import():
    """dot_4's steps undone in reverse order give the imported graph."""
    from snax_forge.sandbox import fuse_map, join_reduce, link, split_reduce, unfuse_map, unlink

    g0 = _dot()
    g = fuse_map(split_reduce(split_map(g0, "mult_map", 4), "sum_map", 4), "mult_map", "sum_map")
    g = bind(g, "mult_map_s", "mac", "chisel_mac", "mac")
    g = link(
        bind(g, "sum_final", "accumulate", "chisel_adder_tree", "sum"), "out_partial", 0, "dot"
    )
    back = unbind(unbind(unlink(g, "out_partial"), "sum_final"), "mult")
    back = join_map(join_reduce(unfuse_map(back, "mult_map"), "sum_map"), "mult_map")
    assert back.to_json() == g0.to_json()
