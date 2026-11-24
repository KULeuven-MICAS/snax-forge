"""Going back: unbind and join_map undo bind and split_map, from the file alone.

A bound graph says what each accelerator computes (the node's ``code``) and
what it replaced (``replaced``), and a split map what it was split from
(``loop.split``), so vecadd_accelerated.snaxdfg alone gives back
vecadd_split.snaxdfg and the imported vecadd, byte for byte.
"""

from __future__ import annotations

import pytest

from snax_forge.dfg import Graph
from snax_forge.sandbox import (
    SandboxError,
    bind,
    join_map,
    split_map,
    unbind,
)
from snax_forge.sandbox import transforms as tr
from tests.helpers import graph_dict

from .helpers import FIXTURES, graph, restricted_brm

BIND = {
    "node": "add",
    "brm": "simd",
    "implementation": "chisel_tiled_spatial",
    "instance": "acc",
}


def imported(n: int = 64) -> Graph:
    d = graph_dict("vecadd")
    d["symbols"]["N"] = n
    return Graph.from_dict(d)


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
    t["inputs"]["in2"]["subset"] = ["n - 2"]
    t["outputs"]["out"]["subset"] = ["n + 0"]
    g = Graph.from_dict(d)
    back = join_map(split_map(g, "add_map", 8), "add_map")
    node = back.node("add")
    assert back.node("add_map").attrs == g.node("add_map").attrs
    assert [m.subset for m in node.inputs.values()] == [["n"], ["n - 2"]]
    assert node.outputs["out"].subset == ["n + 0"]  # split's own expression put back as it was
    assert back.to_json() == g.to_json()


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
    d["body"][0]["body"][0]["body"][0]["inputs"]["in1"]["subset"] = ["n_1_t + 16 * n_2_s"]
    with pytest.raises(SandboxError, match="is not an index split_map wrote"):
        join_map(Graph.from_dict(d), "add_map")


# =============================================================================
# The linking transforms: each undone by its inverse, the graph as it started
# =============================================================================
