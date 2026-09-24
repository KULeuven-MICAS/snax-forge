"""split_map and bind (SBX1, D73, D80) on the vecadd fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.dfg import Graph, execute
from snax_forge.sandbox import SandboxError, bind, split_map
from snax_forge.sandbox import transforms as tr

from .helpers import FIXTURES, graph, graph_dict, restricted_brm

BIND = {
    "node": "add",
    "brm": "elementwise_add",
    "implementation": "chisel_tiled_spatial",
    "instance": "acc",
}


def bound(name: str, n: int = 64) -> Graph:
    d = graph_dict(name)
    d["symbols"]["N"] = n
    return Graph.from_dict(d)


def same_output(a: Graph, b: Graph, n: int = 64) -> bool:
    rng = np.random.default_rng(3)
    inputs = {k: rng.integers(-99, 99, n) for k in "AB"} | {"C": np.zeros(n, np.int64)}
    return np.array_equal(execute(a, inputs)["C"], execute(b, inputs)["C"])


# =============================================================================
# split_map
# =============================================================================


def test_split_gives_the_split_fixture():
    g = split_map(bound("vecadd"), "add_map", 4)
    assert g.to_json() == (FIXTURES / "vecadd_split.snaxdfg").read_text()


def test_split_does_not_change_its_input():
    g = bound("vecadd")
    before = g.to_json()
    split_map(g, "add_map", 4)
    assert g.to_json() == before


def test_split_a_range_that_does_not_start_at_zero():
    d = graph_dict("vecadd")
    d["symbols"]["N"] = 66
    d["body"][0]["attrs"]["range"] = "2:N"
    g = Graph.from_dict(d)
    s = split_map(g, "add_map", 8)
    assert s.node("add_map").attrs["range"] == "0:(N - 2) // 8"
    assert s.node("add").inputs["in1"].subset == ["2 + 8 * i_t + i_s"]
    assert same_output(g, s, 66)


@pytest.mark.parametrize(
    ("make", "args", "message"),
    [
        (lambda: graph("vecadd"), ("add_map", 4), "no value for 'N' .*bind the symbol"),
        (lambda: bound("vecadd", 66), ("add_map", 4), "66 iterations are not a multiple of 4"),
        (lambda: bound("vecadd"), ("add", 4), "a 'tasklet' is not a map"),
        (lambda: bound("vecadd"), ("nope", 4), "no node 'nope'"),
        (lambda: bound("vecadd"), ("add_map", 0), "factor 0"),
        (lambda: graph("vecadd_split"), ("add_map", 4), "already temporal"),
    ],
)
def test_split_rejected(make, args, message):
    with pytest.raises(SandboxError, match=message):
        split_map(make(), *args)


# =============================================================================
# bind
# =============================================================================


def test_bind_gives_the_accelerated_fixture():
    g = bind(graph("vecadd_split"), **BIND)
    assert g.to_json() == (FIXTURES / "vecadd_accelerated.snaxdfg").read_text()


def test_bind_reads_w_off_the_spatial_bound():
    g = bind(split_map(bound("vecadd"), "add_map", 8), **BIND)
    acc = g.node("add")
    assert acc.attrs["params"] == {"W": 8, "op": "add"}
    assert acc.inputs["a"].subset == ["8 * i_t:8 * i_t + 8"]
    assert same_output(graph("vecadd_split"), g)


def test_a_w_the_brm_does_not_allow(monkeypatch):
    monkeypatch.setattr(tr, "load_brm", lambda name: restricted_brm())
    bind(split_map(bound("vecadd"), "add_map", 8), **BIND)  # 8 is allowed
    with pytest.raises(SandboxError, match="W = 2 not in the param's values \\[4, 8\\]"):
        bind(split_map(bound("vecadd"), "add_map", 2), **BIND)


def split_dict() -> dict:
    return graph_dict("vecadd_split")


def tasklet(d: dict) -> dict:
    return d["body"][0]["body"][0]["body"][0]


def _edit(fn):
    d = split_dict()
    fn(d)
    return Graph.from_dict(d)


@pytest.mark.parametrize(
    ("make", "params", "message"),
    [
        (lambda: bound("vecadd"), {}, "only node of a spatial map \\(split the map first\\)"),
        (lambda: graph("vecadd_split"), {"brm": "nope"}, "no BRM 'nope'"),
        (lambda: graph("vecadd_split"), {"implementation": "rtl"}, "no implementation 'rtl'"),
        (
            lambda: graph("vecadd_split"),
            {"params": {"W": 8}},
            "W = 8 given, the spatial bound is 4",
        ),
        (
            lambda: _edit(lambda d: tasklet(d)["attrs"].update(code="out = in1 * in2")),
            {},
            "the tasklet is mul of 2, the pattern is add of 2",
        ),
        (
            lambda: _edit(lambda d: tasklet(d)["attrs"].update(code="out = in1 + in1")),
            {},
            "not one operator over distinct inputs",
        ),
        (
            lambda: _edit(lambda d: [c.update(dtype="int32") for c in d["containers"].values()]),
            {},
            "port 'a' is int64, container 'A' is int32",
        ),
        (
            lambda: _edit(
                lambda d: [
                    m.update(subset=["i_t + 16 * i_s"])
                    for m in (*tasklet(d)["inputs"].values(), *tasklet(d)["outputs"].values())
                ]
            ),
            {},
            "does not give the BRM's order",
        ),
        (
            lambda: _edit(
                lambda d: tasklet(d)["inputs"]["in1"].update(subset=["4 * i_t + 3 - i_s"])
            ),
            {},
            "runs the lanes backwards",
        ),
    ],
)
def test_bind_rejected(make, params, message):
    with pytest.raises(SandboxError, match=message):
        bind(make(), **(BIND | params))


def test_the_order_check_is_about_the_brm_not_the_result():
    """Lanes strided by 16 still add the right elements, but not in elementwise_add's order."""
    d = split_dict()
    for m in (*tasklet(d)["inputs"].values(), *tasklet(d)["outputs"].values()):
        m["subset"] = ["i_t + 16 * i_s"]
    assert same_output(graph("vecadd"), Graph.from_dict(d))
