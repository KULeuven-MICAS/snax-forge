"""split_map and bind on the vecadd fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.brm import Brm
from snax_forge.dfg import Graph, execute
from snax_forge.sandbox import SandboxError, bind, split_map
from snax_forge.sandbox import transforms as tr
from tests.brm.helpers import affine_adder
from tests.helpers import graph_dict

from .helpers import FIXTURES, graph

BIND = {
    "node": "add",
    "brm": "simd",
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


@pytest.mark.parametrize(
    ("make", "args", "message"),
    [
        (lambda: graph("vecadd"), ("add_map", 4), "no value for 'N' .*bind the symbol"),
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
            {"params": {"op": "add"}},
            "op = 'add' given, but the tasklet computes 'out = a \\* b', 'simd''s code at op = 'mul'",
        ),
        (
            lambda: _edit(lambda d: tasklet(d)["attrs"].update(code="out = in1 // in2")),
            {},
            (
                "the tasklet computes 'out = a // b' on the BRM's ports, 'simd' computes one of "
                "'out = a \\+ b' \\(op = 'add', streams = 2\\), 'out = a - b' \\(op = 'sub', "
            ),
        ),
        (
            lambda: _edit(lambda d: tasklet(d)["attrs"].update(code="out = in1 + in1")),
            {},
            "reads in1 more than once; simd reads distinct inputs, each once",
        ),
        (
            lambda: _edit(lambda d: d["containers"]["C"].update(dtype="int32")),
            {},
            "output out is int32, narrower than input a \\(int64\\)",
        ),
        (
            lambda: _edit(
                lambda d: tasklet(d)["inputs"]["in1"].update(subset=["4 * n_1_t + 3 - n_2_s"])
            ),
            {},
            "runs the lanes backwards",
        ),
    ],
)
def test_bind_rejected(make, params, message):
    with pytest.raises(SandboxError, match=message):
        bind(make(), **(BIND | params))


def test_bind_refuses_vmul_into_a_narrower_output_naming_the_accepted_ones():
    def vmul_int32_out(d):
        tasklet(d)["attrs"].update(code="out = in1 * in2")
        d["containers"]["C"].update(dtype="int32")

    with pytest.raises(SandboxError) as e:
        bind(_edit(vmul_int32_out), **(BIND | {"brm": "simd"}))
    assert str(e.value).endswith(
        "output out is int32, narrower than input a (int64): an output is at least as wide as "
        "every input; output out is int32, narrower than input b (int64): an output is "
        "at least as wide as every input; accepted output dtypes for these inputs: "
        "out: int64, uint64"
    )


def _strided() -> Graph:
    """vecadd split, its lanes strided by 16: the right elements, in another order."""
    d = split_dict()
    for m in (*tasklet(d)["inputs"].values(), *tasklet(d)["outputs"].values()):
        m["subset"] = ["n_1_t + 16 * n_2_s"]
    return Graph.from_dict(d)


def test_the_order_check_is_about_the_brm_not_the_result(monkeypatch):
    """An affine BRM's nest takes the elements in order: the strided lanes are refused,
    though they add the right elements."""
    assert same_output(graph("vecadd"), _strided())
    monkeypatch.setattr(tr, "load_brm", lambda name: Brm.from_dict(affine_adder()))
    with pytest.raises(SandboxError, match="does not give the BRM's order"):
        bind(_strided(), **BIND)
