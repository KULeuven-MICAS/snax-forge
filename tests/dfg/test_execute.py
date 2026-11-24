"""The reference executor: .snaxdfg graphs run in NumPy.

Accepted when the imported vecadd equals the kernel's ``reference`` on
``make_inputs``, for N a multiple of the lane count and not, and the plain,
split and accelerated graphs agree; and when the imported dot equals
``np.dot`` on integer inputs, its sum map folding with ``reduce``
from the identity. Also: transients and offset subsets, the
accelerated node's firing order and tasks, and every check named.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from snax_forge.dfg import KINDS, Graph, Memlet, register_kind
from snax_forge.dfg.__main__ import main
from snax_forge.dfg.execute import EXECUTORS, ExecutionError, execute, register_executor
from snax_forge.sdfg.loader import load

from .helpers import NAMES, as_dict, fixture

# the module, not the function the package exports under the same name
ex = importlib.import_module("snax_forge.dfg.execute")


def kernel_run(n: int, seed: int = 0, kernel: str = "vecadd") -> tuple[dict, dict]:
    """A kernel's inputs and its own reference output (vecadd unless named)."""
    spec = load(kernel)
    inputs = spec.make_inputs(np.random.default_rng(seed), n=n)
    ref = {k: v.copy() for k, v in inputs.items()}
    spec.reference(**ref)
    return inputs, ref


# =============================================================================
# vecadd (the acceptance)
# =============================================================================


@pytest.mark.parametrize("name", NAMES)
def test_every_step_equals_the_kernel_reference(name):
    inputs, ref = kernel_run(64)
    assert np.array_equal(execute(Graph.load(fixture(name)), inputs)["C"], ref["C"])


# =============================================================================
# dot: a reduction
# =============================================================================


def _m(*subset):
    return Memlet("x", list(subset))


NAMED_RATES = [
    ({"out": (_m("0:1"), "T")}, [("i_t", 16)], {"T": 16}),  # dot: every firing
    ({"out": (_m("i"), "T")}, [("i", 4), ("j", 6)], {"T": 6}),  # a row sum: the inner loop
    ({"a": (_m("i"), 1), "out": (_m("i"), "T")}, [("i", 4)], {"T": 1}),  # folds nothing
    ({"out": (_m(0), "T"), "c": (_m(1), "T")}, [("i", 2), ("j", 3)], {"T": 6}),
]


def test_named_rates_rejected():
    with pytest.raises(ExecutionError, match="does not use loop 'i', which is outside"):
        ex.named_rates({"out": (_m("j"), "T")}, [("i", 4), ("j", 6)], "t")
    with pytest.raises(ExecutionError, match="rate 'T' is 6 and 3 on two ports"):
        ex.named_rates({"a": (_m(0), "T"), "b": (_m("i"), "T")}, [("i", 2), ("j", 3)], "t")


def test_cli_check(capsys):
    files = [str(fixture(n)) for n in NAMES]
    assert main(["check", *files, "--kernel", "vecadd"]) == 0
    out = capsys.readouterr().out
    assert out.count("OK ") == 3 and "(N = 64)" in out and "(N = 1024)" in out
    assert main(["check", str(fixture("vecadd")), "--kernel", "vecadd", "--n", "10"]) == 0


# =============================================================================
# Beyond vecadd: transients and offsets, imported
# =============================================================================


# =============================================================================
# The accelerated node: firings, tasks and its BRM's function
# =============================================================================


def tiled() -> dict:
    """vecadd_accelerated with a tile map of 2 around a temporal map of 8 beats."""
    d = as_dict("vecadd_accelerated")
    tmap = d["body"][0]
    tmap["attrs"]["range"] = "0:8"
    lanes = "32 * k + 4 * n_1_t:32 * k + 4 * n_1_t + 4"
    for m in (*tmap["body"][0]["inputs"].values(), *tmap["body"][0]["outputs"].values()):
        m["subset"] = [lanes]
    d["body"] = [
        {
            "id": "tile",
            "kind": "map",
            "attrs": {"var": "k", "range": "0:2", "loop.kind": "tile"},
            "body": [tmap],
        }
    ]
    return d


def test_firings_in_order_and_a_task_per_tile(monkeypatch):
    """k counts firings within a task, state lives for one task, n is its firing count."""
    calls = []
    real = ex._accel

    def spy(node, what, ctx):
        instance, cfg, slots = real(node, what, ctx)

        def fn(k, ins, state, params):
            calls.append((k, params["n"], state.get("seen", 0), ins["a"][0]))
            state["seen"] = state.get("seen", 0) + 1
            return cfg.fn(k, ins, state, params)

        return instance, type(cfg)(cfg.ports, fn, cfg.kind), slots

    monkeypatch.setattr(ex, "_accel", spy)
    inputs, ref = kernel_run(64)
    out = execute(Graph.from_dict(tiled()), inputs)
    assert np.array_equal(out["C"], ref["C"])
    assert [c[:3] for c in calls] == [(k, 8, k) for k in range(8)] * 2
    assert [c[3] for c in calls] == list(inputs["A"][::4])  # lane 0 of each beat, in order


# =============================================================================
# Checks, each named
# =============================================================================


def accel(d: dict) -> dict:
    return d["body"][0]["body"][0]


def plain_node(d: dict) -> dict:
    return d["body"][0]["body"][0]


GRAPH_CASES = [
    ("vecadd_accelerated", lambda d: accel(d)["attrs"]["params"].update(W=8), "has 8 lanes, got 4"),
    (
        "vecadd_accelerated",
        lambda d: accel(d)["inputs"].update(c=accel(d)["inputs"]["a"]),
        "inputs.c: neither a port of 'simd' nor an unset slot of it",
    ),
    ("vecadd_accelerated", lambda d: accel(d)["attrs"].update(brm="nope"), "no BRM 'nope'"),
    (
        "vecadd_accelerated",
        lambda d: d["body"][0]["attrs"].update({"loop.kind": "spatial"}),
        "inside a spatial map",
    ),
    ("vecadd", lambda d: plain_node(d)["outputs"]["out"].update(subset=[0]), "write conflict"),
    (
        "vecadd",
        lambda d: plain_node(d)["inputs"]["in1"].update(subset=["n + 1"]),
        "A\\[n \\+ 1\\] leaves 0:64",
    ),
    ("vecadd", lambda d: plain_node(d)["inputs"]["in1"].update(subset=["n - 1"]), "leaves 0:64"),
]


@pytest.mark.parametrize(("name", "edit", "message"), GRAPH_CASES)
def test_rejected(name, edit, message):
    d = as_dict(name)
    edit(d)
    inputs, _ = kernel_run(64)
    with pytest.raises(ExecutionError, match=message):
        execute(Graph.from_dict(d), inputs)


def test_port_dtypes_follow_the_rule():
    """The containers' dtypes pass the BRM's rule: an int32 output under int64 inputs
    is refused, naming the outputs it would take."""
    d = as_dict("vecadd_accelerated")
    d["containers"]["C"]["dtype"] = "int32"
    inputs = kernel_run(64)[0]
    inputs["C"] = inputs["C"].astype(np.int32)
    with pytest.raises(ExecutionError, match="output out is int32, narrower than input a") as e:
        execute(Graph.from_dict(d), inputs)
    assert str(e.value).endswith("accepted output dtypes for these inputs: out: int64, uint64")


def test_symbols_and_inputs():
    g_plain, g_bound = Graph.load(fixture("vecadd")), Graph.load(fixture("vecadd_split"))
    inputs, _ = kernel_run(32)
    with pytest.raises(ExecutionError, match="the graph binds 64, the inputs give 32"):
        execute(g_bound, inputs)
    with pytest.raises(ExecutionError, match="no input for container 'B'"):
        execute(g_plain, {"A": inputs["A"], "C": inputs["C"]})
    with pytest.raises(ExecutionError, match="input 'D' is not a container"):
        execute(g_plain, {**inputs, "D": inputs["A"]})
    with pytest.raises(ExecutionError, match="inputs give both 32 and 31"):
        execute(g_plain, {**inputs, "B": inputs["B"][:31]})
    with pytest.raises(ExecutionError, match="input 'A': int32\\[32\\] given"):
        execute(g_plain, {**inputs, "A": inputs["A"].astype(np.int32)})


def test_an_unregistered_executor():
    register_kind("test_noop")
    try:
        d = as_dict("vecadd")
        d["body"].append({"id": "noop", "kind": "test_noop"})
        with pytest.raises(ExecutionError, match="no executor for kind 'test_noop'"):
            execute(Graph.from_dict(d), kernel_run(8)[0])
        register_executor("test_noop", lambda node, ctx: None)
        execute(Graph.from_dict(d), kernel_run(8)[0])
    finally:
        KINDS.pop("test_noop", None)
        EXECUTORS.pop("test_noop", None)
