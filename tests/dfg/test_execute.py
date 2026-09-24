"""The reference executor (REF1, D79): .snaxdfg graphs run in NumPy.

Accepted when the imported vecadd equals the kernel's ``reference`` on
``make_inputs``, for N a multiple of the lane count and not, and the plain,
split and accelerated graphs agree. Also: transients and offset subsets,
the accelerated node's firing order and tasks, and every check named.
"""

from __future__ import annotations

import copy
import importlib

import numpy as np
import pytest

from snax_forge.dfg import KINDS, Graph, register_kind
from snax_forge.dfg.__main__ import main
from snax_forge.dfg.execute import EXECUTORS, ExecutionError, execute, register_executor
from snax_forge.dfg.import_sdfg import import_sdfg
from snax_forge.sdfg.loader import load

from . import sdfg_programs as progs
from .helpers import NAMES, as_dict, fixture

# the module, not the function the package exports under the same name
ex = importlib.import_module("snax_forge.dfg.execute")


def kernel_run(n: int, seed: int = 0) -> tuple[dict, dict]:
    """vecadd's inputs and the kernel's own reference output."""
    spec = load("vecadd")
    inputs = spec.make_inputs(np.random.default_rng(seed), n=n)
    ref = {k: v.copy() for k, v in inputs.items()}
    spec.reference(**ref)
    return inputs, ref


# =============================================================================
# vecadd (the acceptance)
# =============================================================================


@pytest.mark.parametrize("n", [64, 10, 1024])
def test_imported_vecadd_equals_the_kernel_reference(n):
    """N = 10 is not a multiple of the lane count: only the plain graph runs it."""
    inputs, ref = kernel_run(n)
    out = execute(Graph.load(fixture("vecadd")), inputs)
    assert np.array_equal(out["C"], ref["C"])
    assert np.array_equal(out["A"], inputs["A"]) and np.array_equal(out["B"], inputs["B"])


@pytest.mark.parametrize("name", NAMES)
def test_every_step_equals_the_kernel_reference(name):
    inputs, ref = kernel_run(64)
    assert np.array_equal(execute(Graph.load(fixture(name)), inputs)["C"], ref["C"])


def test_plain_split_and_accelerated_agree():
    inputs, _ = kernel_run(64, seed=7)
    outs = [execute(Graph.load(fixture(n)), inputs)["C"] for n in NAMES]
    assert all(np.array_equal(outs[0], o) for o in outs[1:])


def test_inputs_are_not_changed():
    inputs, _ = kernel_run(64)
    before = copy.deepcopy(inputs)
    execute(Graph.load(fixture("vecadd_accelerated")), inputs)
    assert all(np.array_equal(before[k], inputs[k]) for k in inputs)


def test_int64_wraps_as_in_c():
    big = np.iinfo(np.int64).max
    inputs = {"A": np.array([big, 1]), "B": np.array([1, 1]), "C": np.zeros(2, np.int64)}
    assert execute(Graph.load(fixture("vecadd")), inputs)["C"].tolist() == [-big - 1, 2]


def test_cli_check(capsys):
    files = [str(fixture(n)) for n in NAMES]
    assert main(["check", *files, "--kernel", "vecadd"]) == 0
    out = capsys.readouterr().out
    assert out.count("OK ") == 3 and "(N = 64)" in out and "(N = 1024)" in out
    assert main(["check", str(fixture("vecadd")), "--kernel", "vecadd", "--n", "10"]) == 0


# =============================================================================
# Beyond vecadd: transients and offsets (imported with IMP1)
# =============================================================================


def test_transient_between_two_maps():
    g = import_sdfg(progs.two.to_sdfg(simplify=True), "two")
    rng = np.random.default_rng(1)
    a, b = rng.integers(-99, 99, 20), rng.integers(-99, 99, 20)
    out = execute(g, {"A": a, "B": b, "C": np.zeros(20, np.int64)})
    assert np.array_equal(out["C"], (a + b) * b)
    assert np.array_equal(out["tmp0"], a + b)


def test_offset_subsets():
    g = import_sdfg(progs.shift.to_sdfg(simplify=True), "shift")
    a = np.arange(10) ** 2
    out = execute(g, {"A": a, "B": np.full(10, -1)})
    want = np.full(10, -1)
    want[1:-1] = a[:-2] + a[2:]
    assert np.array_equal(out["B"], want)


# =============================================================================
# The accelerated node: firings, tasks and its BRM's function
# =============================================================================


def tiled() -> dict:
    """vecadd_accelerated with a tile map of 2 around a temporal map of 8 beats."""
    d = as_dict("vecadd_accelerated")
    tmap = d["body"][0]
    tmap["attrs"]["range"] = "0:8"
    lanes = "32 * k + 4 * i_t:32 * k + 4 * i_t + 4"
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

    def spy(node, what):
        instance, cfg = real(node, what)

        def fn(k, ins, state, params):
            calls.append((k, params["n"], state.get("seen", 0), ins["a"][0]))
            state["seen"] = state.get("seen", 0) + 1
            return cfg.fn(k, ins, state, params)

        return instance, type(cfg)(cfg.ports, fn, cfg.latency, cfg.ii, cfg.kind)

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
        lambda d: accel(d)["inputs"].update(x=accel(d)["inputs"].pop("a")),
        "connectors \\['b', 'x'\\], the BRM's ports",
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
        lambda d: plain_node(d)["inputs"]["in1"].update(subset=["i + 1"]),
        "A\\[i \\+ 1\\] leaves 0:64",
    ),
    ("vecadd", lambda d: plain_node(d)["inputs"]["in1"].update(subset=["i - 1"]), "leaves 0:64"),
]


@pytest.mark.parametrize(("name", "edit", "message"), GRAPH_CASES)
def test_rejected(name, edit, message):
    d = as_dict(name)
    edit(d)
    inputs, _ = kernel_run(64)
    with pytest.raises(ExecutionError, match=message):
        execute(Graph.from_dict(d), inputs)


def test_port_dtype_must_be_the_containers():
    d = as_dict("vecadd_accelerated")
    for c in d["containers"].values():
        c["dtype"] = "int32"
    inputs = {k: v.astype(np.int32) for k, v in kernel_run(64)[0].items()}
    with pytest.raises(ExecutionError, match="port 'a' is int64, container 'A' is int32"):
        execute(Graph.from_dict(d), inputs)


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
