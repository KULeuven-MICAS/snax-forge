"""Design point -> task list (LOW1a, D89)."""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.design import DesignPoint, MemorySpec, run_checks
from snax_forge.design.streamers import Loop
from snax_forge.dfg import Graph, Memlet
from snax_forge.lower import (
    Layout,
    LowerError,
    TaskList,
    cluster_file,
    lower_program,
    memlet_values,
    task_list,
)
from snax_forge.lower import __main__ as cli
from snax_forge.lower.derive import addresses
from snax_forge.sandbox import Recipe, apply_recipe
from snax_forge.snax_model.ctrl import Wait
from snax_forge.snax_model.scenario import MemInit, Scenario, run
from tests.design.helpers import FIXTURES, bound_w8, chained, design, two_loops

from .helpers import REPO, SCEN

VECADD = SCEN / "vecadd"
PINS = [("B.l1.base", 576), ("C.l1.base", 1152)]


def point(graph="vecadd_accelerated", memory=(), sets=None, edit_graph=None) -> DesignPoint:
    d = design(graph, sets=sets, edit_graph=edit_graph)
    d.memory_spec = MemorySpec().with_changes(list(memory))
    assert run_checks(d) == []
    return DesignPoint.of(d, "vecadd")


def dot() -> Graph:
    """dot as imported (DFG3)."""
    return Graph.load(FIXTURES / "dot.snaxdfg")


def run_point(p: DesignPoint, seed: int = 3):
    """Run a design point in the model with random inputs; its containers from L2 after."""
    rng = np.random.default_rng(seed)
    written = {
        m.data for n, _ in p.graph.walk() if n.kind == "accelerated" for m in n.outputs.values()
    }
    data, fills = {}, []
    in_l2 = [c for c in p.graph.containers if "l2" in p.memory.layouts[c]]  # not a transient
    for c in in_l2:
        lay = p.memory.layouts[c]["l2"]
        n = int(np.prod(lay.shape))
        if c not in written:
            data[c] = rng.integers(-1000, 1000, n)
            fills.append(MemInit("l2", lay.base, data=data[c].tolist()))
    cl = cluster_file(p)
    sc = Scenario(p.name, cl, fills, lower_program(task_list(p), cl), max_cycles=5000)
    res = run(sc)
    out = {}
    for c in in_l2:
        lay = p.memory.layouts[c]["l2"]
        out[c] = res.l2[lay.base // 8 : lay.base // 8 + int(np.prod(lay.shape)), 0]
    return res, data, out


# =============================================================================
# 1. vecadd
# =============================================================================


def test_vecadd_with_its_l1_places_gives_its_task_list():
    """LOW1a acceptance: equal to the hand-written scenarios/vecadd/tasks.json."""
    got = task_list(point(memory=PINS))
    want = TaskList.load(VECADD / "tasks.json")
    assert got == want and got.to_dict() == want.to_dict()


def test_the_default_plan_differs_only_in_the_l1_bases():
    got = task_list(point()).to_dict()["steps"]
    want = TaskList.load(VECADD / "tasks.json").to_dict()["steps"]
    diff = [(a["task_name"], a["values"]) for a, b in zip(got, want, strict=True) if a != b]
    assert [name for name, _ in diff] == ["load_B", "add_acc_b", "add_acc_out", "store_C"]
    assert diff[0][1]["dst"]["base"] == 512 and diff[1][1]["base"] == 512
    assert diff[2][1]["base"] == 1024 and diff[3][1]["src"]["base"] == 1024


@pytest.mark.parametrize("memory, cycles", [(PINS, 77), ((), 85)])
def test_vecadd_runs_right_in_the_model(memory, cycles):
    """With vecadd's places the cycles of scenarios/vecadd; contiguous, the conflict case."""
    res, data, out = run_point(point(memory=memory))
    assert np.array_equal(out["C"], data["A"] + data["B"]) and res.total_cycles == cycles


def test_w8_runs_right():
    d = design(bound_w8())
    assert run_checks(d) == []
    p = DesignPoint.of(d, "vecadd_w8")
    tl = task_list(p)
    acc = tl.configured()["add_acc"]
    assert acc.values == {"n": 8}
    assert tl.configured()["add_acc_a"].values == {
        "base": 0, "temporal_bounds": [8], "temporal_strides": [64], "spatial_strides": [8]
    }  # fmt: skip
    _, data, out = run_point(p)
    assert np.array_equal(out["C"], data["A"] + data["B"])


# =============================================================================
# 2. More than one loop, more than one node
# =============================================================================


def test_two_temporal_maps_give_two_streamer_loops():
    p = point(edit_graph=two_loops, sets=[("streamers.default.temporal_dims", 2)])
    tl = task_list(p)
    assert tl.configured()["add_acc_a"].values == {
        "base": 0, "temporal_bounds": [8, 2], "temporal_strides": [32, 256], "spatial_strides": [8]
    }  # fmt: skip
    assert tl.configured()["add_acc"].values == {"n": 16}
    _, data, out = run_point(p)
    assert np.array_equal(out["C"], data["A"] + data["B"])


def test_a_chain_of_two_nodes_orders_its_tasks_by_the_data():
    p = point(edit_graph=chained)
    tl = task_list(p)
    names = [s.task_name for s in tl.steps if s.op == "configure"]
    assert names == [
        "load_A", "load_B", "add_acc_a", "add_acc_b", "add_acc_out", "add_acc", "store_C",
        "add2_acc2_a", "add2_acc2_b", "add2_acc2_out", "add2_acc2", "store_D",
    ]  # fmt: skip
    c = tl.configured()
    assert c["add2_acc2_a"].after == ["add_acc_out"]  # C comes from the first node, in L1
    assert c["add2_acc2_b"].after == ["load_B"] and c["store_D"].after == ["add2_acc2_out"]
    assert [s.task for s in tl.steps if s.op == "sync"] == ["store_C", "store_D"]
    _, data, out = run_point(p)
    assert np.array_equal(out["C"], data["A"] + data["B"])
    assert np.array_equal(out["D"], data["A"] + 2 * data["B"])


# =============================================================================
# dot: a multiplier chained into an accumulator through L1 (LOW2, LOW3a, D105)
# =============================================================================


def test_dot_gives_its_task_list():
    """mul, then sum after mul's writer; sum's out one beat, T = N / W; out stored as one beat."""
    tl = task_list(point("dot_accelerated"))
    names = [s.task_name for s in tl.steps if s.op == "configure"]
    assert names == [
        "load_A", "load_B", "mult_mul_a", "mult_mul_b", "mult_mul_out", "mult_mul",
        "sum_sum_a", "sum_sum_out", "sum_sum", "store_out",
    ]  # fmt: skip
    c = tl.configured()
    assert c["sum_sum_a"].after == ["mult_mul_out"]  # tmp0 comes from mul, through L1
    assert c["mult_mul"].values == {"n": 16}
    assert c["sum_sum"].values == {"n": 16, "T": 16}
    assert c["sum_sum_out"].values == {
        "base": 1024, "temporal_bounds": [1], "temporal_strides": [0], "spatial_strides": [8]
    }  # fmt: skip
    assert c["store_out"].values["src"] == {"base": 1024, "bounds": [1], "strides": [64]}
    assert [s.task for s in tl.steps if s.op == "sync"] == ["store_out"]


def test_dot_waits_once_between_its_accelerators():
    """Of the lowered waits, one is on an accelerator's writer: mul_out, before sum starts."""
    p = point("dot_accelerated")
    cl = cluster_file(p)
    waits = [c.block for c in lower_program(task_list(p), cl) if isinstance(c, Wait)]
    assert waits == ["dma", "dma", "mul_out", "sum_out", "dma"]


@pytest.mark.parametrize(("w", "cycles"), [(4, 99), (8, 87), (1, 219)])
def test_dot_runs_right_in_the_model(w, cycles):
    g = apply_recipe(Recipe.load(REPO / "recipes" / "dot.json").with_params({"W": w}), dot())
    res, data, out = run_point(point(g[-1].graph))
    assert out["out"].tolist() == [int(data["A"] @ data["B"])]
    assert res.total_cycles == cycles


def test_a_memlet_the_nest_disagrees_with_is_refused():
    """C written back to front: the addresses are right, the order is not the BRM's."""

    def backwards(g):
        g["body"][0]["body"][0]["outputs"]["out"]["subset"] = ["60 - 4 * i_t:64 - 4 * i_t"]

    with pytest.raises(LowerError, match="different orders"):
        task_list(point(edit_graph=backwards))


# =============================================================================
# 3. memlet_values
# =============================================================================


def test_memlet_values_through_row_and_column_major_layouts():
    m = Memlet("X", ["i_t", "0:8"])
    loops = [Loop("m", "i_t", 0, 4, 1)]
    row, sb = memlet_values(m, loops, Layout.contiguous(0, (4, 8), 8), {}, "x")
    assert row == {"base": 0, "temporal_bounds": [4], "temporal_strides": [64],
                   "spatial_strides": [8]} and sb == [8]  # fmt: skip
    col, _ = memlet_values(m, loops, Layout(1024, (4, 8), (8, 32)), {}, "x")
    assert col == {"base": 1024, "temporal_bounds": [4], "temporal_strides": [8],
                   "spatial_strides": [32]}  # fmt: skip
    assert addresses(col, sb)[1].tolist() == [1032 + 32 * k for k in range(8)]


def test_memlet_values_with_offsets_steps_and_no_loop():
    m = Memlet("X", ["2 * i + 1:2 * i + 9:2"])
    v, sb = memlet_values(m, [Loop("m", "i", 3, 2, 4)], Layout(0, (64,), (8,)), {}, "x")
    # i = 3, 7: first element 7, then 15; lanes step 2
    assert v == {"base": 56, "temporal_bounds": [2], "temporal_strides": [64],
                 "spatial_strides": [16]} and sb == [4]  # fmt: skip
    one, _ = memlet_values(Memlet("X", ["0:4"]), [], Layout(0, (64,), (8,)), {}, "x")
    assert one["temporal_bounds"] == [1] and one["temporal_strides"] == [0]


def test_memlet_values_refuse_what_is_not_affine():
    with pytest.raises(LowerError, match="not affine"):
        memlet_values(Memlet("X", ["i * i"]), [Loop("m", "i", 0, 4, 1)],
                      Layout(0, (64,), (8,)), {}, "x")  # fmt: skip
    with pytest.raises(LowerError, match="changes length"):
        memlet_values(Memlet("X", ["i:2 * i"]), [Loop("m", "i", 0, 4, 1)],
                      Layout(0, (64,), (8,)), {}, "x")  # fmt: skip


# =============================================================================
# 4. Command line
# =============================================================================


def test_cli_writes_the_task_list_beside_the_point(tmp_path, capsys):
    point(memory=PINS).save(tmp_path / "design_point.json")
    assert cli.main(["tasks", str(tmp_path / "design_point.json")]) == 0
    assert TaskList.load(tmp_path / "tasks.json") == TaskList.load(VECADD / "tasks.json")
    out = capsys.readouterr().out
    assert "task list of vecadd (12 steps, 57 commands):" in out
    assert "sync      store_C (poll)" in out


def test_cli_rejects_a_bad_point(tmp_path, capsys):
    (tmp_path / "dp.json").write_text("{}")
    assert cli.main(["tasks", str(tmp_path / "dp.json")]) == 1
    assert "[point.keys]" in capsys.readouterr().err
    assert cli.main(["other"]) == 2


@pytest.mark.parametrize(
    "first",
    ["snax_forge.design", "snax_forge.lower", "snax_forge.design.memory",
     "snax_forge.lower.layout", "snax_forge.design.point", "snax_forge.lower.derive"],
)  # fmt: skip
def test_design_and_lower_import_in_any_order(first):
    """design.memory reads lower.layout and lower reads design: no import cycle either way."""
    import subprocess
    import sys

    code = f"import {first}; import snax_forge.design, snax_forge.lower"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
