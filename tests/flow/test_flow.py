"""The whole path, kernel to model run (E2E1, D90): M3's acceptance."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from snax_forge.design import DesignError
from snax_forge.flow import __main__ as cli
from snax_forge.flow import default_name, functional_check, run_flow
from snax_forge.lower import TaskList
from snax_forge.sandbox import Recipe
from snax_forge.sdfg.loader import load as load_kernel
from snax_forge.snax_model.scenario import Scenario, read_outputs, run
from tests.design.helpers import FIXTURES, RECIPE, SMALL16

SCEN = RECIPE.parents[1] / "scenarios"
PLAIN = FIXTURES / "vecadd.snaxdfg"  # the import of the vecadd kernel (IMP1)
PINS = [("B.l1.base", 576), ("C.l1.base", 1152)]


def flow(tmp_path, **kw):
    kw.setdefault("graph_path", PLAIN)
    return run_flow(RECIPE, SMALL16, out=tmp_path / "flow", **kw)


def profile_without_check(p: dict) -> dict:
    p = dict(p)
    p.pop("functional_check")
    return p


# =============================================================================
# 1. The acceptance
# =============================================================================


def test_vecadd_from_the_kernel(tmp_path):
    """Kernel -> import -> recipe -> design point -> cluster and tasks -> scenario -> run."""
    f = run_flow(RECIPE, SMALL16, out=tmp_path / "flow")
    assert f.passed and f.result.total_cycles == 85  # contiguous: A and B share banks
    c = f.check["containers"]["C"]
    assert c == {"memory": "l2", "elements": 64, "reference": True, "ref1": True, "mismatches": 0}
    out = tmp_path / "flow"
    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert {"sandbox/recipe.json", "sandbox/0_input.snaxdfg", "sandbox/2_bind.snaxdfg",
            "design/platform.json", "design/memory.json", "design/design_point.json",
            "cluster.json", "tasks.json", "scenario.json", "A.npy", "B.npy", "C.npy",
            "run/run.json", "run/profile.json", "run/l2.npy",
            "run/trace.jsonl", "run/trace_meta.json"} <= names  # fmt: skip
    got = read_outputs(out / "run")
    assert got.profile.functional_check == f.check
    assert got.run["trace_level"] == "task"  # the default (D94): the schedule has class runs
    assert {"acc", "acc_a", "acc_b", "acc_out", "dma", "ctl"} <= set(got.trace.intervals)


def test_with_vecadds_places_it_is_scenarios_vecadd(tmp_path):
    """B and C pinned: the cluster, task list, cycles and profile of scenarios/vecadd."""
    f = flow(tmp_path, memory_sets=PINS)
    out = tmp_path / "flow"
    assert f.passed and f.result.total_cycles == 77
    assert f.name == "vecadd_B.l1.base576_C.l1.base1152" and f.point.name == "vecadd"
    assert (out / "tasks.json").read_text() == (SCEN / "vecadd" / "tasks.json").read_text()
    assert (out / "cluster.json").read_text() == (SCEN / "clusters" / "alu4.json").read_text()
    assert TaskList.load(out / "tasks.json") == TaskList.load(SCEN / "vecadd" / "tasks.json")
    want = run(Scenario.load(SCEN / "vecadd" / "scenario.json")).profile.to_dict()
    assert profile_without_check(f.result.profile.to_dict()) == profile_without_check(want)


def test_the_output_is_the_kernels_reference(tmp_path):
    f = flow(tmp_path, seed=5)
    spec = load_kernel("vecadd")
    inputs = spec.make_inputs(np.random.default_rng(5), n=64)
    l2 = np.load(tmp_path / "flow" / "run" / "l2.npy")
    assert np.array_equal(l2[1024 // 8 : 1024 // 8 + 64, 0], inputs["A"] + inputs["B"])
    assert f.check["seed"] == 5 and f.check["symbols"] == {"N": 64}


def test_w8_and_platform_changes_run_right(tmp_path):
    f = flow(tmp_path, recipe_sets={"W": 8},
             platform_sets=[("l1.n_banks", 32), ("streamers.acc_out.fifo_depth", 4)])  # fmt: skip
    assert f.passed and f.point.streamers["acc_out"].n_ports == 8
    assert f.name == "vecadd_W8_l1.n_banks32_streamers.acc_out.fifo_depth4"
    assert f.cluster.l1.n_banks == 32 and f.tasks.configured()["add_acc"].values == {"n": 8}


def test_the_written_scenario_runs_again_the_same(tmp_path):
    f = flow(tmp_path, memory_sets=PINS)
    again = run(Scenario.load(tmp_path / "flow" / "scenario.json"))
    assert again.total_cycles == f.result.total_cycles and np.array_equal(again.l2, f.result.l2)


# =============================================================================
# 2. What fails
# =============================================================================


def test_a_wrong_output_fails_the_check(tmp_path):
    f = flow(tmp_path)
    bad = copy.copy(f.result)
    bad.l2 = f.result.l2.copy()
    bad.l2[1024 // 8 + 3, 0] += 1  # one element of C
    spec = load_kernel("vecadd")
    inputs = spec.make_inputs(np.random.default_rng(0), n=64)
    check = functional_check(spec, f.point, inputs, bad, 0)
    assert not check["passed"]
    assert check["containers"]["C"] == {
        "memory": "l2", "elements": 64, "reference": False, "ref1": False, "mismatches": 1
    }  # fmt: skip


def test_a_design_problem_stops_the_flow_before_the_run(tmp_path):
    with pytest.raises(DesignError) as e:
        flow(tmp_path, platform_sets=[("l1.rows", 8)])
    assert e.value.codes == ["memory.fit"]
    assert not (tmp_path / "flow" / "run").exists()


def test_default_name():
    """Every --set: recipe params, then platform and memory paths without their prefix (D94)."""
    r = Recipe.load(RECIPE)
    assert default_name(r, {}) == "vecadd" and default_name(r, {"W": 8}) == "vecadd_W8"
    assert default_name(r, {}, memory_sets=[("B.l1.base", 576)]) == "vecadd_B.l1.base576"
    assert default_name(r, {}, [("l1.n_banks", 32)]) == "vecadd_l1.n_banks32"
    both = default_name(r, {"W": 8}, [("l1.n_banks", 32)], [("B.l1.base", 576)])
    assert both == "vecadd_W8_l1.n_banks32_B.l1.base576"
    # a value that is not a plain word keeps one plain directory name
    assert default_name(r, {}, [("x.y", [2, 2]), ("z", "a/b c")]) == "vecadd_x.y2,2_zabc"


def test_the_guard_rail_flows_get_names_of_their_own(tmp_path):
    """Default, W = 8 and B pinned no longer share out/flow/vecadd (D94)."""
    names = {
        flow(tmp_path / "0").name,
        flow(tmp_path / "1", recipe_sets={"W": 8}).name,
        flow(tmp_path / "2", memory_sets=[("B.l1.base", 576)]).name,
    }
    assert names == {"vecadd", "vecadd_W8", "vecadd_B.l1.base576"}


def test_the_trace_level_can_still_be_lowered(tmp_path):
    f = flow(tmp_path, trace_level="off")
    assert f.result.total_cycles == 85 and f.result.trace is None
    assert not (tmp_path / "flow" / "run" / "trace_meta.json").exists()


# =============================================================================
# 3. Command line
# =============================================================================


def main(*argv) -> int:
    return cli.main([str(a) for a in argv])


def test_cli_runs_and_says_what_it_did(tmp_path, capsys):
    args = [RECIPE, "--platform", SMALL16, "--graph", PLAIN, "--out", tmp_path / "f"]
    assert main(*args, "--set", "memory.B.l1.base=576", "--set", "memory.C.l1.base=1152") == 0
    out = capsys.readouterr().out
    assert "flow vecadd_B.l1.base576_C.l1.base1152 (vecadd: W=4; N=64) on small16" in out
    assert "lower     cluster.json (7 components), tasks.json (12 steps, 57 commands)" in out
    assert "run       77 cycles" in out
    assert "check     C (64 elements, from l2) equals the vecadd reference and REF1" in out
    d = json.loads((tmp_path / "f" / "design" / "memory.json").read_text())
    assert d["changes"] == {"B.l1.base": 576, "C.l1.base": 1152}


def test_cli_routes_each_set(tmp_path, capsys):
    args = [RECIPE, "--platform", SMALL16, "--graph", PLAIN, "--out", tmp_path / "f"]
    assert main(*args, "--set", "W=8", "--set", "platform.streamers.default.fifo_depth=4") == 0
    out = capsys.readouterr().out
    assert "flow vecadd_W8_streamers.default.fifo_depth4 (vecadd: W=8; N=64)" in out
    assert (tmp_path / "f" / "run" / "trace_meta.json").exists()  # --trace task by default
    pf = json.loads((tmp_path / "f" / "design" / "platform.json").read_text())
    assert pf["changes"] == {"streamers.default.fifo_depth": 4}


def test_cli_reports_a_failing_stage(tmp_path, capsys):
    args = [RECIPE, "--platform", SMALL16, "--graph", PLAIN, "--out", tmp_path / "f"]
    assert main(*args, "--set", "platform.l1.rows=8") == 1
    assert "[memory.fit] C in l1" in capsys.readouterr().err
    assert main(*args, "--set", "Q=1") == 1
    assert "no param 'Q'" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        main(*args, "--set", "l1.rows=8")
    assert e.value.code == 2
