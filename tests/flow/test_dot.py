"""dot 1 end to end (D148): ``recipes/dot_1.json``, a multiplier into tmp0 in L1,
then an adder tree over it, the two groups ``mul`` and ``sum`` of one accelerator ``dot``.

Up to the generator on the host: the task list waits once between the groups (on mul's
writer) and never on an idle mover. On SNAX-GVSoC (marker ``gvsoc``, D134: in the image), at
trace levels off and task: the run checks itself (every output, both groups' counters), each
group's counter is its task's (busy + its stalls, at least its beats), and the reports name
each task's group. No cycle count is compared: the window moves with the code's layout (open
item 65).

dot's walk-through (examples/dot, tests/flow/test_dot_example.py) stays parked until M13's
B3: it compares two accumulators on SNAX-MODEL, which RET1 removed.
"""

from __future__ import annotations

import pytest

from snax_forge.design import DesignPoint
from snax_forge.flow import run_flow, summary
from snax_forge.lower import placed_waits, task_list
from snax_forge.report import reports_of
from snax_forge.run import read_run
from tests.design.helpers import REPO, SNAX_FORGE

from .helpers import needs_gvsoc

DOT1 = REPO / "recipes" / "dot_1.json"


def test_the_task_list_waits_once_between_the_groups(tmp_path):
    from tests.gen.helpers import bound_dot1, point

    p = DesignPoint.load(point(tmp_path, name="dot_1", graph=bound_dot1()))
    tl = task_list(p)
    accel = [s.task_name for s in tl.steps if s.op == "configure" and s.type == "accel"]
    assert accel == ["mult_mul", "sum_sum"]
    assert tl.configured()["sum_sum_a"].after == ["mult_mul_out"]
    waits = [w.component for w in placed_waits(tl, p)]
    assert waits == ["dma", "dma", "mul_out", "sum_out", "dma"]


@pytest.fixture(scope="module")
def flows(tmp_path_factory):
    base = tmp_path_factory.mktemp("dot_1")
    return {lv: run_flow(DOT1, SNAX_FORGE, out=base / lv, trace_level=lv) for lv in ("off", "task")}


@pytest.mark.gvsoc
@needs_gvsoc
@pytest.mark.parametrize("level", ["off", "task"])
def test_dot_1_runs_and_checks_itself(flows, level):
    f = flows[level]
    assert f.passed, summary(f)
    assert f.check["exit_code"] == 0 and f.check["data_errors"] == 0 and f.check["words"] == 1
    tasks = f.check["tasks"]
    assert {t: (x["group"], x["beats"]) for t, x in tasks.items()} == {
        "mult_mul": ("mul", 16), "sum_sum": ("sum", 16)}  # fmt: skip
    assert all(x["counter"] >= x["beats"] for x in tasks.values())
    p = read_run(f.out / "run").profile
    assert {k: a.firings for k, a in p.accelerators.items()} == {"mul": 16, "sum": 16}
    for t, unit in (("mult_mul", "mul"), ("sum_sum", "sum")):
        c = p.accelerators[unit].cycles  # one run each: its counter is its classes but idle
        assert tasks[t]["counter"] == c["busy"] + c["stall_in"] + c["stall_out"] + c["flush"]
    stdout = (f.out / "run" / "stdout.txt").read_text()
    assert "forge data: 0 errors (1 words)\n" in stdout and "forge errors: 0\n" in stdout


@pytest.mark.gvsoc
@needs_gvsoc
def test_the_reports_name_each_tasks_group(flows):
    f = flows["task"]
    r = reports_of(f.out)
    groups = {t.name: t.group for t in r.run.tasks}
    assert (groups["mult_mul"], groups["mult_sum_a"], groups["sum_mul_out"]) == (
        "mul",
        "mul",
        "sum",
    )
    idle = {t.name: t for t in r.run.tasks if t.name in ("mult_sum_a", "sum_mul_a")}
    assert len(idle) == 2
    assert [(a.instance, a.group) for a in r.design.accelerators] == [
        ("mul", "mul"),
        ("sum", "sum"),
    ]
