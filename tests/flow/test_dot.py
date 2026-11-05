"""dot 1 end to end (D148, D157, D158): ``recipes/dot_1.json``, a multiplier into tmp0 in L1,
then an adder tree over it, two accelerators ``vmul`` and ``vsum``, run in series; and
``recipes/dot_1_enc.json``, the two as the groups of one accelerator ``dot``.

Up to the generator on the host: dot 1's task list waits once between the accelerators (the
serial sync on vmul's task) and has no idle mover. On SNAX-GVSoC (marker ``gvsoc``, D134: in the
image), at trace levels off and task: the run checks itself (every output, both counters), each
unit's counter is its task's (busy + its stalls, at least its beats), and dot_1_enc's reports
name each task's group, its idle movers' too. No cycle count is compared: the window moves with the code's layout (open
item 65).

dot's walk-through (examples/dot, tests/flow/test_dot_example.py) runs dot 1–6 and checks
their counters against C§17.
"""

from __future__ import annotations

import pytest

from snax_forge.design import DesignPoint
from snax_forge.flow import run_flow, summary
from snax_forge.lower import placed_waits, task_list
from snax_forge.report import reports_of
from snax_forge.run import read_run
from tests.design.helpers import REPO, SNAX_FORGE
from tests.helpers import dot1_split

from .helpers import needs_gvsoc

DOT1 = REPO / "recipes" / "dot_1.json"
DOT1_ENC = REPO / "recipes" / "dot_1_enc.json"


def test_the_task_list_waits_once_between_the_accelerators(tmp_path):
    from tests.gen.helpers import bound_dot1, point

    p = DesignPoint.load(point(tmp_path, name="dot_1", graph=bound_dot1()))
    tl = task_list(p)
    accel = [s.task_name for s in tl.steps if s.op == "configure" and s.type == "accel"]
    assert accel == ["mult_vmul", "sum_vsum"]
    assert tl.configured()["sum_vsum_a"].after == ["mult_vmul_out"]
    waits = [w.component for w in placed_waits(tl, p)]
    assert waits == ["dma", "dma", "vmul", "vsum_out", "dma"]  # vmul: the serial sync (D158)


@pytest.fixture(scope="module")
def flows(tmp_path_factory):
    base = tmp_path_factory.mktemp("dot_1")
    return {lv: run_flow(DOT1, SNAX_FORGE, out=base / lv, trace_level=lv) for lv in ("off", "task")}


@pytest.fixture(scope="module")
def enc_flow(tmp_path_factory):
    return run_flow(DOT1_ENC, SNAX_FORGE, out=tmp_path_factory.mktemp("dot_1_enc") / "task",
                    trace_level="task")  # fmt: skip


@pytest.mark.gvsoc
@needs_gvsoc
@pytest.mark.parametrize("level", ["off", "task"])
def test_dot_1_runs_and_checks_itself(flows, level):
    f = flows[level]
    assert f.passed, summary(f)
    assert f.check["exit_code"] == 0 and f.check["data_errors"] == 0 and f.check["words"] == 1
    tasks = f.check["tasks"]
    assert {t: (x["group"], x["beats"]) for t, x in tasks.items()} == {
        "mult_vmul": ("vmul", 16), "sum_vsum": ("vsum", 16)}  # fmt: skip
    assert all(x["counter"] >= x["beats"] for x in tasks.values())
    p = read_run(f.out / "run").profile
    assert {k: a.firings for k, a in p.accelerators.items()} == {"vmul": 16, "vsum": 16}
    for t, unit in (("mult_vmul", "vmul"), ("sum_vsum", "vsum")):
        c = p.accelerators[unit].cycles  # one run each: its counter is its classes but idle
        assert tasks[t]["counter"] == c["busy"] + c["stall_in"] + c["stall_out"] + c["flush"]
    stdout = (f.out / "run" / "stdout.txt").read_text()
    assert "forge data: 0 errors (1 words)\n" in stdout and "forge errors: 0\n" in stdout


@pytest.mark.gvsoc
@needs_gvsoc
def test_the_reports_name_each_tasks_group(enc_flow):
    """dot_1_enc: one accelerator, its groups' tasks and their idle movers each by its group."""
    f = enc_flow
    r = reports_of(f.out)
    groups = {t.name: t.group for t in r.run.tasks}
    assert (groups["mult_vmul"], groups["mult_vsum_a"], groups["sum_vmul_out"]) == (
        "vmul",
        "vmul",
        "vsum",
    )
    idle = {t.name: t for t in r.run.tasks if t.name in ("mult_vsum_a", "sum_vmul_a")}
    assert len(idle) == 2
    assert [(a.instance, a.group) for a in r.design.accelerators] == [
        ("vmul", "vmul"),
        ("vsum", "vsum"),
    ]


@pytest.mark.gvsoc
@needs_gvsoc
def test_dot_1_reads_the_muls_pads_through_a_tail(tmp_path):
    """The dot 1 dry run at N = 2048, W_mul = W_sum = 10 (D144, DRY1-2): the mul writes tmp0's
    two pad words from A's and B's pads, 0 × 0 = 0, add's identity, so the sum folds them and
    the run's own check finds every output right."""
    import json
    from dataclasses import replace

    recipe = tmp_path / "dot_1_w10.json"
    recipe.write_text(json.dumps(replace(dot1_split(10, 10), name="dot_1_w10").to_dict()))
    f = run_flow(recipe, SNAX_FORGE, out=tmp_path / "flow")
    assert f.passed, summary(f)
    assert f.check["exit_code"] == 0 and f.check["data_errors"] == 0 and f.check["words"] == 1


@pytest.fixture(scope="module")
def beat_flows(tmp_path_factory):
    """dot 1 at N = 2048, W = 8 untraced and at --trace beat (the DRY1-2 dry run's 8/8)."""
    base = tmp_path_factory.mktemp("dot_1_2048")
    sets = {"N": 2048, "W": 8}
    return {lv: run_flow(DOT1, SNAX_FORGE, recipe_sets=sets, out=base / lv, trace_level=lv)
            for lv in ("off", "beat")}  # fmt: skip


@pytest.mark.gvsoc
@needs_gvsoc
def test_dot_1_converts_at_beat_level(beat_flows):
    """Two groups of one unit each at beat level (DRY1-2 R2, open item 96), dot 1's two
    accelerators: each unit's 256 fires under its own task, tmp0's words written by vmul_out
    and read by vsum_a, and the counters and the window those of the untraced run; vmul's
    COUNTER is the expected 259, its own set's streamer started just before its START (in
    dot_1_enc's shared streamer it was 256, its START 10 cycles after the streamers' start)."""
    off, beat = beat_flows["off"], beat_flows["beat"]
    assert off.passed and beat.passed, summary(beat)
    trace = read_run(beat.out / "run").trace
    fires = {u: sum(1 for e in trace.of_kind("fire") if e["src"] == u) for u in ("vmul", "vsum")}
    assert fires == {"vmul": 256, "vsum": 256}
    tmp0 = [e for e in trace.of_kind("grant") if e["region"] == "tmp0"]
    assert {e["src"].split(".")[0] for e in tmp0 if e["op"] == "w"} == {"vmul_out"}
    assert {e["src"].split(".")[0] for e in tmp0 if e["op"] == "r"} == {"vsum_a"}
    assert len({e["addr"] for e in tmp0 if e["op"] == "w"}) == 2048
    counters = {t: x["counter"] for t, x in beat.check["tasks"].items()}
    assert counters == {t: x["counter"] for t, x in off.check["tasks"].items()}
    assert counters == {"mult_vmul": 259, "sum_vsum": 259}
    assert (
        read_run(beat.out / "run").profile.total_cycles
        == read_run(off.out / "run").profile.total_cycles
    )
