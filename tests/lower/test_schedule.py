"""The design point's schedule: mvt_v3_1 at W = 8, its two halves two accelerators
(``mvt`` and ``mvt_1``) that share no data, under serial, pipeline and pipeline_compute; mvt_v3,
the same halves on one shared group, gives one task list under all three; sequential syncs after
every start."""

from __future__ import annotations

import re
from dataclasses import replace
from functools import cache

import pytest

from snax_forge.design import Design, DesignPoint, Platform, run_checks
from snax_forge.design.check import SCHEDULES
from snax_forge.lower import task_list
from snax_forge.lower.derive import sequential
from snax_forge.lower.tasks import Configure, Start, Sync
from tests.design.helpers import SNAX_FORGE
from tests.helpers import steps

from .test_c import functions, quiet
from .test_c import quiet_build as quiet_text


@cache
def mvt(name: str = "mvt_v3_1") -> DesignPoint:
    d = Design(Platform.load(SNAX_FORGE), steps(name, W=8)[-1])
    assert run_checks(d) == []
    return DesignPoint.of(d, name)


def at(schedule: str, name: str = "mvt_v3_1") -> DesignPoint:
    return replace(mvt(name), schedule=schedule)


def order(schedule: str) -> list[str]:
    """The steps, a half's streamer and accelerator configures as ``conf <acc>``, once."""
    out: list[str] = []
    for s in task_list(at(schedule)).steps:
        if isinstance(s, Configure):
            half_2 = "x2" in s.component or s.component == "mvt_1"
            word = s.task_name if s.type == "dma" else f"conf {'mvt_1' if half_2 else 'mvt'}"
            if not out or out[-1] != word:
                out.append(word)
        elif isinstance(s, Start):
            out.append(f"start {s.tasks[-1]}")
        else:
            out.append(f"sync {s.task}")
    return out


LOADS_1 = ["load_A", "start load_A", "load_y1", "start load_y1", "load_x1", "start load_x1"]
LOADS_2 = ["load_y2", "start load_y2", "load_x2", "start load_x2"]
STORES = ["store_x1", "start store_x1", "store_x2", "start store_x2", "sync store_x1",
          "sync store_x2"]  # fmt: skip


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_every_schedule_keeps_the_tasks_and_their_after(schedule):
    """The same tasks, each configured with the same values and after, each started after its
    after (TaskList's checks) and synced the same; only the order and the syncs of serial
    (on half 1, before half 2) and sequential (after every start) differ."""
    serial, tl = task_list(at("serial")), task_list(at(schedule))
    assert tl.configured() == serial.configured()
    started = {t: i for i, s in enumerate(tl.steps) if isinstance(s, Start) for t in s.tasks}
    for t, c in tl.configured().items():
        assert all(started[a] < started[t] for a in c.after), t
    syncs = [s.task for s in tl.steps if isinstance(s, Sync)]
    if schedule == "sequential":
        assert syncs == [s.tasks[-1] for s in tl.steps if isinstance(s, Start)]
    else:
        assert syncs == (["tmp0_mult_mvt"] if schedule == "serial" else []) + ["store_x1",
                                                                               "store_x2"]  # fmt: skip


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_the_shared_form_gives_one_task_list_under_every_schedule(schedule):
    """The halves on one group: one list, sequential's with a sync after every start."""
    want = task_list(at("serial", "mvt_v3"))
    if schedule == "sequential":
        want = sequential(want)
    assert task_list(at(schedule, "mvt_v3")) == want


def test_sequential_syncs_after_every_start():
    assert order("sequential") == [
        "load_A", "start load_A", "sync load_A", "load_y1", "start load_y1", "sync load_y1",
        "load_x1", "start load_x1", "sync load_x1", "conf mvt", "start tmp0_mult_mvt",
        "sync tmp0_mult_mvt", "store_x1", "start store_x1", "sync store_x1", "load_y2",
        "start load_y2", "sync load_y2", "load_x2", "start load_x2", "sync load_x2", "conf mvt_1",
        "start tmp2_mult_mvt_1", "sync tmp2_mult_mvt_1", "store_x2", "start store_x2",
        "sync store_x2",
    ]  # fmt: skip


def test_the_schedule_round_trips_through_the_design_point():
    p = at("pipeline_compute")
    assert DesignPoint.from_dict(p.to_dict()) == p
    d = p.to_dict()
    del d["schedule"]  # a point written before the field is serial
    assert DesignPoint.from_dict(d).schedule == "serial"


@pytest.mark.parametrize("schedule", ["pipeline", "pipeline_compute"])
def test_the_quiet_build_still_waits_on_every_start_and_each_build_once(tmp_path, schedule,
                                                                         monkeypatch):  # fmt: skip
    """As test_c's quiet-build test, on the pipelined mvt_v3_1: each START followed by its
    polls in the quiet build, and one poll of each half in either build."""
    from tests.gen import helpers as gen_helpers

    real = gen_helpers.DesignPoint.of
    monkeypatch.setattr(gen_helpers.DesignPoint, "of",
                        lambda d, n: replace(real(d, n), schedule=schedule))  # fmt: skip
    src, _ = quiet(tmp_path, graph=steps("mvt_v3_1", W=8)[-1], name="mvt_v3_1")
    q = functions(quiet_text(src), quiet=True)["forge_core_0"]
    default = functions(src)["forge_core_0"]
    for unit in ("MVT", "MVT_1"):
        start = q.index(f"csrw_ss({unit}_RW_START, 1);")
        assert q[start + 1] == f"while (csrr_ss({unit}_RO_STATUS) & 1) {{"
        for lines in (q, default):
            assert lines.count(f"while (csrr_ss({unit}_RO_STATUS) & 1) {{") == 1
    # the default build: both STARTs before either poll
    starts = [i for i, x in enumerate(default) if re.fullmatch(r"csrw_ss\(\w+_RW_START, 1\);", x)]
    polls = [i for i, x in enumerate(default) if x.endswith("_RO_STATUS) & 1) {")]
    assert len(starts) == 2 and max(starts) < min(polls)
