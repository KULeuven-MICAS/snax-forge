"""The whole path, kernel to a SNAX-GVSoC run (D136).

1. up to the generator on vecadd and vsum, SNAX-GVSoC stubbed (helpers.stub):
   main.c without marks and with forge_check, data.h's goldens the kernel's
   reference, the make calls, the folder, run.json
2. the build stage: when it runs, its output live on the terminal and in
   build.log (the stub's, then make's own through a fake make on PATH), the
   checkout's warning
3. the check: the program's output parsed (pass, data errors, a missing
   line, a nonzero exit) and the run's exit code from make
4. what stops the flow: a design problem, the generator's refusals, a make
   failure, the trace level beat, a run without its events
5. names, the log and the command line; the trace level task, stubbed
6. the full flow on SNAX-GVSoC (marker ``gvsoc``), at trace off and task
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from snax_forge.design import DesignError, DesignPoint
from snax_forge.design.check import CHECKS
from snax_forge.flow import (
    STAGE_NAMES,
    Checkout,
    FlowError,
    build_reason,
    check_output,
    checkout,
    default_name,
    run_exit_code,
    run_flow,
    summary,
)
from snax_forge.flow import run as flow_run
from snax_forge.flow.run import with_level
from snax_forge.gen import Manifest
from snax_forge.run import read_run
from snax_forge.sandbox import Recipe
from snax_forge.sdfg.loader import load as load_kernel
from tests.design.helpers import REPO, SMALL16, SNAX_FORGE
from tests.run.helpers import point_symbols

from . import helpers
from .helpers import COMMIT, PLAIN, PLAINS, built, golden, main, needs_gvsoc, stub

RECIPES = REPO / "recipes"


def flow(tmp_path, design="vecadd", **kw):
    kw.setdefault("graph_path", PLAINS[design])
    return run_flow(RECIPES / f"{design}.json", SNAX_FORGE, out=tmp_path / "flow", **kw)


# =============================================================================
# 1. Up to the generator, SNAX-GVSoC stubbed
# =============================================================================


@pytest.mark.parametrize("design", ["vecadd", "vsum"])
def test_up_to_gen_then_the_stubbed_run(tmp_path, monkeypatch, design):
    fake = stub(monkeypatch, tmp_path)
    f = flow(tmp_path, design)
    out = tmp_path / "flow"
    assert f.passed and f.total_cycles == 226 and f.check["exit_code"] == 0
    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert {"sandbox/recipe.json", "sandbox/2_bind.snaxdfg", "design/design_point.json",
            "tasks.json", "gen/snax_forge.hjson", "gen/snax_forge_streamer.hjson",
            "gen/accelerator.json", "gen/main.c", "gen/data.h", "gen/manifest.json",
            "gen/include/acc_csr.h", "sw/forge.elf", "run/stdout.txt", "run/stderr.txt",
            "run/run.json", "run/profile.json", "run/events.log", "flow.log"} <= names  # fmt: skip
    assert not {"run/trace.jsonl", "run/trace_meta.json"} & names  # trace off
    main_c = (out / "gen" / "main.c").read_text()
    assert "FORGE_MARK" not in main_c and "forge_main" not in main_c
    assert "static __attribute__((noinline)) int forge_check(uint32_t t0) {" in main_c
    # data.h's goldens are the kernel's reference on its make_inputs (seed 0)
    spec = load_kernel(design)
    inputs = spec.make_inputs(np.random.default_rng(0), n=64)
    spec.reference(**inputs)
    data_h = (out / "gen" / "data.h").read_text()
    for c in spec.inout:
        want = inputs[c].ravel().tolist()
        assert golden(data_h, f"{c}_golden")[: len(want)] == want
    # build (none yet), sw and run, each through gvsoc/Makefile with the flow's folders
    assert [g for g, _ in fake.calls] == ["build", "sw", "run"]
    sw, run = fake.calls[1][1], fake.calls[2][1]
    assert sw == {"APPS": "forge", "FORGE": str((out / "gen").resolve()), "DESIGN": f.name,
                  "FORGE_BUILD": str((out / "sw").resolve())}  # fmt: skip
    assert run["TARGET"] == "snax_hjson" and run["ELF"] == str((out / "sw/forge.elf").resolve())
    assert run["GVRUN_ARGS"] == f"--param hjson={(out / 'gen/snax_forge.hjson').resolve()}"
    assert run["WORK_DIR"] == str((out / "run" / "work").resolve())
    assert run["EVENTS"] == str((out / "run" / "events.log").resolve())
    assert run["TRACES"] == "summary"
    beats = {t.name: t.beats for t in Manifest.load(out / "gen" / "manifest.json").tasks}
    assert f.check["tasks"] == {t: {"counter": n + 3, "beats": n} for t, n in beats.items()}


def test_the_generated_folder_names_the_design_point_from_the_flows_folder(tmp_path, monkeypatch):
    """gen/ is byte-identical wherever the flow's folder is: its files name the design point
    as design/design_point.json."""
    stub(monkeypatch, tmp_path)
    a = run_flow(RECIPES / "vecadd.json", SNAX_FORGE, graph_path=PLAIN, out=tmp_path / "a" / "x")
    b = run_flow(RECIPES / "vecadd.json", SNAX_FORGE, graph_path=PLAIN, out=tmp_path / "b" / "y")
    assert a.manifest.design_point == b.manifest.design_point == "design/design_point.json"
    for f in a.manifest.files:
        assert (a.out / "gen" / f).read_bytes() == (b.out / "gen" / f).read_bytes(), f


def test_run_json_records_what_ran(tmp_path, monkeypatch):
    stub(monkeypatch, tmp_path, dirty=True)
    f = flow(tmp_path)
    out = tmp_path / "flow"
    rec = json.loads((out / "run" / "run.json").read_text())
    assert rec == {
        "design_point": "design/design_point.json",
        "hjson": [
            "gen/snax_forge.hjson",
            "gen/snax_forge_streamer.hjson",
            "gen/accelerator.json",
        ],
        "elf": "sw/forge.elf",
        "elf_sha256": f.elf_sha256,
        "elf_symbols": {
            c: {"addr": a, "size": n}
            for c, (a, n) in point_symbols(f.point).items()
            if not c.endswith("_golden")
        },
        "target": "snax_hjson",
        "snax_gvsoc": {"commit": COMMIT, "dirty": True},
        "trace_level": "off",
        "trace_sources": [],
        "vcd": False,
    }
    prof = json.loads((out / "run" / "profile.json").read_text())
    assert (prof["total_cycles"], prof["functional_check"]) == (226, f.check)
    assert set(prof) == {"total_cycles", "accelerators", "streamers", "cores", "dmas", "ports",
                         "banks", "functional_check"}  # fmt: skip
    assert f.check == {
        "kernel": "vecadd", "seed": 0, "symbols": {"N": 64}, "exit_code": 0, "data_errors": 0,
        "words": 64, "tasks": {"add_acc": {"counter": 19, "beats": 16}}, "passed": True,
    }  # fmt: skip


def test_an_earlier_runs_files_go_first(tmp_path, monkeypatch):
    stub(monkeypatch, tmp_path)
    out = tmp_path / "flow"
    for stale in ("report/run.md", "run/trace.jsonl", "gen/old.h"):
        (out / stale).parent.mkdir(parents=True, exist_ok=True)
        (out / stale).write_text("old")
    flow(tmp_path)
    for stale in ("run/trace.jsonl", "gen/old.h"):
        assert not (out / stale).exists(), stale
    assert (out / "report" / "run.md").read_text().startswith("# Run report: ")  # written again


# =============================================================================
# 2. The build stage
# =============================================================================


def test_the_build_is_skipped_when_up_to_date(tmp_path, monkeypatch):
    fake = stub(monkeypatch, tmp_path)
    built(fake)
    f = flow(tmp_path)
    assert [g for g, _ in fake.calls] == ["sw", "run"]
    assert f.build == "up to date"
    assert "  build     up to date (snax-gvsoc aaaaaaaa)" in summary(f)


def test_a_build_is_shown_live_and_logged(tmp_path, monkeypatch, capsys):
    fake = stub(monkeypatch, tmp_path)
    f = flow(tmp_path)
    log = tmp_path / "flow" / "build.log"
    first, *lines = capsys.readouterr().out.splitlines()
    assert first == (f"  build     building SNAX-GVSoC: no install in {fake.workdir}; log in {log}")
    assert lines == log.read_text().splitlines() == ["make build"]
    assert fake.live == ["build"]  # sw and gvsoc to their logs only
    assert f.build.startswith("built: no install in ")
    text = (tmp_path / "flow" / "flow.log").read_text()
    assert "building SNAX-GVSoC" not in text and "make build" not in text


@pytest.mark.parametrize("stage", ["build", "sw"])
def test_a_make_failure_names_its_log_and_tail(tmp_path, monkeypatch, capsys, stage):
    """The build's output is shown live too; sw's only in its log."""
    fake = stub(monkeypatch, tmp_path)
    fake.rc[stage] = 2
    fake.stderr[stage] = "cc: error: no such file\n"
    with pytest.raises(FlowError) as e:
        flow(tmp_path)
    msg = str(e.value)
    assert msg.startswith(f"{stage}: make {stage} failed (exit 2); its output is in ")
    assert msg.endswith(f"{stage}.log:\nmake {stage}\ncc: error: no such file")
    assert (tmp_path / "flow" / "flow.log").read_text().startswith(f"flow vecadd FAILED\n{stage}: ")
    if stage == "build":
        assert capsys.readouterr().out.splitlines()[1:] == ["make build", "cc: error: no such file"]


def test_an_up_to_date_build_prints_nothing(tmp_path, monkeypatch, capsys):
    fake = stub(monkeypatch, tmp_path)
    built(fake)
    flow(tmp_path)
    assert capsys.readouterr().out == ""
    assert fake.live == [] and not (tmp_path / "flow" / "build.log").exists()


def fake_make(tmp_path, monkeypatch, script: str) -> Path:
    """A ``make`` on PATH that runs ``script`` (sh) in place of gvsoc/Makefile."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    make = bin_ / "make"
    make.write_text(f"#!/bin/sh\n{script}\n")
    make.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    return make


def test_a_live_make_is_teed_merged_and_unbuffered(tmp_path, monkeypatch, capsys):
    fake_make(
        tmp_path, monkeypatch, 'echo "one $PYTHONUNBUFFERED"\necho two >&2\necho three\nexit 3'
    )
    log = tmp_path / "build.log"
    assert flow_run.run_gvsoc("build", {}, log, live=True) == 3
    assert capsys.readouterr().out == log.read_text() == "one 1\ntwo\nthree\n"


def test_an_interrupted_live_make_is_terminated(tmp_path, monkeypatch):
    pid = tmp_path / "pid"
    fake_make(tmp_path, monkeypatch, f"echo $$ > {pid}\necho started\nexec sleep 60")

    class Interrupt:
        def write(self, line):
            raise KeyboardInterrupt

    monkeypatch.setattr(flow_run.sys, "stdout", Interrupt())
    with pytest.raises(KeyboardInterrupt):
        flow_run.run_gvsoc("build", {}, tmp_path / "build.log", live=True)
    with pytest.raises(ProcessLookupError):  # terminated and waited for
        os.kill(int(pid.read_text()), 0)
    assert (tmp_path / "build.log").read_text() == "started\n"


def test_build_reason(tmp_path):
    work = tmp_path / "w"
    c = Checkout(tmp_path / "sub", COMMIT, False, COMMIT)
    assert build_reason(c, work).startswith("no install in ")
    (work / "install" / "bin").mkdir(parents=True)
    (work / "install" / "bin" / "gvrun").write_text("")
    assert "was built from an unknown checkout" in build_reason(c, work)
    (work / "checkout.txt").write_text(f"{tmp_path / 'other'}\n{COMMIT}\n")
    assert build_reason(c, work).endswith(f"was built from {tmp_path / 'other'}")
    (work / "checkout.txt").write_text(f"{c.path}\n")  # a build that failed: no commit
    assert (
        build_reason(c, work)
        == "built at snax-gvsoc an unknown commit, the checkout is at aaaaaaaa"
    )
    (work / "checkout.txt").write_text(f"{c.path}\n{'b' * 40}\n")
    assert build_reason(c, work) == "built at snax-gvsoc bbbbbbbb, the checkout is at aaaaaaaa"
    (work / "checkout.txt").write_text(f"{c.path}\n{COMMIT}\n")
    assert build_reason(c, work) is None
    dirty = Checkout(c.path, COMMIT, True, COMMIT)
    assert build_reason(dirty, work) == "gvsoc/snax-gvsoc has uncommitted changes"


def test_a_submodule_off_its_pin_is_warned_about(tmp_path, monkeypatch):
    stub(monkeypatch, tmp_path, pinned="b" * 40)
    f = flow(tmp_path)
    line = "  WARNING   gvsoc/snax-gvsoc is at aaaaaaaa, snax-forge's tree pins bbbbbbbb"
    assert line in summary(f).splitlines()
    assert line in (tmp_path / "flow" / "flow.log").read_text().splitlines()


def test_a_submodule_not_checked_out_is_named(tmp_path):
    with pytest.raises(FlowError, match=r"build: .* is not checked out: git submodule update"):
        checkout(tmp_path)


# =============================================================================
# 3. The check
# =============================================================================

PASS = """make: Entering directory '/x/gvsoc'
forge cycles: 226
forge data: 0 errors (64 words)
forge counter add_acc: 19
forge errors: 0
make: Leaving directory '/x/gvsoc'
"""


def test_the_check_of_a_passing_program():
    assert check_output(PASS, 0, {"add_acc": 16}) == (226, {
        "exit_code": 0, "data_errors": 0, "words": 64,
        "tasks": {"add_acc": {"counter": 19, "beats": 16}}, "passed": True,
    })  # fmt: skip


def test_the_check_of_data_errors_and_a_nonzero_exit():
    text = PASS.replace("0 errors (64", "2 errors (64").replace(
        "forge errors: 0", "forge errors: 2"
    )
    cycles, check = check_output(text, 1, {"add_acc": 16})
    assert cycles == 226 and check["data_errors"] == 2 and check["exit_code"] == 1
    assert not check["passed"]
    # the exit code alone decides
    assert not check_output(PASS, 3, {"add_acc": 16})[1]["passed"]


@pytest.mark.parametrize(
    ("drop", "says"),
    [("forge cycles: 226\n", "forge cycles:"), ("forge data: 0 errors (64 words)\n", "forge data:"),
     ("forge counter add_acc: 19\n", "forge counter add_acc:")],
    ids=["cycles", "data", "counter"],
)  # fmt: skip
def test_a_missing_line_is_named(drop, says):
    with pytest.raises(FlowError, match=f"check: no `{says}` line in the program's output"):
        check_output(PASS.replace(drop, ""), 0, {"add_acc": 16})


def test_the_runs_exit_code_from_make():
    assert run_exit_code(0, "") == 0
    assert run_exit_code(2, "Warning: ...\nmake: *** [Makefile:110: run] Error 1\n") == 1
    with pytest.raises(FlowError, match=r"gvsoc: make run failed \(exit 2\) before running gvrun"):
        run_exit_code(2, "make: *** No rule to make target 'run'.  Stop.\n")


def test_a_failing_program_fails_the_flow_not_the_stage(tmp_path, monkeypatch):
    fake = stub(monkeypatch, tmp_path)
    fake.rc["run"] = 2
    fake.stderr["run"] = "make: *** [Makefile:110: run] Error 1\n"
    fake.stdout = PASS.replace("0 errors (64", "2 errors (64")
    f = flow(tmp_path)
    assert not f.passed and f.check["exit_code"] == 1 and f.check["data_errors"] == 2
    assert "  check     FAILED: 226 cycles (main to forge_check); data 2 errors" in summary(f)


# =============================================================================
# 4. What stops it
# =============================================================================


def test_a_design_problem_stops_the_flow_before_gen(tmp_path, monkeypatch):
    fake = stub(monkeypatch, tmp_path)
    with pytest.raises(DesignError) as e:
        flow(tmp_path, platform_sets=[("l1.rows", 1)])
    assert "memory.fit" in e.value.codes
    assert not (tmp_path / "flow" / "gen").exists() and fake.calls == []


def test_a_platform_without_a_source_stops_at_gen(tmp_path, monkeypatch):
    fake = stub(monkeypatch, tmp_path)
    with pytest.raises(FlowError, match=r"gen: \[gen.source\] platform small16 has no source"):
        run_flow(RECIPES / "vecadd.json", SMALL16, graph_path=PLAIN, out=tmp_path / "flow")
    assert fake.calls == []
    assert (tmp_path / "flow" / "flow.log").read_text().startswith("flow vecadd FAILED\n")


def test_trace_sources_go_with_the_beat_levels_and_name_this_run(tmp_path, monkeypatch):
    fake = stub(monkeypatch, tmp_path)
    with pytest.raises(FlowError, match="trace sources \\['acc'\\]: they go with beat"):
        flow(tmp_path, trace_level="task", trace_sources=["acc"])
    with pytest.raises(FlowError, match="trace level 'cycle': one of off, task, beat"):
        flow(tmp_path, trace_level="cycle")
    assert fake.calls == [] and not (tmp_path / "flow").exists()
    with pytest.raises(FlowError, match=r"gvsoc: \[sources.unknown\] sources \['acc_z'\]"):
        flow(tmp_path, trace_level="beat", trace_sources=["acc", "acc_z"])


@pytest.mark.parametrize(
    "change, said",
    [
        (lambda s: {k: v for k, v in s.items() if k != "C"}, "container C lives in L2, the ELF"),
        (lambda s: {**s, "A": (s["A"][0], 8)}, "container A: its symbol is 8 bytes"),
    ],
)
def test_the_elf_must_hold_each_l2_container(tmp_path, monkeypatch, change, said):
    stub(monkeypatch, tmp_path)
    real = helpers.point_symbols
    monkeypatch.setattr(helpers, "point_symbols", lambda point: change(real(point)))
    with pytest.raises(FlowError, match=r"^sw: .*forge\.elf: \[elf\.") as e:
        flow(tmp_path)
    assert said in str(e.value)


def test_a_run_without_its_events_stops_at_convert(tmp_path, monkeypatch):
    stub(monkeypatch, tmp_path)
    monkeypatch.setattr(helpers, "write_events", lambda variables: None)
    with pytest.raises(FlowError, match=r"convert: no .*run/events.log after the run"):
        flow(tmp_path)


def test_events_that_do_not_convert_stop_at_convert(tmp_path, monkeypatch):
    stub(monkeypatch, tmp_path)
    monkeypatch.setattr(
        helpers,
        "write_events",
        lambda variables: Path(variables["EVENTS"]).write_text("forge cycles: 226\n"),
    )
    with pytest.raises(FlowError, match=r"convert: \[events.line\]"):
        flow(tmp_path)


# =============================================================================
# 5. Names, the log, the command line
# =============================================================================


def test_default_name():
    """Every --set: recipe params, then platform and memory paths without their prefix."""
    r = Recipe.load(RECIPES / "vecadd.json")
    assert default_name(r, {}) == "vecadd" and default_name(r, {"W": 8}) == "vecadd_W8"
    assert default_name(r, {}, memory_sets=[("B.l1.base", 576)]) == "vecadd_B.l1.base576"
    assert default_name(r, {}, [("l1.n_banks", 32)]) == "vecadd_l1.n_banks32"
    both = default_name(r, {"W": 8}, [("l1.n_banks", 32)], [("B.l1.base", 576)])
    assert both == "vecadd_W8_l1.n_banks32_B.l1.base576"
    # a value that is not a plain word keeps one plain directory name
    assert default_name(r, {}, [("x.y", [2, 2]), ("z", "a/b c")]) == "vecadd_x.y2,2_zabc"


def test_the_summary_names_every_stage(tmp_path, monkeypatch):
    stub(monkeypatch, tmp_path)
    lines = summary(flow(tmp_path)).splitlines()
    assert [x.split()[0] for x in lines[1:]] == [
        "sandbox", "design", "lower", "gen", "build", "sw", "gvsoc", "check", "convert", "report"
    ]  # fmt: skip
    assert lines[-2] == (
        "  convert   run/events.log -> profile.json (the summary lines; trace off)"
    )
    assert lines[-1] == "  report    report/design.md, report/run.md"
    assert (tmp_path / "flow" / "report" / "run.md").read_text().startswith("# Run report: ")


def test_the_trace_level_task_stubbed(tmp_path, monkeypatch):
    """At task the run turns the events on, the folder's default name carries the level, and
    convert writes the trace (its invariants checked, tests/run)."""
    fake = stub(monkeypatch, tmp_path)
    f = run_flow(RECIPES / "vecadd.json", SNAX_FORGE, graph_path=PLAIN, trace_level="task",
                 out=tmp_path / "flow")  # fmt: skip
    assert fake.calls[-1][1]["TRACES"] == "summary events"
    run = tmp_path / "flow" / "run"
    meta = json.loads((run / "trace_meta.json").read_text())
    assert (meta["level"], meta["total_cycles"], meta["boot"]) == ("task", 226, 1005)
    assert len((run / "trace.jsonl").read_text().splitlines()) == len(f.trace.events) > 0
    assert json.loads((run / "run.json").read_text())["trace_level"] == "task"
    assert f"{len(f.trace.events)} events in the window; boot 1005" in summary(f)
    assert with_level("vecadd", "task") == "vecadd_task" and with_level("vecadd", "off") == "vecadd"


@pytest.mark.parametrize("level", ["beat", "stall", "finegrained"])
def test_the_beat_levels_stubbed(tmp_path, monkeypatch, level):
    """Above task the run turns on the beat traces of its sources and the crossbar's inputs
    property, convert keeps its beat events (their invariants checked) and the filter, and the
    profile is the task run's, byte for byte."""
    fake = stub(monkeypatch, tmp_path)
    run_flow(RECIPES / "vecadd.json", SNAX_FORGE, graph_path=PLAIN, trace_level="task",
             out=tmp_path / "task")  # fmt: skip
    f = run_flow(RECIPES / "vecadd.json", SNAX_FORGE, graph_path=PLAIN, trace_level=level,
                 out=tmp_path / level)  # fmt: skip
    v = fake.calls[-1][1]
    traces = v["TRACES"].split()
    assert traces[:5] == ["summary", "events", "snax_acc0/unit/beats", "snax_acc0/streamer/beats",
                          "tcdm/snax_xbar/beats"]  # fmt: skip
    assert ("snax_acc0/streamer/stalls" in traces) == (level != "beat")
    assert [x for x in traces if x.endswith("/insn")] == (
        ["pe0/insn", "pe1/insn"] if level == "finegrained" else []
    )
    lanes = ",".join(f"snax_acc0.streamer.{m}.lane{i}" for m in ("reader0", "reader1", "writer0")
                     for i in range(4))  # fmt: skip
    assert f"'chip/soc/cluster_0/tcdm/snax_xbar/trace_inputs={lanes},dma'" in v["GVRUN_ARGS"]
    kinds = {e["k"] for e in f.trace.events}
    assert {"grant", "dma", "burst", "fire", "push", "class"} <= kinds
    assert ("fifo" in kinds) == ("stall" in kinds) == (level != "beat")
    assert ("insn" in kinds) == (level == "finegrained")
    sources = ["acc", "acc_a", "acc_b", "acc_out", "dma"]
    sources += ["core0", "core1"] if level == "finegrained" else []
    run = tmp_path / level / "run"
    assert json.loads((run / "trace_meta.json").read_text())["filter"] == sources
    assert json.loads((run / "run.json").read_text())["trace_sources"] == sources
    assert f"sources {', '.join(sources)})" in summary(f)
    task = (tmp_path / "task" / "run" / "profile.json").read_bytes()
    assert (run / "profile.json").read_bytes() == task


def test_named_sources_and_the_vcd_stubbed(tmp_path, monkeypatch):
    """Named sources say whose lines are written, a core's crossbar lines too; --vcd puts
    gvrun's VCD and its gtkwave view into run/."""
    fake = stub(monkeypatch, tmp_path)
    f = run_flow(RECIPES / "vecadd.json", SNAX_FORGE, graph_path=PLAIN, trace_level="stall",
                 trace_sources=["core0", "acc_a", "acc"], vcd=True, out=tmp_path / "f")  # fmt: skip
    v = fake.calls[-1][1]
    assert v["TRACES"].split() == ["summary", "events", "snax_acc0/unit/beats",
                                   "snax_acc0/streamer/beats", "tcdm/snax_xbar/beats",
                                   "snax_acc0/streamer/stalls",
                                   "tcdm/snax_xbar/stalls"]  # fmt: skip
    lanes = ",".join(f"snax_acc0.streamer.reader0.lane{i}" for i in range(4))
    assert f"trace_inputs={lanes},core0'" in v["GVRUN_ARGS"]
    assert (
        "'--vcd' '--event=.*/snax_acc0/unit/.*' '--event=.*/snax_acc0/streamer/.*'"
        in (v["GVRUN_ARGS"])
    )
    assert {e["src"] for e in f.trace.events if e["k"] in ("grant", "fifo")} == {
        "acc_a", *(f"acc_a.lane{i}" for i in range(4))}  # fmt: skip
    run = tmp_path / "f" / "run"
    assert (run / "all.vcd").is_file() and not (run / "work" / "all.vcd").exists()
    assert (run / "view.gtkw").read_text() == f'[dumpfile] "{(run / "all.vcd").resolve()}"\n'
    rec = json.loads((run / "run.json").read_text())
    assert (rec["trace_sources"], rec["vcd"]) == (["acc", "acc_a", "core0"], True)
    assert "run/all.vcd, run/view.gtkw (gtkwave)" in summary(f)


def test_the_log_is_the_summary_the_checks_and_the_stage_times(tmp_path, monkeypatch):
    stub(monkeypatch, tmp_path)
    f = flow(tmp_path)
    log = (tmp_path / "flow" / "flow.log").read_text()
    head, _, rest = log.partition("\n\ndesign checks that ran ")
    checks, _, times = rest.partition("\n\nstage times (wall clock, this machine):\n")
    assert head == summary(f)
    listed = [c for line in checks.splitlines()[1:] for c in line.split(None, 1)[1].split(", ")]
    assert listed == list(CHECKS)
    rows = [line.split() for line in times.splitlines()]
    assert [r[0] for r in rows] == [*STAGE_NAMES, "total"] and all(r[2] == "s" for r in rows)
    secs = [float(r[1]) for r in rows]
    assert all(s >= 0 for s in secs) and abs(sum(secs[:-1]) - secs[-1]) < 0.01
    assert list(f.times) == list(STAGE_NAMES)
    assert "gen" in f.times and "gvsoc" in f.times and "report" in f.times


def test_cli_runs_and_says_what_it_did(tmp_path, monkeypatch, capsys):
    built(stub(monkeypatch, tmp_path))  # a build would print its lines first
    args = [RECIPES / "vecadd.json", "--platform", SNAX_FORGE, "--graph", PLAIN]
    assert main(*args, "--out", tmp_path / "f") == 0
    out = capsys.readouterr().out
    assert out.startswith("flow vecadd (vecadd: W=4, N=64; N=64) on snax_forge -> ")
    assert "  check     passed: 226 cycles (main to forge_check)" in out
    assert (tmp_path / "f" / "flow.log").read_text().startswith(out)
    # the sets route as before; the folder carries them
    assert main(*args, "--out", tmp_path / "g", "--set", "W=8",
                "--set", "platform.streamers.default.fifo_depth=4") == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "flow vecadd_W8_streamers.default.fifo_depth4 (vecadd: W=8, N=64; N=64)" in out
    pf = json.loads((tmp_path / "g" / "design" / "platform.json").read_text())
    assert pf["changes"] == {"streamers.default.fifo_depth": 4}
    assert DesignPoint.load(tmp_path / "g" / "design" / "design_point.json").name == "vecadd_W8"


def test_cli_reports_a_failing_stage_and_a_failing_program(tmp_path, monkeypatch, capsys):
    fake = stub(monkeypatch, tmp_path)
    args = [RECIPES / "vecadd.json", "--platform", SNAX_FORGE, "--graph", PLAIN, "--out", tmp_path / "f"]  # fmt: skip
    with pytest.raises(SystemExit):
        main(*args, "--trace", "task:acc")
    assert "sources go with beat, stall or finegrained" in capsys.readouterr().err
    assert main(*args, "--set", "Q=1") == 1
    assert "no param 'Q'" in capsys.readouterr().err
    fake.rc["run"] = 2
    fake.stderr["run"] = "make: *** [Makefile:110: run] Error 1\n"
    assert main(*args) == 1
    assert "  check     FAILED: " in capsys.readouterr().out


# =============================================================================
# 6. The full flow on SNAX-GVSoC
# =============================================================================


@pytest.mark.gvsoc
@needs_gvsoc
@pytest.mark.parametrize("level", ["off", "task"])
@pytest.mark.parametrize("design", ["vecadd", "vsum"])
def test_the_flow_on_snax_gvsoc(tmp_path, design, level):
    """From the kernel: a checked SNAX-GVSoC run, exit 0, its window and counters printed; its
    events converted (convert's invariants hold, or the flow fails)."""
    f = run_flow(RECIPES / f"{design}.json", SNAX_FORGE, out=tmp_path / design, trace_level=level)
    assert f.passed, summary(f)
    run = read_run(tmp_path / design / "run")
    assert run.profile.total_cycles == f.total_cycles and (run.trace is None) == (level == "off")
    assert run.profile.accelerators["acc"].firings == sum(
        t["beats"] for t in f.check["tasks"].values()
    )
    assert f.check["exit_code"] == 0 and f.check["data_errors"] == 0 and f.total_cycles > 0
    assert all(t["counter"] >= t["beats"] for t in f.check["tasks"].values())
    stdout = (tmp_path / design / "run" / "stdout.txt").read_text()
    assert f"forge cycles: {f.total_cycles}\n" in stdout
    assert flow_run.checkout().commit == f.checkout.commit


@pytest.mark.gvsoc
@needs_gvsoc
def test_matadd_the_first_2d_bind_on_snax_gvsoc(tmp_path):
    """matadd at W = 4, M = 8, N = 16: two firing loops (the rows, the column beats), so each
    streamer has two temporal loops (D84); 32 firings, every output checked."""
    f = run_flow(RECIPES / "matadd.json", SNAX_FORGE, out=tmp_path / "matadd")
    assert f.passed, summary(f)
    assert f.check["exit_code"] == 0 and f.check["data_errors"] == 0 and f.total_cycles > 0
    assert {t: v["beats"] for t, v in f.check["tasks"].items()} == {"add_acc": 32}
    assert all(t["counter"] >= t["beats"] for t in f.check["tasks"].values())
    text = (tmp_path / "matadd" / "gen" / "snax_forge_streamer.hjson").read_text()
    streamer = json.loads("".join(ln for ln in text.splitlines(True) if not ln.startswith("//")))
    assert streamer["data_reader_params"]["temporal_dim"] == [2, 2]
    assert streamer["data_writer_params"]["temporal_dim"] == [2]


@pytest.mark.gvsoc
@needs_gvsoc
@pytest.mark.parametrize("design, beats", [("rowsum", 128), ("rowsum_serial", 64)])
def test_rowsum_restarts_every_row_on_snax_gvsoc(tmp_path, design, beats):
    """rowsum: one sum per row, T = N / W = 16 beats each (open item 81), so the accumulator
    restarts M times in one run; A's reader has two temporal loops, y's writer one."""
    f = run_flow(RECIPES / f"{design}.json", SNAX_FORGE, out=tmp_path / design)
    assert f.passed, summary(f)
    assert f.check["exit_code"] == 0 and f.check["data_errors"] == 0 and f.total_cycles > 0
    assert {t: v["beats"] for t, v in f.check["tasks"].items()} == {"sum_acc": beats}
    assert all(t["counter"] >= t["beats"] for t in f.check["tasks"].values())
    gen = tmp_path / design / "gen"
    text = (gen / "snax_forge_streamer.hjson").read_text()
    streamer = json.loads("".join(ln for ln in text.splitlines(True) if not ln.startswith("//")))
    assert streamer["data_reader_params"]["temporal_dim"] == [2]
    assert streamer["data_writer_params"]["temporal_dim"] == [1]
    assert "csrw_ss(ACC_RW_LEN, 16);" in (gen / "main.c").read_text()


@pytest.mark.gvsoc
@needs_gvsoc
def test_a_64_kib_l1_runs_on_its_own_runtime(tmp_path):
    """vecadd at l1.rows = 256: the program and its runtime take the design point's cluster
    headers (snax-forge D126), so the stacks sit at the end of the 64 KiB L1 the model places."""
    f = run_flow(RECIPES / "vecadd.json", SNAX_FORGE, platform_sets=[("l1.rows", 256)],
                 out=tmp_path / "vecadd")  # fmt: skip
    assert f.passed, summary(f)
    include = tmp_path / "vecadd" / "gen" / "include"
    assert (
        "#define CLUSTER_TCDM_SIZE 0x10000\n" in (include / "snitch_cluster_addrmap.h").read_text()
    )
    assert f.check["exit_code"] == 0 and f.check["data_errors"] == 0 and f.total_cycles > 0
