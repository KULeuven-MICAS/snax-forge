"""Design and run reports (D99) on flow runs (C§7, run/): every number equals its source.

1. the design report against the design point and the cluster (run/cluster.py)
2. the run report against the profile and the trace, at task and at off
3. achieved II from the accelerator's done events
4. round trips, rendering, the recorded vecadd run, the command line
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from snax_forge.design import DesignPoint
from snax_forge.lower.tasks import TaskList
from snax_forge.report import (
    DesignReport,
    ReportError,
    RunReport,
    achieved_ii,
    render_design,
    render_run,
    reports_of,
)
from snax_forge.report.__main__ import main
from snax_forge.report.run import build_run, render_movement
from snax_forge.run import load_cluster, read_events, read_run, write_run
from snax_forge.run.convert import convert
from snax_forge.viz import api, schedule
from tests.flow.helpers import stubbed_flow

FIXTURE = Path(__file__).resolve().parents[1] / "run" / "fixtures" / "vecadd"


@pytest.fixture(scope="module")
def flows(tmp_path_factory):
    """vecadd at task and off and vsum at task, SNAX-GVSoC stubbed, as flow folders."""
    base = tmp_path_factory.mktemp("flows")
    return {(d, lv): stubbed_flow(base, d, lv) for d, lv in
            [("vecadd", "task"), ("vecadd", "off"), ("vsum", "task")]}  # fmt: skip


# =============================================================================
# 1. The design report
# =============================================================================


@pytest.mark.parametrize("key", [("vecadd", "task"), ("vsum", "task")])
def test_the_design_report_equals_its_sources(flows, key):
    folder = flows[key]
    r = reports_of(folder).design
    point = DesignPoint.load(folder / "design" / "design_point.json")
    cluster = api.load_run(folder).cluster
    h = r.header
    assert (h.kernel, h.recipe, h.symbols) == (key[0], key[0], {"N": 64})
    assert (h.platform, h.passes) == ("snax_forge", dict(point.memory.passes))
    assert [c.name for c in r.components] == cluster.sources
    (a,) = r.accelerators
    unit = cluster.accelerator.unit
    assert (a.instance, a.kind, a.target_ii, a.latency) == (
        "acc",
        "SnaxUnit",
        unit["ii"],
        unit["latency"],
    )
    assert a.ports == [f"{m.port} ← {m.name}" for m in cluster.accelerator.movers]
    assert [(s.name, s.lanes, s.fifo_depth) for s in r.streamers] == [
        (m.name, m.lanes, m.fifo_depth) for m in cluster.accelerator.movers
    ]
    l1 = next(m for m in r.memories if m.mem == "l1")
    assert (l1.columns, l1.word_bytes, l1.columns * l1.rows * l1.word_bytes) == (
        cluster.banks,
        cluster.bank_bytes,
        cluster.l1_size,
    )
    assert {x.name: x.base for x in l1.regions} == {
        c: lay["l1"].base for c, lay in point.memory.layouts.items() if "l1" in lay
    }
    assert l1.used_bytes == sum(x.bytes for x in l1.regions)
    # L2 at the program's link: the ELF's addresses run.json records
    l2 = next(m for m in r.memories if m.mem == "l2")
    syms = json.loads((folder / "run" / "run.json").read_text())["elf_symbols"]
    assert {x.name: x.base for x in l2.regions} == {c: s["addr"] for c, s in syms.items()}
    assert r.notes


# =============================================================================
# 2. The run report
# =============================================================================


def test_the_run_report_equals_its_sources(flows):
    folder = flows[("vecadd", "task")]
    r = reports_of(folder).run
    out = read_run(folder / "run")
    p, tr = out.profile, out.trace
    fc = p.functional_check
    assert (r.cycles, r.trace_level, r.boot) == (p.total_cycles, "task", tr.boot)
    assert (r.check.passed, r.check.data_errors, r.check.words) == (
        True,
        fc["data_errors"],
        fc["words"],
    )
    assert r.check.counters == {t: x["counter"] for t, x in fc["tasks"].items()}
    tasks = TaskList.load(folder / "tasks.json")
    assert sorted(t.name for t in r.tasks) == sorted(tasks.configured())
    for t in r.tasks:
        assert t.cycles == t.done - t.start > 0
        assert (t.direction is not None) == (t.block == "dma") == (t.bytes is not None)
    (a,) = r.accelerators
    pa = p.accelerators["acc"]
    assert (a.firings, a.utilisation, a.stall_in, a.stall_out, a.flush) == (
        pa.firings,
        pa.utilisation,
        pa.cycles["stall_in"],
        pa.cycles["stall_out"],
        pa.cycles["flush"],
    )
    (acc,) = [x for x in schedule.lanes(api.load_run(folder).index)["lanes"] if x["name"] == "acc"]
    assert a.span == sum(b["to"] - b["from"] for b in acc["bars"])
    assert a.utilisation_span == pa.cycles["busy"] / a.span
    for c in r.cores:
        pc = p.cores[c.name]
        assert (c.cycles, c.csr_writes, c.csr_reads) == (pc.cycles, pc.csr_writes, pc.csr_reads)
        assert sum(c.cycles.values()) == p.total_cycles
        assert [(w.block, w.first, w.last) for w in c.waits] == [
            (w.block, w.first, w.last) for w in pc.waits
        ]
    (d,) = r.dmas
    pd = p.dmas["dma"]
    assert (d.transfers, d.busy, d.bytes_l2_to_l1, d.bytes_l1_to_l2) == (
        pd.transfers,
        pd.busy,
        pd.bytes_l2_to_l1,
        pd.bytes_l1_to_l2,
    )
    assert [(x.port, x.grants, x.whole_run) for x in r.ports] == [
        (n, x.grants, x.whole_run) for n, x in p.ports.items()
    ]
    assert sum(len(b.banks.split(",")) for b in r.banks) <= len(p.banks.conflicts)
    assert [s.name for s in r.streamers] == list(p.streamers)
    assert r.movement.reason == "Needs --trace beat; this run was traced at task."


def test_a_wait_names_the_task_that_starts_after_it(flows):
    r = reports_of(flows[("vecadd", "task")]).run
    core0 = next(c for c in r.cores if c.name == "core0")
    starts = {t.name: t.start for t in r.tasks}
    for w in core0.waits:
        if w.then is not None:
            assert starts[w.then] > w.last
            assert all(s <= w.last or s >= starts[w.then] for s in starts.values())


def test_off_has_no_tasks_no_classes_no_achieved_ii(flows):
    r = reports_of(flows[("vecadd", "off")]).run
    assert (r.trace_level, r.boot, r.tasks) == ("off", None, None)
    assert all(c.cycles is None and not c.waits for c in r.cores)
    assert r.accelerators[0].achieved_ii is None and r.accelerators[0].span is None
    text = render_run(r)
    assert "Needs at least a task trace" in text and "need at least a task trace" in text


# =============================================================================
# 3. Achieved II
# =============================================================================


def test_achieved_ii():
    assert achieved_ii([]) is None
    assert achieved_ii([{"firings": 1, "first": 5, "last": 5}]) is None
    assert achieved_ii([{"firings": 16, "first": 10, "last": 25}]) == 1.0
    assert achieved_ii([{"firings": 4, "first": 0, "last": 9},
                        {"firings": 4, "first": 20, "last": 23}]) == 2.0  # fmt: skip


def test_achieved_ii_of_a_flow_is_its_done_events(flows):
    folder = flows[("vsum", "task")]
    tr = read_run(folder / "run").trace
    (a,) = reports_of(folder).run.accelerators
    assert a.achieved_ii == achieved_ii(tr.of_kind("done", "acc"))


# =============================================================================
# 4. Round trips, rendering, the recorded run, the command line
# =============================================================================


@pytest.mark.parametrize("key", [("vecadd", "task"), ("vecadd", "off")])
def test_the_reports_round_trip_and_render(flows, key):
    r = reports_of(flows[key])
    assert RunReport.from_dict(json.loads(json.dumps(r.run.to_dict()))) == r.run
    assert DesignReport.from_dict(json.loads(json.dumps(r.design.to_dict()))) == r.design
    run_md, design_md = render_run(r.run), render_design(r.design)
    for section in ("## Tasks", "## Accelerators", "## Cores", "## DMA", "## Memory",
                    "## Data movement"):  # fmt: skip
        assert section in run_md
    for section in ("## Cluster", "## Accelerators", "## Streamers", "## L1", "## L2", "## Notes"):
        assert section in design_md
    with pytest.raises(ValueError, match="unknown keys"):
        RunReport.from_dict({**r.run.to_dict(), "extra": 1})


def test_the_flow_wrote_them(flows):
    folder = flows[("vecadd", "task")]
    r = reports_of(folder)
    assert (folder / "report" / "run.md").read_text() == render_run(r.run)
    assert (folder / "report" / "design.md").read_text() == render_design(r.design)


def test_the_recorded_vecadd_run(flows, tmp_path):
    """The recorded SNAX-GVSoC run (tests/run/fixtures/vecadd) in a flow folder: the reports of
    real events render."""
    folder = tmp_path / "vecadd_task"
    shutil.copytree(flows[("vecadd", "task")], folder)
    for f in ("snax_forge.hjson", "snax_forge_streamer.hjson", "snax_forge_acc.hjson"):
        shutil.copy(FIXTURE / f, folder / "gen" / f)
    shutil.copy(FIXTURE / "tasks.json", folder / "tasks.json")
    check = json.loads((FIXTURE / "check.json").read_text())
    cluster = load_cluster(FIXTURE / "snax_forge.hjson", FIXTURE / "design_point.json")
    profile, trace = convert(read_events(FIXTURE / "events.log"), cluster,
                             TaskList.load(FIXTURE / "tasks.json"), check["total_cycles"], "task",
                             check["functional_check"])  # fmt: skip
    write_run(folder / "run", profile, trace)
    r = reports_of(folder)
    assert r.run.cycles == check["total_cycles"] == trace.total_cycles
    assert "# Run report: vecadd_task" in render_run(r.run)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """The recorded vecadd run converted at task, beat and stall into flow folders."""
    from tests.viz.helpers import recorded_run

    base = tmp_path_factory.mktemp("recorded")
    return {lv: recorded_run(base / f"vecadd_{lv}", lv) for lv in ("task", "beat", "stall")}


@pytest.mark.parametrize("level", ["beat", "stall"])
def test_the_movement_section_is_the_index(recorded, level):
    """At beat and above: per region its spans, per lane its nest and pace (viz/movement.py's
    answers), and at stall the stalls per bank and port, which are the profile's."""
    rv = api.load_run(recorded[level])
    m = build_run(rv).movement
    assert m.reason is None and m.sources is None
    view = api.movement_view(rv)
    assert [(r.region, r.mem, r.arrival, r.use, r.departure) for r in m.residency] == [
        (r["region"], r["mem"], r["arrival"], r["use"], r["departure"]) for r in view["residency"]]  # fmt: skip
    lanes = [p for p in view["patterns"] if p["port"] != "dma"]
    assert [(p.port, p.task, p.base, p.nest, p.beats, p.first, p.last) for p in m.patterns] == [
        (p["port"], p["task"], p["addr"]["base"], p["addr"]["nest"], p["beats"], p["first"],
         p["last"]) for p in lanes]  # fmt: skip
    md = "\n".join(render_movement(m))
    assert (
        "| A | L1 | 64 | 0 |" in md
        and "| acc_a.lane0 | add_acc_a | 0x10000000 | 16 × 32 | 16 |" in md
    )
    if level == "beat":
        assert m.conflicts is None and all(p.held is None for p in m.patterns)
        assert "L1 stalls: Needs --trace stall; this run was traced at beat." in md
    else:
        snax = {n: x.stalls for n, x in rv.outputs.profile.ports.items() if n.startswith("acc_")}
        assert m.conflicts == sum(snax.values()) == 4
        assert [(b.where, b.stalls) for b in m.banks] == [("0–3", 1)]
        assert [(p.where, p.stalls) for p in m.ports] == [(n, x) for n, x in snax.items() if x]
        assert [p.held for p in m.patterns if p.port.startswith("acc_b.")] == [1, 1, 1, 1]
        assert "4 L1 stalls" in md
    assert RunReport.from_dict(json.loads(json.dumps(build_run(rv).to_dict()))) == build_run(rv)


def test_the_movement_section_below_beat(recorded, flows):
    for folder, level in ((recorded["task"], "task"), (flows[("vecadd", "off")], "off")):
        r = build_run(api.load_run(folder))
        assert render_movement(r.movement) == [
            f"Needs --trace beat; this run was traced at {level}."
        ]


def test_a_filtered_trace_says_its_sources(tmp_path):
    folder = stubbed_flow(tmp_path, "vecadd", "beat", trace_sources=["acc", "acc_a"])
    m = build_run(api.load_run(folder)).movement
    assert m.sources == ["acc", "acc_a"]
    assert render_movement(m)[0] == "The beat lines cover acc, acc_a only."


def test_the_command_line(flows, tmp_path, capsys):
    folder = flows[("vecadd", "task")]
    assert main([str(folder), "--out", str(tmp_path / "r")]) == 0
    assert "report vecadd_task:" in capsys.readouterr().out
    assert (tmp_path / "r" / "run.md").is_file() and (tmp_path / "r" / "design.md").is_file()
    assert main([str(folder / "run"), "--out", str(tmp_path / "s")]) == 0  # its run/ too
    assert main([str(tmp_path)]) == 1
    assert "not a flow run" in capsys.readouterr().err
    with pytest.raises(ReportError):
        reports_of(tmp_path)
