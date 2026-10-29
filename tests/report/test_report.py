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
from snax_forge.viz import api, movement, schedule
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
    (unit,) = cluster.accelerator.description["groups"][0]["units"]
    assert (a.instance, a.kind, a.target_ii, a.latency) == (
        "acc",
        unit["datapath"],
        unit["timing"]["ii"],
        unit["timing"]["latency"],
    )
    assert a.ports == [f"{m.port} ← {m.name}" for m in cluster.accelerator.movers]
    assert [(s.name, s.num_ch, s.fifo_depth) for s in r.streamers] == [
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
    for f in ("snax_forge.hjson", "snax_forge_streamer.hjson", "accelerator.json"):
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
    residency, patterns = movement.residency(rv.index), movement.patterns(rv.index)
    assert [(r.region, r.mem, r.arrival, r.use, r.departure) for r in m.residency] == [
        (r["region"], r["mem"], r["arrival"], r["use"], r["departure"]) for r in residency]  # fmt: skip
    lanes = [p for p in patterns if p["port"] != "dma"]
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


# =============================================================================
# 5. Packing and padding (D142, D144), from design points
# =============================================================================


def designed_report(tmp_path, recipe):
    """The design report of ``recipes/<recipe>.json`` designed on snax_forge (no run)."""
    from snax_forge.gen import generate
    from snax_forge.report.design import build_design
    from snax_forge.sandbox import Recipe
    from tests.gen.helpers import REPO, designed

    rel = designed(tmp_path, recipe, recipe)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        generate(rel, tmp_path / "gen", root=tmp_path)
    point = DesignPoint.load(tmp_path / rel)
    cluster = load_cluster(tmp_path / "gen" / "snax_forge.hjson", tmp_path / rel)
    return build_design(recipe, cluster, point, Recipe.load(REPO / "recipes" / f"{recipe}.json"))


def test_a_mover_shows_its_dtype_k_and_channels(tmp_path):
    """vecadd_i32: int32, num_slc 2 to a 64-bit word, so 4 lanes take num_ch 2 (num_channel)."""
    r = designed_report(tmp_path, "vecadd_i32")
    assert [(s.port, s.dtype, s.num_slc, s.num_ch) for s in r.streamers] == [
        ("a", "int32", 2, 2), ("b", "int32", 2, 2), ("out", "int32", 2, 2)
    ]  # fmt: skip
    text = render_design(r)
    assert "| Streamer | Serves | Direction | dtype | num_slc | num_ch | FIFO depth |" in text
    assert "num_ch 2, FIFO depth" in text


def test_the_extents_include_the_padding(tmp_path):
    """rowsum_3x7: A[3, 7] over its padded [3, 8] (three pads, 192 B, whole beats); y, three
    int64 the DMA moves, kept free up to a whole 64-byte beat. The checks' extents (D144)."""
    r = designed_report(tmp_path, "rowsum_3x7")
    l1 = {x.name: x for x in next(m for m in r.memories if m.mem == "l1").regions}
    a, y = l1["A"], l1["y"]
    assert (a.end - a.base, a.data_end, a.pad_elements, a.beat_pad_bytes) == (192, a.end, 3, 0)
    assert (y.data_end - y.base, y.end - y.base, y.pad_elements, y.beat_pad_bytes) == (
        24,
        64,
        0,
        40,
    )
    assert (a.padding, y.padding) == ("3 pad elements", "40 B to a whole beat")
    assert "| Region | Base | End | Bytes | Share | Padding |" in render_design(r)


# =============================================================================
# Several groups (D148): dot 1's recorded run
# =============================================================================


@pytest.fixture(scope="module")
def dot1(tmp_path_factory):
    from tests.viz.helpers import recorded_run

    fixture = Path(__file__).resolve().parents[1] / "run" / "fixtures" / "dot_1"
    return recorded_run(tmp_path_factory.mktemp("dot1") / "dot_1_task", "task", fixture)


def test_the_run_report_names_each_tasks_group(dot1):
    r = reports_of(dot1)
    groups = {t.name: t.group for t in r.run.tasks}
    assert (groups["mult_mul"], groups["mult_mul_a"], groups["mult_sum_a"]) == ("mul",) * 3
    assert (groups["sum_sum"], groups["sum_mul_a"], groups["sum_sum_out"]) == ("sum",) * 3
    assert groups["load_A"] is None
    assert [(a.name, a.group, a.firings) for a in r.run.accelerators] == [
        ("mul", "mul", 16), ("sum", "sum", 16)]  # fmt: skip
    text = render_run(r.run)
    assert "| Block | Group | # | Task | Start | Done | Cycles | Direction | Bytes |" in text
    assert "| mul | mul | 0 | mult_mul |" in text and "| sum_a | mul | 0 | mult_sum_a |" in text
    assert "| Accelerator | Group | Firings |" in text


def test_the_design_report_has_a_row_per_unit(dot1):
    r = reports_of(dot1).design
    assert [(a.instance, a.group, a.kind) for a in r.accelerators] == [
        ("mul", "mul", "ElementwiseLanes"), ("sum", "sum", "TreeAccumulate")]  # fmt: skip
    assert [c.name for c in r.components][:2] == ["mul", "sum"]
    assert [(s.name, s.instance) for s in r.streamers][:3] == [
        ("mul_a", "mul"), ("mul_b", "mul"), ("sum_a", "sum")]  # fmt: skip


# =============================================================================
# A linked group (D148): dot 4's recorded run
# =============================================================================


@pytest.fixture(scope="module")
def dot4(tmp_path_factory):
    from tests.viz.helpers import recorded_run

    fixture = Path(__file__).resolve().parents[1] / "run" / "fixtures" / "dot_4"
    return recorded_run(tmp_path_factory.mktemp("dot4") / "dot_4_task", "task", fixture)


def test_a_linked_group_is_one_accelerator_in_the_run_report(dot4):
    """The group is one accelerator named after it, with its run's classes; its task and its
    movers' tasks are the group's."""
    r = reports_of(dot4)
    assert [(a.name, a.firings, a.stall_in, a.stall_out, a.flush) for a in r.run.accelerators] == [
        ("dot", 128, 3, 0, 1)]  # fmt: skip
    assert [(t.block, t.name) for t in r.run.tasks if t.block != "dma"] == [
        ("mac_a", "mult_mac_a"), ("mac_b", "mult_mac_b"), ("sum_out", "sum_final_sum_out"),
        ("dot", "mult_dot")]  # fmt: skip


def test_a_linked_groups_design_report_has_a_row_per_unit(dot4):
    """The cluster's one component for the group, its datapaths in chain order under the
    group's control unit; an accelerator row per unit, each with its BRM and ports."""
    r = reports_of(dot4).design
    assert (r.components[0].name, r.components[0].detail) == (
        "dot", "Mac → TreeAccumulate on core0, group control unit, slot snax_forge")  # fmt: skip
    assert [(a.instance, a.kind, a.brm, a.latency, a.ports) for a in r.accelerators] == [
        ("mac", "Mac", "mac", 1, ["a ← mac_a", "b ← mac_b"]),
        ("sum", "TreeAccumulate", "accumulate", 1, ["out ← sum_out"]),
    ]  # fmt: skip


def test_one_group_has_no_group_column(flows):
    text = render_run(reports_of(flows[("vecadd", "task")]).run)
    assert "| Block | # | Task |" in text and "Group" not in text
