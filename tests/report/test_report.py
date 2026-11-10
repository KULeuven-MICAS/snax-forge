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
from snax_forge.estimate.cost import sig3
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
from snax_forge.report.run import CheckRow, bank_notes, build_run, counters_text, render_movement
from snax_forge.run import load_cluster, read_events, read_run, write_run
from snax_forge.run.convert import convert
from snax_forge.viz import api, movement, schedule
from tests.flow.helpers import stubbed_flow
from tests.helpers import recipe_path

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
    (unit,) = cluster.accelerator.description["accelerators"][0]["groups"][0]["units"]
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
    for f in (
        "snax_forge.hjson",
        "snax_forge_vecadd_streamer_acc.hjson",
        "accelerator.json",
        "manifest.json",
    ):
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


def designed_report(tmp_path, recipe, sets=()):
    """The design report of ``recipes/<folder>/<recipe>.json`` (``--set`` each of ``sets``) designed on
    snax_forge (no run)."""
    from snax_forge.gen import generate
    from snax_forge.report.design import build_design
    from snax_forge.sandbox import Recipe
    from tests.gen.helpers import designed

    rel = designed(tmp_path, recipe, recipe, sets)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        generate(rel, tmp_path / "gen", root=tmp_path)
    point = DesignPoint.load(tmp_path / rel)
    cluster = load_cluster(tmp_path / "gen" / "snax_forge.hjson", tmp_path / rel)
    return build_design(recipe, cluster, point, Recipe.load(recipe_path(recipe)))


def test_a_mover_shows_its_dtype_k_and_channels(tmp_path):
    """vecadd_i32: int32, num_slc 2 to a 64-bit word, so 4 lanes take num_ch 2 (num_channel)."""
    r = designed_report(tmp_path, "vecadd_i32")
    assert [(s.port, s.dtype, s.num_slc, s.num_ch) for s in r.streamers] == [
        ("a", "int32", 2, 2), ("b", "int32", 2, 2), ("out", "int32", 2, 2)
    ]  # fmt: skip
    text = render_design(r)
    assert "| Streamer | Serves | Direction | dtype | num_slc | num_ch | FIFO depth |" in text
    assert "num_ch 2, FIFO depth" in text


@pytest.mark.parametrize("w", [1, 4])
def test_design_md_lists_the_design_notes(tmp_path, w):
    """dot_5 at W = 1 binds SerialAccumulate at lanes 1: design.md's Notes start with the note
    design.serial_one_lane and its fix; at W = 4 there is none."""
    r = designed_report(tmp_path, "dot_5", ["N=16", f"W={w}"])
    note = ("note [design.serial_one_lane] sum: SerialAccumulate at lanes 1 reads one lane per "
            "beat, the same work as TreeAccumulate at lanes 1 Fix: bind chisel_adder_tree")  # fmt: skip
    assert (r.notes[:1] == [note]) == (w == 1)
    assert (f"- {note}" in render_design(r)) == (w == 1)
    assert not any("design.serial_one_lane" in n for n in r.notes[w == 1 :])


@pytest.mark.parametrize("w", [8, 16])
def test_the_bank_bound_notes_in_design_md_and_run_md(tmp_path, w):
    """vecmul at N = 2048 (D155): at W = 16 design.md's Estimate has the group's bank-bound note
    and run.md puts it beside the task's expected BUSY_COUNTER; at W = 8 neither has one."""
    r = designed_report(tmp_path, "vecmul", ["N=2048", f"W={w}"])
    note = ("acc: the streams move 6144 words over 32 banks, at least 192 cycles of L1 time "
            "against the ideal 128")  # fmt: skip
    assert r.estimate.bank_notes == ([note] if w == 16 else [])
    cycles = 2048 // w  # bound, so no unmapped part (D155)
    assert f"Accelerator cycles: {cycles}. Total: {cycles}. L1 footprint:" in render_design(r)
    assert (f"- {note}." in render_design(r)) == (w == 16)
    point = DesignPoint.load(next(tmp_path.rglob("design_point.json")))
    assert bank_notes(point) == ({"mult_acc": note} if w == 16 else {})
    c = CheckRow(True, "vecmul", 0, 2048, {"mult_acc": 194}, {"mult_acc": 131}, bank_notes(point))
    assert counters_text(c) == (f"mult_acc 194 (expected 131, a guide; {note})" if w == 16
                                else "mult_acc 194 (expected 131, a guide)")  # fmt: skip


def test_design_md_counts_unmapped_work_as_cycles():
    """D155's line: the unmapped part with its assumption when there is unmapped work, left out
    when there is none."""
    from snax_forge.report.design import EstimateTable, cycles_line

    some = EstimateTable(
        [], 0, 256, 0, None, unmapped_cycles=2048, total_cycles=2048, unmapped_iterations=2048
    )  # vecmul split by 8 and tagged: 256 not mapped
    assert cycles_line(some) == ("Accelerator cycles: 0. Unmapped: 2048 cycles (2048 iterations "
                                 "at 1 cycle each, an assumption). Total: 2048.")  # fmt: skip
    none = EstimateTable([], 256, 0, 0, None, unmapped_cycles=0, total_cycles=256)
    assert cycles_line(none) == "Accelerator cycles: 256. Total: 256."


def test_the_extents_include_the_padding(tmp_path):
    """rowsum_3x7: A[3, 7] over its padded [3, 8] (three pads, 192 B, whole beats); y, three
    int64 the DMA moves, kept free up to a whole 64-byte beat. The checks' extents."""
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
# Several groups (D148): dot_1_enc's recorded run
# =============================================================================


@pytest.fixture(scope="module")
def dot1(tmp_path_factory):
    from tests.viz.helpers import recorded_run

    fixture = Path(__file__).resolve().parents[1] / "run" / "fixtures" / "dot_1_enc"
    return recorded_run(tmp_path_factory.mktemp("dot1") / "dot_1_enc_task", "task", fixture)


def test_the_run_report_names_each_tasks_group(dot1):
    r = reports_of(dot1)
    groups = {t.name: t.group for t in r.run.tasks}
    assert (groups["mult_vmul"], groups["mult_vmul_a"], groups["mult_vsum_a"]) == ("vmul",) * 3
    assert (groups["sum_vsum"], groups["sum_vmul_a"], groups["sum_vsum_out"]) == ("vsum",) * 3
    assert groups["load_A"] is None
    assert [(a.name, a.group, a.firings) for a in r.run.accelerators] == [
        ("vmul", "vmul", 16), ("vsum", "vsum", 16)]  # fmt: skip
    text = render_run(r.run)
    assert (
        "| Block | Group | # | Task | Accelerator | Start | Done | Cycles | Direction | Bytes |"
        in text
    )
    assert (
        "| vmul | vmul | 0 | mult_vmul | dot |" in text
        and "| vsum_a | vmul | 0 | mult_vsum_a (idle) | dot |" in text
    )
    assert "| Unit | Group | Firings |" in text
    # the other group's movers, started only for the shared start, tagged idle (C§7)
    idle = sorted(t.name for t in r.run.tasks if t.idle)
    assert idle == ["mult_vsum_a", "mult_vsum_out", "sum_vmul_a", "sum_vmul_b", "sum_vmul_out"]
    # each group's movers in a collapsible under Accelerators
    assert [(g.group, [m.name for m in g.movers]) for g in r.run.group_movers] == [
        ("vmul", ["vmul_a", "vmul_b", "vmul_out"]), ("vsum", ["vsum_a", "vsum_out"])]  # fmt: skip
    assert "<details>\n<summary>vmul ▸ vmul_a, vmul_b, vmul_out</summary>" in text
    assert "<summary>vsum ▸ vsum_a, vsum_out</summary>" in text
    assert text.index("## Accelerators") < text.index("<summary>vmul ▸") < text.index("## Cores")


def test_the_port_table_leaves_out_the_idle_tasks(tmp_path):
    """C§7: dot 1 at stall (fixtures/dot_1): the Data movement port table has no lane of the five
    idle streamer tasks, and one line under it says they are left out."""
    from tests.run.helpers import RECORDED
    from tests.viz.helpers import recorded_run

    r = reports_of(recorded_run(tmp_path / "dot_1_enc_stall", "stall", RECORDED / "dot_1_enc")).run
    tasks = {p.task for p in r.movement.patterns}
    assert tasks == {"mult_vmul_a", "mult_vmul_b", "mult_vmul_out", "sum_vsum_a", "sum_vsum_out"}
    assert r.movement.idle_left_out == 5
    assert ("The 5 idle streamer tasks (every temporal bound 0, started only for the shared start; "
            "tagged idle under Tasks) are left out.") in render_run(r)  # fmt: skip


def test_the_bank_notes_compare_the_streamers_of_one_group(dot1):
    """C§14: dot 1's mul_a and mul_b keep their note (A and B a whole number of bank rows
    apart); nothing pairs a mul streamer with a sum one (the groups run one after the other):
    no note on mul_a and sum_out, none on tmp0 against itself through mul_out and sum_a."""
    notes = reports_of(dot1).design.notes
    assert any(n.startswith("vmul_a and vmul_b: A and B lie ") for n in notes), notes
    assert not any("vsum_" in n and "vmul_" in n for n in notes), notes
    assert not any("tmp0 and tmp0" in n for n in notes), notes


def test_the_design_report_has_a_row_per_unit(dot1):
    r = reports_of(dot1).design
    assert [(a.instance, a.group, a.kind) for a in r.accelerators] == [
        ("vmul", "vmul", "SimdLanes"), ("vsum", "vsum", "TreeAccumulate")]  # fmt: skip
    assert [c.name for c in r.components][:2] == ["vmul", "vsum"]
    assert [(s.name, s.instance) for s in r.streamers][:3] == [
        ("vmul_a", "vmul"), ("vmul_b", "vmul"), ("vsum_a", "vsum")]  # fmt: skip


# =============================================================================
# A linked group: dot 4's recorded run
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
        "dot", "Mac → TreeAccumulate on core0, group control unit, slot snax_forge_dot_4")  # fmt: skip
    assert [(a.instance, a.kind, a.brm, a.latency, a.ports) for a in r.accelerators] == [
        ("mac", "Mac", "mac", 1, ["a ← mac_a", "b ← mac_b"]),
        ("sum", "TreeAccumulate", "accumulate", 1, ["out ← sum_out"]),
    ]  # fmt: skip


def test_one_group_has_no_group_column(flows):
    text = render_run(reports_of(flows[("vecadd", "task")]).run)
    assert "| Block | # | Task |" in text and "| Group |" not in text


def test_design_md_costs_each_group_beside_its_cycles(dot1):
    """dot 1's two groups on nangate45 (C§20): each its area and energy beside its cycles, the
    cost table a row per unit and per control unit (its comb area equal to its flops', D154), the
    L1 accesses' table, the accelerator's totals their sums, every unit held by the table; the
    report round-trips."""
    from snax_forge.report.design import DesignReport

    r = reports_of(dot1).design
    e = r.estimate
    assert e.tech == "nangate45" and e.char_source["snax_forge_experiments"] and not e.missing
    assert [(c.group, c.block, c.datapath) for c in e.cost] == [
        ("vmul", "vmul", "SimdLanes"), ("vmul", "ctrl", "group control unit"),
        ("vsum", "vsum", "TreeAccumulate"), ("vsum", "ctrl", "group control unit")]  # fmt: skip
    assert all(c.found for c in e.cost if c.block != "ctrl")
    # D167: the groups are the datapath and control parts; the total adds the streamers, the
    # crossbar and the banks, and the accelerators' L1 rows are the crossbar's and banks' energy
    parts = {p.part: p for p in e.parts}
    assert list(parts) == ["datapath", "control", "streamers", "crossbar", "memory"]
    assert parts["datapath"].area_um2 + parts["control"].area_um2 == pytest.approx(
        sum(g.area_um2 for g in e.groups))  # fmt: skip
    assert e.area_um2 == pytest.approx(sum(p.area_um2 for p in e.parts))
    assert e.area_design_um2 == pytest.approx(e.area_um2 - parts["memory"].area_um2)
    assert e.energy_nj == pytest.approx(sum(p.energy_nj for p in e.parts))
    assert sum(p.area_share for p in e.parts) == pytest.approx(1.0)
    (l1,) = e.l1
    assert (l1.accelerator, l1.bits) == ("dot", 64 * l1.words) and l1.words > 0
    assert l1.energy_nj == pytest.approx(parts["crossbar"].energy_nj + parts["memory"].energy_nj)
    assert [x.streamer for x in e.streamer_cost] == [
        "vmul_a",
        "vmul_b",
        "vmul_out",
        "vsum_a",
        "vsum_out",
    ]
    assert all(c.comb_um2 == c.flops_um2 for c in e.cost if c.block == "ctrl")
    text = render_design(r)
    assert "| Expected BUSY_COUNTER | Area (µm²) | Energy (nJ) |" in text
    assert (
        "### Cost estimate (first-order) on nangate45\n\nFirst-order: aimed at the large costs"
        in text
    )
    assert (f"Design on nangate45: about {sig3(e.area_um2)} µm² ({sig3(e.area_design_um2)} µm² "
            f"without the banks) and {e.energy_nj:.3f} nJ per run.") in text  # fmt: skip
    # an area shown to 3 significant figures, an energy to 3 decimals
    assert (
        "| vsum | vsum | TreeAccumulate | measured | 16 | 64 | 1 300 | 340 | 2.495 | 0.457 | 0.055 |"
        in text
    )
    # a block with no flops (the SimdLanes mul) holds "–"
    (mul,) = [x for x in text.splitlines()
              if x.startswith("| vmul | vmul | SimdLanes |") and len(x.split(" | ")) == 11]  # fmt: skip
    assert mul.split(" | ")[9] == "–", mul
    assert (f"| dot | L1 accesses (crossbar and banks) | {l1.reads} | {l1.writes} | "
            f"{l1.xbar_nj:.3f} | {l1.banks_nj:.3f} | {l1.energy_nj:.3f} |") in text  # fmt: skip
    m = parts["memory"]
    assert (f"| memory | {sig3(m.area_um2)} | {100 * m.area_share:.1f} % | {m.energy_nj:.3f} | "
            f"{100 * m.energy_share:.1f} % |") in text  # fmt: skip
    assert "Cost by part:" in text and "| without the banks |" in text
    assert "Not characterized" not in text and "Not costed" not in text
    assert DesignReport.from_dict(r.to_dict()).to_dict() == r.to_dict()


def test_design_md_on_no_technology(dot1):
    """The report's technology (``--tech``): None, no cost. (nangate15's case dropped: the
    technology is no longer offered, D167.)"""
    from snax_forge.report.files import reports_of as of

    plain = of(dot1, tech=None).design
    assert plain.estimate.tech is None and plain.estimate.cost == []
    text = render_design(plain)
    assert "Cost estimate" not in text and "Area (µm²)" not in text


def test_design_md_names_what_the_table_does_not_hold(tmp_path):
    """vecadd_i32: an int32 SimdLanes, not in nangate45's int64 table; its comb counted
    as 0, listed under the cost and noted (char.missing)."""
    r = designed_report(tmp_path, "vecadd_i32")
    assert r.estimate.missing == ["acc"]
    text = render_design(r)
    assert "| acc | acc | SimdLanes | not characterized |" in text
    assert "Not characterized on nangate45 (comb counted as 0, flops counted): acc" in text
    assert ("- note [char.missing] acc: not characterized: SimdLanes at lanes 4 on "
            "nangate45") in text  # fmt: skip


def test_the_estimate_beside_the_counters(dot1, dot4):
    """design.md's Estimate section (D155) per group, and run.md's expected BUSY_COUNTER beside each
    measured one: equal on the recorded runs, the cycles plus the streamers' fill."""
    r = reports_of(dot1)
    e = r.design.estimate
    assert [(g.group, g.cycles, g.bottleneck, g.ramp, g.stall, g.counter) for g in e.groups] == [
        ("vmul", 16, 16, 0, 0, 19), ("vsum", 17, 16, 1, 0, 19)]  # fmt: skip
    assert (e.accelerator_cycles, e.not_mapped, e.l1_size) == (33, 0, 128 * 1024)
    text = render_design(r.design)
    assert "## Estimate" in text and "| vmul | 16 | 16 | 0 | 0 | 19 |" in text
    assert r.run.check.expected == r.run.check.counters == {"mult_vmul": 19, "sum_vsum": 19}
    assert "mult_vmul 19 (expected 19, a guide), sum_vsum 19 (expected 19, a guide)" in render_run(
        r.run
    )
    four = reports_of(dot4)
    assert four.run.check.expected == four.run.check.counters == {"mult_dot": 133}


# =============================================================================
# The accelerators, one streamer set each (D157): dot_1 and dot_1_enc
# =============================================================================


@pytest.fixture(scope="module")
def dot1_sets(tmp_path_factory):
    from tests.run.helpers import RECORDED
    from tests.viz.helpers import recorded_run

    base = tmp_path_factory.mktemp("sets")
    return {
        d: recorded_run(base / f"{d}_task", "task", RECORDED / d) for d in ("dot_1", "dot_1_enc")
    }


def test_run_md_has_a_row_per_accelerator(dot1_sets):
    """dot_1: vmul 960-981 and vsum 982-999, each its own streamer busy window and BUSY_COUNTER;
    dot_1_enc: one accelerator dot, 960-996, its streamer started twice; each task names its
    accelerator."""
    one, enc = (reports_of(dot1_sets[d]).run for d in ("dot_1", "dot_1_enc"))
    assert [(a.name, a.groups, a.first, a.last, a.counters) for a in one.sets] == [
        ("vmul", ["vmul"], 960, 981, {"vmul": [19]}), ("vsum", ["vsum"], 982, 999, {"vsum": [19]})]  # fmt: skip
    assert [len(a.busy) for a in one.sets] == [1, 1]
    assert one.sets[0].busy[0][1] < one.sets[1].busy[0][0]  # vsum's starts after vmul's ends
    (dot,) = enc.sets
    assert (dot.name, dot.groups, dot.first, dot.last, dot.counters) == (
        "dot", ["vmul", "vsum"], 960, 996, {"vmul": [19], "vsum": [19]})  # fmt: skip
    assert len(dot.busy) == 2  # the shared streamer, once per group's task
    acc_of = {t.name: t.accelerator for t in one.tasks}
    assert (acc_of["mult_vmul_a"], acc_of["sum_vsum"], acc_of["load_A"]) == ("vmul", "vsum", None)
    text = render_run(one)
    assert "| Accelerator | Groups | CSRs | Streamer busy | BUSY_COUNTER |" in text
    assert "| vmul | vmul | 960–981 |" in text and "| vsum | vsum | 982–999 |" in text
    assert text.index("## Accelerators") < text.index("| Accelerator | Groups") < text.index(
        "### Units")  # fmt: skip


def test_design_md_has_a_row_per_accelerator_and_counts_the_streamers(dot1_sets):
    """The same rows from the design: each set's movers, channels, temporal dims, CSRs and its
    groups' own CSRs; dot_1's 2 streamer sets (40 CSRs, not costed), dot_1_enc's 1 (37)."""
    one, enc = (reports_of(dot1_sets[d]).design for d in ("dot_1", "dot_1_enc"))
    assert [(x.name, x.movers, x.channels, x.temporal_dims, x.first, x.last, x.group_csrs)
            for x in one.sets] == [
        ("vmul", 3, 12, [1, 1, 1], 960, 981, {"vmul": [978, 981]}),
        ("vsum", 2, 5, [1, 1], 982, 999, {"vsum": [995, 999]})]  # fmt: skip
    (dot,) = enc.sets
    assert (dot.movers, dot.channels, dot.first, dot.last, dot.group_csrs) == (
        5, 17, 960, 996, {"vmul": [988, 991], "vsum": [992, 996]})  # fmt: skip
    assert (one.estimate.streamers, one.estimate.csrs) == (2, 40)
    assert (enc.estimate.streamers, enc.estimate.csrs) == (1, 37)
    text = render_design(one)
    assert "2 streamer sets, one per accelerator; 40 CSRs" in text and "not costed." in text
    assert "| vmul | vmul | 3 | 12 | 1, 1, 1 | 960–981 | vmul 978–981 |" in text
    assert "1 streamer set, one per accelerator; 37 CSRs" in render_design(enc)
