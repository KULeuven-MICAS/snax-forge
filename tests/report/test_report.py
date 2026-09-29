"""Design and run reports (REP1, D99): every number equals its source.

1. the design report against the design point, the recipe and the cluster file
2. the run report against the profile, the trace and the movement answers
3. achieved II: 1.44 default, 1.00 pinned, 1.38 at W = 8
4. round trips, rendering for every scenario and flow, and a report that does
   not grow with the elements
5. the command line and where it writes
"""

from __future__ import annotations

import json

import pytest

from snax_forge.design import DesignPoint
from snax_forge.flow import regions_of, run_flow
from snax_forge.report import (
    DesignReport,
    RunReport,
    achieved_ii,
    render_design,
    render_run,
    reports_of,
)
from snax_forge.report.__main__ import main
from snax_forge.report.design import pct
from snax_forge.snax_model.scenario import Scenario, run, write_outputs
from snax_forge.viz import api
from snax_forge.viz import memory as memview
from tests.design.helpers import FIXTURES, RECIPE, SMALL16

from .helpers import SCEN, SCENARIOS

PLAIN = FIXTURES / "vecadd.snaxdfg"
FLOWS = {
    "default": {},
    "pinned": {"memory_sets": [("B.l1.base", 576), ("C.l1.base", 1152)]},
    "W8": {"recipe_sets": {"W": 8}},
}


@pytest.fixture(scope="module")
def flows(tmp_path_factory):
    """The guard-rail flows at beat level, as folders."""
    base = tmp_path_factory.mktemp("flows")
    out = {}
    for name, kw in FLOWS.items():
        f = run_flow(RECIPE, SMALL16, graph_path=PLAIN, out=base / name, trace_level="beat", **kw)
        out[name] = f.out
    return out


@pytest.fixture(scope="module")
def scenario_runs(tmp_path_factory):
    """Every scenario run at task level, as run directories."""
    base = tmp_path_factory.mktemp("runs")
    out = {}
    for name in SCENARIOS:
        sc = Scenario.load(SCEN / name / "scenario.json")
        out[name] = write_outputs(run(sc, trace_level="task"), base / name)
    return out


# =============================================================================
# 1. The design report
# =============================================================================


def test_the_design_report_equals_its_sources(flows):
    folder = flows["default"]
    r = reports_of(folder).design
    point = DesignPoint.load(folder / "design" / "design_point.json")
    recipe = json.loads((folder / "sandbox" / "recipe.json").read_text())
    cluster = json.loads((folder / "cluster.json").read_text())
    h = r.header
    assert (h.kernel, h.recipe, h.params, h.symbols) == (
        recipe["kernel"],
        recipe["name"],
        recipe["params"],
        {"N": 64},
    )
    assert (h.platform, h.platform_base, h.platform_changes) == (
        point.platform.name,
        point.platform.base,
        {},
    )
    assert h.passes == point.memory.passes and h.pins == {}
    assert [c.name for c in r.components] == [c["name"] for c in cluster["components"]]
    (acc,) = r.accelerators
    entry = next(c for c in cluster["components"] if c["kind"] == "accel")
    assert (acc.instance, acc.kind, acc.brm, acc.implementation) == (
        "acc",
        entry["accel"],
        "elementwise_add",
        "chisel_tiled_spatial",
    )
    assert (acc.latency, acc.target_ii, acc.params) == (
        entry["params"]["latency"],
        entry["params"]["ii"],
        {"W": 4, "op": "add"},
    )
    for s in r.streamers:
        cfg = next(c for c in cluster["components"] if c["name"] == s.name)["config"]
        st = point.streamers[s.name]
        assert (s.lanes, s.fifo_depth, s.temporal_dims) == (
            cfg["n_ports"],
            cfg["fifo_depth"],
            cfg["temporal_dims"],
        )
        assert (s.instance, s.port, s.direction) == (
            st.instance,
            st.port,
            "write" if st.write else "read",
        )
    assert {s.name: s.container for s in r.streamers} == {
        "acc_a": "A",
        "acc_b": "B",
        "acc_out": "C",
    }


def test_the_memory_tables_are_the_memory_tabs(flows):
    folder = flows["pinned"]
    r = reports_of(folder).design
    point = DesignPoint.load(folder / "design" / "design_point.json")
    cluster = api.load_run(folder / "run").cluster
    for table in r.memories:
        s = memview.summary(memview.layout_of(table.mem, cluster, regions_of(point)))
        assert table.used_bytes == s["used"]["words"] * s["geometry"]["word_bytes"]
        for row, reg in zip(table.regions, s["regions"], strict=True):
            assert (row.name, row.base, row.end, row.bytes, row.share) == (
                reg["name"],
                reg["start"],
                reg["end"],
                reg["bytes"],
                reg["share"],
            )
    b = next(x for x in r.memories[0].regions if x.name == "B")
    assert (b.base, b.end, b.banks, b.rows) == (576, 1088, "0–15", "4–8")
    assert r.header.pins == {"B.l1.base": 576, "C.l1.base": 1152}
    assert pct(0.0625) == "6.3%"  # halves round up, as the memory tab shows them


def test_the_notes_are_address_facts(flows):
    notes = reports_of(flows["default"]).design.notes
    assert notes[0] == (
        "acc_a and acc_b: A and B lie 512 B apart, a whole number of 128-byte bank rows, "
        "so A[i] and B[i] are in the same bank for every i."
    )
    pinned = reports_of(flows["pinned"]).design.notes
    assert not any("acc_b" in n for n in pinned)  # B at 576 starts in bank 8


# =============================================================================
# 2. The run report
# =============================================================================


def test_the_run_report_equals_its_sources(flows):
    folder = flows["default"]
    rv = api.load_run(folder / "run")
    prof = rv.outputs.profile.to_dict()
    r = reports_of(folder).run
    assert (r.cycles, r.trace_level, r.check.passed) == (prof["total_cycles"], "beat", True)
    assert [(t.block, t.start, t.done) for t in r.tasks] == sorted(
        [(b, t["start"], t["done"]) for b, ts in api.tasks(rv).items() for t in ts],
        key=lambda x: (x[1], x[0]),
    )
    assert [t.name for t in r.tasks] == [
        "load_A",
        "load_B",
        "add_acc_a",
        "add_acc_b",
        "add_acc_out",
        "add_acc",
        "store_C",
    ]
    (a,) = r.accelerators
    pa = prof["accelerators"]["acc"]
    assert (a.firings, a.busy, a.stall_in, a.stall_out) == (
        pa["firings"],
        pa["cycles"]["busy"],
        pa["cycles"]["stall_in"],
        0,
    )
    assert a.window == sum(d - s for s, d in rv.spans["acc"])
    c = prof["controller"]
    assert (r.controller.command, r.controller.wait) == (
        c["cycles"]["command"],
        c["cycles"]["wait"],
    )
    assert [(w.block, w.first, w.last) for w in r.controller.waits] == [
        (w["block"], w["first"], w["last"]) for w in c["waits"]
    ]
    (d,) = r.dmas
    pd = prof["dmas"]["dma"]
    assert (d.bytes_read, d.bytes_written, d.busy) == (
        pd["bytes_read"],
        pd["bytes_written"],
        pd["cycles"]["busy"],
    )
    assert d.bytes_per_busy_cycle == (pd["bytes_read"] + pd["bytes_written"]) / pd["cycles"]["busy"]
    assert [(b.banks, b.conflicts) for b in r.banks] == [("0–3", 3), ("8–11", 4)]
    assert sum(prof["banks"]["conflicts"]) == 3 * 4 + 4 * 4
    by = {s.name: s for s in r.streamers}
    assert (by["acc_b"].stall_xbar, by["acc_b"].port_stalls) == (7, 28)


def test_the_movement_section_is_the_movement_answer(flows):
    rv = api.load_run(flows["default"] / "run")
    mv = api.movement_view(rv)
    m = reports_of(flows["default"]).run.movement
    assert [(x.region, x.mem, x.arrival, x.use) for x in m.residency] == [
        (x["region"], x["mem"], x["arrival"], x["use"]) for x in mv["residency"]
    ]
    kept = [p for p in mv["patterns"] if not (p["owner"] == "dma" and "side" not in p)]
    assert [(p.task, p.beats, p.held_cycles) for p in m.patterns] == [
        (p["task"], p["beats"], p.get("held_cycles", 0)) for p in kept
    ]
    b1 = next(p for p in m.patterns if p.port == "acc_b.1")
    assert (b1.pattern, b1.held_cycles, len(b1.held)) == ("B[4j+1]", 7, 3)
    assert b1.held[0] == "44: B[9] waited, bank 9 served acc_a.1 for A[9]"
    assert (
        next(p for p in m.patterns if p.port.startswith("dma.src") and p.task == 0).pattern
        == "A[8j]"
    )
    assert m.conflicts == 28


# =============================================================================
# 3. Achieved II
# =============================================================================


@pytest.mark.parametrize("name, ii", [("default", "1.44"), ("pinned", "1.00"), ("W8", "1.38")])
def test_achieved_ii(flows, name, ii):
    (a,) = reports_of(flows[name]).run.accelerators
    assert f"{a.achieved_ii:.2f}" == ii and a.target_ii == 1
    assert f"| acc | {a.firings} | 1 | {ii} |" in render_run(reports_of(flows[name]).run)


def test_achieved_ii_from_intervals():
    runs = [["idle", 0, 5], ["busy", 5, 7], ["stall_in", 7, 8], ["busy", 8, 10], ["idle", 10, 12]]
    assert achieved_ii(runs, 4) == 5 / 4
    assert achieved_ii(None, 4) is None and achieved_ii(runs, 0) is None


# =============================================================================
# 4. Round trips, rendering, size
# =============================================================================


def test_the_reports_round_trip(flows, scenario_runs):
    for folder in [*flows.values(), *scenario_runs.values()]:
        r = reports_of(folder)
        for rep, cls in ((r.design, DesignReport), (r.run, RunReport)):
            back = cls.from_dict(json.loads(json.dumps(rep.to_dict())))
            assert back == rep and back.to_dict() == rep.to_dict()
    with pytest.raises(ValueError):
        RunReport.from_dict({**reports_of(flows["default"]).run.to_dict(), "extra": 1})


def test_every_scenario_renders(scenario_runs):
    for name, d in scenario_runs.items():
        r = reports_of(d, tasks_path=SCEN / name / "tasks.json")
        text = render_run(r.run)
        assert text.startswith(f"# Run report: {name}") and "needs a beat-level trace" in text
        assert all(t.name for t in r.run.tasks)  # named from the scenario's task list
        assert render_design(r.design).startswith(f"# Design report: {name}")
        assert r.design.header is None


def test_a_run_without_a_trace(tmp_path):
    sc = Scenario.load(SCEN / "vecadd" / "scenario.json")
    d = write_outputs(run(sc), tmp_path / "off")
    r = reports_of(d).run
    assert r.tasks is None and r.accelerators[0].achieved_ii is None
    assert "Needs at least a task trace" in render_run(r)


def test_the_report_does_not_grow_with_the_elements(tmp_path, flows):
    """N = 256 gives as many lines as N = 64: nothing is listed per element or per cycle."""
    recipe = json.loads(RECIPE.read_text())
    recipe["symbols"]["N"] = 256
    path = tmp_path / "vecadd256.json"
    path.write_text(json.dumps(recipe))
    big = run_flow(path, SMALL16, graph_path=PLAIN, out=tmp_path / "big", trace_level="beat")
    assert big.result.total_cycles > 200
    small, large = reports_of(flows["default"]), reports_of(big.out)
    for a, b in (
        (render_design(small.design), render_design(large.design)),
        (render_run(small.run), render_run(large.run)),
    ):
        assert len(a.splitlines()) == len(b.splitlines())


# =============================================================================
# 5. Command line
# =============================================================================


def test_cli_writes_next_to_the_run(flows, scenario_runs, capsys):
    assert main([str(flows["default"])]) == 0
    assert (flows["default"] / "report" / "design.md").is_file() and (
        flows["default"] / "report" / "run.md"
    ).is_file()
    assert not (flows["default"] / "run" / "run.md").exists()  # never inside run/ (D44)
    assert "report default: 85 cycles, trace beat" in capsys.readouterr().out
    d = scenario_runs["vecadd"]
    assert main([str(d), "--tasks", str(SCEN / "vecadd" / "tasks.json")]) == 0
    assert (d.with_name("vecadd_report") / "run.md").read_text().count("add_acc_a") == 1


def test_cli_rejects_a_folder_that_is_not_a_run(tmp_path, capsys):
    assert main([str(tmp_path)]) == 1
    assert "not a flow folder" in capsys.readouterr().err
