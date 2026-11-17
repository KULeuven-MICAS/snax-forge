"""The energy run (flow/energy.py, D177) and its report (report/energy.py): vecadd's quiet build
written by hand, its snapshots priced by window.

On SNAX-GVSoC (marker ``gvsoc``: in the image), vecadd and dot_3 at N = 2048, W = 8 run with
``energy``: the counts of the ``run`` window that the program fixes are the static estimate's
(each mover's grants its words, each unit's firings times its span its fires), every mover's
requests are at least its grants, and the windows' energies add up to the run's total.
"""

from __future__ import annotations

import json
from itertools import pairwise

import pytest

from snax_forge.design import Platform
from snax_forge.estimate import estimate
from snax_forge.estimate.dynamic import graph_counts
from snax_forge.flow import run_flow
from snax_forge.flow.energy import EnergyRun, counts_of, energy_of
from snax_forge.report.energy import render_energy
from snax_forge.run import RunError, cluster_from, load_cluster, parse_line
from tests.estimate.helpers import SNAX_FORGE
from tests.helpers import recipe_path, steps
from tests.run.helpers import files, snap_lines

from .helpers import needs_gvsoc

MARKS = [{"mark": 1, "kind": "acc_config", "label": "acc config acc"},
         {"mark": 2, "kind": "run", "label": "run acc"},
         {"mark": 3, "kind": "dma_out", "label": "dma store_out"}]  # fmt: skip
MOVERS = ("acc_a", "acc_b", "acc_out")
REAL = {"vecadd": {}, "dot_3": {"N": 2048, "W": 8}}  # the recipes run on SNAX-GVSoC, their --set


def totals(grants: int, stalls: int, firings: int, busy: int) -> dict:
    """Running totals: each of the three movers' 4 lanes ``grants`` and ``stalls``, core0 9 grants,
    the unit ``firings``, the group ``busy`` cycles."""
    return {"inputs": {f"{m}.lane{k}": (grants, stalls) for m in MOVERS for k in range(4)}
            | {"core0": (9, 0)},
            "group": {"busy": busy, "stall_in": 0, "stall_out": 0, "flush": 0},
            "units": {"acc": (firings, firings)}}  # fmt: skip


def run(stalls: int = 0) -> EnergyRun:
    """vecadd at N = 64, W = 4: its run window as the static estimate counts it (16 grants per lane,
    16 firings, 16 busy cycles), with ``stalls`` refusals per lane."""
    cluster = cluster_from(*files())
    graph = steps("vecadd")[-1]
    snaps = [(1, 50, totals(0, 0, 0, 0)), (2, 70, totals(0, 0, 0, 0)),
             (3, 100, totals(16, stalls, 16, 16)), (-1, 140, totals(16, stalls, 16, 16))]  # fmt: skip
    lines = [parse_line(x) for x in snap_lines(cluster, snaps)]
    return energy_of(lines, cluster, MARKS, graph, estimate(graph, Platform.load(SNAX_FORGE)))


def test_a_run_window_counted_as_the_graph_costs_what_the_static_estimate_says():
    e = run()
    assert [(w.kind, w.cycles) for w in e.windows] == [("boot", 50), ("acc_config", 20),
                                                        ("run", 30), ("dma_out", 40)]  # fmt: skip
    ((w, got),) = e.of_kind("run")
    for part, want in e.static.parts.items():
        assert got.parts[part] == pytest.approx(want, rel=1e-12), part
    assert (got.requests, got.cycles, got.fires) == (192, 16, 16)
    assert all(x.total == 0 for v, x in zip(e.windows, e.energies, strict=True) if v is not w)


def test_refusals_cost_the_crossbar_and_the_streamers_and_the_windows_add_up():
    e, base = run(stalls=2), run()
    ((_, got),) = e.of_kind("run")
    ((_, want),) = base.of_kind("run")
    assert got.requests == want.requests + 24 and got.reads == want.reads
    assert got.parts["crossbar"] == pytest.approx(
        want.parts["crossbar"] + (16 * e.prices.e_xro_pj + 8 * e.prices.e_xwo_pj) / 1000
    )
    assert got.parts["memory"] == want.parts["memory"]
    assert sum(e.total.values()) == pytest.approx(sum(x.total for x in e.energies))


def test_the_energy_run_round_trips_and_reports():
    e = run(stalls=1)
    assert EnergyRun.from_dict(e.to_dict()) == e
    text = render_energy(e)
    assert text.startswith("# Energy by window on nangate45")
    assert "| 2 | run | run acc | 30 |" in text
    assert "| static |  | one run of the accelerator | 16 |" in text
    assert "| crossbar: requests | 204 | 192 |" in text


def test_a_run_without_snapshots_is_refused():
    cluster = cluster_from(*files())
    graph = steps("vecadd")[-1]
    with pytest.raises(RunError, match="energy.snapshots"):
        energy_of([], cluster, MARKS, graph, estimate(graph, Platform.load(SNAX_FORGE)))


def test_the_flow_runs_the_quiet_build_again_and_prices_it(tmp_path, monkeypatch):
    """vecadd through the flow with --energy, SNAX-GVSoC stubbed: the quiet build's run window
    counted as the static estimate counts it costs the static estimate's five parts."""
    from snax_forge.flow import summary

    from .helpers import stub
    from .test_flow import flow

    fake = stub(monkeypatch, tmp_path)
    f = flow(tmp_path, energy=True)
    goal, v = fake.calls[-1]
    assert goal == "run" and v["ELF"].endswith("sw/forge_quiet.elf") and v["TRACES"] == "summary"
    out = tmp_path / "flow"
    assert (out / "energy" / "energy.json").is_file() and (
        out / "energy" / "events.log.gz"
    ).is_file()
    assert (out / "report" / "energy.md").read_text() == render_energy(f.energy)
    ((w, got),) = f.energy.of_kind("run")
    for part, want in f.energy.static.parts.items():
        assert got.parts[part] == pytest.approx(want, rel=1e-12), part
    assert f"energy    sw/forge_quiet.elf, {len(f.energy.windows)} windows; window {w.index}" in (
        summary(f)
    )
    assert "energy" in f.times


@pytest.fixture(scope="module", params=sorted(REAL))
def real(request, tmp_path_factory):
    """A recipe of ``REAL`` through the flow on SNAX-GVSoC with ``energy``."""
    name = request.param
    return run_flow(recipe_path(name), SNAX_FORGE, recipe_sets=REAL[name],
                    out=tmp_path_factory.mktemp(name) / "flow", energy=True)  # fmt: skip


@pytest.mark.gvsoc
@needs_gvsoc
def test_a_real_run_window_has_the_static_grants_and_fires(real):
    f, e = real, real.energy
    assert f.passed and e is not None
    point = f.point
    cluster = load_cluster(f.out / "gen" / "snax_forge.hjson",
                           f.out / "design" / "design_point.json")  # fmt: skip
    est = estimate(point.graph, point.platform)
    static = graph_counts(point.graph, est)
    ((w, _),) = e.of_kind("run")
    got = counts_of(w, cluster)
    assert sorted(got.movers) == sorted(static.movers)
    for m, (requests, grants) in got.movers.items():
        assert grants == static.movers[m][1], m
        assert requests >= grants, m
    fires = {u: b.fires for g in est.groups if g.cost for u, b in g.cost.units.items()}
    assert sorted(got.units) == sorted(fires)
    for u, (firings, _) in got.units.items():
        assert firings * e.prices.units[u][0] == fires[u], u


@pytest.mark.gvsoc
@needs_gvsoc
def test_a_real_runs_windows_add_up_to_its_total(real):
    """energy.json as written: its windows follow one another from boot, and per part their
    energies add up to its total."""
    d = json.loads((real.out / "energy" / "energy.json").read_text())
    ws = d["windows"]
    assert ws[0]["first"] == 0 and all(a["last"] == b["first"] for a, b in pairwise(ws))
    for part, total in d["total"].items():
        assert sum(w["energy"]["parts"][part] for w in ws) == pytest.approx(total), part
    assert d["total"]["memory"] > 0
