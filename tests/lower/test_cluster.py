"""Design point -> cluster file (LOW1c, D53, D88)."""

from __future__ import annotations

import itertools
import json

import pytest

from snax_forge.brm import load_brm
from snax_forge.design import DesignPoint, Platform, run_checks
from snax_forge.design.streamers import Streamer
from snax_forge.lower import __main__ as cli
from snax_forge.lower import cluster_config, cluster_file, cluster_of, stub
from snax_forge.snax_model.config import to_json
from snax_forge.snax_model.scenario import ClusterConfig, build_cluster
from tests.design.helpers import SMALL16, bound_w8, design

from .helpers import SCEN

ALU4 = SCEN / "clusters" / "alu4.json"


def point(graph="vecadd_accelerated", sets=None) -> DesignPoint:
    d = design(graph, sets=sets)
    assert run_checks(d) == []
    return DesignPoint.of(d, "vecadd")


# =============================================================================
# 1. vecadd's design point gives alu4
# =============================================================================


def test_vecadd_design_point_gives_alu4_byte_for_byte():
    """LOW1c acceptance."""
    assert to_json(cluster_file(point()).to_dict()) == ALU4.read_text()


def test_the_cluster_of_small16_and_the_brm_is_alu4():
    inst = load_brm("elementwise_add").resolve("chisel_tiled_spatial", {"W": 4})
    cfg = cluster_of(Platform.load(SMALL16), {"acc": inst})
    assert to_json(cfg.to_dict()) == ALU4.read_text()


def test_platform_changes_reach_the_cluster():
    sets = [("streamers.acc_out.fifo_depth", 4), ("l1.n_banks", 32), ("register_window", 64)]
    cfg = cluster_file(point(sets=sets))
    comps = {c.name: c for c in cfg.components}
    assert comps["acc_out"].config["fifo_depth"] == 4 and comps["acc_a"].config["fifo_depth"] == 2
    assert cfg.l1.n_banks == 32 and cfg.register_map.window == 64
    build_cluster(cfg)


def test_w8_gives_8_lane_streamers_and_accelerator():
    d = design(bound_w8())
    assert run_checks(d) == []
    cfg = cluster_file(DesignPoint.of(d, "vecadd_w8"))
    comps = {c.name: c for c in cfg.components}
    assert [comps[s].config["n_ports"] for s in ("acc_a", "acc_b", "acc_out")] == [8, 8, 8]
    assert comps["acc"].params["lanes"] == 8
    build_cluster(cfg)


# =============================================================================
# 2. The assembly rules
# =============================================================================


def test_component_order_and_register_map():
    cfg = cluster_file(point())
    assert [(c.name, c.kind) for c in cfg.components] == [
        ("xbar", "xbar"), ("dma", "dma"), ("acc_a", "streamer"), ("acc_b", "streamer"),
        ("acc_out", "streamer"), ("acc", "accel"), ("ctl", "controller"),
    ]  # fmt: skip
    assert cfg.register_map.blocks == ["dma", "acc_a", "acc_b", "acc_out", "acc"]
    assert cfg.register_map.spatial_bounds == {}


def test_without_an_l2_there_is_no_dma():
    pf = Platform("l1only", default=Platform.load(SMALL16).default)
    inst = load_brm("elementwise_add").resolve("chisel_tiled_spatial", {"W": 4})
    cfg = cluster_of(pf, {"acc": inst})
    assert "dma" not in [c.name for c in cfg.components] and cfg.l2 is None
    assert cfg.register_map.blocks == ["acc_a", "acc_b", "acc_out", "acc"]
    build_cluster(cfg)


def test_spatial_bounds_other_than_the_lanes_go_in_the_register_map():
    pf = Platform.load(SMALL16)
    acc = stub(pf, "acc", "elementwise", {"lanes": 4, "n_inputs": 2, "op": "add"},
               [("a", False, 4), ("b", False, 4), ("out", True, 4)])  # fmt: skip
    a = acc.streamers[0]
    two_d = Streamer(a.name, a.instance, a.port, a.write, (2, 2), a.options)
    acc = type(acc)(acc.name, acc.accel, acc.params, (two_d, *acc.streamers[1:]))
    cfg = cluster_config(pf, [acc])
    assert cfg.register_map.spatial_bounds == {"acc_a": [2, 2]}
    assert build_cluster(cfg).regmap.blocks["acc_a"].adapter.spatial_bounds == (2, 2)


def test_stub_clusters_keep_their_checked_in_form():
    """red4 and mul1 have no BRM: their streamers come from the platform shell all the same."""
    red4 = ClusterConfig.load(SCEN / "clusters" / "red4.json")
    assert [c.name for c in red4.components] == ["xbar", "acc_in", "acc_out", "acc", "ctl"]
    assert [c.config["n_ports"] for c in red4.components[1:3]] == [4, 1]
    mul1 = ClusterConfig.load(SCEN / "clusters" / "mul1.json")
    assert {c.config["temporal_dims"] for c in mul1.components if c.kind == "streamer"} == {2}


# =============================================================================
# 3. What passes the design checks builds
# =============================================================================

GRID = list(itertools.product((1, 2, 4, 8), (1, 2, 6), (8, 16, 32), (16, 32)))


@pytest.fixture(scope="module")
def graphs():
    from snax_forge.dfg import Graph
    from snax_forge.sandbox import Recipe, apply_recipe
    from tests.design.helpers import FIXTURES, RECIPE

    plain = Graph.load(FIXTURES / "vecadd.snaxdfg")
    return {
        w: apply_recipe(Recipe.load(RECIPE).with_params({"W": w}), plain)[-1].graph
        for w in (1, 2, 4, 8)
    }


def test_every_combination_that_passes_the_checks_builds(graphs):
    """W x temporal_dims x n_banks x register_window: every passing design builds; the
    failing ones are 6 temporal loops in a 16-register window (17 registers), by name."""
    passed, failed = 0, set()
    for w, t, banks, window in GRID:
        sets = [("streamers.default.temporal_dims", t), ("l1.n_banks", banks),
                ("register_window", window)]  # fmt: skip
        d = design(graphs[w], sets=sets)
        problems = run_checks(d)
        if problems:
            failed |= {p.code for p in problems}
            continue
        cfg = cluster_file(DesignPoint.of(d, f"w{w}"))
        built = build_cluster(cfg)
        assert built.regmap.blocks["acc_a"].comp.cfg.n_ports == w
        passed += 1
    assert passed == len(GRID) - 4 * 3 and failed == {"connect.regmap"}


# =============================================================================
# 4. Command line
# =============================================================================


def test_cli_writes_the_cluster_beside_the_point(tmp_path, capsys):
    point().save(tmp_path / "design_point.json")
    assert cli.main(["cluster", str(tmp_path / "design_point.json")]) == 0
    assert (tmp_path / "cluster.json").read_text() == ALU4.read_text()
    out = capsys.readouterr().out
    assert "acc        accel elementwise" in out and "attach a <- acc_a" in out


def test_cli_rejects_a_bad_point(tmp_path, capsys):
    d = point().to_dict()
    d["streamers"]["acc_a"]["fifo_depth"] = 4
    (tmp_path / "dp.json").write_text(json.dumps(d))
    assert cli.main(["cluster", str(tmp_path / "dp.json"), "--out", str(tmp_path / "c.json")]) == 1
    assert "[point.streamers]" in capsys.readouterr().err and not (tmp_path / "c.json").exists()
    assert cli.main([]) == 2
