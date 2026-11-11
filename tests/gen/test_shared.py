"""A shared design (D169, PORT1) beyond gen-gvsoc: the bundle refuses it until the shell
wrapper's mux (open item 114), and the run reads its cluster as one mover per shared mover."""

from __future__ import annotations

import shutil

import pytest

from snax_forge.design import DesignPoint
from snax_forge.gen import generate
from snax_forge.gen.blocks import GenError
from snax_forge.gen.rtl import accelerator_spec
from snax_forge.lower import task_list
from snax_forge.report.design import build_design, render_design
from snax_forge.run.beats import mover_tasks, pairing
from snax_forge.run.cluster import Cluster, load_cluster
from snax_forge.sandbox import Recipe
from tests.helpers import recipe_path

from .helpers import designed


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    """dot_1_shared's design point and its gen-gvsoc folder, under one root."""
    tmp = tmp_path_factory.mktemp("shared")
    point = designed(tmp, "dot_1_shared", "dot_1_shared")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        generate(point, tmp / "gvsoc", root=tmp)
    return tmp, tmp / point, tmp / "gvsoc"


def test_the_bundle_refuses_a_shared_design(shared):
    tmp, point, _ = shared
    with pytest.raises(GenError, match="share their movers") as e:
        accelerator_spec(point, tmp)
    assert e.value.code == "bundle.share"


def test_the_run_reads_one_mover_per_shared_mover(shared, tmp_path):
    """The cluster has dot_reader0, dot_reader1 and dot_writer0, each with its users; the task
    pairing takes each mover's configure of its group's task (vsum's reader1 the idle one)."""
    _, point, out = shared
    shutil.copy(point, tmp_path / "design_point.json")
    c = load_cluster(out / "snax_forge.hjson", tmp_path / "design_point.json")
    movers = c.accelerator.movers
    assert [(m.name, m.gvsoc, m.lanes, m.users, m.enabled) for m in movers] == [
        ("dot_reader0", "reader0", 4, ("vmul_a", "vsum_a"), 0),
        ("dot_reader1", "reader1", 4, ("vmul_b",), 0),
        ("dot_writer0", "writer0", 4, ("vmul_out", "vsum_out"), 1),
    ]  # fmt: skip
    assert [m.name for m in c.accelerator.movers_of("vsum")] == ["dot_reader0", "dot_writer0"]
    # a beat between vsum and writer0 holds vsum.out's one word, vmul's its four (D169)
    acc = c.accelerator
    assert [acc.lanes_of(movers[2], u) for u in ("vmul", "vsum")] == [4, 1]
    assert acc.port_of(movers[0], "vsum") == "vsum_a"
    assert Cluster.from_dict(c.to_dict()).accelerator == c.accelerator
    tasks = task_list(DesignPoint.load(point))
    paired = pairing(tasks, c)
    assert [sorted(v) for v in paired.values()] == [
        ["dot_reader0", "dot_reader1", "dot_writer0"]
    ] * 2
    vsum = paired[next(t for t in paired if t.endswith("vsum"))]
    assert vsum["dot_reader0"].endswith("vsum_a") and vsum["dot_writer0"].endswith("vsum_out")
    assert vsum["dot_reader1"].endswith("vmul_b")  # idle in vsum's task
    assert [t.rsplit("_", 1)[-1] for t in mover_tasks(tasks, movers[2])] == ["out", "out"]


def test_design_md_says_the_set_shares_its_movers(shared, tmp_path):
    """design.md: the set dot of three movers, 12 channels, and the line naming share_ports; the
    shared movers listed with no one port's container."""
    _, point, out = shared
    shutil.copy(point, tmp_path / "design_point.json")
    c = load_cluster(out / "snax_forge.hjson", tmp_path / "design_point.json")
    r = build_design("dot_1_shared", c, DesignPoint.load(point),
                     Recipe.load(recipe_path("dot_1_shared")))  # fmt: skip
    ((name, movers, channels, shared_),) = [(x.name, x.movers, x.channels, x.shared)
                                            for x in r.sets]  # fmt: skip
    assert (name, movers, channels, shared_) == ("dot", 3, 12, True)
    assert [(x.name, x.instance, x.num_ch, x.container) for x in r.streamers] == [
        ("dot_reader0", None, 4, None), ("dot_reader1", None, 4, None),
        ("dot_writer0", None, 4, None)]  # fmt: skip
    assert "dot shares its movers among its groups (share_ports, D169)" in render_design(r)
