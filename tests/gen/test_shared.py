"""A shared design beyond gen-gvsoc: gen-rtl and the bundle take it, the shell
wrapper holding its data-side mux (``<tag>_<set>_port_mux``), and the run reads
its cluster as one mover per shared mover."""

from __future__ import annotations

import shutil

import pytest

from snax_forge.design import DesignPoint
from snax_forge.gen import generate
from snax_forge.report.design import build_design, render_design
from snax_forge.run.cluster import load_cluster
from snax_forge.sandbox import Recipe
from tests.helpers import recipe_path

from .helpers import designed


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    """dot_v1_2's design point and its gen-gvsoc folder, under one root."""
    tmp = tmp_path_factory.mktemp("shared")
    point = designed(tmp, "dot_v1_2", "dot_v1_2")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        generate(point, tmp / "gvsoc", root=tmp)
    return tmp, tmp / point, tmp / "gvsoc"


def test_design_md_says_the_set_shares_its_movers(shared, tmp_path):
    """design.md: the set dot of three movers, 12 channels, and the line naming share_ports; the
    shared movers listed with no one port's container."""
    _, point, out = shared
    shutil.copy(point, tmp_path / "design_point.json")
    c = load_cluster(out / "snax_forge.hjson", tmp_path / "design_point.json")
    r = build_design("dot_v1_2", c, DesignPoint.load(point),
                     Recipe.load(recipe_path("dot_v1_2")))  # fmt: skip
    ((name, movers, channels, shared_),) = [(x.name, x.movers, x.channels, x.shared)
                                            for x in r.sets]  # fmt: skip
    assert (name, movers, channels, shared_) == ("dot", 3, 12, True)
    assert [(x.name, x.instance, x.num_ch, x.container) for x in r.streamers] == [
        ("dot_reader0", None, 4, None), ("dot_reader1", None, 4, None),
        ("dot_writer0", None, 4, None)]  # fmt: skip
    assert "dot shares its movers among its groups (share_ports)" in render_design(r)
