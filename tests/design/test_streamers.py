"""The streamer shell against a bound graph (DP1a, D84)."""

from __future__ import annotations

from snax_forge.brm import load_brm
from snax_forge.design import Platform, instances, nest_spatial_bounds, resolve
from snax_forge.dfg import Graph

from .helpers import BOUND, SMALL16, alu4_streamers, bound_w8, platform_dict


def shell(platform: Platform, graph: Graph) -> dict:
    return resolve(platform, instances(graph))


def test_vecadd_on_small16_gives_alu4s_streamers():
    """Names, order and every config field of alu4.json's streamer entries (LOW1c's input)."""
    got = shell(Platform.load(SMALL16), Graph.load(BOUND))
    want = alu4_streamers()
    assert list(got) == list(want) == ["acc_a", "acc_b", "acc_out"]
    assert {k: s.config().to_dict() for k, s in got.items()} == want
    assert [(s.instance, s.port, s.write, s.spatial_bounds) for s in got.values()] == [
        ("acc", "a", False, (4,)), ("acc", "b", False, (4,)), ("acc", "out", True, (4,))
    ]  # fmt: skip


def test_lanes_follow_the_recipe():
    """Bound with W = 8: the same platform gives 8-lane streamers, nothing to edit."""
    got = shell(Platform.load(SMALL16), bound_w8())
    assert {s.n_ports for s in got.values()} == {8}
    assert all(s.spatial_bounds == (8,) for s in got.values())


def test_an_entry_changes_only_its_streamer():
    d = platform_dict()
    d["streamers"]["acc_out"] = {"fifo_depth": 4, "temporal_dims": 2}
    got = shell(Platform.from_dict(d), Graph.load(BOUND))
    assert (got["acc_out"].options.fifo_depth, got["acc_out"].options.temporal_dims) == (4, 2)
    assert got["acc_a"].options == got["acc_b"].options == Platform.from_dict(d).default


def test_given_spatial_bounds_are_used():
    d = platform_dict()
    d["streamers"]["acc_a"] = {"spatial_bounds": [4]}
    got = shell(Platform.from_dict(d), Graph.load(BOUND))
    assert got["acc_a"].spatial_bounds == (4,) and got["acc_a"].n_ports == 4


def test_nest_spatial_bounds_are_the_ports_lanes():
    brm = load_brm("elementwise_add")
    for w in (1, 4, 8):
        inst = brm.resolve("chisel_tiled_spatial", {"W": w})
        assert [nest_spatial_bounds(inst, p) for p in ("a", "b", "out")] == [[w]] * 3


def test_streamer_to_dict():
    s = shell(Platform.load(SMALL16), Graph.load(BOUND))["acc_out"]
    assert s.to_dict() == {
        "instance": "acc", "port": "out", "write": True, "n_ports": 4, "spatial_bounds": [4],
        "temporal_dims": 1, "fifo_depth": 2, "addr_depth": 8, "prio": 0,
    }  # fmt: skip
