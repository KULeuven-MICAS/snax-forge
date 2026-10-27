"""The streamer shell against a bound graph (D84)."""

from __future__ import annotations

from snax_forge.brm import Brm
from snax_forge.design import Platform, instances, nest_spatial_bounds, resolve
from snax_forge.dfg import Graph
from tests.brm.helpers import affine_adder

from .helpers import (
    BOUND,
    SMALL16,
    VECADD_STREAMERS,
    bound_matadd,
    bound_vsum,
    bound_w8,
    platform_dict,
)


def shell(platform: Platform, graph: Graph) -> dict:
    return resolve(platform, graph, instances(graph))


def test_vecadd_on_small16_gives_its_streamers():
    """Names, order and every field of vecadd's three streamers."""
    got = shell(Platform.load(SMALL16), Graph.load(BOUND))
    assert list(got) == list(VECADD_STREAMERS) == ["acc_a", "acc_b", "acc_out"]
    assert {k: s.to_dict() for k, s in got.items()} == VECADD_STREAMERS
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
    d["streamers"]["acc_out"] = {"fifo_depth": 4}
    got = shell(Platform.from_dict(d), Graph.load(BOUND))
    assert got["acc_out"].options.fifo_depth == 4
    assert got["acc_a"].options == got["acc_b"].options == Platform.from_dict(d).default


def test_given_spatial_bounds_are_used():
    d = platform_dict()
    d["streamers"]["acc_a"] = {"spatial_bounds": [4]}
    got = shell(Platform.from_dict(d), Graph.load(BOUND))
    assert got["acc_a"].spatial_bounds == (4,) and got["acc_a"].n_ports == 4


def test_nest_spatial_bounds_are_the_ports_lanes():
    """An affine BRM: the spatial loops of each port's nest."""
    brm = Brm.from_dict(affine_adder())
    for w in (1, 4, 8):
        inst = brm.resolve("chisel_tiled_spatial", {"W": w})
        assert [nest_spatial_bounds(inst, p) for p in ("a", "b", "out")] == [[w]] * 3


def test_stream_spatial_bounds_are_the_memlets_ranges():
    """elementwise_add is in the stream notation: matadd's A[i, 4 * j_t:4 * j_t + 4] has one
    range of 4, so each streamer has spatial bounds [4]."""
    got = shell(Platform.load(SMALL16), bound_matadd())
    assert {k: s.spatial_bounds for k, s in got.items()} == {
        "acc_a": (4,), "acc_b": (4,), "acc_out": (4,)
    }  # fmt: skip


def test_temporal_dims_come_from_the_graph():
    """One streamer loop per firing loop the port's memlet runs over (D84): vecadd 1; vsum's
    out, which folds the one firing loop (rate T), 1; matadd, a row and a column loop, 2."""
    pf = Platform.load(SMALL16)
    dims = {
        name: {k: s.temporal_dims for k, s in shell(pf, g).items()}
        for name, g in (("vecadd", Graph.load(BOUND)), ("vsum", bound_vsum()),
                        ("matadd", bound_matadd()))
    }  # fmt: skip
    assert dims == {
        "vecadd": {"acc_a": 1, "acc_b": 1, "acc_out": 1},
        "vsum": {"acc_a": 1, "acc_out": 1},
        "matadd": {"acc_a": 2, "acc_b": 2, "acc_out": 2},
    }


def test_a_streamer_takes_its_containers_dtype():
    """int32 containers: two elements per word, the wire default W / k channels (D142)."""
    d = Graph.load(BOUND).to_dict()
    for c in d["containers"].values():
        c["dtype"] = "int32"
    got = shell(Platform.load(SMALL16), Graph.from_dict(d))
    assert {k: (s.dtype, s.k, s.n_ports, s.channels) for k, s in got.items()} == {
        "acc_a": ("int32", 2, 4, 2), "acc_b": ("int32", 2, 4, 2), "acc_out": ("int32", 2, 4, 2)
    }  # fmt: skip
    pd = platform_dict()
    pd["streamers"]["acc_out"] = {"channels": 4}
    assert shell(Platform.from_dict(pd), Graph.from_dict(d))["acc_out"].channels == 4


def test_streamer_to_dict():
    s = shell(Platform.load(SMALL16), Graph.load(BOUND))["acc_out"]
    assert s.to_dict() == {
        "instance": "acc", "port": "out", "write": True, "n_ports": 4, "spatial_bounds": [4],
        "temporal_dims": 1, "dtype": "int64", "k": 1, "channels": 4, "fifo_depth": 2,
    }  # fmt: skip
