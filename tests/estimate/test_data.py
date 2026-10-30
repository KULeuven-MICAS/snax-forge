"""The estimate's data side and its work not mapped (D151, C§19), and the round trip."""

from __future__ import annotations

import pytest

from snax_forge.estimate import Estimate, estimate
from snax_forge.estimate.cycles import not_mapped
from snax_forge.sandbox import split, split_map, split_reduce
from snax_forge.sandbox.transforms import tag_map

from .helpers import estimated, imported, steps


@pytest.mark.slow
def test_a_link_takes_tmp0_out_of_l1():
    """dot 1 against dot 2 at N = 512, W = 4: tmp0 written and read again through L1 against a
    link; A, B and tmp0 are 512 int64 each, out one, none padded."""
    one, two = estimated("dot_1", N=512, W=4), estimated("dot_2", N=512, W=4)
    traffic = [(sum(c.reads for c in e.containers.values()),
                sum(c.writes for c in e.containers.values())) for e in (one, two)]  # fmt: skip
    assert traffic == [(1536, 513), (1024, 1)]
    assert (one.l1_footprint, two.l1_footprint, one.l1_size) == (12296, 8200, 128 * 1024)
    assert (two.containers["tmp0"].placement, two.containers["tmp0"].depth) == ("link", 1)
    assert two.ports["mul.a"].to_dict() == {"container": "A", "loops": [["i_t", 128]],
                                            "beats": 128, "num_ch": 4}  # fmt: skip
    assert [(g.channels, g.words_per_cycle) for g in two.groups] == [(9, 1025 / 130)]


def _bound(g):
    d = g.to_dict()
    d["symbols"] = {"N": 512} if g.name == "dot" else {"N": 64}
    return type(g).from_dict(d)


@pytest.mark.slow
def test_not_mapped_counts_the_iterations_no_unit_runs():
    """dot at N = 512: both nests (1024); mult_map split by 4 and tagged, its spatial map left
    out (128 + 512); sum_map split into its partials and final sum (128 + 128 + 1). vecadd at
    N = 64: split but untagged, all 64; the inner map tagged spatial, 16."""
    dot = _bound(imported("dot"))
    a = split_map(dot, "mult_map", 4)
    b = split_reduce(a, "sum_map", 4)
    assert [not_mapped(g) for g in (dot, a, b)] == [1024, 640, 257]
    vecadd = _bound(imported("vecadd"))
    s = split(vecadd, "add_map", 4)
    t = tag_map(s, "add_map_s", "spatial")
    assert [not_mapped(g) for g in (vecadd, s, t)] == [64, 64, 16]


@pytest.mark.slow
def test_a_tail_uses_part_of_its_lanes():
    e = estimate(steps("vecadd_n7")[-1])
    assert e.l1_size is None
    ((name, s),) = e.splits.items()
    assert (name, s.n, s.w, s.lane_use) == ("add_map", 7, 4, 7 / 8)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["dot_1", "dot_4", "vecadd_n7"])
def test_the_estimate_round_trips(name):
    e = estimated(name)
    assert Estimate.from_dict(e.to_dict()).to_dict() == e.to_dict()
    assert Estimate.from_dict(e.to_dict()) == e
