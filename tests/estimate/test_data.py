"""The estimate's data side and its work not mapped (D155, C§19), and the round trip."""

from __future__ import annotations

import pytest

from snax_forge.estimate import Estimate, estimate
from snax_forge.estimate.cycles import not_mapped
from snax_forge.sandbox import split, split_map, split_reduce
from snax_forge.sandbox.transforms import tag_map
from tests.helpers import imported, steps

from .helpers import estimated


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
    t = tag_map(s, "add_map_inner", "spatial")
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


@pytest.mark.slow
def test_the_bank_bound_of_vecmul():
    """vecmul at N = 2048 (D155): three ports of W words a beat over 32 banks. At W = 16, 6144
    words take at least 192 cycles against the ideal 128 (the run's COUNTER is 194), a note; at
    W = 8 the same 192 is below the ideal 256, no note. Without a platform, no bound."""
    (sixteen,) = estimated("vecmul", N=2048, W=16).groups
    assert (sixteen.cycles, sixteen.bank_bound, sixteen.counter) == (128, 192, 131)
    assert sixteen.bank_bound_note == ("acc: the outer ports move 6144 words over 32 banks, at "
                                       "least 192 cycles against the ideal 128")  # fmt: skip
    (eight,) = estimated("vecmul", N=2048, W=8).groups
    assert (eight.cycles, eight.bank_bound, eight.bank_bound_note) == (256, 192, None)
    (alone,) = estimate(steps("vecmul", N=2048, W=16)[-1]).groups
    assert (alone.bank_bound, alone.bank_bound_note) == (None, None)


@pytest.mark.slow
def test_the_bank_bound_of_dot_1_per_group():
    """dot 1 at N = 2048, W = 8: the mul's 3 ports of 256 beats x 8 channels, 6144 words, 192
    banked cycles under its 256; the sum's in (256 x 8) and out (1 word), 2049 words, 65."""
    mul, total = estimated("dot_1", N=2048, W=8).groups
    assert [(g.name, g.cycles, g.bank_bound) for g in (mul, total)] == [
        ("mul", 256, 192), ("sum", 257, 65)]  # fmt: skip
    assert mul.bank_bound_note is None and total.bank_bound_note is None


def test_an_untagged_tail_split_pads_and_checks():
    """vecadd at N = 60 split by 8 with ``split`` alone (both maps untagged): W comes from the
    inner map (dfg/pad.py's split_factor), so the reference check passes and 60 of 64 lanes
    are used."""
    from snax_forge.dfg import Graph
    from snax_forge.sandbox import Recipe, Step, apply_recipe
    from tests.helpers import FIXTURES

    r = Recipe(
        "vecadd_60", "vecadd", {}, {"N": 60}, [Step("split", {"map": "add_map", "factor": 8})]
    )
    last = apply_recipe(r, Graph.load(FIXTURES / "vecadd.snaxdfg"))[-1].graph  # checked
    s = estimate(last).splits["add_map"]
    assert (s.n, s.w, s.lane_use) == (60, 8, 60 / 64)
    assert estimate(last).containers["A"].bytes == 64 * 8
