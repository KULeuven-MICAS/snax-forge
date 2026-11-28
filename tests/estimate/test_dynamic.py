"""The dynamic energy (estimate/dynamic.py): a run's counts priced at the instances' own
prices; with the graph's counts, the static estimate to the last digit."""

from __future__ import annotations

import pytest

from snax_forge.estimate.dynamic import (
    Counts,
    Energy,
    Prices,
    graph_counts,
    port_movers,
    price,
    prices_of,
)
from tests.helpers import steps

from .helpers import estimated


def _static(graph, est):
    return prices_of(graph, est, port_movers(est)), graph_counts(graph, est)


def test_a_sliced_seed_pops_once_a_job():
    """A seed's slicer pops on its unit's pushes (one result a job), a plain input's on its
    firings."""
    p = Prices(
        {"sum": (1, 2.0)},
        {},
        0,
        0.0,
        {},
        0.0,
        0.0,
        0.0,
        0.0,
        8,
        {"sum.seed": ("sum", "pushes", 0.5), "sum.a": ("sum", "firings", 0.7)},
    )
    e = price(p, Counts({"sum": (40, 4)}, busy=42))
    assert e.parts["datapath"] == pytest.approx((40 * 2.0 + 4 * 0.5 + 40 * 0.7) / 1000)
    assert e.fires == 40 + 4 + 40


def test_dot_3_at_w8_counts_and_a_refusal_at_its_fraction():
    """dot 3 at W = 8, N = 2048: 4097 words through the crossbar, 4096 read at a read-only input's
    price and 1 written at a write-only one's, the banks also 512 load beats and 1 store beat of a
    superbank (8 banks each), as SNAX-GVSoC counts them; 100 refused requests on reader A's lanes add
    r_xbar of 100 read-only accesses, A's own r (its reader's) of 100 / 8 of its beats, and no bank access."""
    graph = steps("dot_v3", W=8, N=2048)[-1]
    est = estimated("dot_v3", W=8, N=2048)
    p, c = _static(graph, est)
    base = price(p, c)
    assert est.transfers == (512, 1)
    assert (base.requests, base.reads, base.writes, base.cycles) == (4097, 4096 + 8, 1 + 4096, 257)
    floor = p.floor["crossbar"] * c.cycles
    assert base.parts["crossbar"] == pytest.approx((4096 * p.e_xro_pj + p.e_xwo_pj + floor) / 1000)
    assert (round(p.e_xro_pj, 6), round(p.e_xwo_pj, 6)) == (1.839299, 1.454513)
    a = next(m for m in c.movers if m.endswith("_a"))
    r, g = c.movers[a]
    c.movers[a] = (r + 100, g)
    got = price(p, c)
    _, channels, e_beat = p.movers[a]
    assert got.parts["crossbar"] - base.parts["crossbar"] == pytest.approx(
        p.r_xbar * 100 * p.e_xro_pj / 1000
    )
    assert got.parts["streamers"] - base.parts["streamers"] == pytest.approx(
        p.r_movers[a] * 100 / channels * e_beat / 1000
    )
    assert 0.05 < p.r_movers[a] < 0.3 and p.r_movers[a] != p.r_strm and 0 <= p.r_xbar < 0.01
    assert got.parts["memory"] == base.parts["memory"]


def test_prices_counts_and_energies_round_trip():
    graph = steps("dot_v3", W=8, N=2048)[-1]
    p, c = _static(graph, estimated("dot_v3", W=8, N=2048))
    e = price(p, c)
    assert Prices.from_dict(p.to_dict()) == p
    assert Counts.from_dict(c.to_dict()) == c
    assert Energy.from_dict(e.to_dict()) == e


def test_a_shared_movers_beats_at_the_channels_each_user_takes():
    """gesummv_v2_1_2's reader0, 4 channels, shared by two dots reading A and B on 1 channel and
    the axpby reading tmp1_sum on 4: each dot's 512 beats on lane 0 alone, the axpby's 16 on all
    four. Priced at the channels each user takes, as the static estimate prices each port, the
    beats are 1040, not the 1088 / 4 = 272 of the mover's 4 channels; a refusal on a wider
    lane than lane 0 counts no narrow beat below zero."""
    graph = steps("gesummv_v2_1_2", N=64, W=8)[-1]
    est = estimated("gesummv_v2_1_2", N=64, W=8)
    from snax_forge.estimate.dynamic import MoverSpec, mover_beats

    spec = MoverSpec("gesummv_reader0", False, 8, 1, 4, (1, 1, 4))
    p = prices_of(graph, est, [spec])
    _, channels, e4 = p.movers["gesummv_reader0"]
    (e1,) = p.narrow["gesummv_reader0"].values()
    assert channels == 4 and 0 < e1 < e4
    lanes = (512 + 512 + 16, 16, 16, 16)
    c = Counts(movers={"gesummv_reader0": (sum(lanes), sum(lanes))},
               lanes={"gesummv_reader0": lanes})  # fmt: skip
    got = price(p, c)
    assert got.beats == 1040
    assert got.parts["streamers"] == pytest.approx((1024 * e1 + 16 * e4) / 1000)
    assert mover_beats(4, e4, {1: e1}, (10, 12, 12, 12), 46) == (12, 12 * e4)
    assert mover_beats(4, e4, {}, None, 48) == (12, 12 * e4)  # unshared: over its channels
    assert Prices.from_dict(p.to_dict()) == p and Counts.from_dict(c.to_dict()) == c
