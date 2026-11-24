"""The dynamic energy (estimate/dynamic.py, D177): a run's counts priced at the instances' own
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


def test_dot_3_at_w8_counts_and_a_refusal_at_a_grants_price():
    """dot 3 at W = 8, N = 2048: 4097 words through the crossbar, 4096 read at a read-only input's
    price and 1 written at a write-only one's (D178); 100 refused requests on reader A's lanes add
    100 read-only accesses and 100 / 8 of A's beats, and no bank access."""
    graph = steps("dot_v3", W=8, N=2048)[-1]
    est = estimated("dot_v3", W=8, N=2048)
    p, c = _static(graph, est)
    base = price(p, c)
    assert (base.requests, base.reads, base.writes, base.cycles) == (4097, 4096, 1, 257)
    assert base.parts["crossbar"] == pytest.approx((4096 * p.e_xro_pj + p.e_xwo_pj) / 1000)
    assert (round(p.e_xro_pj, 6), round(p.e_xwo_pj, 6)) == (1.839299, 1.454513)
    a = next(m for m in c.movers if m.endswith("_a"))
    r, g = c.movers[a]
    c.movers[a] = (r + 100, g)
    got = price(p, c)
    _, channels, e_beat = p.movers[a]
    assert got.parts["crossbar"] - base.parts["crossbar"] == pytest.approx(100 * p.e_xro_pj / 1000)
    assert got.parts["streamers"] - base.parts["streamers"] == pytest.approx(
        100 / channels * e_beat / 1000
    )
    assert got.parts["memory"] == base.parts["memory"]


def test_prices_counts_and_energies_round_trip():
    graph = steps("dot_v3", W=8, N=2048)[-1]
    p, c = _static(graph, estimated("dot_v3", W=8, N=2048))
    e = price(p, c)
    assert Prices.from_dict(p.to_dict()) == p
    assert Counts.from_dict(c.to_dict()) == c
    assert Energy.from_dict(e.to_dict()) == e
