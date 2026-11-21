"""The dynamic energy (estimate/dynamic.py, D177): a run's counts priced at the instances' own
prices; with the graph's counts, the static estimate to the last digit."""

from __future__ import annotations

import pytest

from snax_forge.design import DesignPoint
from snax_forge.estimate import estimate
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


@pytest.mark.parametrize("name, params", [
    ("dot_v3", {"W": 8, "N": 2048}),  # a linked group, wired
    ("dot_v2", {"W": 8, "N": 2048}),  # a depth-1 buffer link: its producer's pushes
    ("dot_v5", {"W": 8, "N": 2048}),  # a serial sum of span 8: its firings times its span
    ("vecmul_v1", {"W": 8, "N": 2048}),  # one unit, a writer of 8 channels
])  # fmt: skip
def test_the_graphs_counts_give_the_static_estimate(name, params):
    graph = steps(name, **params)[-1]
    est = estimated(name, **params)
    got = price(*_static(graph, est))
    for part, want in est.parts.items():
        assert got.parts[part] == pytest.approx(want.energy_nj, rel=1e-12), part


def test_a_packed_outputs_packer_pushes_its_owners_pushes(tmp_path):
    """matvec_v3 at i8, W = 8: the sum's int32 results packed 2 to a word."""
    from tests.gen.helpers import designed

    point = DesignPoint.load(tmp_path / designed(tmp_path, "matvec_v3", "m3", ["types=i8", "W=8"]))
    est = estimate(point.graph, point.platform)
    p, c = _static(point.graph, est)
    assert p.links["sum.out"][0] == "sum" and c.units["sum"][1] == 16
    for part, want in est.parts.items():
        assert price(p, c).parts[part] == pytest.approx(want.energy_nj, rel=1e-12), part


def test_a_sliced_inputs_slicer_pops_its_units_firings(tmp_path):
    """vecadd_v1_1 at W = 1: each slicer pops on the unit's 64 firings; with the graph's counts the
    static estimate, and 10 more busy cycles add the control unit's clock only."""
    from tests.gen.helpers import designed

    point = DesignPoint.load(tmp_path / designed(tmp_path, "vecadd_v1_1", "v", ["W=1"]))
    est = estimate(point.graph, point.platform)
    p, c = _static(point.graph, est)
    assert set(p.slicers) == {"acc.a", "acc.b"} and set(p.links) == {"acc.out"}
    u, count, e_pop = p.slicers["acc.a"]
    assert (u, count, c.units["acc"], c.busy) == ("acc", "firings", (64, 64), 65)
    s = est.groups[0].cost.links["acc.a"]
    assert e_pop == s.e_fire_pj
    base = price(p, c)
    for part, want in est.parts.items():
        assert base.parts[part] == pytest.approx(want.energy_nj, rel=1e-12), part
    assert base.fires == 64 + 64 + 2 * 64  # the unit's, the packer's pushes, the slicers' pops
    c.busy += 10
    more = price(p, c)
    assert more.parts["datapath"] == base.parts["datapath"]
    assert more.parts["control"] - base.parts["control"] == pytest.approx(
        10 * p.ctrl_bits * p.e_ctrl_pj / 1000
    )
    assert Prices.from_dict(p.to_dict()) == p


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


def test_a_count_with_no_price_is_refused():
    graph = steps("vecmul_v1", W=8, N=2048)[-1]
    p, c = _static(graph, estimated("vecmul_v1", W=8, N=2048))
    c.units["ghost"] = (1, 1)
    with pytest.raises(ValueError, match="ghost"):
        price(p, c)


def test_prices_counts_and_energies_round_trip():
    graph = steps("dot_v3", W=8, N=2048)[-1]
    p, c = _static(graph, estimated("dot_v3", W=8, N=2048))
    e = price(p, c)
    assert Prices.from_dict(p.to_dict()) == p
    assert Counts.from_dict(c.to_dict()) == c
    assert Energy.from_dict(e.to_dict()) == e


def test_an_estimate_without_a_platform_has_no_prices():
    graph = steps("vecmul_v1", W=8, N=2048)[-1]
    with pytest.raises(ValueError, match="not costed"):
        prices_of(graph, estimate(graph), [])
