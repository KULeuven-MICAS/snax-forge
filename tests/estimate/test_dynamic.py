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
    ("dot_3", {"W": 8, "N": 2048}),  # a linked group, wired
    ("dot_2", {"W": 8, "N": 2048}),  # a depth-1 buffer link: its producer's pushes
    ("dot_5", {"W": 8, "N": 2048}),  # a serial sum of span 8: its firings times its span
    ("vecmul", {"W": 8, "N": 2048}),  # one unit, a writer of 8 channels
])  # fmt: skip
def test_the_graphs_counts_give_the_static_estimate(name, params):
    graph = steps(name, **params)[-1]
    est = estimated(name, **params)
    got = price(*_static(graph, est))
    for part, want in est.parts.items():
        assert got.parts[part] == pytest.approx(want.energy_nj, rel=1e-12), part


def test_a_packed_outputs_packer_pushes_its_owners_pushes(tmp_path):
    """matvec_3 at i8, W = 8: the sum's int32 results packed 2 to a word."""
    from tests.gen.helpers import designed

    point = DesignPoint.load(tmp_path / designed(tmp_path, "matvec_3", "m3", ["types=i8", "W=8"]))
    est = estimate(point.graph, point.platform)
    p, c = _static(point.graph, est)
    assert p.links["sum.out"][0] == "sum" and c.units["sum"][1] == 16
    for part, want in est.parts.items():
        assert price(p, c).parts[part] == pytest.approx(want.energy_nj, rel=1e-12), part


def test_dot_3_at_w8_counts_and_a_refusal_at_a_grants_price():
    """dot 3 at W = 8, N = 2048: 4097 words through the crossbar; 100 refused requests on reader
    A's lanes add 100 accesses and 100 / 8 of A's beats, and no bank access."""
    graph = steps("dot_3", W=8, N=2048)[-1]
    est = estimated("dot_3", W=8, N=2048)
    p, c = _static(graph, est)
    base = price(p, c)
    assert (base.requests, base.reads, base.writes, base.cycles) == (4097, 4096, 1, 257)
    a = next(m for m in c.movers if m.endswith("_a"))
    r, g = c.movers[a]
    c.movers[a] = (r + 100, g)
    got = price(p, c)
    _, channels, e_beat = p.movers[a]
    assert got.parts["crossbar"] - base.parts["crossbar"] == pytest.approx(
        100 * p.e_access_pj / 1000
    )
    assert got.parts["streamers"] - base.parts["streamers"] == pytest.approx(
        100 / channels * e_beat / 1000
    )
    assert got.parts["memory"] == base.parts["memory"]


def test_a_count_with_no_price_is_refused():
    graph = steps("vecmul", W=8, N=2048)[-1]
    p, c = _static(graph, estimated("vecmul", W=8, N=2048))
    c.units["ghost"] = (1, 1)
    with pytest.raises(ValueError, match="ghost"):
        price(p, c)


def test_prices_counts_and_energies_round_trip():
    graph = steps("dot_3", W=8, N=2048)[-1]
    p, c = _static(graph, estimated("dot_3", W=8, N=2048))
    e = price(p, c)
    assert Prices.from_dict(p.to_dict()) == p
    assert Counts.from_dict(c.to_dict()) == c
    assert Energy.from_dict(e.to_dict()) == e


def test_an_estimate_without_a_platform_has_no_prices():
    graph = steps("vecmul", W=8, N=2048)[-1]
    with pytest.raises(ValueError, match="not costed"):
        prices_of(graph, estimate(graph), [])
