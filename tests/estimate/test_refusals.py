"""The estimate package's refusals, one table: each case builds its input and makes the call that
is refused, with the error and the message it must give."""

from __future__ import annotations

import pytest

from snax_forge.estimate import estimate
from snax_forge.estimate.dynamic import (
    Counts,
    Prices,
    price,
    prices_of,
)
from tests.estimate.helpers import estimated
from tests.estimate.test_dynamic import _static
from tests.helpers import steps


def _each_control_unit_clocks_its_own_groups_cycles():
    graph = steps("dot_v1", W=8, N=2048)[-1]
    est = estimated("dot_v1", W=8, N=2048)
    p, c = _static(graph, est)
    assert c.groups == {g.name: g.cycles for g in est.groups} and len(c.groups) == 2
    assert p.group_bits == {g.name: g.cost.ctrl_bits for g in est.groups}
    want = sum(g.cost.ctrl_bits * g.cycles for g in est.groups) * p.e_ctrl_pj / 1000
    assert price(p, c).parts["control"] == pytest.approx(want, rel=1e-12)
    assert sum(g.cost.ctrl_nj for g in est.groups) == pytest.approx(want, rel=1e-12)
    first = est.groups[0]
    alone = Counts(c.units, 100, c.movers, groups={first.name: 100})
    assert price(p, alone).parts["control"] == pytest.approx(
        first.cost.ctrl_bits * 100 * p.e_ctrl_pj / 1000
    )
    assert Counts.from_dict(c.to_dict()) == c and Prices.from_dict(p.to_dict()) == p
    price(p, Counts(groups={"nope": 1}))


def _a_count_with_no_price_is_refused():
    graph = steps("vecmul_v1", W=8, N=2048)[-1]
    p, c = _static(graph, estimated("vecmul_v1", W=8, N=2048))
    c.units["ghost"] = (1, 1)
    price(p, c)


def _an_estimate_without_a_platform_has_no_prices():
    graph = steps("vecmul_v1", W=8, N=2048)[-1]
    prices_of(graph, estimate(graph), [])


CASES = [
    pytest.param(_each_control_unit_clocks_its_own_groups_cycles, ValueError, 'have counts and no price', id='each_control_unit_clocks_its_own_groups_cycles'),
    pytest.param(_a_count_with_no_price_is_refused, ValueError, 'ghost', id='a_count_with_no_price_is_refused'),
    pytest.param(_an_estimate_without_a_platform_has_no_prices, ValueError, 'not costed', id='an_estimate_without_a_platform_has_no_prices'),
]  # fmt: skip


@pytest.mark.parametrize(("case", "error", "message"), CASES)
def test_refusal(case, error, message):
    with pytest.raises(error, match=message):
        case()
