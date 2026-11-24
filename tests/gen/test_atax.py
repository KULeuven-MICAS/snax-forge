"""accelerator.json for atax's one group: t over a link into the bank's held, broadcast input,
and A's fork with a buffered branch to the `Mac` (``fork_links``). The rule's counter on the
description is the estimate's. Imported through DaCe, hence slow."""

from __future__ import annotations

from dataclasses import replace

import pytest

from snax_forge.design import DesignPoint
from snax_forge.estimate import estimate
from snax_forge.gen.accelerator import Description, ForkLink, GenError, _check_links, describe
from snax_forge.gen.rtl import group_counter
from tests.gen.helpers import designed

pytestmark = pytest.mark.slow
SETS = ("M=4", "N=8", "W=4", "R=2", "D=4")  # T = N / W = 2 beats a row of A


@pytest.fixture(scope="module")
def point(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("atax")
    rel = designed(tmp, "atax_v4", "atax_v4", SETS)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        p = DesignPoint.load(tmp / rel)
        return p, describe(p, rel)


def test_t_comes_over_a_link_into_a_held_broadcast_port(point):
    _, desc = point
    (acc,) = desc.accelerators
    (g,) = acc.groups
    mac = next(u for u in g.units if u.name == "mac")
    assert (mac.ports["b"].broadcast, mac.ports["b"].hold, mac.ports["b"].lanes) == (True, True, 4)
    assert [(k.from_, k.to, k.depth) for k in g.links] == [
        ("mul.out", "sum.a", 1),
        ("sum.out", "mac.b", 1),
    ]


def test_a_buffered_branch_is_a_fork_link_and_round_trips(point):
    _, desc = point
    (acc,) = desc.accelerators
    assert acc.inputs == [["mul.a", "mac.a"], "mul.b"] and acc.outputs == ["mac.out"]
    (g,) = acc.groups
    assert g.fork_links == [ForkLink("A", "mac.a", 4)]
    assert Description.from_dict(desc.to_dict()) == desc
    assert "fork_links" not in replace(g, fork_links=[]).to_dict()  # absent without one


def test_the_counter_is_the_estimates(point):
    p, desc = point
    (acc,) = desc.accelerators
    (g,) = acc.groups
    (est,) = estimate(p.graph, p.platform).groups
    counter, _ = group_counter(g, 1, {"sum": 2, "mac": 8}, acc.forks)
    assert counter == est.cycles == 4 * 2 + 2 + 3 + 1  # the slicers' cycle (int8, W = 4)


def test_a_fork_link_into_the_reader_or_of_no_depth_is_refused(point):
    _, desc = point
    (acc,) = desc.accelerators
    (g,) = acc.groups
    for bad, text in (
        (ForkLink("A", "mul.a", 4), "not its reader's first"),
        (ForkLink("A", "mac.a", 0), "depth 0; a fork's branch is a buffer"),
    ):
        with pytest.raises(GenError, match=text):
            _check_links(replace(acc, groups=[replace(g, fork_links=[bad])]))
