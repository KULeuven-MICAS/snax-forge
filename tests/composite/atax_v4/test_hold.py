"""The held link (D197): the vsum's t[i] over a link of depth 1 into the bank's held,
broadcast b, taken on a row's first firing, so the sum gives one result a row of ROWS of the
bank's firings. At M = 4, N = 8, W = 4: T = ROWS = 2, the bank's T = M · ROWS = 8."""

from snax_forge.estimate.cycles import group_rule
from snax_forge.gen.accelerator import Description
from snax_forge.gen.rtl import rule_graph
from tests.helpers import requires_submodule

pytestmark = requires_submodule

TS = [1, 2, 8]  # mul, sum, mac


def rule(walked, **depths):
    """The group as the rule takes it (gen/rtl.py's rule_graph), its fork link's depth set."""
    desc = Description.from_dict(walked.accelerator)
    (acc,) = desc.accelerators
    units, links, owner, forks, branches = rule_graph(acc.groups[0], acc.forks)
    branches = [dict(b, **depths) for b in branches]
    return group_rule().GroupRule(units, links, owner, forks, branches)


def test_the_link_into_mac_b_is_held_and_one_lane(walked):
    (g,) = walked.groups
    assert {"name": "tmp1", "from": "sum.out", "to": "mac.b", "depth": 1} in g["links"]
    b, out = walked.units["mac"]["ports"]["b"], walked.units["sum"]["ports"]["out"]
    assert b["broadcast"] and b["hold"] and b["lanes"] == 4 and out["lanes"] == 1


def test_the_sum_gives_one_t_a_row(walked):
    g = rule(walked)
    assert g.links[1]["hold"]
    assert g.firings_per_run(1, TS) == [8, 8, 8]
    rows, end = g.run(1, TS, lambda c: (1, 1), lambda c: 1, 200)
    assert end is not None
    gives = [sum(r[2][k][1] for r in rows) for k in range(3)]
    assert gives == [8, 4, 2]  # mul a product a beat, sum t[0..3], the bank its 2 rows
