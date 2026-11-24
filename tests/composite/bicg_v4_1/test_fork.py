"""fork_read and unfork_read: one read of A given to bicg's two halves (D189)."""

from __future__ import annotations

import pytest

from snax_forge.design.streamers import forked_ports, group_chains, group_runs, groups
from snax_forge.sandbox import (
    TRANSFORMS,
    Recipe,
    apply_recipe,
    fork_read,
    link,
    unfork_read,
    unlink,
)
from snax_forge.sandbox.fork import ForkError
from tests.helpers import imported, recipe_path

SMALL = {"M": 8, "N": 8, "W": 8}  # bicg_v3 at N = W, small


def halves(**params: int):
    """bicg_v3's graph before its fork_read: q's dot_v2 group and s's Mac, each reading A."""
    r = Recipe.load(recipe_path("bicg_v3")).with_params(params or SMALL).to_dict()
    r["steps"] = r["steps"][:-1]
    r = Recipe.from_dict(r)
    return [x.graph for x in apply_recipe(r, imported(r.kernel, r.dtypes, r.types))][-1]


def test_fork_read_joins_the_halves_and_unfork_read_gives_them_back():
    g = halves()
    assert TRANSFORMS["fork_read"].site(g, "A") == {"container": "A", "group": "bicg_types"}
    assert TRANSFORMS["fork_read"].site(g, "s_mult") == {"container": "A", "group": "bicg_types"}
    forked = fork_read(g, "A", "bicg")
    assert groups(forked) == {"bicg": ["mul", "sum", "mac"]}
    assert group_chains(forked, ["mul", "sum", "mac"]) == [["mul", "sum"], ["mac"]]
    assert forked_ports(forked) == {"mac.a": "mul.a"}
    (node,) = [n for n, _ in forked.walk() if n.kind == "fork"]
    assert node.attrs == {"container": "A", "readers": ["mul.a", "mac.a"],
                          "groups": {"mul": "matvec", "sum": "matvec", "mac": None}}  # fmt: skip
    (run,) = group_runs(forked)["bicg"]
    assert list(run) == ["mul", "sum", "mac"]
    assert TRANSFORMS["unfork_read"].site(forked, node.id) == {"container": "A"}
    assert unfork_read(forked, "A").to_dict() == g.to_dict()


def test_the_beats_in_another_order_name_interchange_map():
    """At N = 2W the column form reads A strip by strip, the row form row by row."""
    g = halves(M=8, N=16, W=8)
    with pytest.raises(ForkError, match="another order: interchange_map its loops first") as e:
        fork_read(g, "A", "bicg")
    assert e.value.code == "fork.order"


@pytest.mark.parametrize(
    "selected, text, code",
    [
        ("nope", "no container of that name", "fork.container"),
        ("p", r"read by \['q_mult'\]; a fork gives one read to two or more", "fork.container"),
    ],
)  # fmt: skip
def test_refused_where_it_does_not_apply(selected, text, code):
    with pytest.raises(ForkError, match=text) as e:
        TRANSFORMS["fork_read"].site(halves(), selected)
    assert e.value.code == code


def test_a_fork_twice_and_a_bad_group():
    forked = fork_read(halves(), "A", "bicg")
    with pytest.raises(ForkError, match="already forked"):
        fork_read(forked, "A", "bicg")
    with pytest.raises(ForkError, match="'1x' is not a name"):
        fork_read(halves(), "A", "1x")


def test_unfork_read_after_the_group_changed():
    """unlink takes sum out of the forked group; unfork_read then refuses, naming the fix."""
    forked = fork_read(halves(), "A", "bicg")
    cut = unlink(forked, "q_prod")
    assert groups(cut) == {"bicg": ["mul", "mac"], "sum": ["sum"]}
    with pytest.raises(ForkError, match="unlink the others first") as e:
        unfork_read(cut, "A")
    assert e.value.code == "fork.changed"
    assert groups(link(cut, "q_prod", 1, "bicg")) == groups(forked)
