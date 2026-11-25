"""atax's one group: t linked into the bank's held input, then A read once and buffered on the
`Mac`'s branch (`link` on a forked container)."""

from __future__ import annotations

import pytest

from snax_forge.design.streamers import forked_ports, groups
from snax_forge.sandbox import TRANSFORMS, Recipe, apply_recipe, link, unlink
from snax_forge.sandbox.compose import LinkError, beats
from tests.helpers import imported, recipe_path

SMALL = {"M": 4, "N": 8, "W": 4, "R": 2, "D": 4}  # T = N / W = 2 beats a row of A


def steps(recipe: str, upto: int | None = None, **params: int):
    """``recipe``'s graphs at ``SMALL`` (or ``params``), its last ``upto`` steps left out."""
    r = Recipe.load(recipe_path(recipe))
    r = r.with_params({k: v for k, v in (params or SMALL).items() if k in r.params}).to_dict()
    r["steps"] = r["steps"][:upto]
    r = Recipe.from_dict(r)
    return [x.graph for x in apply_recipe(r, imported(r.kernel, r.dtypes, r.types))]


def test_a_held_input_takes_one_beat_a_bank_row():
    """The vsum gives t[i] once a row of A; the bank's t port, held for its R rows, takes it
    once a row too, so the link passes M beats against M."""
    *_, before, g = steps("atax_v3")
    nodes = {n.attrs["instance"]: n for n, _ in before.walk() if n.kind == "accelerated"}
    assert len(beats(before, nodes["sum"], "out")) == SMALL["M"]
    assert len(beats(before, nodes["mac"], "b")) == SMALL["M"]
    assert groups(g) == {"matvec": ["mul", "sum", "mac"]}


def test_link_on_a_fork_buffers_the_branch_down_the_chain():
    *_, forked, buffered, g = steps("atax_v4")
    (fork,) = [n for n, _ in g.walk() if n.kind == "fork"]
    assert fork.attrs["readers"] == ["mul.a", "mac.a"] and fork.attrs["buffers"] == {"mac.a": 4}
    assert TRANSFORMS["link"].site(forked, "A") == {"container": "A", "group": "atax"}
    assert TRANSFORMS["link"].site(forked, fork.id) == {"container": "A", "group": "atax"}
    with pytest.raises(
        LinkError, match="the fork gives"
    ):  # every branch buffered: nothing to offer
        TRANSFORMS["link"].site(buffered, "A")
    assert groups(g) == {"atax": ["mul", "sum", "mac"]}
    assert forked_ports(g) == {"mac.a": "mul.a"}
    assert unlink(buffered, "A").to_dict() == forked.to_dict()


def test_the_link_into_a_forked_reader_comes_after_its_buffer():
    *_, forked, _, _ = steps("atax_v4")
    with pytest.raises(LinkError, match=r"mac\.a takes A from a fork; link A first") as e:
        link(forked, "tmp1", 1, "atax")
    assert e.value.code == "link.fork"


def test_unlink_of_the_buffer_waits_for_the_link_into_its_reader():
    *_, g = steps("atax_v4")
    with pytest.raises(LinkError, match="mac.a is fed by a link too; unlink that first") as e:
        unlink(g, "A")
    assert e.value.code == "unlink.fork"
    assert unlink(unlink(g, "tmp1"), "A").to_dict() == steps("atax_v4", -1)[-2].to_dict()


def test_a_fork_branch_is_a_buffer():
    *_, forked, _, _ = steps("atax_v4")
    with pytest.raises(LinkError, match="a forked read's branch is a buffer, depth 1 or more"):
        link(forked, "A", 0, "atax")
    with pytest.raises(LinkError, match="the fork's group is 'atax', not 'other'"):
        link(forked, "A", 4, "other")
