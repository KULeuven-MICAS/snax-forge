"""split_reduce, fuse_map and link (D148, D150): dot's variants spelled step by step, each step
checked by the reference executor, and every refusal with its fix."""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.dfg import Graph, execute
from snax_forge.sandbox import (
    SandboxError,
    bind,
    fuse_map,
    join_reduce,
    link,
    split_map,
    split_reduce,
    unfuse_map,
    unlink,
)
from snax_forge.sandbox.compose import LinkError, beats

from .helpers import graph_dict

MUL = ("elementwise_mul", "chisel_tiled_spatial", "mul")
TREE = ("accumulate", "chisel_adder_tree")


def dot(n: int = 8) -> Graph:
    d = graph_dict("dot")
    d["symbols"]["N"] = n
    return Graph.from_dict(d)


def inputs(n: int = 8) -> dict:
    rng = np.random.default_rng(1)
    return {"A": rng.integers(-99, 99, n), "B": rng.integers(-99, 99, n),
            "out": np.zeros(1, np.int64)}  # fmt: skip


def same(g: Graph, n: int = 8) -> None:
    """``g`` gives dot's sum on the reference executor."""
    x = inputs(n)
    assert execute(g, x)["out"][0] == int(np.dot(x["A"], x["B"]))


def fused(n: int = 8) -> Graph:
    """dot after split_map, split_reduce and fuse_map at W = 4 (dot_4's first steps)."""
    g = split_reduce(split_map(dot(n), "mult_map", 4), "sum_map", 4)
    return fuse_map(g, "mult_map", "sum_map")


def dot2(depth: int = 1) -> Graph:
    g = bind(split_map(dot(), "mult_map", 4), "mult", *MUL)
    g = bind(split_map(g, "sum_map", 4), "sum", *TREE, "sum")
    return link(g, "tmp0", depth, "dot")


def dot4(depth: int = 0) -> Graph:
    g = bind(fused(), "mult_map_s", "mac", "chisel_mac", "mac")
    g = bind(g, "sum_final", *TREE, "sum")
    return link(g, "out_partial", depth, "dot")


def test_split_reduce_gives_lane_partials_and_a_final_sum():
    g = split_reduce(split_map(dot(), "mult_map", 4), "sum_map", 4)
    assert [n.id for n in g.body] == ["mult_map", "sum_map", "sum_map_final"]
    assert g.containers["out_partial"].shape == [4] and g.containers["out_partial"].transient
    assert g.node("sum").outputs["out"].subset == ["i_s_1"]  # split_map's fresh variable
    assert g.node("sum_final").inputs["in1"].subset == ["i_f"]
    assert g.node("sum_final").outputs["out"].subset == [0]
    assert g.node("sum_map").attrs["loop.split_reduce"] == {"partial": "out_partial",
                                                            "final": "sum_map_final"}  # fmt: skip
    same(g)


def test_fuse_map_keeps_the_first_ids_and_makes_the_transient_local():
    g = fused()
    assert [n.id for n in g.body] == ["mult_map", "sum_map_final"]
    assert [n.id for n in g.node("mult_map_s").body] == ["mult", "sum"]
    assert g.node("sum").inputs["in1"].subset == g.node("mult").outputs["out"].subset
    assert g.containers["tmp0"].attrs == {"loop.local": "mult_map"}
    same(g)


def test_bind_takes_the_fused_nests_spatial_map():
    g = bind(fused(), "mult_map_s", "mac", "chisel_mac", "mac")
    mac = g.node("mult")
    assert (mac.kind, mac.attrs["code"]) == ("accelerated", "out = a * b")
    assert mac.outputs["out"].subset == ["0:4"]
    assert mac.attrs["replaced"]["id"] == "mult_map_s"
    same(g)


@pytest.mark.parametrize("depth", [0, 1])
def test_link_on_dot2_and_dot4(depth):
    for g, container, units in ((dot2(depth), "tmp0", ("mul", "sum")),
                                (dot4(depth), "out_partial", ("mac", "sum"))):  # fmt: skip
        (k,) = [n for n, _ in g.walk() if n.kind == "link"]
        assert k.attrs == {"container": container, "depth": depth, "from": units[0], "to": units[1]}
        assert {n.attrs["group"] for n, _ in g.walk() if n.kind == "accelerated"} == {"dot"}
        assert [n.id for n in g.body][-2] == k.id  # before the reader, in the body they share
        same(g)


def test_a_links_ends_give_the_same_beats():
    g = dot4()
    assert beats(g, g.node("mult"), "out") == beats(g, g.node("sum_final"), "a")
    assert len(beats(g, g.node("mult"), "out")) == 1  # one beat per T firings
    g = dot2()
    assert len(beats(g, g.node("mult"), "out")) == 2  # N / W


# =============================================================================
# Refusals, each with the fix
# =============================================================================


def test_bind_names_the_fix():
    """D150: the multiply of a fusable pair is 'fuse_map first', a fold into one element
    'split_reduce first'."""
    g = split_reduce(split_map(dot(), "mult_map", 4), "sum_map", 4)
    with pytest.raises(SandboxError, match="fuse_map first"):
        bind(g, "mult", "mac", "chisel_mac", "mac")
    one = fuse_map(split_map(split_map(dot(), "mult_map", 4), "sum_map", 4), "mult_map", "sum_map")
    with pytest.raises(SandboxError, match="split_reduce first"):
        bind(one, "mult_map_s", "mac", "chisel_mac", "mac")


@pytest.mark.parametrize(
    ("step", "text"),
    [
        (lambda: split_reduce(dot(), "mult_map", 4), "must fold its one output"),
        (lambda: split_reduce(dot(6), "sum_map", 4), "not a multiple"),
        (lambda: join_reduce(dot(), "sum_map"), "no loop.split_reduce"),
        (lambda: join_reduce(bind(fused(), "mult_map_s", "mac", "chisel_mac", "mac"), "sum_map"),
         "no node 'sum_map'"),
        (lambda: join_reduce(bind(split_reduce(dot(), "sum_map", 4), "sum_final", *TREE, "s"),
                             "sum_map"), "bound"),
        (lambda: fuse_map(split_map(dot(), "mult_map", 4), "mult_map", "sum_map"),
         "same depth"),
        (lambda: fuse_map(split_map(split_map(dot(), "mult_map", 4), "sum_map", 2), "mult_map",
                          "sum_map"), "same ranges"),
        (lambda: fuse_map(fused(), "sum_map_final", "mult_map"), "right after"),
        (lambda: unfuse_map(dot(), "mult_map"), "no loop.fused"),
        (lambda: unfuse_map(bind(fused(), "mult_map_s", "mac", "chisel_mac", "mac"), "mult_map"),
         "changed since"),
    ],
    ids=["not-a-fold", "tail", "join-unsplit", "join-fused", "join-bound", "depth", "ranges",
         "adjacent", "unfuse-unfused", "unfuse-bound"],
)  # fmt: skip
def test_split_reduce_and_fuse_map_refuse(step, text):
    with pytest.raises(SandboxError, match=text):
        step()


def _unbound() -> Graph:
    """dot 2 with the sum still a tasklet."""
    return split_map(bind(split_map(dot(), "mult_map", 4), "mult", *MUL), "sum_map", 4)


@pytest.mark.parametrize(
    ("step", "code"),
    [
        (lambda: link(dot2(), "A", 0, "dot"), "link.container"),  # not a transient
        (lambda: link(dot2(), "tmp0", 0, "dot"), "link.container"),  # already a link
        (lambda: link(dot4(), "tmp0", 0, "dot"), "link.container"),  # a local value
        (lambda: link(_unbound(), "tmp0", 0, "dot"), "link.unbound"),
        (lambda: link(bind(_unbound(), "sum", "accumulate", "chisel_adder_tree", "sum"), "tmp0",
                      0, "x y"), "link.group"),
    ],
    ids=["transient", "linked", "local", "unbound", "name"],
)  # fmt: skip
def test_link_refuses(step, code):
    with pytest.raises(LinkError) as e:
        step()
    assert e.value.code == code


def test_link_refuses_another_order():
    """The reader takes tmp0 backwards: the same elements, another order."""
    g = split_map(bind(split_map(dot(), "mult_map", 4), "mult", *MUL), "sum_map", 4)
    d = g.to_dict()
    temporal = d["body"][1]
    v_t, v_s = temporal["attrs"]["var"], temporal["body"][0]["attrs"]["var"]
    total = temporal["body"][0]["body"][0]
    total["inputs"]["in1"]["subset"] = [f"4 * (1 - {v_t}) + {v_s}"]
    g = bind(Graph.from_dict(d), "sum", *TREE, "sum")
    with pytest.raises(LinkError) as e:
        link(g, "tmp0", 0, "dot")
    assert e.value.code == "link.order"


def test_unlink_takes_the_group_off_a_unit_left_alone():
    g = unlink(dot2(), "tmp0")
    assert {n.attrs.get("group") for n, _ in g.walk() if n.kind == "accelerated"} == {None}
    assert not [n for n, _ in g.walk() if n.kind == "link"]
