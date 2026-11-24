"""split_reduce, fuse_map and link: dot's variants spelled step by step, each step
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
)
from snax_forge.sandbox.compose import LinkError
from tests.helpers import graph_dict

MUL = ("simd", "chisel_tiled_spatial", "mul")
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
    """dot after split_map, split_reduce and fuse_map at W = 4 (dot_v4's first steps)."""
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


# =============================================================================
# Refusals, each with the fix
# =============================================================================


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
         r"mult_map runs 0:2 and its child 0:4, sum_map runs 0:8: split_map sum_map 4 first"),
        (lambda: fuse_map(split_map(split_map(dot(), "mult_map", 4), "sum_map", 2), "mult_map",
                          "sum_map"), "same ranges"),
        (lambda: fuse_map(fused(), "sum_map_final", "mult_map"), "right after"),
        (lambda: unfuse_map(dot(), "mult_map"), "no loop.fused"),
        (lambda: unfuse_map(bind(fused(), "mult_map_s", "mac", "chisel_mac", "mac"), "mult_map"),
         "changed since"),
    ],
    ids=["not-a-fold", "tail", "join-unsplit", "join-fused", "join-bound", "split-one-side", "ranges",
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


def _join() -> Graph:
    """Two multiplies into one add, each bound: out = A * B + C * D at W = 4, N = 8."""

    def acc(node, brm, code, ins, out, op):
        return {"id": node, "kind": "accelerated",
                "inputs": {p: {"data": c, "subset": ["4 * i_t:4 * i_t + 4"]} for p, c in ins.items()},
                "outputs": {"out": {"data": out, "subset": ["4 * i_t:4 * i_t + 4"]}},
                "attrs": {"instance": node, "brm": brm, "implementation": "chisel_tiled_spatial",
                          "code": code, "params": {"W": 4, "op": op}, "replaced": None},
                "body": []}  # fmt: skip

    def loop(name, node):
        return {"id": name, "kind": "map", "inputs": {}, "outputs": {},
                "attrs": {"var": "i_t", "range": "0:N // 4", "loop.kind": "temporal"},
                "body": [node]}  # fmt: skip

    cont = {"shape": ["N"], "dtype": "int64", "transient": False, "attrs": {}}
    d = {"name": "join", "symbols": {"N": 8},
         "containers": {**{c: dict(cont) for c in ("A", "B", "C", "D", "out")},
                        **{c: dict(cont, transient=True) for c in ("t1", "t2")}},
         "body": [loop("m1", acc("mul1", "simd", "out = a * b", {"a": "A", "b": "B"},
                                 "t1", "mul")),
                  loop("m2", acc("mul2", "simd", "out = a * b", {"a": "C", "b": "D"},
                                 "t2", "mul")),
                  loop("m3", acc("add", "simd", "out = a + b", {"a": "t1", "b": "t2"},
                                 "out", "add"))]}  # fmt: skip
    return Graph.from_dict(d)


def test_link_refuses_a_join():
    """link.chain: mul2 into the add's other input makes {mul1, mul2, add} a join, not a chain
    (SNAX-GVSoC's limit, gen.chain's check made early; chains meet only at a fork)."""
    g = link(_join(), "t1", 0, "j")
    with pytest.raises(LinkError) as e:
        link(g, "t2", 0, "j")
    assert e.value.code == "link.chain" and "one chain, or chains joined by a fork" in str(e.value)


# =============================================================================
# fuse_map with an untagged side, and its range refusal's fix
# =============================================================================


def matvec_acc() -> Graph:
    d = graph_dict("matvec_acc")
    d["symbols"] = {"M": 8, "N": 512}
    return Graph.from_dict(d)


def test_an_untagged_map_takes_the_other_sides_tag():
    """matvec_acc: tag_map sum_map temporal then fuse_map mult_map sum_map is the graph
    fuse_map then tag_map mult_map temporal gives, the fused map temporal (m_t); its record
    the two nests tagged, so unfuse_map gives both back temporal and untag_map the plain fuse;
    the result still computes q."""
    from snax_forge.sandbox.transforms import tag_map, untag_map

    g = matvec_acc()
    tagged_first = fuse_map(tag_map(g, "sum_map", "temporal"), "mult_map", "sum_map")
    fused_first = tag_map(fuse_map(g, "mult_map", "sum_map"), "mult_map", "temporal")
    assert tagged_first == fused_first
    m = tagged_first.node("mult_map")
    assert (m.attrs["loop.kind"], m.attrs["var"]) == ("temporal", "m_t")
    both = tag_map(tag_map(g, "mult_map", "temporal"), "sum_map", "temporal")
    assert unfuse_map(tagged_first, "mult_map") == both
    assert untag_map(fused_first, "mult_map") == fuse_map(g, "mult_map", "sum_map")
    x = {"A": np.arange(8 * 512).reshape(8, 512), "p": np.arange(512), "q": np.arange(8)}
    want = execute(g, {k: v.copy() for k, v in x.items()})["q"]
    assert np.array_equal(execute(tagged_first, {k: v.copy() for k, v in x.items()})["q"], want)
