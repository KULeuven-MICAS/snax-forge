"""split_reduce, fuse_map and link (D148, D150): dot's variants spelled step by step, each step
checked by the reference executor, and every refusal with its fix."""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.dfg import DfgError, Graph, execute
from snax_forge.sandbox import (
    SandboxError,
    bind,
    fuse_map,
    join_reduce,
    link,
    split_map,
    split_reduce,
    unbind,
    unfuse_map,
    unlink,
)
from snax_forge.sandbox.compose import LinkError, beats
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
    assert g.node("sum").outputs["out"].subset == ["n_2_s"]  # its dimension's second piece
    assert g.node("sum_final").inputs["in1"].subset == ["n_f_s"]  # a dimension of its own
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


def unreferenced(g: Graph) -> set[str]:
    """The containers no live node's memlet names."""
    used = {m.data for n, _ in g.walk() for m in (*n.inputs.values(), *n.outputs.values())}
    return set(g.containers) - used


def test_bind_takes_the_fused_nests_local_value_out_of_the_graph():
    before = fused()
    g = bind(before, "mult_map_s", "mac", "chisel_mac", "mac")
    assert "tmp0" not in g.containers and unreferenced(g) == set()
    (rec,) = g.node("mult").attrs["locals"]
    assert rec == {"name": "tmp0", "index": list(before.containers).index("tmp0"),
                   "container": before.containers["tmp0"].to_dict()}  # fmt: skip
    assert Graph.from_dict(g.to_dict()).to_json() == g.to_json()  # round-trips with its locals


def test_a_local_value_may_not_also_be_a_live_container():
    d = bind(fused(), "mult_map_s", "mac", "chisel_mac", "mac").to_dict()
    d["containers"]["tmp0"] = d["body"][0]["body"][0]["attrs"]["locals"][0]["container"]
    with pytest.raises(DfgError, match="'tmp0' is a container of the graph too"):
        Graph.from_dict(d)


@pytest.mark.parametrize("name", ["dot_4", "dot_5", "dot_6", "matvec_4", "matvec_5", "matvec_6"])
def test_unbind_then_unfuse_map_gives_the_graph_before_fuse_map(name):
    """On each recipe that fuses: after the fused nest's bind no container is unreferenced, and
    unbind then unfuse_map give the graphs before the bind and before fuse_map exactly."""
    from tests.sandbox.helpers import recipe_graphs

    r, graphs = recipe_graphs(name)
    steps = [s.transform for s in r.steps]
    k = steps.index("fuse_map") + 1  # graphs[0] is the input, graphs[k] after fuse_map
    b = k + steps[k:].index("bind") + 1
    acc = graphs[b].node(next(n.id for n, _ in graphs[b].walk() if "locals" in n.attrs))
    assert [x["name"] for x in acc.attrs["locals"]] == ["tmp0"]
    assert unreferenced(graphs[b]) == set()
    back = unbind(graphs[b], acc.id)
    assert back.to_json() == graphs[k].to_json()
    first = r.steps[k - 1].params["first"]
    assert unfuse_map(back, first).to_json() == graphs[k - 1].to_json()


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


def test_unlink_takes_the_group_off_a_unit_left_alone():
    g = unlink(dot2(), "tmp0")
    assert {n.attrs.get("group") for n, _ in g.walk() if n.kind == "accelerated"} == {None}
    assert not [n for n, _ in g.walk() if n.kind == "link"]


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


def test_link_refuses_a_join_until_tier_a():
    """link.chain: mul2 into the add's other input makes {mul1, mul2, add} a join, not a chain
    (SNAX-GVSoC's limit, gen.chain's check made early)."""
    g = link(_join(), "t1", 0, "j")
    with pytest.raises(LinkError) as e:
        link(g, "t2", 0, "j")
    assert e.value.code == "link.chain" and "until tier A" in str(e.value)


# =============================================================================
# fuse_map with an untagged side, and its range refusal's fix (B2 R5 B0)
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


def test_temporal_against_spatial_is_refused():
    from snax_forge.sandbox.transforms import tag_map

    g = tag_map(tag_map(matvec_acc(), "mult_map", "temporal"), "sum_map", "spatial")
    with pytest.raises(SandboxError, match=r"fuse_map 'mult_map' 'sum_map': mult_map is "
                       r"temporal, sum_map spatial; fused nests have the same tags"):  # fmt: skip
        fuse_map(g, "mult_map", "sum_map")


def test_different_ranges_name_the_split_that_fixes_them():
    """After split_map sum_map_1 4 the refusal names split_map mult_map_1 4, a FuseRangeError
    with that fix; applied, the two fuse (fuse_near gives the pair before)."""
    from snax_forge.sandbox.compose import FuseRangeError, fuse_near, fuse_pairs
    from snax_forge.sandbox.transforms import tag_map

    t = split_map(tag_map(matvec_acc(), "sum_map", "temporal"), "sum_map_1", 4)
    with pytest.raises(FuseRangeError) as e:
        fuse_map(t, "mult_map", "sum_map")
    assert str(e.value) == ("fuse_map 'mult_map' 'sum_map': mult_map_1 runs 0:512, sum_map_1 "
                            "runs 0:128 and its child 0:4: split_map mult_map_1 4 first")  # fmt: skip
    assert e.value.fix == {"map": "mult_map_1", "factor": 4}
    assert fuse_pairs(t) == [] and fuse_near(t) == [("mult_map", "sum_map", str(e.value))]
    fixed = split_map(t, "mult_map_1", 4)
    assert fuse_pairs(fixed) == [("mult_map", "sum_map")] and fuse_near(fixed) == []
    fuse_map(fixed, "mult_map", "sum_map")
