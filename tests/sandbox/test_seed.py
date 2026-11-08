"""The fold's seed: ``fuse_seed`` and ``unfuse_seed`` on matvec_acc (C§11, C§12, M13 B2 R5).

matvec_acc (``q += A p``, ``tests/dfg/fixtures/matvec_acc.snaxdfg``) imports as matvec's
multiply into ``tmp0``, the sum folding into the transient ``tmp1[m]``, then the elementwise
``q[m] = q[m] + tmp1[m]``. ``fuse_seed`` takes the last into the fold: the sum writes ``q[m]``
from the seed ``q[m]``, and ``tmp1`` and its nest go. Accepted when the worked example (W = 2:
A's rows (1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), p = (1, 2, 0, 1), q = (1, 1, 1)) gives
q' = (10, 26, 42) at every step and bound (row 0: 1 + 5 = 6 after the first beat, then
6 + 4 = 10), ``unfuse_seed`` gives the graph back, the seed follows the splits, tags and
interchanges in any order before ``bind``, and each refusal names what it found.
"""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.dfg import Graph, execute
from snax_forge.dfg.graph import SEEDED
from snax_forge.sandbox import (
    Recipe,
    SandboxError,
    apply_recipe,
    bind,
    fuse_seed,
    join_map,
    join_reduce,
    split,
    split_map,
    split_reduce,
    unbind,
    unfuse_seed,
)
from snax_forge.sandbox.seed import seed_link, seed_pairs, seed_seconds, seeded_choices
from snax_forge.sandbox.session import form
from snax_forge.sandbox.transforms import interchange, site_of, tag_map, untag_map
from tests.helpers import FIXTURES, graph_dict

A = np.arange(1, 13, dtype=np.int64).reshape(3, 4)
P = np.array([1, 2, 0, 1], np.int64)
Q = np.array([1, 1, 1], np.int64)
WANT = [10, 26, 42]
TREE = {"brm": "accumulate", "implementation": "chisel_adder_tree", "instance": "sum"}
FOLD = "sum_map_1"
EW = "augassign_19_4_map"  # the elementwise nest q = q + tmp1


def matvec_acc(m: int = 3, n: int = 4) -> Graph:
    d = graph_dict("matvec_acc")
    d["symbols"] = {"M": m, "N": n}
    return Graph.from_dict(d)


def run(g: Graph) -> list[int]:
    return execute(g, {"A": A, "p": P, "q": Q.copy()})["q"].tolist()


def tagged(g: Graph, w: int = 2) -> Graph:
    """The sum's row map temporal and its column map split by w (matvec_seed's steps)."""
    return split_map(tag_map(g, "sum_map", "temporal"), FOLD, w)


# =============================================================================
# fuse_seed, the worked example, unfuse_seed
# =============================================================================


def test_fuse_seed_takes_the_elementwise_nest_and_the_transient():
    g = matvec_acc()
    s = fuse_seed(g, FOLD, EW)
    assert "tmp1" not in s.containers and list(s.containers) == ["A", "p", "q", "tmp0"]
    assert [n.id for n in s.body] == ["mult_map", "sum_map"]
    out = s.node("sum").outputs["out"].to_dict()
    assert out == {"data": "q", "subset": ["m"],
                   "reduce": {"op": "add", "seed": {"data": "q", "subset": ["m"]}}}  # fmt: skip
    rec = s.node("sum").attrs[SEEDED]
    assert (rec["container"], rec["index"], rec["identity"], rec["after"]) == ("tmp1", 4, 0, None)
    assert rec["nest"]["id"] == "augassign_19_4_map"
    assert Graph.from_dict(s.to_dict()) == s  # the stored form round-trips


@pytest.mark.parametrize(
    "make",
    [
        matvec_acc,
        lambda: fuse_seed(matvec_acc(), FOLD, EW),
        lambda: tagged(fuse_seed(matvec_acc(), FOLD, EW)),
        lambda: fuse_seed(tagged(matvec_acc()), FOLD, EW),
        lambda: bind(fuse_seed(tagged(matvec_acc()), FOLD, EW), node="sum", **TREE),
        lambda: bind(fuse_seed(tagged(matvec_acc(), 1), FOLD, EW), node="sum", **TREE
                     | {"implementation": "chisel_adder_tree"}),
        lambda: bind(fuse_seed(tagged(matvec_acc()), FOLD, EW), node="sum", **TREE
                     | {"implementation": "chisel_serial"}),
        lambda: split_reduce(fuse_seed(matvec_acc(), FOLD, EW), FOLD, 2),
    ],
    ids=["imported", "seeded", "seeded-split", "split-seeded", "bound-tree", "bound-w1",
         "bound-serial", "split_reduce"],
)  # fmt: skip
def test_the_worked_example_gives_10_26_42(make):
    assert run(make()) == WANT


def test_the_bound_unit_takes_the_seed_on_its_port_once_per_job():
    b = bind(fuse_seed(tagged(matvec_acc()), FOLD, EW), node="sum", **TREE)
    n = b.node("sum")
    assert n.attrs["params"] == {"W": 2, "op": "add", "seed": 1}
    assert {c: m.to_dict() for c, m in n.inputs.items()} == {
        "a": {"data": "tmp0", "subset": ["m_t", "2 * n_1_t:2 * n_1_t + 2"]},
        "seed": {"data": "q", "subset": ["m_t:m_t + 1"]},
    }
    assert n.outputs["out"].to_dict() == {"data": "q", "subset": ["m_t:m_t + 1"]}
    plain = bind(tagged(matvec_acc()), node="sum", **TREE).node("sum")
    assert plain.attrs["params"] == {"W": 2, "op": "add"} and "seed" not in plain.inputs
    assert run(unbind(b, "sum")) == WANT


def test_seed_then_unseed_gives_the_graph_back():
    g = matvec_acc()
    assert unfuse_seed(fuse_seed(g, FOLD, EW), FOLD) == g
    t = tagged(g)
    assert unfuse_seed(fuse_seed(t, FOLD, EW), FOLD) == t
    assert unfuse_seed(tagged(fuse_seed(g, FOLD, EW)), FOLD) == t  # the splits since kept
    sr = split_reduce(g, FOLD, 2)
    assert unfuse_seed(fuse_seed(sr, FOLD, EW), FOLD) == sr


def test_split_reduce_seeds_the_final_sum_only():
    """The partials keep the identity, the final sum takes the seed and its record; seeding
    before or after split_reduce gives one graph, and join_reduce gives the seeded fold back."""
    s = fuse_seed(matvec_acc(), FOLD, EW)
    sr = split_reduce(s, FOLD, 2)
    assert sr.node("sum").outputs["out"].to_dict() == {
        "data": "tmp1_partial", "subset": ["m", "n_2_s"],
        "reduce": {"op": "add", "identity": 0}}  # fmt: skip
    assert sr.node("sum_final").outputs["out"].to_dict()["reduce"]["seed"] == {
        "data": "q", "subset": ["m"]}  # fmt: skip
    assert SEEDED not in sr.node("sum").attrs and SEEDED in sr.node("sum_final").attrs
    assert sr == fuse_seed(split_reduce(matvec_acc(), FOLD, 2), FOLD, EW)
    assert join_reduce(sr, FOLD) == s


def test_the_seed_follows_split_tag_interchange_and_join():
    g = matvec_acc()
    s = fuse_seed(g, FOLD, EW)
    assert join_map(split(s, FOLD, 2), FOLD) == s
    assert untag_map(tag_map(s, "sum_map", "temporal"), "sum_map") == s
    x = interchange(s, "sum_map", FOLD)
    assert x == fuse_seed(interchange(g, "sum_map", FOLD), FOLD, EW) and run(x) == WANT
    assert interchange(x, FOLD, "sum_map") == s
    sp = split(s, "sum_map", 3)  # the row map split: the seed's index moves with it
    assert sp.node("sum").outputs["out"].reduce.seed.subset == ("3 * m_1 + m_2",)
    assert run(sp) == WANT


STEPS = {
    "tag": {"transform": "tag_map", "params": {"map": "sum_map", "kind": "temporal"}},
    "split": {"transform": "split_map", "params": {"map": FOLD, "factor": "W"}},
    "seed": {"transform": "fuse_seed", "params": {"first": FOLD, "second": EW}},
}
BIND = {"transform": "bind", "params": {"node": "sum", **TREE}}


@pytest.mark.parametrize("order", [["seed", "tag", "split"], ["tag", "seed", "split"],
                                   ["tag", "split", "seed"]])  # fmt: skip
def test_fuse_seed_goes_anywhere_before_bind(order):
    """fuse_seed before, between or after the tag and the split: the same bound unit, every
    step checked against the reference on the seeded inputs (run.py)."""
    r = Recipe.from_dict({"name": "t", "kernel": "matvec_acc", "params": {"W": 4},
                          "symbols": {"M": 8, "N": 64},
                          "steps": [*(STEPS[k] for k in order), BIND]})  # fmt: skip
    got = apply_recipe(r, Graph.load(FIXTURES / "matvec_acc.snaxdfg"))[-1].graph
    want = Recipe.from_dict({**r.to_dict(), "steps": [STEPS["tag"], STEPS["split"],
                                                      STEPS["seed"], BIND]})  # fmt: skip
    assert got == apply_recipe(want, Graph.load(FIXTURES / "matvec_acc.snaxdfg"))[-1].graph


# =============================================================================
# The refusals
# =============================================================================


def _second_reader() -> Graph:
    d = graph_dict("matvec_acc")
    d["symbols"] = {"M": 3, "N": 4}
    nest = graph_dict("matvec_acc")["body"][2]
    nest["id"], nest["body"][0]["id"] = "again_map", "again"
    d["body"].append(nest)
    return Graph.from_dict(d)


def _edited(fn) -> Graph:
    d = graph_dict("matvec_acc")
    d["symbols"] = {"M": 3, "N": 4}
    fn(d)
    return Graph.from_dict(d)


def _reader(d):
    return d["body"][2]["body"][0]


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (_second_reader, r"tmp1 has another reader: \['augassign_19_4', 'again'\]"),
        (lambda: _edited(lambda d: _reader(d)["attrs"].update(code="out = in1 * in2")),
         r"computes 'out = in1 \* in2', a different op from the fold's add"),
        (lambda: _edited(lambda d: _reader(d)["inputs"]["in2"].update(subset=[0])),
         r"reads tmp1\[0\] and q\[m\], writes q\[m\]: a different subset"),
        (lambda: _edited(lambda d: _reader(d)["inputs"]["in1"].update(subset=["M - 1 - m"])),
         r"a different subset"),
        (lambda: _edited(lambda d: d["containers"]["tmp1"].update(transient=False)),
         r"folds into tmp1, which is not a transient"),
        (lambda: _edited(lambda d: d["containers"]["tmp1"].update(dtype="float64")),
         r"tmp1 is float64, a float op type"),
        (lambda: fuse_seed(matvec_acc(), FOLD, EW), r"sum is seeded from q already \(unfuse_seed first\)"),
        (lambda: bind(tagged(matvec_acc()), node="sum", **TREE),
         r"its fold is bound as 'sum' \(unbind first\)"),
        (matvec_acc, None),
    ],
    ids=["another-reader", "another-op", "another-subset", "x-elsewhere", "not-transient",
         "float", "seeded", "bound", "ok"],
)  # fmt: skip
def test_fuse_seed_refuses(make, message):
    g = make()
    if message is None:
        assert site_of("fuse_seed", g, FOLD) == {"first": FOLD, "second": EW}
        return
    with pytest.raises(SandboxError, match=message):
        fuse_seed(g, FOLD, EW)
    with pytest.raises(SandboxError, match=message):  # its site says the same
        site_of("fuse_seed", g, FOLD)


def test_unfuse_seed_refuses_a_fold_not_seeded():
    with pytest.raises(SandboxError, match=r"sum's fold is not seeded by fuse_seed"):
        unfuse_seed(matvec_acc(), FOLD)


def test_bind_refuses_a_seed_on_a_brm_without_a_seed_port():
    s = fuse_seed(tagged(matvec_acc()), FOLD, EW)
    with pytest.raises(SandboxError, match=r"seeded from q; 'simd' has no seed port"):
        bind(s, node="sum", brm="simd", implementation="chisel_tiled_spatial", instance="x")
    with pytest.raises(SandboxError, match=r"seed = 0 given, but the fold is seeded"):
        bind(s, node="sum", **TREE, params={"seed": 0})


# =============================================================================
# The editor
# =============================================================================


def test_the_site_takes_one_map_or_two_as_fuse_map_does():
    """One map of the fold's nest fills ``second`` with the nest's outermost map; two maps are
    first and second in the order selected; any map of either chain goes, others are refused."""
    g = tagged(matvec_acc())
    for first in ("sum_map", FOLD, "sum_map_1_s"):
        assert site_of("fuse_seed", g, first) == {"first": first, "second": EW}
        assert site_of("fuse_seed", g, [first, EW]) == {"first": first, "second": EW}
    assert seed_seconds(g, "sum_map") == [EW]
    assert seed_pairs(g) == [("sum_map", EW), (FOLD, EW), ("sum_map_1_s", EW)]
    assert fuse_seed(g, "sum_map", EW) == fuse_seed(g, FOLD, EW)
    with pytest.raises(SandboxError, match=r"fuse_seed 'sum_map_1' 'mult_map': 'mult_map' is not "
                       r"a map of the elementwise nest that reads the fold's result "
                       r"\(augassign_19_4_map\)"):  # fmt: skip
        fuse_seed(g, FOLD, "mult_map")
    with pytest.raises(SandboxError, match=r"is not a map of the elementwise nest"):
        site_of("fuse_seed", g, [FOLD, "mult_map"])
    with pytest.raises(SandboxError, match=r"fuse_seed: takes one map or two, not 3"):
        site_of("fuse_seed", g, [FOLD, EW, "mult_map"])
    with pytest.raises(SandboxError, match=r"fuse_seed 'augassign_19_4_map': its tasklet augassign_19_4 "
                       r"must fold its one output"):  # fmt: skip
        site_of("fuse_seed", g, EW)


def test_the_link_names_the_loops_matched():
    """The fold's kept loop against the nest's: m = m at import, m_t = m after the tag; a row
    split of the nest leaves a loop of its own."""
    g = matvec_acc(8, 512)
    assert seed_link(g, FOLD, EW) == {"container": "tmp1", "subset": ["m"], "writer": "sum",
                                      "loops": [["m", "m", "0:8"]]}  # fmt: skip
    t = tagged(g, 4)
    assert seed_link(t, FOLD, EW)["loops"] == [["m_t", "m", "0:8"]]
    sp = split(t, EW, 2)
    assert seed_link(sp, FOLD, EW)["loops"] == [["m_t", "m_1", "0:8 = 0:4"], ["", "m_2", "0:2"]]
    assert run(fuse_seed(split(tagged(matvec_acc()), EW, 3), FOLD, EW)) == WANT


# =============================================================================
# The editor
# =============================================================================


def test_the_editor_offers_the_elementwise_nests_maps():
    g = tagged(matvec_acc(8, 512), 4)
    first, second = form("fuse_seed", g, {"params": site_of("fuse_seed", g, "sum_map")}, {})
    assert (first["name"], first["value"], first["readonly"]) == ("first", "sum_map", True)
    assert (second["name"], second["type"], second["value"], second["readonly"]) == (
        "second", "choice", EW, False)  # fmt: skip
    assert second["choices"] == [{"value": EW, "ok": True, "label": "augassign_19_4_map · "
                                  "tmp1[m] · m_t = m (0:8)"}]  # fmt: skip
    assert "seed" in second["hint"]
    s = fuse_seed(g, FOLD, EW)
    assert seed_pairs(s) == [] and [c["map"] for c in seeded_choices(s)] == [FOLD]
    (u,) = form("unfuse_seed", s, {"params": site_of("unfuse_seed", s, "sum_map_1_s")}, {})
    assert (u["value"], u["choices"][0]["value"]) == (FOLD, FOLD)
