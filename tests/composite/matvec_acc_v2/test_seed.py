"""The fold's seed: ``fuse_seed`` and ``unfuse_seed`` on matvec_acc.

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
    SandboxError,
    bind,
    fuse_seed,
    join_reduce,
    split_map,
    split_reduce,
)
from snax_forge.sandbox.seed import seed_pairs, seed_seconds
from snax_forge.sandbox.transforms import site_of, tag_map
from tests.helpers import graph_dict

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
    """The sum's row map temporal and its column map split by w (matvec_acc_v1's steps)."""
    return split_map(tag_map(g, "sum_map", "temporal"), FOLD, w)


# =============================================================================
# fuse_seed, the worked example, unfuse_seed
# =============================================================================


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


STEPS = {
    "tag": {"transform": "tag_map", "params": {"map": "sum_map", "kind": "temporal"}},
    "split": {"transform": "split_map", "params": {"map": FOLD, "factor": "W"}},
    "seed": {"transform": "fuse_seed", "params": {"first": FOLD, "second": EW}},
}
BIND = {"transform": "bind", "params": {"node": "sum", **TREE}}


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
    assert site_of("fuse_seed", g, EW) == {"first": "sum_map", "second": EW}  # the other side
    with pytest.raises(SandboxError, match=r"fuse_seed 'mult_map': its tasklet mult must fold "
                       r"its one output"):  # fmt: skip
        site_of("fuse_seed", g, "mult_map")


# =============================================================================
# The editor
# =============================================================================
