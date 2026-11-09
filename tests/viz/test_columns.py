"""The columns of the DFG view (api.py's ``_layout``): the top-level nodes' independent chains
drawn side by side, a node reading from two of them below both."""

from __future__ import annotations

from pathlib import Path

from snax_forge.dfg import Graph
from snax_forge.viz.dfg.api import Entry, columns, graph_view
from tests.helpers import imported

MVT_HALF_1 = ["tmp0_mult_map", "sum_map", "augassign_18_4_map"]
MVT_HALF_2 = ["tmp2_mult_map", "sum_map_2", "augassign_19_4_map"]


def view_of(g: Graph) -> dict:
    return graph_view(Entry(g.name, Path(g.name), g, None))


def drawn(rows: list[dict]) -> list:
    """Rows as their boxes (a list), their node's id, or ``{column id: its rows}``."""
    out: list = []
    for r in rows:
        if r["kind"] == "containers":
            out.append(r["boxes"])
        elif r["kind"] == "node":
            out.append(r["node"]["id"])
        else:
            out.append({c["id"]: drawn(c["rows"]) for c in r["columns"]})
    return out


def pairs(v: dict) -> list[tuple[str, str]]:
    return [(e["from"], e["to"]) for e in v["edges"]]


def test_mvt_draws_its_two_halves_side_by_side():
    """mvt: x1 += A y1 and x2 += Aᵀ y2 share only a read of A, so each half is a column of
    its three nests in order, with A boxed in each."""
    v = view_of(imported("mvt"))
    assert v["columns"] == [{"id": 0, "tops": MVT_HALF_1, "joins": []},
                            {"id": 1, "tops": MVT_HALF_2, "joins": []}]  # fmt: skip
    assert drawn(v["rows"]) == [{
        0: [["A@0", "y1@0"], "tmp0_mult_map", ["tmp0_prod@1"], "sum_map", ["x1@0", "tmp0@1"],
            "augassign_18_4_map", ["x1@1"]],
        1: [["A@0~3", "y2@0"], "tmp2_mult_map", ["tmp2_prod@1"], "sum_map_2", ["x2@0", "tmp2@1"],
            "augassign_19_4_map", ["x2@1"]],
    }]  # fmt: skip
    a = {k: (b["column"], b["again"]) for k, b in v["boxes"].items() if b["container"] == "A"}
    assert a == {"A@0": (0, False), "A@0~3": (1, False)}  # each column's own input, not dimmed
    assert {("A@0", "tmp0_mult.in1"), ("A@0~3", "tmp2_mult.in1")} <= set(pairs(v))
    (block,) = v["rows"]
    assert block["column"] is None and all(
        r["column"] == c["id"] for c in block["columns"] for r in c["rows"]
    )


def test_one_chain_draws_as_before():
    """vecadd, dot and matvec: one column, the rows as they always were."""
    want = {
        "vecadd": [["A@0", "B@0"], "add_map", ["C@1"]],
        "dot": [["A@0", "B@0"], "mult_map", ["tmp0@1"], "sum_map", ["out@1"]],
        "matvec": [["A@0", "p@0"], "mult_map", ["tmp0@1"], "sum_map", ["q@1"]],
    }
    for kernel, rows in want.items():
        g = imported(kernel)
        v = view_of(g)
        assert v["columns"] == [{"id": 0, "tops": [t.id for t in g.body], "joins": []}], kernel
        assert drawn(v["rows"]) == rows, kernel
        assert {r["column"] for r in v["rows"]} == {0}, kernel


def test_gesummv_joins_its_two_columns_at_its_last_node():
    """gesummv: alpha A x and beta B x are two columns, each ending in what it wrote; add_map
    reads both, so it is a third column joining them, drawn below them."""
    v = view_of(imported("gesummv"))
    half_1, half_2 = (
        ["mult_map", "tmp1_mult_map", "sum_map"],
        ["mult_map_2", "tmp3_mult_map", "sum_map_2"],
    )
    assert v["columns"] == [{"id": 0, "tops": half_1, "joins": []},
                            {"id": 1, "tops": half_2, "joins": []},
                            {"id": 2, "tops": ["add_map"], "joins": [0, 1]}]  # fmt: skip
    block, *below = drawn(v["rows"])
    assert below == ["add_map", ["out@1"]]
    assert [x for x in block[0] if isinstance(x, str)] == half_1 and block[0][-1] == ["tmp1@1"]
    assert [x for x in block[1] if isinstance(x, str)] == half_2 and block[1][-1] == ["tmp3@1"]
    assert block[1][2] == ["tmp2@1", "x@0~4"]  # x, read by both, drawn in each
    assert {("tmp1@1", "add.in1"), ("tmp3@1", "add.in2")} <= set(pairs(v))


def _map(mid: str, ins: list[str], out: str) -> dict:
    tasklet = {"id": f"{mid}_t", "kind": "tasklet", "attrs": {"code": "out = in1"},
               "inputs": {f"in{i}": {"data": d, "subset": ["i"]} for i, d in enumerate(ins, 1)},
               "outputs": {"out": {"data": out, "subset": ["i"]}}}  # fmt: skip
    return {"id": mid, "kind": "map", "attrs": {"var": "i", "range": "0:N"}, "body": [tasklet]}


def test_a_chain_is_a_write_the_other_reads_or_writes():
    """Two reads of A do not link two nests; a write of what another read (or wrote) does,
    whichever comes first."""
    cont = {k: {"shape": ["N"], "dtype": "int64", "transient": False} for k in "ABCD"}

    def cols(*body: dict) -> list[list[str]]:
        g = Graph.from_dict(
            {"name": "g", "symbols": {"N": 8}, "containers": cont, "body": list(body)}
        )
        return [[g.body[k].id for k in c["tops"]] for c in columns(g.body)[1]]

    assert cols(_map("a", ["A"], "B"), _map("b", ["A"], "C")) == [["a"], ["b"]]
    assert cols(_map("a", ["A"], "B"), _map("b", ["C"], "A")) == [["a", "b"]]  # writes what a read
    assert cols(_map("a", ["A"], "B"), _map("b", ["C"], "B")) == [["a", "b"]]  # both write B
    assert cols(_map("a", ["A"], "B"), _map("b", ["A"], "C"), _map("c", ["D"], "D")) == [
        ["a"], ["b"], ["c"]]  # fmt: skip
