"""The transpose kind: a whole container copied, permuted, into another (CONTRACTS, "SNAX-DFG");
its check, the reference executor and the round trip."""

from __future__ import annotations

import json

import numpy as np
import pytest

from snax_forge.dfg import DfgError, Graph
from snax_forge.dfg.execute import execute


def copy_graph(**edit) -> dict:
    """A [M, N] int64 into its transient A_T [N, M] by one transpose node, then B = A_T + 1."""
    d = {
        "name": "t",
        "symbols": {"M": 3, "N": 5},
        "containers": {
            "A": {"shape": ["M", "N"], "dtype": "int64"},
            "A_T": {"shape": ["N", "M"], "dtype": "int64", "transient": True},
            "B": {"shape": ["N", "M"], "dtype": "int64"},
        },
        "body": [
            {"id": "transpose_A", "kind": "transpose",
             "inputs": {"src": {"data": "A", "subset": ["0:M", "0:N"]}},
             "outputs": {"dst": {"data": "A_T", "subset": ["0:N", "0:M"]}},
             "attrs": {"perm": [1, 0], "by": "dma"}},
            {"id": "add_map", "kind": "map", "attrs": {"var": "n", "range": "0:N"}, "body": [
                {"id": "add_map_1", "kind": "map", "attrs": {"var": "m", "range": "0:M"},
                 "body": [{"id": "add", "kind": "tasklet", "attrs": {"code": "out = in1 + 1"},
                           "inputs": {"in1": {"data": "A_T", "subset": ["n", "m"]}},
                           "outputs": {"out": {"data": "B", "subset": ["n", "m"]}}}]}]},
        ],
    }  # fmt: skip
    for path, value in edit.items():
        *keys, last = path.split("__")
        x = d
        for k in keys:
            x = x[int(k)] if isinstance(x, list) else x[k]
        x[last] = value
    return d


def test_the_kind_round_trips(tmp_path):
    g = Graph.from_dict(copy_graph())
    node = g.node("transpose_A")
    assert node.attrs == {"perm": [1, 0], "by": "dma"}
    assert Graph.from_dict(json.loads(g.to_json())) == g
    g.save(tmp_path / "t.snaxdfg")
    assert Graph.load(tmp_path / "t.snaxdfg").to_json() == g.to_json()


def test_the_executor_copies_the_input_permuted():
    a = np.arange(15, dtype=np.int64).reshape(3, 5)
    out = execute(Graph.from_dict(copy_graph()), {"A": a, "B": np.zeros((5, 3), np.int64)})
    assert np.array_equal(out["A_T"], a.T) and np.array_equal(out["B"], a.T + 1)


@pytest.mark.parametrize(("edit", "message"), [
    ({"body__0__attrs__perm": [0, 0]}, r"perm: \[0, 0\] is not a permutation of 'A''s 2"),
    ({"body__0__attrs__perm": [1, 0, 2]}, "is not a permutation"),
    ({"body__0__attrs__perm": [0, 1]}, (r"'A_T' is \['N', 'M'\], 'A' permuted by \[0, 1\] is "
                                        r"\['M', 'N'\]")),
    ({"containers__A_T__dtype": "int8"}, "'A_T' is int8, 'A' int64"),
    ({"body__0__attrs__by": "cpu"}, r"by: 'cpu' is not one of \['dma', 'core'\]"),
    ({"body__0__inputs__src__subset": ["0:M", "1:N"]}, "moves the whole of 'A'"),
    ({"body__0__outputs": {}}, "one input and one output, got 1 and 0"),
])  # fmt: skip
def test_the_check_refuses(edit, message):
    with pytest.raises(DfgError, match=message):
        Graph.from_dict(copy_graph(**edit))
