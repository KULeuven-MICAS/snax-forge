"""The reference executor on a split with a tail (open item 31, D144): the valid lanes only,
a pad lane read as the container's pad value; the pad helpers (dfg/pad.py)."""

from __future__ import annotations

import copy
import sys

import numpy as np
import pytest

from snax_forge.dfg import DfgError, ExecutionError, Graph, execute
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.dfg.pad import pad_value, padded_shape
from snax_forge.sandbox import Recipe, apply_recipe
from tests.helpers import FIXTURES, REPO

execute_mod = sys.modules["snax_forge.dfg.execute"]  # the package exports the function by name


def steps(name: str, **params) -> list[Graph]:
    recipe = Recipe.load(REPO / "recipes" / f"{name}.json").with_params(params)
    return [s.graph for s in apply_recipe(recipe, import_kernel(recipe.kernel))]


def lane_sums(monkeypatch, g: Graph, inputs: dict) -> tuple[list[int], dict]:
    """The sum of each input beat the accelerator's function gets, firing by firing (pad lanes
    included), and the run's outputs."""
    seen = []
    real = execute_mod._lanes

    def record(x, port, what):
        flat = real(x, port, what)
        if port.name == "a":
            seen.append(int(flat.sum()))
        return flat

    monkeypatch.setattr(execute_mod, "_lanes", record)
    return seen, execute(g, inputs)


def test_vsum_n21_w4(monkeypatch):
    x = 7 * np.arange(21) - 60
    graphs = steps("vsum_tree", W=4, N=21)
    sums, out = lane_sums(monkeypatch, graphs[-1], {"A": x, "out": np.zeros(1, np.int64)})
    assert sums == [-198, -86, 26, 138, 250, 80]  # the last: one lane and 3 pad lanes of 0
    assert out["out"].tolist() == [210]
    for g in graphs[:-1]:
        assert execute(g, {"A": x, "out": np.zeros(1, np.int64)})["out"].tolist() == [210]


def test_rowsum_3x7_w4(monkeypatch):
    a = np.array([[10 * i + j - 12 for j in range(7)] for i in range(3)])
    graphs = steps("rowsum", W=4, M=3, N=7)
    sums, out = lane_sums(monkeypatch, graphs[-1], {"A": a, "y": np.zeros(3, np.int64)})
    assert sums == [-42, -21, -2, 9, 38, 39]  # two firings per row
    assert out["y"].tolist() == [-63, 7, 77]
    for g in graphs[:-1]:
        assert execute(g, {"A": a, "y": np.zeros(3, np.int64)})["y"].tolist() == [-63, 7, 77]


def test_padded_shape_rounds_the_split_dimension_up():
    g = steps("rowsum", W=4, M=3, N=7)[-1]
    assert padded_shape(g, "A") == (3, 8) and padded_shape(g, "y") == (3,)
    g = steps("vecadd", W=4, N=7)[-1]
    assert padded_shape(g, "A") == (8,)
    g = steps("vecadd", W=4, N=8)[-1]
    assert padded_shape(g, "A") == (8,)  # no tail, no padding


def _vsum(dtype: str, op: str, identity: int) -> Graph:
    d = Graph.load(FIXTURES / "vsum.snaxdfg").to_dict()
    d["symbols"]["N"] = 8
    d["containers"]["A"]["dtype"] = dtype
    d["containers"]["out"]["dtype"] = dtype
    d["body"][0]["body"][0]["outputs"]["out"]["reduce"] = {"op": op, "identity": identity}
    return Graph.from_dict(d)


def test_pad_value_is_a_folds_identity():
    assert pad_value(_vsum("int8", "max", -128), "A") == -128
    assert pad_value(_vsum("int8", "min", 127), "A") == 127
    assert pad_value(_vsum("int8", "add", 0), "A") == 0
    assert pad_value(_vsum("int8", "max", -128), "out") == 0  # read by nothing that folds


def test_readers_that_need_different_pads_are_refused():
    d = _vsum("int8", "max", -128).to_dict()
    second = copy.deepcopy(d["body"][0])
    second["id"], second["body"][0]["id"] = "min_map", "smallest"
    second["body"][0]["outputs"]["out"].update(data="low", reduce={"op": "min", "identity": 127})
    d["containers"]["low"] = copy.deepcopy(d["containers"]["out"])
    d["body"].append(second)
    with pytest.raises(DfgError, match="container A: its readers need different pads: "
                       "sum \\(max\\) need -128; smallest \\(min\\) need 127"):  # fmt: skip
        pad_value(Graph.from_dict(d), "A")


def test_a_memlet_past_the_padding_is_still_an_error():
    """vecadd N = 7, W = 4, A padded to 8: a memlet reaching element 8 leaves the padding."""
    d = steps("vecadd", W=4, N=7)[-1].to_dict()
    d["body"][0]["body"][0]["inputs"]["a"]["subset"] = ["4 * i_t + 4:4 * i_t + 8"]
    a = np.arange(7)
    with pytest.raises(ExecutionError, match="leaves 0:"):
        execute(Graph.from_dict(d), {"A": a, "B": a, "C": np.zeros(7, np.int64)})
