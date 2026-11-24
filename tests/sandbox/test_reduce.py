"""dot through SNAX-SANDBOX: a multiplier and an accumulator bound in one recipe.

Accepted when ``recipes/dot/dot_v1.json`` gives ``dot_accelerated.snaxdfg`` at its
last bind, the fixture's instances ``mul`` and ``sum`` named ``vmul`` and ``vsum``
(the multiply map bound to simd, the sum map to accumulate, each step
passing the reference check), ``recipes/dot/dot_v1_1.json`` the same and then both
groups in the accelerator ``dot``, and the bound graph equals ``np.dot``; the
accumulate binding with one lane (the Chisel Accumulator) works as well, the
reduce matcher rejects every mismatch by name, and unbind and join_map take
dot back to the imported graph.
"""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.dfg import Graph, execute
from snax_forge.sandbox import (
    Recipe,
    SandboxError,
    apply_recipe,
    bind,
    split_map,
)
from tests.helpers import graph_dict, recipe_path

from .helpers import FIXTURES, graph

RECIPE = recipe_path("dot_v1")
ENC = recipe_path("dot_v1_1")
MUL = {
    "node": "mult",
    "brm": "simd",
    "implementation": "chisel_tiled_spatial",
    "instance": "mul",
}
SUM = {"node": "sum", "brm": "accumulate", "implementation": "chisel_adder_tree", "instance": "sum"}


def imported(n: int = 64) -> Graph:
    d = graph_dict("dot")
    d["symbols"]["N"] = n
    return Graph.from_dict(d)


def dot_inputs(n: int = 64, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a, b = rng.integers(-1000, 1000, n), rng.integers(-1000, 1000, n)
    return {"A": a, "B": b, "out": np.array([7])}


def split_sum(w: int = 4) -> Graph:
    """dot with the multiplier bound at W = 4 and the sum map split by ``w``."""
    return split_map(bind(split_map(imported(), "mult_map", 4), **MUL), "sum_map", w)


# =============================================================================
# The recipe (the acceptance)
# =============================================================================


def test_the_dot_recipe_gives_the_fixture():
    """dot 1 ends at its second bind, two accelerators vmul and vsum: the fixture, its instances
    renamed."""
    results = apply_recipe(Recipe.load(RECIPE), graph("dot"))
    assert [r.file_name for r in results] == [
        "0_input.snaxdfg",
        "1_split_map.snaxdfg",
        "2_bind.snaxdfg",
        "3_split_map.snaxdfg",
        "4_bind.snaxdfg",
    ]
    fixture = (FIXTURES / "dot_accelerated.snaxdfg").read_text()
    renamed = fixture.replace('"instance": "mul"', '"instance": "vmul"').replace(
        '"instance": "sum"', '"instance": "vsum"'
    )
    assert renamed != fixture and results[-1].graph.to_json() == renamed


@pytest.mark.parametrize("w", [1, 2, 8, 16, 64])
def test_every_w_that_divides_n(w):
    """One W for both accelerators; the recipe checks each step, and the result is np.dot."""
    g = apply_recipe(Recipe.load(RECIPE).with_params({"W": w}), graph("dot"))[-1].graph
    assert g.node("mult").attrs["params"] == {"W": w, "op": "mul", "streams": 2}
    assert g.node("sum").attrs["params"] == {"W": w, "op": "add"}
    x = dot_inputs(seed=w)
    assert execute(g, x)["out"].tolist() == [int(x["A"] @ x["B"])]


def test_the_sum_folds_every_lane_into_one():
    """out[0] becomes the one lane out[0:1], with no reduce: the BRM folds."""
    s = Graph.load(FIXTURES / "dot_accelerated.snaxdfg").node("sum")
    assert s.outputs["out"].to_dict() == {"data": "out", "subset": ["0:1"]}
    assert s.inputs["a"].subset == ["4 * n_1_t:4 * n_1_t + 4"]
    assert s.attrs["code"] == "out = a"
    assert s.attrs["replaced"]["body"][0]["outputs"]["out"]["reduce"] == {
        "op": "add",
        "identity": 0,
    }


# =============================================================================
# The reduce matcher and bind, each rejection named
# =============================================================================


def _edit_sum(edit):
    def make() -> Graph:
        d = split_sum().to_dict()
        edit(d["body"][1]["body"][0]["body"][0])
        return Graph.from_dict(d)

    return make


REJECTED = [
    (_edit_sum(lambda t: t["attrs"].update(code="out = in1 * in1")), SUM, "is not a copy"),
    (_edit_sum(lambda t: t["outputs"]["out"].pop("reduce")), SUM, "does not fold \\(no reduce"),
    (
        _edit_sum(lambda t: t["outputs"]["out"]["reduce"].update(op="mul", identity=1)),
        SUM,
        "folds with 'mul', the pattern with 'add'",
    ),
    (
        _edit_sum(lambda t: t["outputs"]["out"]["reduce"].update(identity=5)),
        SUM,
        "starts from 5, the BRM from add's identity 0",
    ),
    (split_sum, SUM | {"brm": "simd"}, "the tasklet computes 'out = a' on the BRM's ports"),
    (lambda: split_map(imported(), "mult_map", 4), MUL | {"brm": "accumulate"}, "reduce folds one"),
]


@pytest.mark.parametrize(("make", "params", "message"), REJECTED)
def test_bind_rejected(make, params, message):
    with pytest.raises(SandboxError, match=message):
        bind(make(), **params)
