"""Running a recipe (SBX1, D80): the vecadd recipe gives the split and accelerated fixtures.

Accepted when the vecadd recipe gives vecadd_accelerated.snaxdfg (a temporal
loop of N / W, a spatial loop of W, bound to elementwise_add), W = 4 and
W = 8 both pass the reference check, and a bound that is not a multiple of W
and a W the BRM does not allow are rejected (tests/sandbox/test_transforms.py
for the last one).
"""

from __future__ import annotations

import json

import pytest

from snax_forge.dfg import Graph
from snax_forge.sandbox import TRANSFORMS, Recipe, SandboxError, apply_recipe, register_transform
from snax_forge.sandbox.__main__ import main

from .helpers import FIXTURES, RECIPE, graph, recipe


def test_the_vecadd_recipe_gives_the_fixtures():
    results = apply_recipe(recipe(), graph("vecadd"))
    assert [r.file_name for r in results] == [
        "0_input.snaxdfg",
        "1_split_map.snaxdfg",
        "2_bind.snaxdfg",
    ]
    assert results[0].graph.symbols == {"N": 64}
    for r, name in zip(results[1:], ("vecadd_split", "vecadd_accelerated"), strict=True):
        assert r.graph.to_json() == (FIXTURES / f"{name}.snaxdfg").read_text()


@pytest.mark.parametrize("w", [4, 8, 16, 64])
def test_every_w_that_divides_n_passes(w):
    results = apply_recipe(recipe().with_params({"W": w}), graph("vecadd"))
    acc = results[-1].graph.node("add")
    assert acc.attrs["params"]["W"] == w
    assert results[-1].graph.node("add_map").attrs["range"] == f"0:N // {w}"


def test_a_w_that_does_not_divide_n():
    with pytest.raises(SandboxError, match="step 1 \\(split_map\\).*not a multiple of 5"):
        apply_recipe(recipe().with_params({"W": 5}), graph("vecadd"))


def test_the_reference_check_stops_a_wrong_step():
    def wrong(g: Graph) -> Graph:
        d = g.to_dict()
        d["body"][0]["body"][0]["body"][0]["attrs"]["code"] = "out = in1 - in2"
        return Graph.from_dict(d)

    register_transform("test_wrong", wrong)
    try:
        r = recipe()
        r.steps.insert(1, type(r.steps[0])("test_wrong", {}))
        with pytest.raises(
            SandboxError, match="step 2 \\(test_wrong\\): the reference check fails, 'C'"
        ):
            apply_recipe(r, graph("vecadd"))
    finally:
        TRANSFORMS.pop("test_wrong")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d["steps"][0].update(transform="tile"), "unknown transform"),
        (lambda d: d["steps"][0]["params"].update(size=2), "unexpected keyword argument 'size'"),
        (lambda d: d["steps"][0]["params"].pop("factor"), "missing a required argument: 'factor'"),
        (lambda d: d["steps"][0]["params"].update(factor="V"), "no value for 'V'"),
        (lambda d: d["symbols"].update(M=3), "the graph has no symbol 'M'"),
    ],
)
def test_bad_steps(edit, message):
    d = json.loads(RECIPE.read_text())
    edit(d)
    with pytest.raises(SandboxError, match=message):
        apply_recipe(Recipe.from_dict(d), graph("vecadd"))


def test_a_bound_symbol_must_agree():
    d = graph("vecadd").to_dict()
    d["symbols"]["N"] = 32
    with pytest.raises(SandboxError, match="N is bound to 32 already, not 64"):
        apply_recipe(recipe(), Graph.from_dict(d))


def test_cli_writes_every_step(tmp_path, capsys):
    out = tmp_path / "w8"
    assert (
        main(
            [
                str(RECIPE),
                "--graph",
                str(FIXTURES / "vecadd.snaxdfg"),
                "--set",
                "W=8",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert sorted(p.name for p in out.iterdir()) == [
        "0_input.snaxdfg",
        "1_split_map.snaxdfg",
        "2_bind.snaxdfg",
        "recipe.json",
    ]
    assert Recipe.load(out / "recipe.json").params == {"W": 8}
    assert Graph.load(out / "2_bind.snaxdfg").node("add").attrs["params"]["W"] == 8
    assert "2 steps" in capsys.readouterr().out


def test_cli_fails_by_name(tmp_path, capsys):
    args = [
        str(RECIPE),
        "--graph",
        str(FIXTURES / "vecadd.snaxdfg"),
        "--set",
        "W=5",
        "--out",
        str(tmp_path),
    ]
    assert main(args) == 1
    assert "not a multiple of 5" in capsys.readouterr().err


def test_cli_starts_from_the_kernel_import(tmp_path):
    """Without --graph the recipe starts from IMP1's import of its kernel."""
    assert main([str(RECIPE), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "2_bind.snaxdfg").read_text() == (
        FIXTURES / "vecadd_accelerated.snaxdfg"
    ).read_text()
