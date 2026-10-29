"""A recipe's ``dtypes`` (D113, D142): containers at another integer dtype, applied to the
kernel's descriptors before DaCe builds the SDFG; the inputs cast to them."""

from __future__ import annotations

import json

import numpy as np
import pytest

from snax_forge.dfg import DfgError
from snax_forge.dfg.execute import execute
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.dfg.reference import kernel_inputs, kernel_spec
from snax_forge.sandbox import Recipe, SandboxError, apply_recipe, write_steps

from .helpers import RECIPE

RECIPES = RECIPE.parent

NARROW = {
    "vecadd_i32": {"A": "int32", "B": "int32", "C": "int32"},
    "vecmul_i32": {"A": "int32", "B": "int32", "C": "int32"},
    "vsum_tree_i32": {"A": "int32", "out": "int64"},
    "vecadd_i8": {"A": "int8", "B": "int8", "C": "int8"},
    "vecmul_i32_i64": {"A": "int32", "B": "int32", "C": "int64"},
}


@pytest.mark.parametrize("name", list(NARROW))
def test_a_narrow_recipe_imports_at_its_dtypes_and_passes_the_reference(name):
    recipe = Recipe.load(RECIPES / f"{name}.json")
    assert recipe.dtypes == NARROW[name]
    graph = import_kernel(recipe.kernel, recipe.dtypes)
    assert {c: graph.containers[c].dtype for c in NARROW[name]} == NARROW[name]
    results = apply_recipe(recipe, graph)  # each step against the reference executor
    assert results[-1].graph.containers["A"].dtype == NARROW[name]["A"]


def test_an_int32_product_is_sign_extended_into_an_int64_output():
    """vecmul_i32_i64 (D143): the product runs at int32, the op type, and is sign-extended into
    C: lane 2 wraps to -2, not 4294967294."""
    recipe = Recipe.load(RECIPES / "vecmul_i32_i64.json")
    graph = apply_recipe(recipe, import_kernel(recipe.kernel, recipe.dtypes))[-1].graph
    a, b = np.zeros(64, np.int32), np.zeros(64, np.int32)
    a[:4], b[:4] = [3, -2, 2**31 - 1, -5], [4, 7, 2, -1]
    c = execute(graph, {"A": a, "B": b, "C": np.zeros(64, np.int64)})["C"]
    assert c.dtype == np.int64
    assert c[:4].tolist() == [12, -14, -2, 5]


def test_the_kernels_inputs_are_cast_to_the_containers_dtypes():
    recipe = Recipe.load(RECIPES / "vecadd_i8.json")
    graph = import_kernel(recipe.kernel, recipe.dtypes)
    inputs = kernel_inputs(kernel_spec("vecadd", "test"), graph, 0)
    assert {c: str(a.dtype) for c, a in inputs.items()} == NARROW["vecadd_i8"]
    raw = kernel_spec("vecadd", "test").make_inputs(np.random.default_rng(0))  # N unbound yet
    assert np.array_equal(inputs["A"], raw["A"].astype(np.int8))  # wrapping, as astype


def test_dtypes_round_trip_and_are_written_only_when_given(tmp_path):
    recipe = Recipe.load(RECIPES / "vecadd_i32.json")
    assert Recipe.from_dict(recipe.to_dict()) == recipe
    assert (RECIPES / "vecadd_i32.json").read_text() == recipe.to_json()
    assert "dtypes" not in Recipe.load(RECIPE).to_dict()
    results = apply_recipe(recipe, import_kernel(recipe.kernel, recipe.dtypes))
    write_steps(recipe, results, tmp_path)
    recorded = json.loads((tmp_path / "recipe.json").read_text())
    assert recorded["dtypes"] == NARROW["vecadd_i32"]  # the record keeps them, so a replay does


def test_an_unknown_dtype_or_container_is_named():
    d = Recipe.load(RECIPES / "vecadd_i32.json").to_dict()
    d["dtypes"]["A"] = "int12"
    with pytest.raises(SandboxError, match="vecadd_i32.dtypes.A: 'int12' is not a NumPy integer"):
        Recipe.from_dict(d)
    with pytest.raises(DfgError, match="recipe dtypes for kernel vecadd: 'D' is not a container"):
        import_kernel("vecadd", {"D": "int32"})
