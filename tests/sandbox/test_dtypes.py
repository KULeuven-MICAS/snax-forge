"""A recipe's ``dtypes``: containers at another integer dtype, applied to the
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
from tests.helpers import recipe_path

from .helpers import RECIPE

NARROW = {
    "vecadd_i32": {"A": "int32", "B": "int32", "C": "int32"},
    "vecmul_i32": {"A": "int32", "B": "int32", "C": "int32"},
    "vsum_tree_i32": {"A": "int32", "out": "int64"},
    "vecadd_i8": {"A": "int8", "B": "int8", "C": "int8"},
    "vecmul_i32_i64": {"A": "int32", "B": "int32", "C": "int64"},
}


@pytest.mark.parametrize("name", list(NARROW))
def test_a_narrow_recipe_imports_at_its_dtypes_and_passes_the_reference(name):
    recipe = Recipe.load(recipe_path(name))
    assert {c: d for c, d in recipe.dtypes.items() if c != "ACC"} == NARROW[name]
    graph = import_kernel(recipe.kernel, recipe.dtypes)
    assert {c: graph.containers[c].dtype for c in NARROW[name]} == NARROW[name]
    results = apply_recipe(recipe, graph)  # each step against the reference executor
    assert results[-1].graph.containers["A"].dtype == NARROW[name]["A"]


def test_an_int32_product_is_sign_extended_into_an_int64_output():
    """vecmul_i32_i64: the product runs at int32, the op type, and is sign-extended into
    C: lane 2 wraps to -2, not 4294967294."""
    recipe = Recipe.load(recipe_path("vecmul_i32_i64"))
    graph = apply_recipe(recipe, import_kernel(recipe.kernel, recipe.dtypes))[-1].graph
    a, b = np.zeros(64, np.int32), np.zeros(64, np.int32)
    a[:4], b[:4] = [3, -2, 2**31 - 1, -5], [4, 7, 2, -1]
    c = execute(graph, {"A": a, "B": b, "C": np.zeros(64, np.int64)})["C"]
    assert c.dtype == np.int64
    assert c[:4].tolist() == [12, -14, -2, 5]


def test_the_kernels_inputs_are_cast_to_the_containers_dtypes():
    recipe = Recipe.load(recipe_path("vecadd_i8"))
    graph = import_kernel(recipe.kernel, recipe.dtypes)
    inputs = kernel_inputs(kernel_spec("vecadd", "test"), graph, 0)
    assert {c: str(a.dtype) for c, a in inputs.items()} == NARROW["vecadd_i8"]
    raw = kernel_spec("vecadd", "test").make_inputs(np.random.default_rng(0))  # N unbound yet
    assert np.array_equal(inputs["A"], raw["A"].astype(np.int8))  # wrapping, as astype


def test_dtypes_round_trip_and_are_written_only_when_given(tmp_path):
    recipe = Recipe.load(recipe_path("vecadd_i32"))
    assert Recipe.from_dict(recipe.to_dict()) == recipe
    assert (recipe_path("vecadd_i32")).read_text() == recipe.to_json()
    assert "dtypes" not in Recipe.load(RECIPE).to_dict()
    results = apply_recipe(recipe, import_kernel(recipe.kernel, recipe.dtypes))
    write_steps(recipe, results, tmp_path)
    recorded = json.loads((tmp_path / "recipe.json").read_text())
    assert recorded["dtypes"] == recipe.dtypes  # the record keeps them, so a replay does


def test_an_unknown_dtype_or_container_is_named():
    d = Recipe.load(recipe_path("vecadd_i32")).to_dict()
    d["dtypes"]["A"] = "int12"
    with pytest.raises(SandboxError, match="vecadd_i32.dtypes.A: 'int12' is not a NumPy integer"):
        Recipe.from_dict(d)
    with pytest.raises(DfgError, match="vecadd: 'D' is not a container of vecadd .* nor a type"):
        import_kernel("vecadd", {"D": "int32"})
    with pytest.raises(
        DfgError, match="vecadd has no preset 'i4' \\(presets: \\['i16', 'i32', 'i8'\\]"
    ):
        import_kernel("vecadd", None, "i4")


# =============================================================================
# Type presets and casts (D143 · D175)
# =============================================================================


def test_types_and_dtypes_are_set_like_params():
    """``--set types=i8`` picks the kernel's preset, ``--set dtypes.C=int16`` one dtype on top;
    both round-trip and are written only when given."""
    recipe = Recipe.load(recipe_path("vecmul")).with_params(
        {"types": "i8", "dtypes.C": "int16", "W": 8}
    )
    assert (recipe.types, recipe.dtypes, recipe.params["W"]) == ("i8", {"C": "int16"}, 8)
    assert Recipe.from_dict(recipe.to_dict()) == recipe
    assert "types" not in Recipe.load(recipe_path("vecmul")).to_dict()
    with pytest.raises(SandboxError, match="no param 'X' .*also types and dtypes.<name>"):
        recipe.with_params({"X": 1})
    g = import_kernel(recipe.kernel, recipe.dtypes, recipe.types)
    assert {c: x.dtype for c, x in g.containers.items()} == {
        "A": "int8", "B": "int8", "C": "int16"}  # fmt: skip
    assert g.type_params == {"ACC": "int32"}
    assert g.node("mult").attrs["code"] == "out = int32(in1) * int32(in2)"


def test_a_cast_sets_the_op_type_and_is_stripped_at_bind():
    """dot 4 at i8: the MAC binds ``out = in1 * in2`` with opType int32 (its ports int8 in,
    int32 out); ``replaced`` keeps the casts, so unbind gives the import back."""
    from snax_forge.sandbox.transforms import unbind

    recipe = Recipe.load(recipe_path("dot_4")).with_params({"types": "i8", "W": 8})
    g0 = import_kernel(recipe.kernel, recipe.dtypes, recipe.types)
    results = apply_recipe(recipe, g0)
    mac = next(n for n, _ in results[-1].graph.walk() if n.kind == "accelerated"
               and n.attrs["brm"] == "mac")  # fmt: skip
    assert (mac.attrs["opType"], mac.attrs["code"]) == ("int32", "out = a * b")
    assert "int32(in1) * int32(in2)" in json.dumps(mac.attrs["replaced"])
    bound = next(r.graph for r in results if any(n.kind == "accelerated"
                                                  for n, _ in r.graph.walk()))  # fmt: skip
    node = next(n for n, _ in bound.walk() if n.kind == "accelerated")
    assert "int32(in1)" in json.dumps(unbind(bound, node.id).to_dict())


def test_the_plain_form_has_no_op_type():
    """vecmul with A, B int8 and ACC int8 (NumPy's wrapped product): no cast, no opType."""
    recipe = Recipe.load(recipe_path("vecmul")).with_params(
        {"types": "i8", "dtypes.ACC": "int8", "W": 8}
    )
    g = apply_recipe(recipe, import_kernel(recipe.kernel, recipe.dtypes, recipe.types))[-1].graph
    (acc,) = [n for n, _ in g.walk() if n.kind == "accelerated"]
    assert "opType" not in acc.attrs and g.type_params == {"ACC": "int8"}


def test_a_narrowing_cast_is_refused_at_bind():
    recipe = Recipe.load(recipe_path("vecmul")).with_params(
        {"dtypes.A": "int32", "dtypes.B": "int32", "dtypes.C": "int32", "dtypes.ACC": "int8"}
    )
    g = import_kernel(recipe.kernel, recipe.dtypes)
    assert g.node("mult").attrs["code"] == "out = int8(in1) * int8(in2)"
    with pytest.raises(SandboxError, match="the cast int8\\(in1\\) of int32 does not hold every "
                       "value of its input; the hardware only extends"):  # fmt: skip
        apply_recipe(recipe, g)
