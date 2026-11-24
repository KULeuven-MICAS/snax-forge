"""A recipe's ``dtypes``: containers at another integer dtype, applied to the
kernel's descriptors before DaCe builds the SDFG; the inputs cast to them."""

from __future__ import annotations

import json

import pytest

from snax_forge.dfg import DfgError
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.sandbox import Recipe, SandboxError, apply_recipe, write_steps
from tests.helpers import recipe_path

from .helpers import RECIPE

NARROW = {
    "vecadd_v1_1": {"A": "int32", "B": "int32", "C": "int32"},
    "vecmul_v1_1": {"A": "int32", "B": "int32", "C": "int32"},
    "vsum_v1_1_1": {"A": "int32", "out": "int64"},
    "vecadd_v1_2": {"A": "int8", "B": "int8", "C": "int8"},
    "vecmul_v1_1_1": {"A": "int32", "B": "int32", "C": "int64"},
}


def test_dtypes_round_trip_and_are_written_only_when_given(tmp_path):
    recipe = Recipe.load(recipe_path("vecadd_v1_1"))
    assert Recipe.from_dict(recipe.to_dict()) == recipe
    assert (recipe_path("vecadd_v1_1")).read_text() == recipe.to_json()
    assert "dtypes" not in Recipe.load(RECIPE).to_dict()
    results = apply_recipe(recipe, import_kernel(recipe.kernel, recipe.dtypes))
    write_steps(recipe, results, tmp_path)
    recorded = json.loads((tmp_path / "recipe.json").read_text())
    assert recorded["dtypes"] == recipe.dtypes  # the record keeps them, so a replay does


def test_an_unknown_dtype_or_container_is_named():
    d = Recipe.load(recipe_path("vecadd_v1_1")).to_dict()
    d["dtypes"]["A"] = "int12"
    with pytest.raises(SandboxError, match="vecadd_v1_1.dtypes.A: 'int12' is not a NumPy integer"):
        Recipe.from_dict(d)
    with pytest.raises(DfgError, match="vecadd: 'D' is not a container of vecadd .* nor a type"):
        import_kernel("vecadd", {"D": "int32"})
    with pytest.raises(
        DfgError, match="vecadd has no preset 'i4' \\(presets: \\['i16', 'i32', 'i8'\\]"
    ):
        import_kernel("vecadd", None, "i4")


# =============================================================================
# Type presets and casts
# =============================================================================
