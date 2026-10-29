"""The linked recipes (D150, C§12): dot 2–6 and matvec 1–6 run step by step with the reference
check, at W = 4 and 8, and end in the groups and links C§17's table has. At N = 32 (and M = 2
for matvec): the groups and links do not depend on the sizes, and the reference executor runs
at every step."""

from __future__ import annotations

import pytest

from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.sandbox import Recipe, apply_recipe

from .helpers import REPO

# Each recipe: its last graph's groups (name -> instances) and its link (container, depth)
WANT = {
    "dot_2": ({"dot": ["mul", "sum"]}, ("tmp0", 1)),
    "dot_3": ({"dot": ["mul", "sum"]}, ("tmp0", 0)),
    "dot_4": ({"dot": ["mac", "sum"]}, ("out_partial", 0)),
    "dot_5": ({"dot": ["mac", "sum"]}, ("out_partial", 0)),
    "dot_6": ({"dot": ["mac", "sum"]}, ("out_partial", 1)),
    "matvec_1": ({"mul": ["mul"], "sum": ["sum"]}, None),
    "matvec_2": ({"matvec": ["mul", "sum"]}, ("tmp0", 1)),
    "matvec_3": ({"matvec": ["mul", "sum"]}, ("tmp0", 0)),
    "matvec_4": ({"matvec": ["mac", "sum"]}, ("q_partial", 0)),
    "matvec_5": ({"matvec": ["mac", "sum"]}, ("q_partial", 0)),
    "matvec_6": ({"matvec": ["mac", "sum"]}, ("q_partial", 1)),
}


@pytest.mark.parametrize("w", [4, 8])
@pytest.mark.parametrize("name", list(WANT))
def test_the_recipe_runs_and_ends_in_its_groups(name, w):
    small = {"N": 32, **({"M": 2} if name.startswith("matvec") else {})}
    recipe = Recipe.load(REPO / "recipes" / f"{name}.json").with_params({"W": w, **small})
    last = apply_recipe(recipe, import_kernel(recipe.kernel))[-1].graph  # each step checked
    groups: dict[str, list[str]] = {}
    for n, _ in last.walk():
        if n.kind == "accelerated":
            groups.setdefault(n.attrs.get("group", n.attrs["instance"]), []).append(
                n.attrs["instance"]
            )
    links = [(n.attrs["container"], n.attrs["depth"]) for n, _ in last.walk() if n.kind == "link"]
    want_groups, want_link = WANT[name]
    assert groups == want_groups
    assert links == ([want_link] if want_link else [])
