"""What every test package shares: the repository's paths, the snax-gvsoc submodule and its skip
marker, and the graph fixtures read as dicts or bound by their recipes.

    REPO, FIXTURES (tests/dfg/fixtures), SUB (gvsoc/snax-gvsoc)
    FORGE_FIXTURES          the submodule's forge/ folders: each its recipe and --set values
    recipe_path(name)       ``recipes/<folder>/<name>.json`` (``find_recipe``)
    requires_submodule      skips where the submodule is not checked out (SUBMODULE_MISSING)
    graph_dict(name)        a fixture's .snaxdfg as a dict
    bound_vsum(w, impl)     vsum bound by its recipe, or at W = ``w`` to ``impl``
    imported(kernel, dtypes)  a kernel's import (DaCe), once per kernel and dtypes
    steps(name, **params)   every graph of the recipe ``name`` at ``params``, the import first
    dot1_split(w_mul, w_sum, n)  ``recipes/dot/dot_v1.json`` with a factor param per split, ``W_mul``
                            and ``W_sum`` (the dot 1 dry run's)
    bound_mvt_column(m, n)  mvt_types at i8, half 2 in the column form (``MVT_COLUMN_STEPS``)
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from snax_forge.dfg import Graph
from snax_forge.sandbox import Recipe, apply_recipe
from snax_forge.sandbox.recipe import find_recipe

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "dfg" / "fixtures"
SUB = REPO / "gvsoc" / "snax-gvsoc"

SUBMODULE_MISSING = (
    "the snax-gvsoc submodule is not checked out (git submodule update --init gvsoc/snax-gvsoc)"
)
requires_submodule = pytest.mark.skipif(not (SUB / "snax").is_dir(), reason=SUBMODULE_MISSING)

SEQUENTIAL = "design.schedule=sequential"
# snax-gvsoc's forge/ folders, one per integration feature, each its recipe and --set values at
# the smallest size that still shows it; the RTL fixtures and the recorded runs of the same names
# are made at the same sizes
FORGE_FIXTURES: dict[str, tuple[str, tuple[str, ...]]] = {
    "vecadd_v1": ("vecadd_v1", ()),  # B on A's banks: conflicts at level stall
    "vecadd_v1_1_w1": ("vecadd_v1_1", ("W=1", "N=8")),  # inputs sliced, the output packed
    "axpy_v1": ("axpy_v1", ("N=8",)),  # a CSR scalar, a padded tail
    "rowsum_v2_padded": ("rowsum_v2", ("M=4", "N=8", "memory.passes.layout=padded_rows")),
    "dot_v4": ("dot_v4", ("N=8",)),  # a linked group, a wire
    "matvec_acc_v2": ("matvec_acc_v2", ("M=2", "N=16")),  # two groups, a seeded fold
    "mvt_v1_2_2": ("mvt_v1_2_2", ("W=8", "M=8", "N=8")),  # shared movers, the port mux
    "mvt_v4_1_1_sequential": ("mvt_v4_1_1", ("M=16", "N=16", SEQUENTIAL)),  # the DMA's transpose
    "mvt_v5_sequential": ("mvt_v5", ("M=16", "N=16", SEQUENTIAL)),  # two accelerators
    "mvt_v6": ("mvt_v6", ("M=8",)),  # a fork, a seeded bank, a hold
    "bicg_v4_1": ("bicg_v4_1", ("M=40", "N=32", "R=4")),  # two strips, a carried seed
}


def recipe_path(name: str) -> Path:
    """The recipe ``name``'s file, ``recipes/<folder>/<name>.json`` (``find_recipe``)."""
    return find_recipe(name)


def graph_dict(name: str = "vecadd_accelerated") -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.snaxdfg").read_text())


def bound_vsum(w: int = 1, impl: str = "chisel_adder_tree") -> Graph:
    """vsum bound by the checked-in recipe (W = 1, N = 64; accumulate's chisel_adder_tree,
    instance acc), or with W = ``w`` to ``impl``."""
    d = Recipe.load(recipe_path("vsum_v1")).with_params({"W": w}).to_dict()
    d["steps"][-1]["params"]["implementation"] = impl
    return apply_recipe(Recipe.from_dict(d), Graph.load(FIXTURES / "vsum.snaxdfg"))[-1].graph


def imported(
    kernel: str,
    dtypes: Mapping[str, str] | Iterable[tuple[str, str]] | None = None,
    types: str | None = None,
) -> Graph:
    """A kernel's import, once per kernel, dtypes (a dict or its items) and type preset; no
    caller changes it."""
    return _imported(kernel, tuple(sorted(dict(dtypes or {}).items())), types)


@cache
def _imported(kernel: str, dtypes: tuple[tuple[str, str], ...], types: str | None) -> Graph:
    from snax_forge.dfg.import_sdfg import import_kernel

    return import_kernel(kernel, dict(dtypes) or None, types)


def dot1_split(w_mul: int, w_sum: int, n: int = 2048) -> Recipe:
    """dot 1's recipe with ``mult_map`` split by ``W_mul`` and ``sum_map`` by ``W_sum``."""
    from snax_forge.sandbox.recipe import Step

    d = Recipe.load(recipe_path("dot_v1"))
    factor = {"mult_map": "W_mul", "sum_map": "W_sum"}
    return Recipe(d.name, d.kernel, {"N": n, "W_mul": w_mul, "W_sum": w_sum}, d.symbols,
                  [Step(x.transform, {**x.params, "factor": factor[x.params["map"]]})
                   if "factor" in x.params else x for x in d.steps])  # fmt: skip


@cache
def steps(name: str, **params: int) -> tuple[Graph, ...]:
    """Every graph of recipe ``name`` at ``params`` (the recipe's defaults for the rest), the
    import first, at the recipe's ``types`` as the sandbox imports it."""
    r = Recipe.load(recipe_path(name)).with_params(params)
    return tuple(x.graph for x in apply_recipe(r, imported(r.kernel, r.dtypes, r.types)))


def _step(transform: str, **params: Any) -> dict[str, Any]:
    return {"transform": transform, "params": params}


# mvt_types at i8 with half 2 in the column form (D186, D187): half 1 as mvt_v4_1_1's,
# half 2's lanes on n (strips of W_x1 columns, m in time between the strip and the lanes), its
# fold seeded from x2, fused and bound as a mac reading y2 as a broadcast
MVT_COLUMN_STEPS = [
    _step("tag_map", map="tmp1_mult_map", kind="temporal"),
    _step("split_map", map="tmp1_mult_map_1", factor="W_x1"),
    _step("bind", node="tmp1_mult", brm="simd", implementation="chisel_tiled_spatial",
          instance="vmul_x1"),
    _step("tag_map", map="sum_map", kind="temporal"),
    _step("split_map", map="sum_map_1", factor="W_x1"),
    _step("fuse_seed", first="sum_map_1_s", second="augassign_18_4_map"),
    _step("bind", node="sum", brm="accumulate", implementation="chisel_adder_tree",
          instance="vsum_x1"),
    _step("link", container="tmp1_prod", depth="D_2", group="mvt"),
    _step("tag_map", map="tmp4_mult_map_1", kind="temporal"),
    _step("split_map", map="tmp4_mult_map", factor="W_x1"),
    _step("interchange_map", map="tmp4_mult_map_s", **{"with": "tmp4_mult_map_1"}),
    _step("tag_map", map="sum_map_3", kind="temporal"),
    _step("split_map", map="sum_map_2", factor="W_x1"),
    _step("interchange_map", map="sum_map_2_s", **{"with": "sum_map_3"}),
    _step("fuse_seed", first="sum_map_2_s", second="augassign_19_4_map"),
    _step("fuse_map", first="tmp4_mult_map", second="sum_map_2"),
    _step("bind", node="tmp4_mult_map_s", brm="mac", implementation="chisel_mac",
          instance="mac_x2"),
]  # fmt: skip


@cache
def bound_mvt_column(m: int = 16, n: int = 16) -> Graph:
    """mvt_types at i8, ``MVT_COLUMN_STEPS`` at W_x1 = 8: its bound graph (no caller changes it)."""
    from snax_forge.sandbox import bind_recipe_symbols

    r = Recipe.from_dict({"name": "mvt_col", "kernel": "mvt_types", "types": "i8",
                          "params": {"M": m, "N": n, "W_x1": 8, "D_2": 1},
                          "symbols": {"M": "M", "N": "N"}, "steps": MVT_COLUMN_STEPS})  # fmt: skip
    return apply_recipe(r, bind_recipe_symbols(r, imported("mvt_types", types="i8")))[-1].graph
