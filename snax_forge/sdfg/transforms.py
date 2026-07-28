"""SDFG transformation: enumerate what is applicable, and apply recipes.

Two halves:
  - analysis  -- which transformations DaCe could apply (static, non-mutating)
  - recipes   -- ordered, named transformation sequences loaded from transforms/

A recipe is an INPUT to the toolchain, like a kernel. It lives in transforms/,
not here; this module only knows how to read and run one.
"""

import importlib.util
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dace
from dace.transformation.optimizer import Optimizer

from .build import OUT as _SDFG_OUT
from .build import build, verify
from .loader import load as load_kernel
from .paths import _repo_root
from .spec import KernelSpec

# Derived from build.OUT, so fixing that path fixes this one too.
OUT = _SDFG_OUT.parent / "transforms"
# Recipes are inputs, not outputs -- they live beside kernels/ at the repo root.
RECIPE_ROOT = _repo_root() / "transforms"

# Wrong target entirely -- SNAX is neither GPU, FPGA, nor a cluster of nodes.
OFF_TARGET_PREFIXES = ("GPU", "FPGA", "MPI")
# Excluded by ADR: streamer-fed datapaths, no accelerator-internal storage.
OFF_SCOPE_NAMES = frozenset(
    {"InLocalStorage", "OutLocalStorage", "AccumulateTransient", "BufferTiling"}
)


def is_relevant(name: str) -> bool:
    return not name.startswith(OFF_TARGET_PREFIXES) and name not in OFF_SCOPE_NAMES


def transformation_matches(sdfg: dace.SDFG) -> list[dict]:
    """Every pattern transformation DaCe can currently apply to this SDFG.

    Static analysis: nothing is applied, the graph is not modified. state_id
    of -1 means a control-flow/SDFG-level match with no dataflow nodes.
    """
    rows = []
    for x in Optimizer(sdfg).get_pattern_matches():
        name = type(x).__name__
        state = sdfg.node(x.state_id) if getattr(x, "state_id", -1) >= 0 else None
        nodes = []
        if state is not None:
            for v in x.subgraph.values():
                try:
                    nodes.append(str(state.node(v)))
                except Exception:  # noqa: BLE001  (subgraph may hold non-node ids)
                    print(f"WARNING: {name} subgraph has non-node id {v} in state {state.label}")
        rows.append(
            {
                "transformation": name,
                "module": type(x).__module__,
                "relevant": is_relevant(name),
                "scope": "dataflow" if state is not None else "control",
                "state": state.label if state is not None else None,
                "nodes": nodes,
            }
        )
    return rows


def report_transformations(spec: KernelSpec, sdfg: dace.SDFG | None = None) -> dict:
    """Print and return the applicable-transformation profile for one kernel."""
    sdfg = sdfg if sdfg is not None else build(spec, simplify=True)
    rows = transformation_matches(sdfg)
    rel = [r for r in rows if r["relevant"]]
    off = [r for r in rows if not r["relevant"]]

    print(
        f"{spec.name}  ({len(list(sdfg.states()))} states)  {len(rel)} relevant / {len(rows)} total"
    )
    for name, n in Counter(r["transformation"] for r in rel).most_common():
        example = next((r["nodes"] for r in rel if r["transformation"] == name and r["nodes"]), [])
        print(f"    {name:28} x{n:<3} {example[:2]}")
    if off:
        print(f"    -- filtered: {dict(Counter(r['transformation'] for r in off))}")

    report = {
        "name": spec.name,
        "states": len(list(sdfg.states())),
        "relevant": dict(Counter(r["transformation"] for r in rel)),
        "filtered": dict(Counter(r["transformation"] for r in off)),
        "matches": rel,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{spec.name}.transforms.json").write_text(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------
# Recipes: declarative transformation sequences, loaded from transforms/
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One transformation application.

    target  -- map label to apply to. Use for single-node patterns whose only
               pattern node is `map_entry` (MapTiling, StripMining,
               Vectorization).
    repeat  -- apply until no match remains. Use for multi-node patterns like
               MapFusion, which needs three nodes and so cannot be targeted by
               a single label. NEVER use with MapTiling: a tiled map is always
               tileable again, so it will not terminate.
    """

    xform: type
    target: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    repeat: bool = False


@dataclass(frozen=True)
class TransformRecipe:
    """Contract between a recipe module and the toolchain.

    Same role as KernelSpec -- a declarative object that a recipe module
    exposes as RECIPE. Not a function.
    """

    name: str
    kernel: str
    steps: tuple[Step, ...]
    notes: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError(f"{self.name}: recipe has no steps")
        for s in self.steps:
            if s.repeat and s.target:
                raise ValueError(
                    f"{self.name}: step {s.xform.__name__} sets both repeat and target"
                )


def _map_entries(sdfg: dace.SDFG):
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                yield state, node


def _shape(sdfg: dace.SDFG) -> dict:
    return {
        "states": len(list(sdfg.states())),
        "maps": sum(1 for _ in _map_entries(sdfg)),
        "transients": sum(1 for _, d in sdfg.arrays.items() if d.transient),
    }


def apply_step(sdfg: dace.SDFG, step: Step) -> int:
    """Apply one step in place. Returns how many times it fired."""
    if step.target is not None:
        hits = [n for _, n in _map_entries(sdfg) if n.label == step.target]
        if not hits:
            avail = sorted({n.label for _, n in _map_entries(sdfg)})
            raise KeyError(f"no map entry labelled {step.target!r}; available: {avail}")
        for me in hits:
            step.xform.apply_to(sdfg, map_entry=me, options=step.options)
        return len(hits)
    if step.repeat:
        return sdfg.apply_transformations_repeated(step.xform, options=step.options or None)
    return sdfg.apply_transformations(step.xform, options=step.options or None)


def apply_recipe(recipe: TransformRecipe, verify_each: bool = True) -> tuple[dace.SDFG, list[dict]]:
    """Build the kernel, apply every step, verify bit-exactness as we go."""
    spec = load_kernel(recipe.kernel)
    sdfg = build(spec, simplify=True)
    sdfg.name = f"{spec.name}_{recipe.name}"  # own .dacecache dir; no collision

    log = []
    print(f"{recipe.name}  (kernel: {recipe.kernel})  start {_shape(sdfg)}")
    for i, step in enumerate(recipe.steps):
        n = apply_step(sdfg, step)
        entry = {
            "step": i,
            "xform": step.xform.__name__,
            "target": step.target,
            "options": step.options,
            "applied": n,
            "shape": _shape(sdfg),
        }
        if verify_each:
            entry["bitexact"] = verify(spec, sdfg)
        log.append(entry)
        ok = "  bitexact" if entry.get("bitexact") else ""
        bad = "" if entry.get("bitexact", True) else "   *** BIT-EXACTNESS BROKEN ***"
        print(f"  [{i}] {step.xform.__name__:20} x{n}  -> {entry['shape']}{ok}{bad}")

    OUT.mkdir(parents=True, exist_ok=True)
    sdfg.save(OUT / f"{recipe.name}.sdfg")
    (OUT / f"{recipe.name}.log.json").write_text(json.dumps(log, indent=2, default=str))
    return sdfg, log


def recipe_paths() -> dict[str, Path]:
    return {p.stem: p for p in sorted(RECIPE_ROOT.rglob("*.py")) if not p.stem.startswith("_")}


def load_recipe(name: str) -> TransformRecipe:
    paths = recipe_paths()
    if name not in paths:
        raise KeyError(f"unknown recipe {name!r}; available: {sorted(paths)}")
    path = paths[name]
    spec = importlib.util.spec_from_file_location(f"_recipe_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "RECIPE"):
        raise AttributeError(f"{path} defines no RECIPE")
    return module.RECIPE
