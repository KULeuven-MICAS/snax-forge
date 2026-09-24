"""Running a recipe (SBX1, D72, D80): every step applied, written and checked.

``apply_recipe(recipe, graph)`` binds the recipe's symbols in the input
graph (step 0, ``input``), then applies the steps in order. After every
step the reference executor (D79) runs the new graph and the input graph on
the same seeded inputs, and every non-transient container must come out
equal: a transform that changes what the graph computes stops the recipe at
that step. The inputs are random integers of each container's dtype and
shape, independent of the kernel; that the input graph computes what the
kernel does is REF1's check (``pixi run check-dfg``).

The result is one ``Result`` per step (index, transform, graph);
``write_steps`` writes them as ``<i>_<transform>.snaxdfg`` next to the
recipe it ran (``recipe.json``, with the param values used), so every step
can be loaded, run and viewed on its own (D72).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from snax_forge.dfg import DfgError, Graph, execute
from snax_forge.dfg.execute import bind_symbols

from .recipe import Recipe, SandboxError
from .transforms import TRANSFORMS


@dataclass
class Result:
    index: int
    transform: str
    graph: Graph

    @property
    def file_name(self) -> str:
        return f"{self.index}_{self.transform}.snaxdfg"


def bind_recipe_symbols(recipe: Recipe, g: Graph) -> Graph:
    """``g`` with the recipe's symbol values; a symbol already bound must agree."""
    values = recipe.bound_symbols()
    unknown = sorted(set(values) - set(g.symbols))
    if unknown:
        raise SandboxError(f"{recipe.name}: the graph has no symbol {unknown[0]!r}")
    symbols = dict(g.symbols)
    for s, v in values.items():
        if symbols[s] is not None and symbols[s] != v:
            raise SandboxError(f"{recipe.name}: {s} is bound to {symbols[s]} already, not {v}")
        symbols[s] = v
    d = g.to_dict()
    d["symbols"] = symbols
    return Graph.from_dict(d)


def step_params(recipe: Recipe, index: int) -> tuple[str, dict[str, Any]]:
    """A step's transform and its params, int params evaluated over the recipe params."""
    step = recipe.steps[index]
    what = f"{recipe.name} step {index + 1} ({step.transform})"
    if step.transform not in TRANSFORMS:
        raise SandboxError(f"{what}: unknown transform (registered: {sorted(TRANSFORMS)})")
    spec = TRANSFORMS[step.transform]
    params = {
        k: recipe.value(v, f"{what}.params.{k}") if k in spec.ints else v
        for k, v in step.params.items()
    }
    try:
        inspect.signature(spec.fn).bind(None, **params)
    except TypeError as e:
        raise SandboxError(f"{what}: {e}") from None
    return step.transform, params


def random_inputs(g: Graph, seed: int) -> dict[str, np.ndarray]:
    """Seeded integers for every non-transient container, at the graph's bound sizes."""
    rng = np.random.default_rng(seed)
    env = {k: v for k, v in g.symbols.items() if v is not None}
    from snax_forge import expr

    out = {}
    for name, c in g.containers.items():
        if c.transient:
            continue
        shape = tuple(expr.evaluate(s, env) for s in c.shape)
        out[name] = rng.integers(-1000, 1000, size=shape).astype(c.dtype)
    return out


def apply_recipe(recipe: Recipe, graph: Graph, seed: int = 0) -> list[Result]:
    """Every step's graph, from the input with its symbols bound; each one checked."""
    g = bind_recipe_symbols(recipe, graph)
    try:
        inputs = random_inputs(g, seed)
        bind_symbols(g, inputs)
        want = execute(g, inputs)
    except (DfgError, KeyError) as e:
        raise SandboxError(f"{recipe.name}: the input graph does not run: {e}") from None
    results = [Result(0, "input", g)]
    for i in range(len(recipe.steps)):
        name, params = step_params(recipe, i)
        what = f"{recipe.name} step {i + 1} ({name})"
        try:
            g = TRANSFORMS[name].fn(g, **params)
            got = execute(g, inputs)
        except (SandboxError, DfgError) as e:
            raise SandboxError(f"{what}: {e}") from None
        for k, c in g.containers.items():
            if not c.transient and not np.array_equal(got[k], want[k]):
                raise SandboxError(f"{what}: the reference check fails, {k!r} differs")
        results.append(Result(i + 1, name, g))
    return results


def write_steps(recipe: Recipe, results: list[Result], out: Path) -> list[Path]:
    """Every step's graph and the recipe that made them, in ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.snaxdfg"):
        old.unlink()
    recipe.save(out / "recipe.json")
    paths = []
    for r in results:
        r.graph.save(out / r.file_name)
        paths.append(out / r.file_name)
    return paths
