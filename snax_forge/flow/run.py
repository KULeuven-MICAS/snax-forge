"""The whole path, kernel to model run (E2E1, D90).

    recipe + platform
      -> SNAX-SANDBOX   the kernel's import (IMP1), the recipe's steps, each
                        checked by the reference executor         sandbox/
      -> SNAX-DESIGN    the last graph on the platform: the checks, the
                        memory plan, the design point               design/
      -> SNAX-LOWER     the cluster file and the task list          cluster.json
                        (lowered once to the program)               tasks.json
      -> scenario       the kernel's make_inputs, each input at its
                        container's layout in its first memory,    scenario.json
                        and one region per container and memory    <C>.npy
                        from the memory plan (D95)
      -> SNAX-MODEL     the run                                     run/
      -> check          every ``inout`` container, read back from its last
                        memory through its layout, against the kernel's
                        reference and the reference executor (REF1) on the
                        same inputs; written into the profile's
                        ``functional_check`` (open item 13)

Everything goes to one folder, ``out/flow/<name>/``, overwritten on every
run; the default name carries every ``--set``, so runs that differ only in
the platform or the memory plan land in folders of their own. The run is
traced at ``task`` level by default, so ``run/`` opens in the run viewer
(``pixi run view``) with its schedule; the data movement views need
``trace_level="beat"``. ``scenario.json`` runs again on its own (``pixi run
model-run``). Nothing is decided here: each stage is the tool of that name,
called as its CLI would (D90, D94).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from snax_forge.design import DesignError, DesignPoint, load, run_checks
from snax_forge.dfg import Graph
from snax_forge.dfg.execute import execute
from snax_forge.lower import TaskList, cluster_file, lower_program, task_list
from snax_forge.sandbox import Recipe, apply_recipe, write_steps
from snax_forge.sdfg.loader import load as load_kernel
from snax_forge.sdfg.paths import _repo_root
from snax_forge.snax_model.scenario import (
    ClusterConfig,
    MemInit,
    Region,
    RunResult,
    Scenario,
    run,
    write_outputs,
)

OUT = _repo_root() / "out" / "flow"


class FlowError(ValueError):
    """A stage of the flow that failed; the message names the stage."""


@dataclass
class Flow:
    """What one run of the flow made."""

    name: str
    out: Path
    recipe: Recipe
    steps: list[Path]
    point: DesignPoint
    cluster: ClusterConfig
    tasks: TaskList
    n_commands: int
    scenario: Scenario
    result: RunResult
    check: dict[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.check["passed"])


def _name_part(path: str, value: Any) -> str:
    """One ``--set`` as part of a name: the path, then the value (compact JSON unless text),
    keeping only letters, digits and ``._,+-`` so the name stays one plain directory."""
    text = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    return re.sub(r"[^\w.,+-]", "", f"{path}{text}")


def default_name(
    recipe: Recipe,
    recipe_sets: Mapping[str, Any],
    platform_sets: Sequence[tuple[str, Any]] = (),
    memory_sets: Sequence[tuple[str, Any]] = (),
) -> str:
    """The recipe's name with every ``--set`` appended (D94): recipe params, then platform and
    memory paths without their ``platform.`` / ``memory.`` prefix, each in the order given
    (``vecadd_W8``, ``vecadd_l1.n_banks32``, ``vecadd_B.l1.base576``)."""
    sets = [*recipe_sets.items(), *platform_sets, *memory_sets]
    return "_".join([recipe.name, *(_name_part(k, v) for k, v in sets)])


def kernel_inputs(spec: Any, graph: Graph, seed: int) -> dict[str, np.ndarray]:
    """The kernel's ``make_inputs`` at the size the graph binds (its one bound symbol)."""
    bound = [v for v in graph.symbols.values() if v is not None]
    kw = {"n": bound[0]} if len(bound) == 1 else {}
    inputs = spec.make_inputs(np.random.default_rng(seed), **kw)
    for c, arr in inputs.items():
        cont = graph.containers.get(c)
        if cont is None:
            raise FlowError(
                f"inputs: {spec.name}'s make_inputs gives {c!r}, the graph has no such container"
            )
        if str(arr.dtype) != cont.dtype:
            raise FlowError(f"inputs: {c} is {arr.dtype}, the container is {cont.dtype}")
    return inputs


def regions_of(point: DesignPoint) -> list[Region]:
    """One region per container per memory, from the memory plan (D95), in its order."""
    return [
        Region(c, mem, lay.base, lay.shape, lay.strides)
        for c, mems in point.memory.layouts.items()
        for mem, lay in mems.items()
    ]


def read_container(point: DesignPoint, result: RunResult, c: str) -> tuple[str, np.ndarray]:
    """Container ``c`` read back from its last memory (L2 if it has one) through its layout."""
    mems = point.memory.layouts[c]
    mem = "l2" if "l2" in mems else "l1"
    lay = mems[mem]
    cfg = point.platform.l2 if mem == "l2" else point.platform.l1
    words = result.l2 if mem == "l2" else result.l1
    idx = np.indices(lay.shape).reshape(len(lay.shape), -1).T
    at = (lay.address(idx) - cfg.base_addr) // (cfg.width_bits // 8)
    dtype = point.graph.containers[c].dtype
    return mem, words[at, 0].astype(dtype).reshape(lay.shape)


def functional_check(
    spec: Any, point: DesignPoint, inputs: Mapping[str, np.ndarray], result: RunResult, seed: int
) -> dict[str, Any]:
    """Every ``inout`` container of the run against the kernel's reference and REF1."""
    ref = {k: np.copy(v) for k, v in inputs.items()}
    spec.reference(**ref)
    ref1 = execute(point.graph, {k: np.copy(v) for k, v in inputs.items()})
    containers = {}
    for c in spec.inout:
        mem, got = read_container(point, result, c)
        containers[c] = {
            "memory": mem,
            "elements": int(got.size),
            "reference": bool(np.array_equal(got, ref[c])),
            "ref1": bool(np.array_equal(got, ref1[c])),
            "mismatches": int(np.count_nonzero(got != ref[c])),
        }
    return {
        "kernel": spec.name,
        "seed": seed,
        "symbols": {k: v for k, v in point.graph.symbols.items() if v is not None},
        "containers": containers,
        "passed": all(x["reference"] and x["ref1"] for x in containers.values()),
    }


def run_flow(
    recipe_path: str | Path,
    platform_path: str | Path,
    *,
    recipe_sets: Mapping[str, Any] | None = None,
    platform_sets: Sequence[tuple[str, Any]] = (),
    memory_path: str | Path | None = None,
    memory_sets: Sequence[tuple[str, Any]] = (),
    graph_path: str | Path | None = None,
    name: str | None = None,
    out: str | Path | None = None,
    seed: int = 0,
    trace_level: str = "task",
) -> Flow:
    """Run the whole path; see the module doc. Raises FlowError or DesignError."""
    recipe_sets = dict(recipe_sets or {})
    recipe = Recipe.load(recipe_path).with_params(recipe_sets)
    # The design point, and with it the task list, is named after the recipe and its params
    # only, so pinning B and C still gives scenarios/vecadd's tasks.json byte for byte; the
    # folder carries every --set (D94).
    point_name = name or default_name(recipe, recipe_sets)
    name = name or default_name(recipe, recipe_sets, platform_sets, memory_sets)
    out = Path(out) if out is not None else OUT / name
    out.mkdir(parents=True, exist_ok=True)

    # SNAX-SANDBOX
    if graph_path is not None:
        graph = Graph.load(graph_path)
    else:
        from snax_forge.dfg.import_sdfg import import_kernel

        graph = import_kernel(recipe.kernel)
    steps = write_steps(recipe, apply_recipe(recipe, graph, seed=seed), out / "sandbox")

    # SNAX-DESIGN
    design = load(steps[-1], platform_path, platform_sets, memory_path, memory_sets)
    problems = run_checks(design)
    if problems:
        raise DesignError(problems, f"design check of {steps[-1]} on {platform_path}")
    point = DesignPoint.of(design, point_name)
    ddir = out / "design"
    ddir.mkdir(exist_ok=True)
    design.platform.save(ddir / "platform.json")
    design.memory.save(ddir / "memory.json")
    point.save(ddir / "design_point.json")

    # SNAX-LOWER
    cluster = cluster_file(point)
    tasks = task_list(point)
    program = lower_program(tasks, cluster)
    cluster.save(out / "cluster.json")
    tasks.save(out / "tasks.json")

    # Scenario: the kernel's inputs at their layouts
    spec = load_kernel(recipe.kernel)
    inputs = kernel_inputs(spec, point.graph, seed)
    fills = []
    for c, arr in inputs.items():
        mems = point.memory.layouts.get(c, {})
        mem = "l2" if "l2" in mems else "l1"
        np.save(out / f"{c}.npy", np.ascontiguousarray(arr), allow_pickle=False)
        fills.append(MemInit(mem, mems[mem].base, npy=f"{c}.npy"))
    scenario = Scenario(
        name, cluster, fills, program, cluster_ref="cluster.json", base_dir=out,
        regions=regions_of(point),
    )  # fmt: skip
    scenario.save(out / "scenario.json")

    # SNAX-MODEL and the check
    result = run(scenario, trace_level=trace_level)
    check = functional_check(spec, point, inputs, result, seed)
    result.profile.functional_check = check
    write_outputs(result, out / "run")
    return Flow(
        name, out, recipe, steps, point, cluster, tasks, len(program), scenario, result, check
    )
