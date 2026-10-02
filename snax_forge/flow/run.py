"""The whole path, kernel to model run (E2E1, D90), or from a bound graph on (FLOW2, D108).

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
      -> reports        design.md and run.md (REP1, D99)           report/
      -> log            the summary and the design checks that ran flow.log

Everything goes to one folder, ``out/flow/<name>/``, overwritten on every
run; the default name carries every ``--set``, so runs that differ only in
the platform or the memory plan land in folders of their own. The run is
traced at ``task`` level by default, so ``run/`` opens in the run viewer
(``pixi run view``) with its schedule; the data movement views need
``trace_level="beat"``. ``scenario.json`` runs again on its own (``pixi run
model-run``). Nothing is decided here: each stage is the tool of that name,
called as its CLI would (D90, D94).

``report/`` and ``flow.log`` sit next to ``run/``, never inside it (D44,
D99). ``flow.log`` is what the command line prints plus the design checks
that ran; a flow that fails writes the failure there instead, and the reports
of an earlier run in the same folder are removed first, so the folder never
holds reports of a run other than its last. A report that cannot be built is
named in the summary; the run and its check stand without it.

**Stage times** (D107). ``flow.log`` ends with the wall-clock seconds of each
stage (``import``, ``sandbox``, ``design``, ``lower``, ``scenario``, ``run``,
``check``, ``report``) and their total, on this machine. They change from run
to run, so they are in the log only: the command line, ``run/`` (byte
identical on every run, D44) and the reports never hold them. They are the
raw material of the paper's turnaround baseline (BASE1, C1).

**From a bound graph** (FLOW2, D108). ``run_bound(graph, platform, ...)`` is
the same path without its first stage: the graph is a bound ``.snaxdfg``
(the last step of a sandbox run, or one edited by hand) and goes straight to
SNAX-DESIGN, as ``steps[-1]`` does above. No recipe is read and ``sandbox/``
is not written, so the per-step reference check does not run; the design
checks and the check of the run's output against the kernel's reference and
REF1 do, which is what stops a graph that computes something else. The
kernel, needed for ``make_inputs`` and the reference only, is the graph's
``name`` unless one is given. The default name is the design step's
(``design_name``: the sandbox folder of a step's graph, else the file's
stem; a flow's own ``sandbox/`` gives its flow folder) with the platform and
memory ``--set`` appended as above. The recipe and step graphs an earlier
flow left in the folder's ``sandbox/`` are removed, unless the graph is one
of them, so the folder never shows a recipe its run did not come from.
``flow.log`` has no ``import`` or ``sandbox`` time.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from snax_forge.design import DesignError, DesignPoint, design_name, load, run_checks
from snax_forge.design.check import CHECKS, STAGES
from snax_forge.dfg import DfgError, Graph
from snax_forge.dfg.execute import execute
from snax_forge.lower import LowerError, TaskList, cluster_file, lower_program, task_list
from snax_forge.report import DESIGN_MD, RUN_MD, write_reports
from snax_forge.sandbox import Recipe, SandboxError, apply_recipe, write_steps
from snax_forge.sdfg.loader import load as load_kernel
from snax_forge.sdfg.paths import _repo_root
from snax_forge.snax_model.scenario import (
    ClusterConfig,
    MemInit,
    Region,
    RunResult,
    Scenario,
    ScenarioError,
    run,
    write_outputs,
)

OUT = _repo_root() / "out" / "flow"
LOG = "flow.log"
STAGE_NAMES = ("import", "sandbox", "design", "lower", "scenario", "run", "check", "report")
REPORT = "report"


class FlowError(ValueError):
    """A stage of the flow that failed; the message names the stage."""


@dataclass
class Flow:
    """What one run of the flow made."""

    name: str
    out: Path
    recipe: Recipe | None  # None for a flow from a bound graph (D108)
    steps: list[Path]  # the sandbox's graphs; none for a flow from a bound graph
    point: DesignPoint
    cluster: ClusterConfig
    tasks: TaskList
    n_commands: int
    scenario: Scenario
    result: RunResult
    check: dict[str, Any]
    checks: dict[str, list[str]] = field(default_factory=dict)  # stage -> codes that ran
    report_error: str | None = None  # why report/ could not be written, if it could not
    times: dict[str, float] = field(default_factory=dict)  # stage -> wall-clock seconds (D107)

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


def bound_name(
    graph_path: str | Path,
    platform_sets: Sequence[tuple[str, Any]] = (),
    memory_sets: Sequence[tuple[str, Any]] = (),
) -> str:
    """The default name of a flow from a bound graph (D108): the design step's name for the
    graph (its sandbox folder, else the file's stem; for a flow's own ``sandbox/`` the flow
    folder above it), then every ``--set`` as ``default_name`` appends them."""
    p = Path(graph_path).resolve()
    base = design_name(p)
    if p.parent.name == "sandbox" and base != p.stem:  # a step in a flow's own sandbox/
        base = p.parent.parent.name
    return "_".join([base, *(_name_part(k, v) for k, v in [*platform_sets, *memory_sets])])


def kernel_spec(kernel: str, why: str) -> Any:
    """The kernel's spec; an unknown name is a FlowError that says where the name came from."""
    try:
        return load_kernel(kernel)
    except KeyError as e:
        raise FlowError(f"kernel ({why}): {e.args[0]}") from None


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


def checks_that_ran() -> dict[str, list[str]]:
    """The registered design checks per stage, in the order they run. The flow only goes on
    when every stage passed, so after a flow these are exactly the checks that ran."""
    return {st: [c.code for c in CHECKS.values() if c.stage == st] for st in STAGES}


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def summary(f: Flow) -> str:
    """What the command line prints: one line per stage."""
    p, check = f.point, f.check
    syms = ", ".join(f"{k}={v}" for k, v in check["symbols"].items())
    if f.recipe is not None:
        params = ", ".join(f"{k}={v}" for k, v in f.recipe.params.items())
        what = f"{f.recipe.name}: {params}; {syms}"
        sandbox = (
            f"{len(f.steps) - 1} steps, each equal to the input graph on the reference "
            "check -> sandbox/"
        )
    else:
        what = f"graph {_rel(Path(p.graph_from))}; kernel {check['kernel']}; {syms}"
        sandbox = "skipped: the bound graph is the input (D108)"
    plan = f.point.memory
    inputs = ", ".join(m.npy for m in f.scenario.memory)
    n_checks = sum(len(v) for v in f.checks.values())
    lines = [f"flow {f.name} ({what}) on {p.platform.name} -> {_rel(f.out)}/"]
    lines.append(f"  sandbox   {sandbox}")
    lines.append(
        f"  design    {n_checks} checks passed; platform base {p.platform.base}, "
        f"{len(p.platform.changes)} changes; memory {plan.passes['placement']}, "
        f"{len(plan.changes)} changes -> design/"
    )
    lines.append(
        f"  lower     cluster.json ({len(f.cluster.components)} components), tasks.json "
        f"({len(f.tasks.steps)} steps, {f.n_commands} commands)"
    )
    lines.append(f"  scenario  scenario.json, inputs {inputs} (make_inputs, seed {check['seed']})")
    lines.append(
        f"  run       {f.result.total_cycles} cycles -> run/  (pixi run view {_rel(f.out / 'run')})"
    )
    for c, x in check["containers"].items():
        if x["reference"] and x["ref1"]:
            lines.append(
                f"  check     {c} ({x['elements']} elements, from {x['memory']}) equals the "
                f"{check['kernel']} reference and REF1"
            )
        else:
            lines.append(
                f"  check     {c} ({x['elements']} elements, from {x['memory']}) DIFFERS: "
                f"reference {x['reference']}, REF1 {x['ref1']}, {x['mismatches']} elements wrong"
            )
    if f.report_error is None:
        lines.append(f"  report    {REPORT}/{DESIGN_MD}, {REPORT}/{RUN_MD}, {LOG}")
    else:
        lines.append(f"  report    NOT WRITTEN: {f.report_error}")
    return "\n".join(lines)


def log_text(f: Flow) -> str:
    """``flow.log``: the summary, the design checks that ran per stage, the stage times."""
    n = sum(len(v) for v in f.checks.values())
    lines = [summary(f), "", f"design checks that ran ({n}, all passed):"]
    lines += [f"  {st:<9} {', '.join(codes)}" for st, codes in f.checks.items() if codes]
    if f.times:
        lines += ["", "stage times (wall clock, this machine):"]
        lines += [f"  {st:<9} {t:.3f} s" for st, t in f.times.items()]
        lines.append(f"  {'total':<9} {sum(f.times.values()):.3f} s")
    return "\n".join(lines) + "\n"


@contextmanager
def _timed(times: dict[str, float], stage: str) -> Iterator[None]:
    """Add the wall-clock seconds of the block to ``times[stage]`` (D107)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        times[stage] = times.get(stage, 0.0) + time.perf_counter() - t0


def _clear(out: Path) -> None:
    """Remove the log and reports of an earlier run in ``out`` (module doc)."""
    for p in (out / LOG, out / REPORT / DESIGN_MD, out / REPORT / RUN_MD):
        p.unlink(missing_ok=True)


def _clear_sandbox(out: Path, graph_path: Path) -> None:
    """Remove what an earlier flow's sandbox stage left in ``out`` (its recipe and step
    graphs), unless ``graph_path`` is one of them (module doc, D108)."""
    folder = out / "sandbox"
    if not folder.is_dir() or folder.resolve() == graph_path.resolve().parent:
        return
    for p in [folder / "recipe.json", *folder.glob("*.snaxdfg")]:
        p.unlink(missing_ok=True)
    if not any(folder.iterdir()):
        folder.rmdir()


FAILURES = (DesignError, FlowError, SandboxError, DfgError, LowerError, ScenarioError, OSError)


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
    _clear(out)
    times: dict[str, float] = {}

    def stages() -> Flow:
        # SNAX-SANDBOX
        with _timed(times, "import"):
            if graph_path is not None:
                graph = Graph.load(graph_path)
            else:
                from snax_forge.dfg.import_sdfg import import_kernel

                graph = import_kernel(recipe.kernel)
        with _timed(times, "sandbox"):
            steps = write_steps(recipe, apply_recipe(recipe, graph, seed=seed), out / "sandbox")
        f = _from_graph(
            steps[-1], recipe.kernel, f"recipe {recipe.name}", name, point_name, out,
            platform_path, platform_sets, memory_path, memory_sets, seed, trace_level, times,
        )  # fmt: skip
        f.recipe, f.steps = recipe, steps
        return f

    return _finish(stages, name, out, times)


def run_bound(
    graph_path: str | Path,
    platform_path: str | Path,
    *,
    kernel: str | None = None,
    platform_sets: Sequence[tuple[str, Any]] = (),
    memory_path: str | Path | None = None,
    memory_sets: Sequence[tuple[str, Any]] = (),
    name: str | None = None,
    out: str | Path | None = None,
    seed: int = 0,
    trace_level: str = "task",
) -> Flow:
    """The path from a bound ``.snaxdfg`` on, without the sandbox (FLOW2, D108); see the
    module doc. ``kernel`` names the kernel when the graph's ``name`` does not. Raises
    FlowError or DesignError."""
    graph_path = Path(graph_path)
    # As in run_flow: the design point keeps the graph's name, the folder carries every --set.
    point_name = name or bound_name(graph_path)
    name = name or bound_name(graph_path, platform_sets, memory_sets)
    out = Path(out) if out is not None else OUT / name
    out.mkdir(parents=True, exist_ok=True)
    _clear(out)
    _clear_sandbox(out, graph_path)
    times: dict[str, float] = {}

    def stages() -> Flow:
        why = "--kernel" if kernel is not None else f"the name of {graph_path}; --kernel K names it"
        return _from_graph(
            graph_path, kernel, why, name, point_name, out, platform_path, platform_sets,
            memory_path, memory_sets, seed, trace_level, times,
        )  # fmt: skip

    return _finish(stages, name, out, times)


def _finish(stages: Callable[[], Flow], name: str, out: Path, times: dict[str, float]) -> Flow:
    """Run the stages, then write the reports and the log; a failing stage is logged."""
    try:
        f = stages()
    except FAILURES as e:
        (out / LOG).write_text(f"flow {name} FAILED\n{e}\n")
        raise
    try:
        with _timed(times, "report"):
            write_reports(out)
    except (ValueError, KeyError, OSError) as e:
        f.report_error = f"{type(e).__name__}: {e}"
    f.times = {st: times[st] for st in STAGE_NAMES if st in times}
    (out / LOG).write_text(log_text(f))
    return f


def _from_graph(
    graph_path: str | Path,
    kernel: str | None,
    kernel_why: str,
    name: str,
    point_name: str,
    out: Path,
    platform_path: str | Path,
    platform_sets: Sequence[tuple[str, Any]],
    memory_path: str | Path | None,
    memory_sets: Sequence[tuple[str, Any]],
    seed: int,
    trace_level: str,
    times: dict[str, float],
) -> Flow:
    """The stages from a bound graph to the check, shared by ``run_flow`` and ``run_bound``;
    each stage's seconds go to ``times``. ``kernel`` None means the graph's name."""
    # SNAX-DESIGN
    with _timed(times, "design"):
        design = load(graph_path, platform_path, platform_sets, memory_path, memory_sets)
        problems = run_checks(design)
        if problems:
            raise DesignError(problems, f"design check of {graph_path} on {platform_path}")
        point = DesignPoint.of(design, point_name)
        ddir = out / "design"
        ddir.mkdir(exist_ok=True)
        design.platform.save(ddir / "platform.json")
        design.memory.save(ddir / "memory.json")
        point.save(ddir / "design_point.json")

    # SNAX-LOWER
    with _timed(times, "lower"):
        cluster = cluster_file(point)
        tasks = task_list(point)
        program = lower_program(tasks, cluster)
        cluster.save(out / "cluster.json")
        tasks.save(out / "tasks.json")

    # Scenario: the kernel's inputs at their layouts
    with _timed(times, "scenario"):
        spec = kernel_spec(kernel or point.graph.name, kernel_why)
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
    with _timed(times, "run"):
        result = run(scenario, trace_level=trace_level)
    with _timed(times, "check"):
        check = functional_check(spec, point, inputs, result, seed)
    result.profile.functional_check = check
    with _timed(times, "run"):  # the check goes into the profile first, so run/ is written after
        write_outputs(result, out / "run")
    return Flow(
        name, out, None, [], point, cluster, tasks, len(program), scenario, result, check,
        checks_that_ran(),
    )  # fmt: skip
