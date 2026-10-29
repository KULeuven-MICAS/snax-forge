"""The whole path, kernel to a SNAX-GVSoC run (D136), or from a bound graph on (D108).

    recipe + platform
      -> SNAX-SANDBOX   the kernel's import, the recipe's steps, each
                        checked by the reference executor         sandbox/
      -> SNAX-DESIGN    the last graph on the platform: the checks, the
                        memory plan, the design point               design/
      -> SNAX-LOWER     the task list                               tasks.json
      -> gen            the generator (gen/gvsoc.py): the three
                        description files, the headers, data.h (the kernel's
                        make_inputs, the reference executor's goldens)
                        and main.c                                  gen/
      -> build          SNAX-GVSoC, when needed (below)            out/snax-gvsoc/
      -> sw             the program, by snax-gvsoc's generic app    sw/forge.elf
      -> gvsoc          the run on snax_hjson with gen/'s cluster, its
                        events into run/events.log (D137)           run/
      -> check          the program's own check: its exit code and
                        what it printed (``check_output``)          run/run.json
      -> convert        the events into the profile and the trace
                        (snax_forge/run/, C§7)                      run/profile.json
      -> report         design.md and run.md (snax_forge/report/)  report/
      -> log            the summary and the design checks that ran flow.log

The flow runs SNAX-GVSoC only (D136). ``build``, ``sw`` and ``gvsoc`` each call
``run_gvsoc``, ``make`` on ``gvsoc/Makefile`` through a subprocess; that
Makefile names the image's tools, so the flow calls it from pixi. Nothing is
decided here: each stage is the tool of that name, called as its command
line would be (D136).

**Build.** GVSoC is built into ``out/snax-gvsoc/`` (``gvsoc/Makefile``'s
``build``) when its install is missing, when the checkout it was built from
(``checkout.txt``: the path, then the snax-gvsoc commit) is not the
submodule's path and HEAD, or when the submodule's tree is dirty; otherwise
the stage says it is up to date. A build first prints why and where its log
is, then make's output, stdout and stderr merged, line by line as it comes,
to the terminal and to ``build.log`` (``run_gvsoc``'s ``live``); neither line
goes to ``flow.log``, and ``sw`` and ``gvsoc`` write their logs only. An
interrupted build is not up to date: ``checkout.txt`` gets its commit only
once the build passed. When the submodule's HEAD is not the commit
snax-forge's tree pins (``git ls-tree HEAD gvsoc/snax-gvsoc``), the summary
and ``flow.log`` warn.

**Goldens.** data.h's goldens are the reference executor's on the design
point's graph; before writing them, gen checks that result against the
kernel's reference on the same inputs (``dfg.reference.reference_check``), so a graph
that computes something else stops there (D108).

**Check** (C§7). The program checks itself (C§16's ``main.c``): its exit
code is the run's (the ``Error N`` make prints for the run's recipe, 0 when
it passed), and its lines give ``total_cycles`` (``forge cycles:``, the
window from main's entry to ``forge_check()``'s entry), the data errors
over the output words and each accelerator task's counter, beside its beats
from the manifest and its group (D148). ``passed`` is an exit code of 0. A missing line is a
FlowError that names it. ``run/run.json`` records what ran: the design
point, the three description files (``hjson``), the ELF's sha256 and the symbols of its L2 containers'
arrays (``elf_symbols``, ``run/elf.py``: their L2 regions), the target, the snax-gvsoc
commit and whether its tree was dirty, the trace level, its sources and whether a VCD was
written;
``run/profile.json`` holds ``total_cycles`` and ``functional_check``, with
what ``convert`` adds.

**Trace** (D137). The run writes the SNAX components' ``summary`` traces into
``run/events.log`` at every level, their ``events`` traces from ``task``, and
from ``beat``, ``stall`` and ``finegrained`` the ``beats``, ``stalls`` and the
cores' ``insn`` lines of the trace's sources (``trace_sources``, by default the
accelerator, its movers and the DMA, at ``finegrained`` every core too); the
cluster, read from gen's files, gives ``gvsoc/Makefile``'s ``TRACES`` and the
crossbar's ``trace_inputs`` (``run/sources.py``). With ``vcd`` the run also
writes gtkwave's VCD of the same sources: ``run/all.vcd`` and ``run/view.gtkw``,
never read here. ``convert`` reads the log, names it after the cluster
(``run/cluster.py``: the generator's hjson files and the design point), the
task list and the design point's regions, and writes the profile and, above
``off``, the trace (``run/convert.py``, whose invariants hold or the stage
fails); ``trace_meta.json`` keeps the sources as its ``filter``. ``report``
then writes ``report/design.md`` and ``report/run.md`` from them (D99).

Everything goes to one folder, ``out/flow/<name>/``, overwritten on every
run; the default name carries every ``--set``, so runs that differ only in
the platform or the memory plan land in folders of their own, then the
trace level when it is not ``off`` (``vecadd_task``). ``flow.log``
is what the command line prints plus the design checks that ran; a flow
that fails writes the failure there instead. The log, the reports, ``gen/``
and ``run/`` of an earlier run in the same folder are removed first.

**Stage times** (D136). ``flow.log`` ends with the wall-clock seconds of each
stage (``import``, ``sandbox``, ``design``, ``lower``, ``gen``, ``build``,
``sw``, ``gvsoc``, ``check``, ``convert``, ``report``) and their total, on this machine. They change
from run to run, so they are in the log only. They are the raw material of
the paper's turnaround baseline (claim C1).

**From a bound graph** (D108). ``run_bound(graph, platform, ...)`` is
the same path without its first stage: the graph is a bound ``.snaxdfg``
(the last step of a sandbox run, or one edited by hand) and goes straight to
SNAX-DESIGN, as ``steps[-1]`` does above. No recipe is read and ``sandbox/``
is not written, so the per-step reference check does not run; the design
checks, gen's reference check and the program's check do; the reference
check is what stops a graph that computes something else. The kernel, needed
for ``make_inputs`` and its reference only, is the graph's ``name`` unless
one is given. The
default name is the design step's (``design_name``: the sandbox folder of a
step's graph, else the file's stem; a flow's own ``sandbox/`` gives its flow
folder) with the platform and memory ``--set`` appended as above. The recipe
and step graphs an earlier flow left in the folder's ``sandbox/`` are
removed, unless the graph is one of them, so the folder never shows a recipe
its run did not come from. ``flow.log`` has no ``import`` or ``sandbox``
time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from snax_forge.common import cwd_relative, git, repo_root, to_json
from snax_forge.design import DesignError, DesignPoint, design_name, load, run_checks
from snax_forge.design.check import CHECKS, STAGES
from snax_forge.dfg import DfgError, Graph
from snax_forge.dfg.reference import KernelError, kernel_spec, reference_check
from snax_forge.gen import GenError, generate
from snax_forge.lower import CError, LowerError, TaskList, task_list
from snax_forge.lower.tasks import Configure
from snax_forge.run import (
    LEVELS,
    RunError,
    Trace,
    at_least,
    load_cluster,
    read_events,
    regions_of,
    write_run,
)
from snax_forge.run.convert import convert
from snax_forge.run.elf import l2_symbols, read_symbols
from snax_forge.run.outputs import EVENTS_FILE
from snax_forge.run.sources import gvsoc_options, sources_for
from snax_forge.sandbox import Recipe, SandboxError, apply_recipe, write_steps

if TYPE_CHECKING:  # gen imports the flow's kernel helpers (gen/data.py)
    from snax_forge.gen import Manifest

ROOT = repo_root()
OUT = ROOT / "out" / "flow"
LOG = "flow.log"
STAGE_NAMES = (
    "import",
    "sandbox",
    "design",
    "lower",
    "gen",
    "build",
    "sw",
    "gvsoc",
    "check",
    "convert",
    "report",
)
REPORT = "report"
# SNAX-GVSoC: gvsoc/Makefile, the submodule and the build's folder (that Makefile's defaults)
GVSOC_MAKE = ROOT / "gvsoc"
SNAX_GVSOC = GVSOC_MAKE / "snax-gvsoc"
SUBMODULE = "gvsoc/snax-gvsoc"
GVSOC_WORKDIR = ROOT / "out" / "snax-gvsoc"
TARGET = "snax_hjson"


def description_files(files: list[str]) -> list[str]:
    """The generator's description files of a run (D134): the cluster and streamer hjson and
    accelerator.json."""
    return [x for x in files if x.endswith(".hjson") or x == "accelerator.json"]


ELF = "forge.elf"
VCD, GTKW = "all.vcd", "view.gtkw"


class FlowError(ValueError):
    """A stage of the flow that failed; the message names the stage."""


@dataclass(frozen=True)
class Checkout:
    """The snax-gvsoc checkout the flow builds and runs: its path, HEAD, whether its tree is
    dirty, and the commit snax-forge's tree pins (None when git cannot say)."""

    path: Path
    commit: str
    dirty: bool
    pinned: str | None


@dataclass
class Flow:
    """What one run of the flow made."""

    name: str
    out: Path
    recipe: Recipe | None  # None for a flow from a bound graph
    steps: list[Path]  # the sandbox's graphs; none for a flow from a bound graph
    point: DesignPoint
    tasks: TaskList
    manifest: Manifest
    checkout: Checkout
    build: str  # "built: <why>" or "up to date"
    elf_sha256: str
    total_cycles: int
    check: dict[str, Any]
    checks: dict[str, list[str]] = field(default_factory=dict)  # stage -> codes that ran
    times: dict[str, float] = field(default_factory=dict)  # stage -> wall-clock seconds
    trace_level: str = "off"
    trace: Trace | None = None  # convert's, above off
    trace_sources: list[str] = field(default_factory=list)
    vcd: bool = False

    @property
    def passed(self) -> bool:
        return bool(self.check["passed"])

    @property
    def warnings(self) -> list[str]:
        c = self.checkout
        if c.pinned is None or c.pinned == c.commit:
            return []
        return [f"{SUBMODULE} is at {c.commit[:8]}, snax-forge's tree pins {c.pinned[:8]}"]


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
    """The recipe's name with every ``--set`` appended (D136): recipe params, then platform and
    memory paths without their ``platform.`` / ``memory.`` prefix, each in the order given
    (``vecadd_W8``, ``vecadd_l1.n_banks32``, ``vecadd_B.l1.base576``)."""
    sets = [*recipe_sets.items(), *platform_sets, *memory_sets]
    return "_".join([recipe.name, *(_name_part(k, v) for k, v in sets)])


def with_level(name: str, level: str) -> str:
    """A default name with the trace level appended when it is not ``off`` (``vecadd_task``)."""
    return name if level == "off" else f"{name}_{level}"


def bound_name(
    graph_path: str | Path,
    platform_sets: Sequence[tuple[str, Any]] = (),
    memory_sets: Sequence[tuple[str, Any]] = (),
) -> str:
    """The default name of a flow from a bound graph: the design step's name for the
    graph (its sandbox folder, else the file's stem; for a flow's own ``sandbox/`` the flow
    folder above it), then every ``--set`` as ``default_name`` appends them."""
    p = Path(graph_path).resolve()
    base = design_name(p)
    if p.parent.name == "sandbox" and base != p.stem:  # a step in a flow's own sandbox/
        base = p.parent.parent.name
    return "_".join([base, *(_name_part(k, v) for k, v in [*platform_sets, *memory_sets])])


def checks_that_ran() -> dict[str, list[str]]:
    """The registered design checks per stage, in the order they run. The flow only goes on
    when every stage passed, so after a flow these are exactly the checks that ran."""
    return {st: [c.code for c in CHECKS.values() if c.stage == st] for st in STAGES}


# =============================================================================
# SNAX-GVSoC
# =============================================================================


def run_gvsoc(goal: str, variables: Mapping[str, str], log: Path, err: Path | None = None,
              live: bool = False) -> int:  # fmt: skip
    """``make -C gvsoc <goal> NAME=VALUE ...`` (gvsoc/Makefile, D136): stdout to ``log``,
    stderr to ``err`` (into ``log`` when None); make's exit code. ``live``: stdout and stderr,
    merged, to ``log`` and to the terminal line by line as they come, as tee would, with
    PYTHONUNBUFFERED=1; on KeyboardInterrupt make is terminated first."""
    cmd = ["make", "-C", str(GVSOC_MAKE), goal, *(f"{k}={v}" for k, v in variables.items())]
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("w") as o:
            if live:
                return _tee(cmd, o)
            if err is None:
                return subprocess.run(
                    cmd, stdout=o, stderr=subprocess.STDOUT, check=False
                ).returncode
            with err.open("w") as e:
                return subprocess.run(cmd, stdout=o, stderr=e, check=False).returncode
    except FileNotFoundError:
        raise FlowError(
            f"{goal}: no make on PATH; run the flow in snax-forge's image (D118)"
        ) from None


def _tee(cmd: list[str], log: IO[str]) -> int:
    """``cmd``'s stdout and stderr, merged, to ``log`` and ``sys.stdout``, each line flushed;
    its exit code."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                          text=True, errors="replace") as p:  # fmt: skip
        try:
            assert p.stdout is not None
            for line in p.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            return p.wait()
        except KeyboardInterrupt:
            p.terminate()
            p.wait()
            raise


def checkout(path: Path = SNAX_GVSOC) -> Checkout:
    """The submodule's checkout (``Checkout``); a FlowError when it is not checked out."""
    commit = git(path, "rev-parse", "HEAD") if (path / "snax").is_dir() else None
    if commit is None:
        raise FlowError(
            f"build: {cwd_relative(path)} is not checked out: git submodule update --init "
            f"{SUBMODULE}"
        )
    status = git(path, "status", "--porcelain")
    pin = git(ROOT, "ls-tree", "HEAD", SUBMODULE)  # "160000 commit <sha>\t<path>"
    pinned = pin.split()[2] if pin and len(pin.split()) >= 3 else None
    return Checkout(path.resolve(), commit, status is None or status != "", pinned)


def build_reason(c: Checkout, workdir: Path) -> str | None:
    """Why GVSoC must be built into ``workdir`` for checkout ``c``, or None (module doc)."""
    if not (workdir / "install" / "bin" / "gvrun").is_file():
        return f"no install in {cwd_relative(workdir)}"
    built = (workdir / "checkout.txt").read_text().split() if (workdir / "checkout.txt").is_file() else []  # fmt: skip
    if not built or Path(built[0]) != c.path:
        return (
            f"{cwd_relative(workdir)} was built from {built[0] if built else 'an unknown checkout'}"
        )
    if len(built) < 2 or built[1] != c.commit:
        was = built[1][:8] if len(built) >= 2 else "an unknown commit"
        return f"built at snax-gvsoc {was}, the checkout is at {c.commit[:8]}"
    if c.dirty:
        return f"{SUBMODULE} has uncommitted changes"
    return None


def _make(stage: str, goal: str, variables: Mapping[str, str], log: Path,
          live: bool = False) -> int:  # fmt: skip
    """``run_gvsoc``; a nonzero exit is a FlowError naming the log and its last lines."""
    rc = run_gvsoc(goal, variables, log, None, live)
    if rc != 0:
        tail = "\n".join(log.read_text().splitlines()[-15:])
        raise FlowError(f"{stage}: make {goal} failed (exit {rc}); its output is in "
                        f"{cwd_relative(log)}:\n{tail}")  # fmt: skip
    return rc


MAKE_ERROR = re.compile(r"^make: \*\*\* \[[^\]]*\] Error (\d+)$", re.MULTILINE)


def run_exit_code(rc: int, stderr: str) -> int:
    """The program's exit code from ``make run``: 0 when make passed, else the ``Error N`` make
    prints for the run's recipe (gvrun's exit, the program's return); a FlowError when make
    failed before it ran gvrun."""
    if rc == 0:
        return 0
    m = MAKE_ERROR.search(stderr)
    if m is None:
        raise FlowError(f"gvsoc: make run failed (exit {rc}) before running gvrun:\n"
                        + "\n".join(stderr.splitlines()[-15:]))  # fmt: skip
    return int(m[1])


# =============================================================================
# The check
# =============================================================================

CYCLES = re.compile(r"^forge cycles: (\d+)$", re.MULTILINE)
DATA = re.compile(r"^forge data: (\d+) errors \((\d+) words\)$", re.MULTILINE)


def check_output(
    stdout: str, exit_code: int, beats: Mapping[str, int]
) -> tuple[int, dict[str, Any]]:
    """The program's own check (C§7): ``total_cycles`` from ``forge cycles:``, and the
    ``functional_check`` fields it prints and returns: ``exit_code``, ``data_errors`` and
    ``words``, per accelerator task (``beats``: its beats, in order) its ``counter`` and
    ``beats``, and ``passed`` (exit code 0). A missing line is a FlowError naming it."""

    def need(rx: re.Pattern[str], what: str) -> re.Match[str]:
        m = rx.search(stdout)
        if m is None:
            raise FlowError(f"check: no `{what}` line in the program's output")
        return m

    cycles = int(need(CYCLES, "forge cycles:")[1])
    data = need(DATA, "forge data:")
    tasks = {}
    for t, n in beats.items():
        m = need(re.compile(rf"^forge counter {re.escape(t)}: (\d+)$", re.MULTILINE),
                 f"forge counter {t}:")  # fmt: skip
        tasks[t] = {"counter": int(m[1]), "beats": n}
    return cycles, {
        "exit_code": exit_code,
        "data_errors": int(data[1]),
        "words": int(data[2]),
        "tasks": tasks,
        "passed": exit_code == 0,
    }


# =============================================================================
# Summary and log
# =============================================================================


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
        what = f"graph {cwd_relative(Path(p.graph_from))}; kernel {check['kernel']}; {syms}"
        sandbox = "skipped: the bound graph is the input (D108)"
    plan = f.point.memory
    n_checks = sum(len(v) for v in f.checks.values())
    m = f.manifest
    hjson = ", ".join(description_files(m.files))
    tasks = ", ".join(f"{t} {x['counter']} ({x['beats']} beats)" for t, x in check["tasks"].items())
    lines = [f"flow {f.name} ({what}) on {p.platform.name} -> {cwd_relative(f.out)}/"]
    lines.append(f"  sandbox   {sandbox}")
    lines.append(
        f"  design    {n_checks} checks passed; platform base {p.platform.base}, "
        f"{len(p.platform.changes)} changes; memory {plan.passes['placement']}, "
        f"{len(plan.changes)} changes -> design/"
    )
    lines.append(f"  lower     tasks.json ({len(f.tasks.steps)} steps)")
    lines.append(f"  gen       {hjson}, {m.program}, data.h (seed {check['seed']}) -> gen/")
    lines.append(f"  build     {f.build} (snax-gvsoc {f.checkout.commit[:8]}"
                 f"{', dirty' if f.checkout.dirty else ''})")  # fmt: skip
    lines.append(f"  sw        sw/{ELF} (sha256 {f.elf_sha256[:16]})")
    lines.append(f"  gvsoc     {TARGET}, exit {check['exit_code']} -> run/")
    verdict = "passed" if f.passed else "FAILED"
    lines.append(
        f"  check     {verdict}: {f.total_cycles} cycles (main to forge_check); data "
        f"{check['data_errors']} errors ({check['words']} words); counters {tasks}"
    )
    if f.trace is None:
        conv = "run/events.log -> profile.json (the summary lines; trace off)"
    else:
        conv = (
            f"run/events.log -> profile.json, trace.jsonl ({len(f.trace.events)} events in "
            f"the window; boot {f.trace.boot}"
            + (f"; sources {', '.join(f.trace_sources)}" if f.trace_sources else "")
            + ")"
        )
    if f.vcd:
        conv += f"; run/{VCD}, run/{GTKW} (gtkwave)"
    lines.append(f"  convert   {conv}")
    lines.append(f"  report    {REPORT}/design.md, {REPORT}/run.md")
    lines += [f"  WARNING   {w}" for w in f.warnings]
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
    """Add the wall-clock seconds of the block to ``times[stage]``."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        times[stage] = times.get(stage, 0.0) + time.perf_counter() - t0


def _keep_vcd(run: Path) -> None:
    """gvrun's VCD and its gtkwave view, from the work directory into ``run/``, the view
    pointing at the VCD's new place (its map files stay in ``work/``)."""
    work = run / "work"
    if not (work / VCD).is_file() or not (work / GTKW).is_file():
        raise FlowError(f"gvsoc: --vcd, but no {VCD} and {GTKW} in {cwd_relative(work)}")
    (work / VCD).replace(run / VCD)
    view = (work / GTKW).read_text()
    view = view.replace(
        f'[dumpfile] "{(work / VCD).resolve()}"', f'[dumpfile] "{(run / VCD).resolve()}"'
    )
    (run / GTKW).write_text(view)
    (work / GTKW).unlink()


def _clear(out: Path) -> None:
    """Remove what an earlier run left in ``out`` (module doc)."""
    for p in (out / LOG, out / REPORT / "design.md", out / REPORT / "run.md"):
        p.unlink(missing_ok=True)
    for d in (out / "gen", out / "run"):
        shutil.rmtree(d, ignore_errors=True)


def _clear_sandbox(out: Path, graph_path: Path) -> None:
    """Remove what an earlier flow's sandbox stage left in ``out`` (its recipe and step
    graphs), unless ``graph_path`` is one of them (module doc)."""
    folder = out / "sandbox"
    if not folder.is_dir() or folder.resolve() == graph_path.resolve().parent:
        return
    for p in [folder / "recipe.json", *folder.glob("*.snaxdfg")]:
        p.unlink(missing_ok=True)
    if not any(folder.iterdir()):
        folder.rmdir()


def _trace(level: str, sources: Sequence[str] | None) -> None:
    if level not in LEVELS:
        raise FlowError(f"trace level {level!r}: one of {', '.join(LEVELS)}")
    if sources is not None and not at_least(level, "beat"):
        raise FlowError(f"trace sources {list(sources)}: they go with beat, stall or finegrained")


FAILURES = (DesignError, FlowError, SandboxError, DfgError, LowerError, RunError, OSError)


def _folder(name: str, out: str | Path | None) -> Path:
    """The flow's folder (``out``, else ``out/flow/<name>``), made and cleared."""
    folder = Path(out) if out is not None else OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    _clear(folder)
    return folder


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
    trace_level: str = "off",
    trace_sources: Sequence[str] | None = None,
    vcd: bool = False,
) -> Flow:
    """Run the whole path; see the module doc. Raises FlowError or DesignError."""
    _trace(trace_level, trace_sources)
    recipe_sets = dict(recipe_sets or {})
    recipe = Recipe.load(recipe_path).with_params(recipe_sets)
    # The design point, and with it the task list, is named after the recipe and its params
    # only; the folder carries every --set (D136).
    point_name = name or default_name(recipe, recipe_sets)
    name = name or with_level(default_name(recipe, recipe_sets, platform_sets, memory_sets),
                              trace_level)  # fmt: skip
    out = _folder(name, out)
    times: dict[str, float] = {}

    def stages() -> Flow:
        # SNAX-SANDBOX
        with _timed(times, "import"):
            if graph_path is not None:
                graph = Graph.load(graph_path)
            else:
                from snax_forge.dfg.import_sdfg import import_kernel

                graph = import_kernel(recipe.kernel, recipe.dtypes)
        with _timed(times, "sandbox"):
            steps = write_steps(recipe, apply_recipe(recipe, graph, seed=seed), out / "sandbox")
        f = _from_graph(
            steps[-1], recipe.kernel, f"recipe {recipe.name}", name, point_name, out,
            platform_path, platform_sets, memory_path, memory_sets, seed, trace_level, times,
            trace_sources, vcd,
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
    trace_level: str = "off",
    trace_sources: Sequence[str] | None = None,
    vcd: bool = False,
) -> Flow:
    """The path from a bound ``.snaxdfg`` on, without the sandbox; see the
    module doc. ``kernel`` names the kernel when the graph's ``name`` does not. Raises
    FlowError or DesignError."""
    _trace(trace_level, trace_sources)
    graph_path = Path(graph_path)
    # As in run_flow: the design point keeps the graph's name, the folder carries every --set.
    point_name = name or bound_name(graph_path)
    name = name or with_level(bound_name(graph_path, platform_sets, memory_sets), trace_level)
    out = _folder(name, out)
    _clear_sandbox(out, graph_path)
    times: dict[str, float] = {}

    def stages() -> Flow:
        why = "--kernel" if kernel is not None else f"the name of {graph_path}; --kernel K names it"
        return _from_graph(
            graph_path, kernel, why, name, point_name, out, platform_path, platform_sets,
            memory_path, memory_sets, seed, trace_level, times, trace_sources, vcd,
        )  # fmt: skip

    return _finish(stages, name, out, times)


def _finish(stages: Callable[[], Flow], name: str, out: Path, times: dict[str, float]) -> Flow:
    """Run the stages, then write the log; a failing stage is logged."""
    try:
        f = stages()
    except FAILURES as e:
        (out / LOG).write_text(f"flow {name} FAILED\n{e}\n")
        raise
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
    trace_sources: Sequence[str] | None = None,
    vcd: bool = False,
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

    # SNAX-LOWER: the task list only; the generator lowers it to main.c
    with _timed(times, "lower"):
        tasks = task_list(point)
        tasks.save(out / "tasks.json")

    # gen: the hjson files, the headers, data.h and main.c, named from the flow's folder
    with _timed(times, "gen"):
        try:
            spec = kernel_spec(kernel or point.graph.name, kernel_why)
            reference_check(spec, point.graph, seed)
        except KernelError as e:
            raise FlowError(str(e)) from None
        gen = out / "gen"
        try:
            manifest = generate(ddir / "design_point.json", gen, ROOT, named_from=out,
                                kernel=spec.name, seed=seed)  # fmt: skip
        except (GenError, CError) as e:
            raise FlowError(f"gen: {e}") from None

    # build: SNAX-GVSoC, when needed
    with _timed(times, "build"):
        c = checkout()
        why = build_reason(c, GVSOC_WORKDIR)
        if why is not None:
            print(f"  build     building SNAX-GVSoC: {why}; log in "
                  f"{cwd_relative(out / 'build.log')}", flush=True)  # fmt: skip
            _make("build", "build", {}, out / "build.log", live=True)
            if not (GVSOC_WORKDIR / "install" / "bin" / "gvrun").is_file():
                raise FlowError(f"build: no GVSoC install in {cwd_relative(GVSOC_WORKDIR)} after "
                                f"the build; see {cwd_relative(out / 'build.log')}")  # fmt: skip
        build = "up to date" if why is None else f"built: {why}"

    # sw: the program, by snax-gvsoc's generic app
    sw = out / "sw"
    with _timed(times, "sw"):
        variables = {"APPS": "forge", "FORGE": str(gen.resolve()), "DESIGN": name,
                     "FORGE_BUILD": str(sw.resolve())}  # fmt: skip
        _make("sw", "sw", variables, out / "sw.log")
        elf = sw / ELF
        if not elf.is_file():
            raise FlowError(f"sw: no {cwd_relative(elf)} after make sw; see "
                            f"{cwd_relative(out / 'sw.log')}")  # fmt: skip
        sha = hashlib.sha256(elf.read_bytes()).hexdigest()
        try:
            elf_symbols = l2_symbols(point, read_symbols(elf))
        except RunError as e:
            raise FlowError(f"sw: {cwd_relative(elf)}: {e}") from None

    # gvsoc: the run on snax_hjson with gen/'s cluster file, its traces from the cluster
    run = out / "run"
    with _timed(times, "gvsoc"):
        try:
            cluster = load_cluster(gen / manifest.cluster, ddir / "design_point.json")
            sources = sources_for(cluster, trace_level,
                                  list(trace_sources) if trace_sources is not None else None)  # fmt: skip
        except RunError as e:
            raise FlowError(f"gvsoc: {e}") from None
        opts = gvsoc_options(cluster, trace_level, sources, vcd, trace_sources is not None)
        args = " ".join([f"--param hjson={(gen / manifest.cluster).resolve()}",
                         *(f"'{a}'" for a in opts.args)])  # fmt: skip
        variables = {
            "TARGET": TARGET, "DESIGN": name, "ELF": str(elf.resolve()), "GVRUN_ARGS": args,
            "WORK_DIR": str((run / "work").resolve()),
            "EVENTS": str((run / EVENTS_FILE).resolve()), "TRACES": " ".join(opts.traces),
        }  # fmt: skip
        rc = run_gvsoc("run", variables, run / "stdout.txt", run / "stderr.txt")
        exit_code = run_exit_code(rc, (run / "stderr.txt").read_text())
        if vcd:
            _keep_vcd(run)

    # check: the program's own
    with _timed(times, "check"):
        cycles, result = check_output((run / "stdout.txt").read_text(), exit_code,
                                      {t.name: t.beats for t in manifest.tasks})  # fmt: skip
        groups = {u.name: u.group for u in cluster.accelerator.units}
        units = {s.task_name: s.component for s in tasks.steps if isinstance(s, Configure)}
        for t, x in result["tasks"].items():  # each task's group (D148)
            x["group"] = groups[units[t]]
        check = {
            "kernel": spec.name,
            "seed": seed,
            "symbols": point.graph.bound_symbols,
            **result,
        }
        hjson = description_files(manifest.files)
        record = {
            "design_point": "design/design_point.json",
            "hjson": [f"gen/{x}" for x in hjson],
            "elf": f"sw/{ELF}",
            "elf_sha256": sha,
            "elf_symbols": elf_symbols,
            "target": TARGET,
            "snax_gvsoc": {"commit": c.commit, "dirty": c.dirty},
            "trace_level": trace_level,
            "trace_sources": sources,
            "vcd": vcd,
        }
        (run / "run.json").write_text(to_json(record))

    # convert: the events into the profile and the trace (D137)
    with _timed(times, "convert"):
        events = run / EVENTS_FILE
        if not events.is_file():
            raise FlowError(f"convert: no {cwd_relative(events)} after the run; see "
                            f"{cwd_relative(run / 'stderr.txt')}")  # fmt: skip
        try:
            profile, trace = convert(read_events(events), cluster, tasks, cycles, trace_level,
                                     check, regions_of(point, elf_symbols), sources)  # fmt: skip
        except RunError as e:
            raise FlowError(f"convert: {e}") from None
        write_run(run, profile, trace)

    # report: design.md and run.md from what convert wrote
    with _timed(times, "report"):
        from snax_forge.report import ReportError, write_reports

        try:
            write_reports(out)
        except ReportError as e:
            raise FlowError(f"report: {e}") from None
    return Flow(
        name, out, None, [], point, tasks, manifest, c, build, sha, cycles, check,
        checks_that_ran(), trace_level=trace_level, trace=trace, trace_sources=sources, vcd=vcd,
    )  # fmt: skip
