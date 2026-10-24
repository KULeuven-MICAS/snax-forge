"""The flow command line.

    python -m snax_forge.flow RECIPE --platform P [--set NAME=VALUE ...]
                              [--set platform.PATH=VALUE ...] [--set memory.PATH=VALUE ...]
                              [--memory M] [--graph FILE] [--name NAME] [--out DIR]
                              [--seed S] [--trace LEVEL[:SOURCES]] [--vcd]
    python -m snax_forge.flow GRAPH.snaxdfg --platform P [--kernel K]
                              [--set platform.PATH=VALUE ...] [--set memory.PATH=VALUE ...]
                              [--memory M] [--name NAME] [--out DIR] [--seed S] [--trace ...]

Runs the recipe on its kernel's import (or ``--graph``), pairs the result
with the platform, lowers the design point to its task list, generates its
hjson files, headers, data (the kernel's ``make_inputs``, the reference
executor's goldens) and ``main.c``, builds SNAX-GVSoC when needed and the
program, runs it on ``snax_hjson`` and takes the program's own check: its
exit code and what it printed (run.py, D136). ``--set W=8`` sets a recipe param,
``--set platform.l1.n_banks=32`` a platform field and
``--set memory.B.l1.base=576`` pins a base, as the sandbox and design
commands take them. Everything goes to ``out/flow/<name>/``. The default
name is the recipe's with every ``--set`` appended: recipe params, then
platform and memory paths without their prefix, each in the order given
(``vecadd_W8``, ``vecadd_l1.n_banks32``, ``vecadd_B.l1.base576``), then the
trace level when it is not ``off``. ``--trace off`` (the default) gives the
profile from the components' counters, ``--trace task`` the trace of every
start, done, CSR access, wait and DMA copy too (D137, ``run/``), ``beat`` every
grant, DMA beat, firing and push, ``stall`` every refused request and queue
count, ``finegrained`` every instruction; ``--trace stall:acc,acc_a`` names the
sources of those lines (C§7 names: ``acc``, the movers, ``dma``, ``core<i>``,
``axi``; ``run/sources.py``). ``--vcd`` also writes gtkwave's VCD of them,
``run/all.vcd`` and ``run/view.gtkw``. What it prints is also in ``flow.log``,
with the design checks that ran.

The second form starts from a bound graph instead (D108): a file ending in
``.snaxdfg`` is the last step of a sandbox run, or a graph edited by hand,
and goes straight to the design step; any other file is a recipe. Nothing
else changes: the same checks, lowering, run and check, in
``out/flow/<name>/``, where the default name is the design step's for the
graph (its sandbox folder, else the file's stem) with every ``--set``
appended. The kernel is the graph's ``name`` unless ``--kernel`` names it.
There is no recipe, so a recipe param (``--set W=8``) and ``--graph`` are
refused, as ``--kernel`` is with a recipe.

Exit 0 when the program's check passed, 1 when a stage fails (its problems
named, a make failure with its log) or the check failed, 2 on bad arguments.
pixi: ``flow``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from snax_forge.design import DesignError
from snax_forge.design.__main__ import setting as design_setting
from snax_forge.dfg import DfgError
from snax_forge.lower import LowerError
from snax_forge.run import RunError
from snax_forge.run.sources import parse_trace
from snax_forge.sandbox import SandboxError
from snax_forge.sandbox.__main__ import setting as sandbox_setting

from .run import FlowError, run_bound, run_flow, summary

BOUND = ".snaxdfg"


def setting(text: str) -> tuple[str, str, Any]:
    """A dotted path as the design step reads it (platform. or memory.), else a recipe
    param as the sandbox reads it."""
    path = text.partition("=")[0]
    if "." not in path:
        return ("recipe", *sandbox_setting(text))
    if path.partition(".")[0] not in ("platform", "memory"):
        raise argparse.ArgumentTypeError(
            f"{text!r}: a dotted path starts with platform. or memory."
        )
    return design_setting(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.flow")
    ap.add_argument("recipe", type=Path, metavar="RECIPE|GRAPH.snaxdfg")
    ap.add_argument("--platform", type=Path, required=True)
    ap.add_argument("--set", type=setting, action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--memory", type=Path, help="memory working copy to start from")
    ap.add_argument("--graph", type=Path, help="start from this .snaxdfg instead of the import")
    ap.add_argument("--kernel", help="with a bound graph: its kernel (default: the graph's name)")
    ap.add_argument("--name", help="flow name (default: recipe name and every --set)")
    ap.add_argument("--out", type=Path, help="output directory (default out/flow/<name>)")
    ap.add_argument("--seed", type=int, default=0, help="seed of make_inputs and the checks")
    ap.add_argument("--trace", default="off", metavar="LEVEL[:SOURCES]",
                    help="off, task, beat, stall or finegrained, and from beat on the sources "
                    "of the beat lines (acc,acc_a,dma,core0,...)")  # fmt: skip
    ap.add_argument("--vcd", action="store_true",
                    help="also write gtkwave's VCD of the sources into run/")  # fmt: skip
    args = ap.parse_args(argv)
    try:
        level, sources = parse_trace(args.trace)
    except RunError as e:
        ap.error(f"--trace: {e}")
    sets: dict[str, list[tuple[str, Any]]] = {"recipe": [], "platform": [], "memory": []}
    for head, path, value in args.set:
        sets[head].append((path, value))
    bound = args.recipe.suffix == BOUND
    if bound and sets["recipe"]:
        ap.error(
            f"--set {sets['recipe'][0][0]}=...: {args.recipe} is a bound graph, it has no recipe "
            "params (run the recipe, or --set platform.PATH / memory.PATH)"
        )
    if bound and args.graph is not None:
        ap.error(f"--graph: {args.recipe} is a bound graph already (--graph goes with a recipe)")
    if not bound and args.kernel is not None:
        ap.error(f"--kernel: {args.recipe} is a recipe, which names its kernel")
    common: dict[str, Any] = {
        "platform_sets": sets["platform"],
        "memory_path": args.memory,
        "memory_sets": sets["memory"],
        "name": args.name,
        "out": args.out,
        "seed": args.seed,
        "trace_level": level,
        "trace_sources": sources,
        "vcd": args.vcd,
    }
    try:
        if bound:
            f = run_bound(args.recipe, args.platform, kernel=args.kernel, **common)
        else:
            f = run_flow(
                args.recipe, args.platform,
                recipe_sets=dict(sets["recipe"]), graph_path=args.graph, **common,
            )  # fmt: skip
    except DesignError as e:
        print(e, file=sys.stderr)
        return 1
    except (FlowError, SandboxError, DfgError, LowerError, OSError) as e:
        print(f"FAILED {args.recipe}: {e}", file=sys.stderr)
        return 1
    print(summary(f))
    return 0 if f.passed else 1


if __name__ == "__main__":
    sys.exit(main())
