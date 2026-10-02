"""The flow command line (E2E1, FLOW2).

    python -m snax_forge.flow RECIPE --platform P [--set NAME=VALUE ...]
                              [--set platform.PATH=VALUE ...] [--set memory.PATH=VALUE ...]
                              [--memory M] [--graph FILE] [--name NAME] [--out DIR]
                              [--seed S] [--trace off|task|beat]  (default task)
    python -m snax_forge.flow GRAPH.snaxdfg --platform P [--kernel K]
                              [--set platform.PATH=VALUE ...] [--set memory.PATH=VALUE ...]
                              [--memory M] [--name NAME] [--out DIR] [--seed S] [--trace ...]

Runs the recipe on its kernel's import (or ``--graph``), pairs the result
with the platform, lowers the design point, runs it in SNAX-MODEL on the
kernel's ``make_inputs`` and checks the output against the kernel's
reference and REF1 (run.py). ``--set W=8`` sets a recipe param,
``--set platform.l1.n_banks=32`` a platform field and
``--set memory.B.l1.base=576`` pins a base, as the sandbox and design
commands take them. Everything goes to ``out/flow/<name>/``. The default
name is the recipe's with every ``--set`` appended: recipe params, then
platform and memory paths without their prefix, each in the order given
(``vecadd_W8``, ``vecadd_l1.n_banks32``, ``vecadd_B.l1.base576``). The run is
traced at ``task`` level unless ``--trace`` says otherwise; the data
movement views need ``--trace beat`` (D94). What it prints is also in
``flow.log``, with the design checks that ran, next to ``report/`` (D99).

The second form starts from a bound graph instead (D108): a file ending in
``.snaxdfg`` is the last step of a sandbox run, or a graph edited by hand,
and goes straight to the design step; any other file is a recipe. Nothing
else changes: the same checks, lowering, run and output check, in
``out/flow/<name>/``, where the default name is the design step's for the
graph (its sandbox folder, else the file's stem) with every ``--set``
appended. The kernel is the graph's ``name`` unless ``--kernel`` names it.
There is no recipe, so a recipe param (``--set W=8``) and ``--graph`` are
refused, as ``--kernel`` is with a recipe.

Exit 0 when the output matches, 1 when a stage fails (its problems named) or
the output does not match, 2 on bad arguments. pixi: ``flow``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from snax_forge.design import DesignError, parse_value
from snax_forge.dfg import DfgError
from snax_forge.lower import LowerError
from snax_forge.sandbox import SandboxError
from snax_forge.snax_model.scenario import ScenarioError

from .run import FlowError, run_bound, run_flow, summary

BOUND = ".snaxdfg"


def _setting(text: str) -> tuple[str, str, Any]:
    path, sep, value = text.partition("=")
    if not sep or not path:
        raise argparse.ArgumentTypeError(f"{text!r} is not NAME=VALUE")
    head, dot, rest = path.partition(".")
    if dot and head in ("platform", "memory") and rest:
        return head, rest, parse_value(value)
    if dot:
        raise argparse.ArgumentTypeError(
            f"{text!r}: a dotted path starts with platform. or memory."
        )
    try:
        return "recipe", path, int(value)
    except ValueError:
        return "recipe", path, value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.flow")
    ap.add_argument("recipe", type=Path, metavar="RECIPE|GRAPH.snaxdfg")
    ap.add_argument("--platform", type=Path, required=True)
    ap.add_argument("--set", type=_setting, action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--memory", type=Path, help="memory working copy to start from")
    ap.add_argument("--graph", type=Path, help="start from this .snaxdfg instead of the import")
    ap.add_argument("--kernel", help="with a bound graph: its kernel (default: the graph's name)")
    ap.add_argument("--name", help="flow name (default: recipe name and every --set)")
    ap.add_argument("--out", type=Path, help="output directory (default out/flow/<name>)")
    ap.add_argument("--seed", type=int, default=0, help="seed of make_inputs and the checks")
    ap.add_argument("--trace", default="task", choices=["off", "task", "beat"])
    args = ap.parse_args(argv)
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
        "trace_level": args.trace,
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
    except (FlowError, SandboxError, DfgError, LowerError, ScenarioError, OSError) as e:
        print(f"FAILED {args.recipe}: {e}", file=sys.stderr)
        return 1
    print(summary(f))
    return 0 if f.passed else 1


if __name__ == "__main__":
    sys.exit(main())
