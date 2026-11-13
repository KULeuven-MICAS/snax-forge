"""The flow command line.

    python -m snax_forge.flow RECIPE --platform P [--set NAME=VALUE ...]
                              [--set platform.PATH=VALUE ...] [--set memory.PATH=VALUE ...]
                              [--memory M] [--graph FILE] [--name NAME] [--out DIR]
                              [--seed S] [--trace LEVEL[:SOURCES]] [--vcd] [--keep-work]
    python -m snax_forge.flow GRAPH.snaxdfg --platform P [--kernel K]
                              [--set platform.PATH=VALUE ...] [--set memory.PATH=VALUE ...]
                              [--memory M] [--name NAME] [--out DIR] [--seed S] [--trace ...]

Runs run.py's ``run_flow`` on a recipe, or ``run_bound`` on a file ending in ``.snaxdfg``, into
``out/flow/<name>/``. ``--set`` takes recipe params, platform fields and memory pins as the sandbox
and design commands do. ``--trace`` picks the level (``off``, ``task``, ``beat``, ``stall``,
``finegrained``) and, after a colon, the sources (``run/sources.py``); ``--vcd`` adds gtkwave's
VCD. With a graph, the kernel is its ``name`` unless ``--kernel`` names it, and a recipe param or
``--graph`` is refused, as ``--kernel`` is with a recipe. Exit 0 when the program's check passed,
1 when a stage or the check failed, 2 on bad arguments. pixi: ``flow``.
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

from .run import FlowError, run_bound, run_flow, split_sets, summary

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
    ap.add_argument("--keep-work", action="store_true",
                    help="keep gvrun's run/work/ after a passing run")  # fmt: skip
    args = ap.parse_args(argv)
    try:
        level, sources = parse_trace(args.trace)
    except RunError as e:
        ap.error(f"--trace: {e}")
    sets = split_sets(args.set)
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
        "keep_work": args.keep_work,
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
