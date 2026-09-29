"""The flow command line (E2E1).

    python -m snax_forge.flow RECIPE --platform P [--set NAME=VALUE ...]
                              [--set platform.PATH=VALUE ...] [--set memory.PATH=VALUE ...]
                              [--memory M] [--graph FILE] [--name NAME] [--out DIR]
                              [--seed S] [--trace off|task|beat]  (default task)

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
movement views need ``--trace beat`` (D94).

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

from .run import Flow, FlowError, run_flow


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


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def summary(f: Flow) -> str:
    p, check = f.point, f.check
    params = ", ".join(f"{k}={v}" for k, v in f.recipe.params.items())
    syms = ", ".join(f"{k}={v}" for k, v in check["symbols"].items())
    plan = f.point.memory
    inputs = ", ".join(m.npy for m in f.scenario.memory)
    lines = [
        f"flow {f.name} ({f.recipe.name}: {params}; {syms}) on {p.platform.name} -> {_rel(f.out)}/"
    ]
    lines.append(
        f"  sandbox   {len(f.steps) - 1} steps, each equal to the input graph on the reference "
        "check -> sandbox/"
    )
    lines.append(
        f"  design    checks passed; platform base {p.platform.base}, "
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
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.flow")
    ap.add_argument("recipe", type=Path)
    ap.add_argument("--platform", type=Path, required=True)
    ap.add_argument("--set", type=_setting, action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--memory", type=Path, help="memory working copy to start from")
    ap.add_argument("--graph", type=Path, help="start from this .snaxdfg instead of the import")
    ap.add_argument("--name", help="flow name (default: recipe name and every --set)")
    ap.add_argument("--out", type=Path, help="output directory (default out/flow/<name>)")
    ap.add_argument("--seed", type=int, default=0, help="seed of make_inputs and the checks")
    ap.add_argument("--trace", default="task", choices=["off", "task", "beat"])
    args = ap.parse_args(argv)
    sets: dict[str, list[tuple[str, Any]]] = {"recipe": [], "platform": [], "memory": []}
    for head, path, value in args.set:
        sets[head].append((path, value))
    try:
        f = run_flow(
            args.recipe, args.platform,
            recipe_sets=dict(sets["recipe"]), platform_sets=sets["platform"],
            memory_path=args.memory, memory_sets=sets["memory"], graph_path=args.graph,
            name=args.name, out=args.out, seed=args.seed, trace_level=args.trace,
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
