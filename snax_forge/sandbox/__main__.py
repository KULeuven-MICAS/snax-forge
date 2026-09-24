"""SNAX-SANDBOX command line (SBX1).

    python -m snax_forge.sandbox RECIPE [--graph FILE] [--set NAME=VALUE ...]
                                        [--out DIR] [--seed S]

Applies RECIPE to the import of its kernel (IMP1; ``--graph`` starts from a
``.snaxdfg`` file instead, without DaCe), checks every step against the
reference executor and writes ``DIR/<i>_<transform>.snaxdfg`` for every step
plus the recipe it ran, ``recipe.json`` (default ``out/sandbox/<recipe
name>``). ``--set W=8`` overrides a recipe param, so one recipe runs one
point of a sweep. Exit 0 when every step passed, 1 when a step failed (the
error names it), 2 on bad arguments. pixi: ``sandbox``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from snax_forge.dfg import DfgError, Graph
from snax_forge.sdfg.paths import _repo_root

from .recipe import Recipe, SandboxError
from .run import apply_recipe, write_steps

OUT = _repo_root() / "out" / "sandbox"


def _setting(text: str) -> tuple[str, int | str]:
    name, sep, value = text.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"{text!r} is not NAME=VALUE")
    try:
        return name, int(value)
    except ValueError:
        return name, value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.sandbox")
    ap.add_argument("recipe", type=Path)
    ap.add_argument("--graph", type=Path, help="start from this .snaxdfg instead of the import")
    ap.add_argument("--set", type=_setting, action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--out", type=Path, help="output directory (default out/sandbox/<name>)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the reference-check inputs")
    args = ap.parse_args(argv)

    try:
        recipe = Recipe.load(args.recipe).with_params(dict(args.set))
        if args.graph:
            graph = Graph.load(args.graph)
        else:
            from snax_forge.dfg.import_sdfg import import_kernel

            graph = import_kernel(recipe.kernel)
        results = apply_recipe(recipe, graph, seed=args.seed)
    except (SandboxError, DfgError, OSError) as e:
        print(f"FAILED {args.recipe}: {e}", file=sys.stderr)
        return 1
    out = args.out or OUT / recipe.name
    for path in write_steps(recipe, results, out):
        print(path)
    print(f"{len(results) - 1} steps, each equal to the input graph on the reference check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
