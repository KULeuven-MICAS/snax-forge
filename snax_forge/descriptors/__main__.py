"""Emit hardware descriptors from raised SDFGs.

pixi run descriptors                       # every raised recipe
pixi run descriptors vecadd_loop
pixi run descriptors --list
pixi run descriptors vecadd_loop --print   # to stdout, nothing written
"""

import argparse
import json
import os
import sys

import dace

from snax_forge.libnodes.libnodes import SnaxVectorOp
from snax_forge.sdfg.paths import _repo_root
from snax_forge.sdfg.recipes import OUT as TRANSFORM_OUT

from .descriptors import OUT, emit, emit_recipe, validate


def raised_recipes() -> list[str]:
    """Recipes whose stored SDFG contains a raised library node.

    Loading each SDFG to find out is slower than reading the filename, and it
    is the only way to know: a recipe that stops at MapTiling produces a
    perfectly good .sdfg with nothing to describe.
    """
    found = []
    for path in sorted(TRANSFORM_OUT.glob("*.sdfg")):
        try:
            sdfg = dace.SDFG.from_file(str(path))
        except Exception as exc:  # noqa: BLE001 - from_file raises anything
            # Broad on purpose: out/transforms is a directory, not a curated
            # list, and a stale or foreign .sdfg is not a reason to abort the
            # scan. Reported rather than swallowed, because a file that should
            # have been readable failing here is otherwise invisible.
            print(f"  skipping {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if any(isinstance(n, SnaxVectorOp) for s in sdfg.states() for n in s.nodes()):
            found.append(path.stem)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(prog="snax-forge descriptors", description=__doc__)
    ap.add_argument("recipes", nargs="*", help="recipe names; empty = every raised recipe")
    ap.add_argument("--list", action="store_true", help="list recipes with a raised node")
    ap.add_argument("--print", action="store_true", help="write to stdout instead of a file")
    args = ap.parse_args()

    if not TRANSFORM_OUT.exists():
        print(f"{TRANSFORM_OUT} does not exist -- run: python -m snax_forge.sdfg --recipe")
        return 2

    available = raised_recipes()
    if args.list:
        print("raised recipes (each yields one descriptor):")
        for name in available:
            print(f"  {name}")
        return 0

    chosen = args.recipes or available
    if not chosen:
        print(f"no raised recipes in {TRANSFORM_OUT}; run a recipe with raise_vector_ops first")
        return 1

    failed = []
    for name in chosen:
        try:
            if args.print:
                sdfg = dace.SDFG.from_file(str(TRANSFORM_OUT / f"{name}.sdfg"))
                print(json.dumps(validate(emit(sdfg, recipe=name)), indent=2))
            else:
                path = emit_recipe(name)
                rel = path.relative_to(_repo_root())
                print(f"  {name:24} -> {rel}")
        # Broad, and deliberately so: this is the top of a batch command, and
        # one unemittable recipe must not stop the other five. Every failure
        # is reported and the exit code reflects them.
        except Exception as exc:  # noqa: BLE001 - batch driver, reports and continues
            print(f"  {name:24} FAILED: {exc}", file=sys.stderr)
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    if not args.print:
        print(f"\n{len(chosen)} descriptor(s) in {OUT.relative_to(_repo_root())}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
