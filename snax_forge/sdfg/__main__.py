import argparse
import json
import os
import sys
from pathlib import Path

from .build import run
from .loader import kernel_paths, load


def parse_size(tok: str):
    """'4096' -> 4096;  'm=256,n=512' -> {'m': 256, 'n': 512}"""
    if "=" in tok:
        return {k: int(v) for k, v in (p.split("=") for p in tok.split(","))}
    return int(tok)


def main() -> int:
    ap = argparse.ArgumentParser(prog="snax-forge")
    ap.add_argument("kernels", nargs="*", help="kernel names; empty = all")
    ap.add_argument("--list", action="store_true", help="list available kernel names")
    ap.add_argument("--profile", action="store_true", help="run a size sweep instead of ingest")
    ap.add_argument("--sizes", type=parse_size, nargs="+", help="sweep sizes; int or m=256,n=512")
    ap.add_argument("--reps", type=int, default=50, help="repetitions per size (default: 50)")
    ap.add_argument(
        "--instrument", action="store_true", help="also emit per-state DaCe timers (tier 1)"
    )
    ap.add_argument(
        "--transforms", action="store_true", help="list applicable transformations (applies none)"
    )
    ap.add_argument(
        "--recipe",
        nargs="*",
        metavar="NAME",
        help="apply recipes from transforms/; bare flag = all",
    )
    ap.add_argument("--list-recipes", action="store_true", help="list available recipe names")
    ap.add_argument(
        "--no-verify", action="store_true", help="skip per-step bit-exact check (faster)"
    )
    ap.add_argument(
        "--sdfg",
        nargs="+",
        metavar="PATH",
        help="profile stored .sdfg files; two or more are compared side by side",
    )
    ap.add_argument(
        "--symbols",
        type=parse_size,
        nargs="+",
        metavar="SPEC",
        help="symbol bindings for --sdfg, e.g. N=4096 N=65536 (bare ints ok if "
        "the SDFG has exactly one free symbol)",
    )
    args = ap.parse_args()

    if args.list:
        print("\n".join(sorted(kernel_paths())))
        return 0

    # --- recipe mode: iterates RECIPES, not kernels -----------------------
    if args.list_recipes or args.recipe is not None:
        from .recipes import apply_recipe, load_recipe, recipe_paths

        if args.list_recipes:
            print("\n".join(sorted(recipe_paths())))
            return 0

        failed = []
        for recipe_name in args.recipe or sorted(recipe_paths()):
            try:
                apply_recipe(load_recipe(recipe_name), verify_each=not args.no_verify)
            except Exception as exc:  # noqa: BLE001
                failed.append(recipe_name)
                print(f"FAILED {recipe_name}: {exc!r}")
        return 1 if failed else 0

    # --- stored-SDFG mode: iterates PATHS, not kernels --------------------
    if args.sdfg:
        import dace

        from .profile import compare_sdfgs, sweep_sdfg

        # Load once; sweep_sdfg/compare_sdfgs accept objects as well as paths.
        variants = {Path(p).stem: dace.SDFG.from_file(p) for p in args.sdfg}
        free = sorted({str(s) for g in variants.values() for s in g.free_symbols})

        if not args.symbols:
            print(f"--symbols is required. These SDFGs need: {free or '(none)'}")
            print(f"  e.g.  --symbols {' '.join(f'{s}=4096' for s in free) or '4096'}")
            return 2

        def as_symbols(entry):
            """Bare ints are only unambiguous when there is one free symbol."""
            if isinstance(entry, dict):
                return entry
            if len(free) == 1:
                return {free[0]: int(entry)}
            raise SystemExit(f"bare size {entry!r} is ambiguous; SDFG needs {free}")

        symbol_sets = [as_symbols(s) for s in args.symbols]
        kw = {"symbol_sets": symbol_sets, "reps": args.reps, "instrument": args.instrument}
        if len(variants) == 1:
            label, sdfg = next(iter(variants.items()))
            sweep_sdfg(sdfg, label=label, **kw)
        else:
            compare_sdfgs(variants, **kw)
        return 0

    # --- kernel mode ------------------------------------------------------
    names = args.kernels or sorted(kernel_paths())
    failed = []
    for name in names:
        try:
            if args.profile:
                from .profile import sweep

                sweep(
                    load(name),
                    sizes=args.sizes,
                    reps=args.reps,
                    instrument=args.instrument,
                )
            elif args.transforms:
                from .recipes import report_transformations

                report_transformations(load(name))
            else:
                report = run(load(name))
                print(json.dumps(report, indent=2))

        except Exception as exc:  # noqa: BLE001
            failed.append((name, repr(exc)))
            print(f"FAILED {name}: {exc!r}")

    return 1 if failed else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
