"""SNAX-DFG command line.

    python -m snax_forge.dfg import KERNEL [KERNEL ...] [--out DIR]
    python -m snax_forge.dfg check FILE [FILE ...] --kernel KERNEL [--n N]

``import`` writes ``DIR/<kernel>.snaxdfg`` (default ``out/dfg``): it builds
each kernel's simplified SDFG in-process, as ``pixi run forge <kernel>``
does. Exit 0 when everything was written, 1 when a construct is not
supported (the error names it), 2 on bad arguments. pixi: ``import-dfg``.

``check`` runs each file in the reference executor (execute.py) on the
kernel's ``make_inputs`` and compares its output containers (the kernel's
``inout``) with the kernel's own reference. ``--n`` is the size given to
``make_inputs``; without it, the size a file binds (its one bound symbol),
else ``make_inputs``' default. Exit 0 when every file matches, 1 otherwise.
pixi: ``check-dfg``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from snax_forge.common import repo_root

from .kinds import DfgError

OUT = repo_root() / "out" / "dfg"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.dfg")
    sub = ap.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="SDFG -> .snaxdfg")
    imp.add_argument("kernels", nargs="+", help="kernel names under kernels/")
    imp.add_argument("--out", type=Path, default=OUT, help="output directory (default out/dfg)")
    chk = sub.add_parser("check", help="run .snaxdfg files against the kernel reference")
    chk.add_argument("files", nargs="+", type=Path)
    chk.add_argument(
        "--kernel", required=True, help="kernel under kernels/ (its inputs, reference)"
    )
    chk.add_argument("--n", type=int, help="size passed to make_inputs")
    args = ap.parse_args(argv)

    if args.command == "check":
        return _check(args)

    from .import_sdfg import import_kernel  # slow (DaCe); only once the arguments are good

    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for k in args.kernels:
        try:
            g = import_kernel(k)
        except DfgError as e:
            print(f"FAILED {k}: {e}", file=sys.stderr)
            failed += 1
            continue
        path = args.out / f"{g.name}.snaxdfg"
        g.save(path)
        print(path)
    return 1 if failed else 0


def _check(args: argparse.Namespace) -> int:
    import numpy as np

    from snax_forge.sdfg.loader import load

    from .execute import against_reference, bind_symbols, execute
    from .graph import Graph

    spec = load(args.kernel)
    failed = 0
    for path in args.files:
        try:
            g = Graph.load(path)
            n = args.n
            bound = list(g.bound_symbols.values())
            if n is None and len(bound) == 1:
                n = bound[0]
            inputs = spec.make_inputs(np.random.default_rng(0), **({} if n is None else {"n": n}))
            pairs = against_reference(spec, inputs, lambda x, g=g: execute(g, x))
        except (DfgError, OSError, ValueError) as e:
            print(f"FAILED {path}: {e}", file=sys.stderr)
            failed += 1
            continue
        bad = [k for k, (got, ref) in pairs.items() if not np.array_equal(got, ref)]
        sizes = ", ".join(f"{s} = {v}" for s, v in bind_symbols(g, inputs).items())
        if bad:
            print(f"MISMATCH {path}: {bad} differ from the {spec.name} reference ({sizes})")
            failed += 1
        else:
            print(f"OK {path}: {list(spec.inout)} equal the {spec.name} reference ({sizes})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
