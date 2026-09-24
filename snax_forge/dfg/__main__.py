"""SNAX-DFG command line (IMP1, REF1).

    python -m snax_forge.dfg import KERNEL [KERNEL ...] [--out DIR]
    python -m snax_forge.dfg import --sdfg PATH --name NAME [--out DIR]
    python -m snax_forge.dfg check FILE [FILE ...] --kernel KERNEL [--n N] [--seed S]

``import`` writes ``DIR/<kernel>.snaxdfg`` (default ``out/dfg``, D71). Given
kernel names it builds each kernel's simplified SDFG in-process, as ``pixi
run forge <kernel>`` does; given ``--sdfg`` it reads a stored ``.sdfg``
instead. Exit 0 when everything was written, 1 when a construct is not
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

from snax_forge.sdfg.paths import _repo_root

from .kinds import DfgError

OUT = _repo_root() / "out" / "dfg"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.dfg")
    sub = ap.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import", help="SDFG -> .snaxdfg (IMP1)")
    imp.add_argument("kernels", nargs="*", help="kernel names under kernels/")
    imp.add_argument("--sdfg", type=Path, help="import a stored .sdfg instead of building one")
    imp.add_argument("--name", help="graph name for --sdfg (default: the SDFG's name)")
    imp.add_argument("--out", type=Path, default=OUT, help="output directory (default out/dfg)")
    chk = sub.add_parser("check", help="run .snaxdfg files against the kernel reference (REF1)")
    chk.add_argument("files", nargs="+", type=Path)
    chk.add_argument(
        "--kernel", required=True, help="kernel under kernels/ (its inputs, reference)"
    )
    chk.add_argument("--n", type=int, help="size passed to make_inputs")
    chk.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.command == "check":
        return _check(args)
    if bool(args.kernels) == bool(args.sdfg):
        ap.error("import takes kernel names or --sdfg, not both and not neither")

    import dace  # slow; only once the arguments are good

    from .import_sdfg import import_kernel, import_sdfg

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = (
        [(k, lambda k=k: import_kernel(k)) for k in args.kernels]
        if args.kernels
        else [(args.name, lambda: import_sdfg(dace.SDFG.from_file(args.sdfg), args.name))]
    )
    failed = 0
    for label, make in jobs:
        try:
            g = make()
        except DfgError as e:
            print(f"FAILED {label or args.sdfg}: {e}", file=sys.stderr)
            failed += 1
            continue
        path = args.out / f"{g.name}.snaxdfg"
        g.save(path)
        print(path)
    return 1 if failed else 0


def _check(args: argparse.Namespace) -> int:
    import numpy as np

    from snax_forge.sdfg.loader import load

    from .execute import bind_symbols, execute
    from .graph import Graph

    spec = load(args.kernel)
    failed = 0
    for path in args.files:
        try:
            g = Graph.load(path)
            n = args.n
            bound = [v for v in g.symbols.values() if v is not None]
            if n is None and len(bound) == 1:
                n = bound[0]
            inputs = spec.make_inputs(
                np.random.default_rng(args.seed), **({} if n is None else {"n": n})
            )
            ref = {k: np.copy(v) for k, v in inputs.items()}
            spec.reference(**ref)
            out = execute(g, inputs)
        except (DfgError, OSError, ValueError) as e:
            print(f"FAILED {path}: {e}", file=sys.stderr)
            failed += 1
            continue
        bad = [k for k in spec.inout if not np.array_equal(out[k], ref[k])]
        sizes = ", ".join(f"{s} = {v}" for s, v in bind_symbols(g, inputs).items())
        if bad:
            print(f"MISMATCH {path}: {bad} differ from the {spec.name} reference ({sizes})")
            failed += 1
        else:
            print(f"OK {path}: {list(spec.inout)} equal the {spec.name} reference ({sizes})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
