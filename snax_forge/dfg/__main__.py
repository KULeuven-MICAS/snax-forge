"""SNAX-DFG command line (IMP1).

    python -m snax_forge.dfg import KERNEL [KERNEL ...] [--out DIR]
    python -m snax_forge.dfg import --sdfg PATH --name NAME [--out DIR]

``import`` writes ``DIR/<kernel>.snaxdfg`` (default ``out/dfg``, D71). Given
kernel names it builds each kernel's simplified SDFG in-process, as ``pixi
run forge <kernel>`` does; given ``--sdfg`` it reads a stored ``.sdfg``
instead. Exit 0 when everything was written, 1 when a construct is not
supported (the error names it), 2 on bad arguments. pixi: ``import-dfg``.
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
    args = ap.parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())
