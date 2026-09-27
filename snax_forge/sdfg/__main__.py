"""SDFG ingest (pixi ``forge``): each kernel's raw and simplified SDFG in out/sdfg/.

    python -m snax_forge.sdfg [KERNEL ...]   ingest the named kernels (all if none)
    python -m snax_forge.sdfg --list         list the kernels in kernels/

The simplified SDFG is the input of the SNAX-DFG importer (IMP1, D71, D78).
"""

import argparse
import json
import os
import sys

from .build import run
from .loader import kernel_paths, load


def main() -> int:
    ap = argparse.ArgumentParser(prog="snax-forge")
    ap.add_argument("kernels", nargs="*", help="kernel names; empty = all")
    ap.add_argument("--list", action="store_true", help="list available kernel names")
    args = ap.parse_args()

    if args.list:
        print("\n".join(sorted(kernel_paths())))
        return 0

    failed = []
    for name in args.kernels or sorted(kernel_paths()):
        try:
            print(json.dumps(run(load(name)), indent=2))
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"FAILED {name}: {exc!r}")
    return 1 if failed else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
