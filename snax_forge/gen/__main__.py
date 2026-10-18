"""The GVSoC generator's command line (GEN3 step 3).

    python -m snax_forge.gen DESIGN_POINT [--out DIR] [--install DIR]

Writes the composite, the target and ``manifest.json`` of the design point
(gvsoc.py) to ``DIR`` (default ``out/gvsoc/<design>/``) and prints the
target and what to pass gvrun. ``--install`` is snax-gvsoc's install
folder, checked for names the generated modules would be shadowed by and
for the modules they import; default ``out/snax-gvsoc/install`` when it
exists, else no check.

Exit 0 when written, 1 when refused (the rule named), 2 on bad arguments.
pixi: ``gen-gvsoc``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from snax_forge.design import DesignError
from snax_forge.design.point import DesignPoint

from .blocks import GenError
from .gvsoc import INSTALL, OUT, generate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.gen")
    ap.add_argument("design_point", type=Path, help="a design_point.json (the design step's)")
    ap.add_argument("--out", type=Path, help="output directory (default out/gvsoc/<design>)")
    ap.add_argument("--install", type=Path, help="snax-gvsoc's install folder to check against")
    args = ap.parse_args(argv)
    install = args.install or (INSTALL if INSTALL.is_dir() else None)
    try:
        out = args.out or OUT / DesignPoint.load(args.design_point).name
        manifest = generate(args.design_point, out, install)
    except (GenError, DesignError) as e:
        print(e, file=sys.stderr)
        return 1
    print(f"target {manifest.target} from {args.design_point}"
          + (f", checked against {install}" if install else ", no install checked"))  # fmt: skip
    for t in manifest.tasks:
        print(f"  task {t.name}: {t.beats} beats, {t.csrs}")
    for f in manifest.files:
        print(f"wrote {out / f}")
    print(f"run: make -C gvsoc run TARGET={manifest.target} TARGET_DIRS={out.resolve()} ELF=...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
