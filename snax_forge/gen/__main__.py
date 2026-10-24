"""The generator's command line.

    python -m snax_forge.gen DESIGN_POINT [--out DIR]

Writes the three hjson files, the headers, ``data.h``, ``main.c`` and
``manifest.json`` of the design point (gvsoc.py) to ``DIR`` (default ``out/gvsoc/<design>/``) and
prints the cluster file and what to pass gvrun.

Exit 0 when written, 1 when refused (the rule named), 2 on bad arguments.
pixi: ``gen-gvsoc``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .gvsoc import OUT, cli, generate, out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.gen")
    ap.add_argument("design_point", type=Path, help="a design_point.json (the design step's)")
    ap.add_argument("--out", type=Path, help="output directory (default out/gvsoc/<design>)")
    args = ap.parse_args(argv)

    def write() -> int:
        out = out_dir(args.out, OUT, args.design_point)
        manifest = generate(args.design_point, out)
        print(f"cluster {manifest.cluster} from {args.design_point}")
        for t in manifest.tasks:
            print(f"  task {t.name}: {t.beats} beats, {t.csrs}")
        for f in manifest.files:
            print(f"wrote {out / f}")
        cluster = (out / manifest.cluster).resolve()
        print(
            f'run: make -C gvsoc run TARGET=snax_hjson GVRUN_ARGS="--param hjson={cluster}" ELF=...'
        )
        return 0

    return cli(write)


if __name__ == "__main__":
    sys.exit(main())
