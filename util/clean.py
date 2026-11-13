"""Remove every generated file, so the repository starts from a clean slate.

    python util/clean.py             remove them (pixi run clean)
    python util/clean.py --all       the GVSoC build and the editor's sessions too
    python util/clean.py --dry-run   only list what would go (with --all or without)

What goes (``targets``): ``out/`` but ``out/snax-gvsoc/`` and ``out/sessions/`` (those two too
with ``--all``: GVSoC's build, minutes to make again, and the editor's sessions, which no
command makes again), a stray top-level ``.dacecache/``, every ``__pycache__/``, the pytest and
ruff caches, and the Chisel build trees of ``pixi run chisel-clean``. Tracked files, ``.pixi/``
and caches outside the repository stay. The list is explicit rather than ``git clean -X``,
which would also remove ``.pixi/``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories removed whole, relative to the repository root; out/ is OUT.
DIRS = [
    ".dacecache",
    ".pytest_cache",
    ".ruff_cache",
    # as pixi run chisel-clean (out/hw is already under out/)
    "hw/chisel/target",
    "hw/chisel/project/target",
    "hw/chisel/project/project",
    "hw/chisel/test_run_dir",
]

# Generated files, and what in it only --all removes: the GVSoC build and the editor's sessions
OUT = "out"
GVSOC_BUILD = "out/snax-gvsoc"
SESSIONS = "out/sessions"
KEPT = (GVSOC_BUILD, SESSIONS)

# Never walked into when looking for __pycache__.
SKIP = {".git", ".pixi"}


def targets(root: Path = ROOT, everything: bool = False) -> list[Path]:
    """Everything that would be removed, existing paths only, in a stable order: out/ whole with
    ``everything``, else out/ but KEPT."""
    out, kept = root / OUT, {root / k for k in KEPT}
    found = [root / d for d in DIRS if (root / d).exists()]
    if everything and out.exists():
        found.append(out)
    elif out.is_dir():
        found += [p for p in sorted(out.iterdir()) if p not in kept]
    stack = [root]
    while stack:
        d = stack.pop()
        for p in sorted(d.iterdir()):
            if not p.is_dir() or p.is_symlink() or p.name in SKIP or p in found or p in kept:
                continue
            if p.name == "__pycache__":
                found.append(p)
            else:
                stack.append(p)
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="list, remove nothing")
    ap.add_argument("--all", action="store_true", help=f"remove {GVSOC_BUILD}/ and {SESSIONS}/ too")
    args = ap.parse_args(argv)
    paths = targets(everything=args.all)
    for p in paths:
        rel = p.relative_to(ROOT)
        if args.dry_run:
            print(f"would remove {rel}")
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"removed {rel}")
    if not paths:
        print("nothing to remove")
    return 0


if __name__ == "__main__":
    sys.exit(main())
