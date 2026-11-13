"""HWCOST's tables from snax-forge-experiments into ``snax_forge/char/<tech>/``.

    pixi run import-costs [--experiments DIR] [--tech nangate45]

Copies ``results/ram/<tech>/ram_table.csv``, ``results/strm/<tech>/streamer.csv``,
``results/strm/<tech>/xbar.csv`` and ``results/strm/<tech>/xbar_kinds.json`` (the crossbar's
read-only and write-only inputs, X71) as they are, and writes ``SOURCE`` with the repository, its
commit and each file's path and sha256 (no date, so a re-run that brings nothing new changes
nothing). By default it fetches the latest main of
snax-forge-experiments (a shallow ``git clone`` into a temporary folder); ``--experiments DIR``
takes a local checkout instead, which must be clean. Run by hand when the tables change, never
in CI; a new commit shows in the PR's diff as a changed ``SOURCE`` and changed tables.
``tests/char/test_tables.py`` checks the files against ``SOURCE``.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
URL = "https://github.com/KULeuven-MICAS/snax-forge-experiments"
PATHS = {
    "ram_table.csv": "results/ram/{tech}/ram_table.csv",
    "streamer.csv": "results/strm/{tech}/streamer.csv",
    "xbar.csv": "results/strm/{tech}/xbar.csv",
    "xbar_kinds.json": "results/strm/{tech}/xbar_kinds.json",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(src: Path, tech: str, out: Path) -> str:
    """Copy the four files of ``tech`` from the checkout ``src`` to ``out`` and write SOURCE;
    returns the commit."""
    if git(src, "status", "--porcelain"):
        sys.exit(f"{src} has uncommitted changes: import from a clean checkout")
    commit = git(src, "rev-parse", "HEAD")
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HWCOST's tables, copied as they are by util/import_costs.py (pixi run import-costs)",
        f"repo: {URL}",
        f"commit: {commit}",
    ]
    for name, rel in PATHS.items():
        f = src / rel.format(tech=tech)
        if not f.is_file():
            sys.exit(f"{f} is missing in {src} at {commit[:8]}")
        shutil.copyfile(f, out / name)
        lines.append(f"{name}: {rel.format(tech=tech)} sha256 {sha256(out / name)}")
    (out / "SOURCE").write_text("\n".join(lines) + "\n")
    return commit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiments", type=Path, help="a clean local checkout (default: fetch main)")
    ap.add_argument("--tech", default="nangate45")
    a = ap.parse_args()
    out = REPO / "snax_forge" / "char" / a.tech
    if a.experiments:
        commit = copy(a.experiments.resolve(), a.tech, out)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "clone", "-q", "--depth", "1", URL, tmp], check=True)
            commit = copy(Path(tmp), a.tech, out)
    print(f"{out.relative_to(REPO)}: {', '.join(PATHS)} at {commit[:8]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
