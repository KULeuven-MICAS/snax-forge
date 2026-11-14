"""The cost model's kernel energy against SYN2, the fixture ``tests/estimate/fixtures/char_model.json``.

    pixi run python util/char_model.py [--experiments DIR] [--out FILE]

Read once from snax-forge-experiments (default ``../snax-forge-experiments``, a clean
checkout): the Model section of ``results/char/char_table.md``, its table "Against SYN2" (vecmul,
vsum, dot 4 and dot 5 at W = 8 and 32, n = 2048, on nangate45 and nangate15). Per row: ``tech``,
``recipe`` (``vsum`` is ``vsum_tree``, ``dot 4`` ``dot_4``), ``W``, ``c_acc``, ``ctrl_bits``
and ``kernel_nj``, the model's kernel energy per task (datapath and ctrl). The file records the
experiments commit; ``tests/estimate/test_cost.py`` compares the estimate's energy with it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "estimate" / "fixtures" / "char_model.json"
TABLE = Path("results") / "char" / "char_table.md"
HEAD = "| tech | design | W | units, n_fire | C_acc | ctrl bits |"
RECIPES = {"vecmul": "vecmul", "vsum": "vsum_tree", "dot 4": "dot_4", "dot 5": "dot_5"}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiments", type=Path, default=REPO.parent / "snax-forge-experiments")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    exp = args.experiments.resolve()
    if git(exp, "status", "--porcelain", "--", str(TABLE)):
        sys.exit(f"{exp / TABLE}: not clean")
    lines = (exp / TABLE).read_text().splitlines()
    start = next((i for i, x in enumerate(lines) if x.startswith(HEAD)), None)
    if start is None:
        sys.exit(f"{TABLE}: no table starting {HEAD!r}")
    head = [c.strip() for c in lines[start].strip("|").split("|")]
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        r = dict(zip(head, (c.strip() for c in line.strip("|").split("|")), strict=True))
        rows.append({"tech": r["tech"], "recipe": RECIPES[r["design"]], "W": int(r["W"]),
                     "c_acc": int(r["C_acc"]), "ctrl_bits": int(r["ctrl bits"]),
                     "kernel_nj": float(r["kernel model"])})  # fmt: skip
    if len(rows) != 16:
        sys.exit(f"{TABLE}: {len(rows)} rows, 16 expected")
    doc = {"source": {"snax_forge_experiments": git(exp, "rev-parse", "HEAD"), "file": str(TABLE),
                      "table": "Model, Against SYN2: kernel model"}, "rows": rows}  # fmt: skip
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
