"""SYN2's synthesized flops, the fixture ``tests/estimate/fixtures/syn2_flops.json`` (D154).

    pixi run python util/syn2_flops.py [--experiments DIR] [--out FILE]

Read once from snax-forge-experiments (default ``../snax-forge-experiments``, a clean
checkout): ``results/forge/nangate45/synth_dc.csv`` (each point's recipe, W and the snax-forge
commit SYN2 built it from) and ``synth_dc_modules.csv`` (each hierarchical cell's area), DC's
48 points of SYN2. A module's flops are its ``seq_um2`` / 5.32, nangate45's one flop (DFFR_X1,
5.32 µm²), which must be a whole number for ctrl and every block. Per point: ``point``,
``recipe``, ``W``, ``ctrl`` (``ctrl/ctrl``, the group's ``<tag>_<group>_ctrl``) and
``instances``, per datapath instance of ``dp`` (a unit's ``blocks_<k>_*``, a buffer's
``bufs_<k>_*``) by its name, the wrapper's module ``snax_forge_<name>`` without the tag:
``kind`` (``unit`` or ``link``), ``module`` (the block, its ``/m`` cell) and ``flops``. The file
records the experiments commit and the snax-forge commit; ``tests/estimate/test_bits.py``
compares it with the estimate's ``bits`` and ``ctrl_bits``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "estimate" / "fixtures" / "syn2_flops.json"
RESULTS = Path("results") / "forge" / "nangate45"
FLOP_UM2 = 5.32
TAG = "snax_forge_"
INSTANCE = re.compile(r"dp/(blocks|bufs)_\d+_\d+")


def flops(row: dict, what: str) -> int:
    n = float(row["seq_um2"]) / FLOP_UM2
    if abs(n - round(n)) > 1e-3:
        sys.exit(f"{row['point']} {what}: seq_um2 {row['seq_um2']} is not a whole number of flops")
    return round(n)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiments", type=Path, default=REPO.parent / "snax-forge-experiments")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    exp = args.experiments.resolve()
    if git(exp, "status", "--porcelain", "--", str(RESULTS)):
        sys.exit(f"{exp / RESULTS}: not clean")
    with (exp / RESULTS / "synth_dc.csv").open(newline="") as fh:
        top = {r["point"]: r for r in csv.DictReader(fh)}
    with (exp / RESULTS / "synth_dc_modules.csv").open(newline="") as fh:
        mods = list(csv.DictReader(fh))
    commits = {r["snax_forge_commit"] for r in top.values()}
    if len(commits) != 1:
        sys.exit(f"SYN2's points name {len(commits)} snax-forge commits: {sorted(commits)}")
    by = {(r["point"], r["path"]): r for r in mods}
    points = []
    for p, r in top.items():
        ctrls = [m for m in mods if m["point"] == p and re.fullmatch(r"ctrl/[^/]+", m["path"])]
        if len(ctrls) != 1:
            sys.exit(f"{p}: {len(ctrls)} group control units, one expected")
        inst = {}
        for m in mods:
            k = INSTANCE.fullmatch(m["path"]) if m["point"] == p else None
            if k:
                block = by[p, m["path"] + "/m"]
                inst[m["design"].removeprefix(TAG)] = {
                    "kind": "unit" if k.group(1) == "blocks" else "link",
                    "module": block["design"], "flops": flops(block, m["path"])}  # fmt: skip
        points.append({"point": p, "recipe": r["recipe"], "W": int(r["W"]),
                       "ctrl": flops(ctrls[0], ctrls[0]["path"]), "instances": inst})  # fmt: skip
    points.sort(key=lambda x: (x["recipe"], x["W"]))
    doc = {
        "source": {
            "snax_forge_experiments": git(exp, "rev-parse", "HEAD"),
            "files": [str(RESULTS / "synth_dc.csv"), str(RESULTS / "synth_dc_modules.csv")],
            "snax_forge": commits.pop(),
            "flop_um2": FLOP_UM2,
        },
        "points": points,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.out} ({len(points)} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
