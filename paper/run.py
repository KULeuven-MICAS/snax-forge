"""The paper's data: a set of design points through the tool, into one CSV.

    pixi run paper SET [--run] [-j N] [--only ID ...]

``SET`` names ``paper/<set>.json``: a platform, ``--set``s for every point, and the points, each
a recipe with its own ``--set``s and the labels the figures use. Every point gets the design
step and the ideal estimate in this process (its cycles, and area and energy by part,
CONTRACTS "Estimate" and "Cost model"). With ``--run`` each point also runs through ``pixi run
flow --energy`` (``-j`` at once, into ``out/paper/<set>/<id>/``), adding the run's window and
dynamic energy by part (CONTRACTS, "The energy run"). The rows go to ``paper/data/<set>.csv``
in the set's order (with ``--only``, those points' rows replaced in the file as it is); a point
that fails keeps its row with the reason in ``status``. A point with ``"run": false`` gets the
estimate only (one SNAX-GVSoC cannot run, such as a crossbar above its inputs); without ``--run``
a row keeps its run columns from the file while its recipe and sets are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from snax_forge.design import DesignError, DesignPoint, Platform, load, run_checks
from snax_forge.design.check import xbar_inputs
from snax_forge.dfg import DfgError
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.estimate import estimate
from snax_forge.flow.__main__ import setting
from snax_forge.flow.run import split_sets
from snax_forge.sandbox import Recipe, SandboxError, apply_recipe

PAPER = ROOT / "paper"
OUT = ROOT / "out" / "paper"
PARTS = ("datapath", "control", "streamers", "crossbar", "memory")
LABELS = ("set", "id", "kernel", "branch", "label", "row", "layout", "W", "banks", "schedule",
          "recipe", "sets")  # fmt: skip
STATIC = ("est_cycles", "est_accelerator_cycles", "est_copy_cycles", "est_bank_bound", "xbar_inputs",
          *(f"static_{p}_nj" for p in PARTS), "static_nj",
          *(f"area_{p}_um2" for p in PARTS), "area_um2", "area_without_banks_um2")  # fmt: skip
RUN = ("cycles", *(f"dynamic_{p}_nj" for p in PARTS), "dynamic_nj", "passed")
COLUMNS = (*LABELS, *STATIC, *RUN, "status")


@dataclass
class Point:
    """One point of a set: its labels, its recipe and every ``--set`` (the set's, then its
    own)."""

    set: str
    id: str
    recipe: Path
    sets: list[str]
    labels: dict[str, Any] = field(default_factory=dict)
    run: bool = True


def load_set(name: str) -> tuple[Path, list[Point]]:
    """``paper/<name>.json``: its platform and points."""
    d = json.loads((PAPER / f"{name}.json").read_text())
    common = list(d.get("sets", []))
    points = []
    for p in d["points"]:
        labels = {k: v for k, v in p.items() if k not in ("id", "recipe", "sets", "run")}
        points.append(Point(d["name"], p["id"], ROOT / p["recipe"], common + list(p["sets"]),
                            labels, p.get("run", True)))  # fmt: skip
    ids = [p.id for p in points]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"paper/{name}.json: an id is used twice")
    return ROOT / d["platform"], points


def static_row(point: Point, platform: Path) -> dict[str, Any]:
    """The design point's ideal estimate: cycles, and area and energy by part."""
    by = split_sets([setting(s) for s in point.sets])
    recipe = Recipe.load(point.recipe).with_params(dict(by["recipe"]))
    graph = import_kernel(recipe.kernel, recipe.dtypes, recipe.types)
    last = apply_recipe(recipe, graph)[-1].graph
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bound.snaxdfg"
        last.save(path)
        design = load(path, platform, by["platform"], None, by["memory"], by["design"])
        problems = run_checks(design)
        if problems:
            raise DesignError(problems, f"{point.id}: design check")
        dp = DesignPoint.of(design, recipe.name)
        counted = xbar_inputs(design)
    est = estimate(dp.graph, dp.platform, schedule=dp.schedule).to_dict()
    cost = est["cost"]
    row = {"est_cycles": est["total_cycles"], "est_accelerator_cycles": est["accelerator_cycles"],
           "est_copy_cycles": est["copy_cycles"], "static_nj": cost["energy"],
           "est_bank_bound": max((g["bank_bound"] or 0 for g in est["groups"]), default=0),
           "area_um2": cost["area"], "area_without_banks_um2": cost["area_design"],
           "xbar_inputs": sum(counted) if counted else 0}  # fmt: skip
    for p in PARTS:
        row[f"static_{p}_nj"] = cost["parts"][p]["energy"]
        row[f"area_{p}_um2"] = cost["parts"][p]["area"]
    return row


def run_row(point: Point, platform: Path) -> dict[str, Any]:
    """The flow with ``--energy``: the window and the dynamic energy by part."""
    out = OUT / point.set / point.id
    cmd = [sys.executable, "-m", "snax_forge.flow", str(point.recipe), "--platform",
           str(platform), "--name", point.id, "--out", str(out), "--energy"]  # fmt: skip
    for s in point.sets:
        cmd += ["--set", s]
    done = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    if done.returncode:
        tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
        return {"passed": False, "status": f"flow failed: {tail[0][:200]}"}
    profile = json.loads((out / "run" / "profile.json").read_text())
    energy = json.loads((out / "energy" / "energy.json").read_text())
    row = {"cycles": profile["total_cycles"], "passed": True,
           "dynamic_nj": sum(energy["total"][p] for p in PARTS)}  # fmt: skip
    for p in PARTS:
        row[f"dynamic_{p}_nj"] = energy["total"][p]
    return row


def row_of(point: Point, platform: Path, run: bool) -> dict[str, Any]:
    """One CSV row; a failing stage is the row's ``status``, never an exception."""
    row: dict[str, Any] = {"set": point.set, "id": point.id, "recipe":
                           str(point.recipe.relative_to(ROOT)), "sets": " ".join(point.sets)}  # fmt: skip
    row |= {k: v for k, v in point.labels.items() if k in LABELS}
    try:
        row |= static_row(point, platform)
        row["status"] = "ok"
    except DesignError as e:
        row["status"] = "refused: " + "; ".join(str(p).splitlines()[0] for p in e.problems)[:300]
        return row
    except (DfgError, SandboxError, OSError, ValueError) as e:
        row["status"] = f"refused: {str(e).splitlines()[0][:300]}"
        return row
    if not point.run:
        limit = Platform.load(platform).l1.max_xbar_inputs
        row["status"] = (
            f"estimate only: {row['xbar_inputs']} crossbar inputs, SNAX-GVSoC takes {limit}"
        )
    elif run:
        row |= run_row(point, platform)
        row.setdefault("status", "ok")
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """The rows, numbers rounded so a rerun with the same numbers gives the same file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pixi run paper")
    ap.add_argument("set", help="a point set: paper/<set>.json")
    ap.add_argument("--run", action="store_true", help="also run each point on SNAX-GVSoC")
    ap.add_argument("-j", type=int, default=4, help="flows at a time with --run (default 4)")
    ap.add_argument("--only", nargs="+", metavar="ID", help="these points only")
    args = ap.parse_args(argv)
    platform, points = load_set(args.set)
    if args.only:
        unknown = set(args.only) - {p.id for p in points}
        if unknown:
            ap.error(f"no such point in paper/{args.set}.json: {', '.join(sorted(unknown))}")
        points = [p for p in points if p.id in args.only]
    with ThreadPoolExecutor(max_workers=args.j if args.run else 1) as pool:
        rows = list(pool.map(lambda p: row_of(p, platform, args.run), points))
    out = PAPER / "data" / f"{args.set}.csv"
    old = list(csv.DictReader(out.open())) if out.is_file() else []
    if not args.run:  # a row's run columns stay while its recipe and sets are the same
        was = {r["id"]: r for r in old}
        for r in rows:
            o = was.get(r["id"])
            if o and (o["recipe"], o["sets"]) == (r["recipe"], r["sets"]) and r["status"] == "ok":
                r |= {k: o[k] for k in RUN if o.get(k, "") != ""}
    if args.only and old:  # the other points' rows stay as they were
        new = {r["id"]: r for r in rows}
        rows = [new.pop(r["id"], r) for r in old] + list(new.values())
    write_csv(out, rows)
    bad = [r for r in rows if r.get("status") != "ok"]
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"{shown}: {len(rows)} points, {len(bad)} not ok")
    for r in bad:
        print(f"  {r['id']}: {r['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
