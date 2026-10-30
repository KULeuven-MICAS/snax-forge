"""A set of bundles for the local RTL check, in one command.

    python -m snax_forge.flow.forge_set quick|full [--sim vsim|vlt] [-j N]    (pixi forge-set)

Runs ``pixi run forge`` (flow/forge.py) on each recipe of the set at its default sizes, ``-j``
at a time (default the CPU count), each in its own folders (``out/flow/<name>/``,
``out/bundle/<name>/``, ``out/forge/<name>/``), each forge's terminal output in
``out/forge-set/<set>/<design>.txt``. Emit is made ready once first (``rtl.warm_emit``: its
classpath and firtool), so the bundles do not all start sbt or extract firtool at once. Then it writes ``out/forge-set/<set>.md``: per design its bundle folder and
the lines its RTL run must print (``forge_expected.txt``, from its SNAX-GVSoC run); the one
line that checks them all from snax_cluster's ``target/snitch_cluster`` (``make
snax-forge-check BUNDLES=... SIM=... SNAX_CLUSTER_IMAGE=<image>``, the image named in
``IMAGE_HINT``); and per design the commands forge printed, for running one by hand.

A forge that fails (a SNAX-GVSoC run that fails its own check among them) stops the set: exit
1, each failed design and its output named, no ``<set>.md``. Exit 2 on bad arguments.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from snax_forge.common import cwd_relative
from snax_forge.gen import bundle
from snax_forge.gen.rtl import warm_emit

ROOT = bundle.ROOT
OUT = ROOT / "out" / "forge-set"

QUICK = ["vecadd_i32", "vsum_tree_i8", "dot_1", "dot_4"]
SETS = {
    "quick": QUICK,
    "full": [*QUICK, "vecadd", "vsum", "vecmul_i32_i64", "dot_2", "dot_3", "dot_5", "dot_6",
             "matvec_5", "matvec_6"],
}  # fmt: skip


def run_forge(recipe: str, output: Path) -> int:
    """``pixi run forge recipes/<recipe>.json``, its terminal output into ``output``; its exit."""
    with output.open("w") as f:
        cmd = [
            sys.executable,
            "-m",
            "snax_forge.flow.forge",
            str(ROOT / "recipes" / f"{recipe}.json"),
        ]
        return subprocess.run(
            cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=False
        ).returncode


def bundle_dir(output: str) -> Path:
    """The bundle folder forge's ``bundle`` line names, absolute."""
    (line,) = (x for x in output.splitlines() if x.startswith("  bundle    "))
    return (ROOT / line.split()[1].rstrip("/:")).resolve()


# The image snax-forge-check runs both simulators' steps in: a placeholder, with where to find it
IMAGE_HINT = (
    "`<image>` is snax_cluster's development image (podman), which both simulators need; on the "
    "MICAS servers the one docs/RUN.md section 8 names."
)


def check_line(dirs: list[Path], sim: str) -> str:
    """The one ``make snax-forge-check`` line for ``dirs``."""
    every = " ".join(str(d) for d in dirs)
    return f'make snax-forge-check BUNDLES="{every}" SIM={sim} SNAX_CLUSTER_IMAGE=<image>'


def by_hand(output: str) -> str:
    """The commands forge printed for snax_cluster: its lines from ``snax_cluster on``."""
    lines = output.splitlines()
    at = next(i for i, x in enumerate(lines) if x.startswith("snax_cluster on "))
    return "\n".join(lines[at:])


def table(name: str, designs: list[str], outputs: dict[str, str], sim: str) -> str:
    """``<set>.md``; see the module doc. One line per paragraph, table row and list item."""
    dirs = {d: bundle_dir(outputs[d]) for d in designs}
    commit, dirty = bundle.provenance(ROOT)
    made = f"snax-forge {commit[:12] if commit else 'outside git'}{', dirty' if dirty else ''}"
    lines = [
        f"# forge-set {name}",
        "",
        (
            f"The bundles of the {name} set ({', '.join(designs)}), made by `pixi run forge-set "
            f"{name}` from {made}. Each one's SNAX-GVSoC run passed, and its `{bundle.EXPECTED}` "
            "holds the lines a run of the same program on snax_cluster's RTL must print."
        ),
        "",
        "| Design | Bundle | Expected lines |",
        "|---|---|---|",
    ]
    for d in designs:
        expected = (dirs[d] / bundle.EXPECTED).read_text().splitlines()
        lines.append(f"| {d} | `{dirs[d]}` | {'<br>'.join(f'`{x}`' for x in expected)} |")
    lines += [
        "",
        (
            "Check them all from snax_cluster's `target/snitch_cluster` (`snax-forge-check`, "
            "snax_cluster's): each run's `forge counter` and `forge data` lines must equal its "
            f"bundle's `{bundle.EXPECTED}`. {IMAGE_HINT}"
        ),
        "",
        "```",
        check_line([dirs[d] for d in designs], sim),
        "```",
        "",
        "## One design by hand",
        "",
        "As `pixi run forge` printed it for each design.",
    ]
    for d in designs:
        lines += ["", f"### {d}", "", "```", by_hand(outputs[d]), "```"]
    return "\n".join(lines) + "\n"


def forge_set(name: str, sim: str, jobs: int) -> int:
    """Run the set and write its file; see the module doc."""
    designs = SETS[name]
    logs = OUT / name
    logs.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.md").unlink(missing_ok=True)
    warm_emit()
    print(f"forge-set {name}: {len(designs)} designs, {jobs} at a time; each forge's output in "
          f"{cwd_relative(logs)}/<design>.txt", flush=True)  # fmt: skip
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rcs = dict(zip(designs, pool.map(lambda d: run_forge(d, logs / f"{d}.txt"), designs),
                       strict=True))  # fmt: skip
    failed = [d for d in designs if rcs[d] != 0]
    for d in designs:
        print(f"  {d:<16} {'FAILED' if rcs[d] else 'passed'}", flush=True)
    if failed:
        for d in failed:
            print(f"forge-set {name} FAILED: {d} (exit {rcs[d]}), its output in "
                  f"{cwd_relative(logs / f'{d}.txt')}", file=sys.stderr)  # fmt: skip
        return 1
    outputs = {d: (logs / f"{d}.txt").read_text() for d in designs}
    path = OUT / f"{name}.md"
    path.write_text(table(name, designs, outputs, sim))
    print(f"wrote {cwd_relative(path)}; from snax_cluster's target/snitch_cluster:")
    print(f"  {check_line([bundle_dir(outputs[d]) for d in designs], sim)}")
    print(f"  {IMAGE_HINT}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.flow.forge_set")
    ap.add_argument("set", choices=sorted(SETS), help="the set of recipes")
    ap.add_argument(
        "--sim", choices=["vsim", "vlt"], default="vsim", help="snax_cluster's simulator"
    )
    ap.add_argument(
        "-j", type=int, default=os.cpu_count() or 1, help="forges at a time (CPU count)"
    )
    args = ap.parse_args(argv)
    if args.j < 1:
        ap.error("-j: 1 or more")
    return forge_set(args.set, args.sim, args.j)


if __name__ == "__main__":
    sys.exit(main())
