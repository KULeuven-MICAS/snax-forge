"""A set of bundles for the local RTL check, in one command.

    python -m snax_forge.flow.forge_set quick|full|<task> [--sim vsim|vlt] [-j N]    (pixi forge-set)

``forge_set`` runs ``pixi run forge`` (flow/forge.py) on each ``Design`` of the set in ``SETS``
(a recipe and its ``--set`` values, named as forge's folder), ``-j`` at a time (default the CPU
count), each forge's terminal output in ``out/forge-set/<set>/<design>.txt``. Emit is made ready
once first (``rtl.warm_emit``), so the bundles do not all start sbt or extract firtool at once.
Then it writes ``out/forge-set/<set>.md``: per design its bundle folder and expected lines
(``forge_expected.txt``), the one ``make snax-forge-check`` line that checks them all from
snax_cluster's ``target/snitch_cluster`` (``check_line``), and forge's commands per design.

A failed forge stops the set: exit 1, each failed design and its output named, no ``<set>.md``.
Exit 2 on bad arguments.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from snax_forge.common import cwd_relative
from snax_forge.gen import bundle
from snax_forge.gen.rtl import warm_emit
from snax_forge.sandbox.recipe import Recipe, find_recipe

ROOT = bundle.ROOT
OUT = ROOT / "out" / "forge-set"


@dataclass(frozen=True)
class Design:
    """One design of a set: a recipe and the ``--set`` values forge runs it with, each
    ``NAME=VALUE`` as on forge's command line (none: the recipe's defaults)."""

    recipe: str
    sets: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        """Forge's folder name for it: the recipe's, with every ``--set`` appended
        (flow/run.py's ``default_name``)."""
        if not self.sets:
            return self.recipe
        from snax_forge.flow import run as flow_run
        from snax_forge.flow.__main__ import setting

        by_head = flow_run.split_sets([setting(t) for t in self.sets])
        recipe_sets = dict(by_head["recipe"])
        r = Recipe.load(find_recipe(self.recipe)).with_params(recipe_sets)
        return flow_run.default_name(
            r, recipe_sets, by_head["platform"], by_head["memory"], by_head["design"]
        )


def designs(*recipes: str | Design) -> list[Design]:
    """A set's designs: a name is its recipe at its defaults."""
    return [r if isinstance(r, Design) else Design(r) for r in recipes]


PADDED = "memory.passes.layout=padded_rows"  # each row a word longer
SEQUENTIAL = "design.schedule=sequential"
# One point per integration feature, the forge/ folders' at the recipes' sizes
QUICK = designs(Design("vecadd_v1_1", ("W=1",)),  # sliced inputs, a packed output
                "axpy_v1",  # a CSR scalar
                "dot_v4",  # a linked group
                Design("mvt_v1_2_2", ("W=8",)),  # two groups on shared movers, the port mux
                "bicg_v4_1")  # a fork, a bank, a held input, a carried seed  # fmt: skip
SETS = {
    "quick": QUICK,
    # The quick set's features and the rest, then one point each at i8, i16 and i32
    "full": [*QUICK,
             *designs("vecadd_v1",
                      Design("rowsum_v2", (PADDED,)),  # padded rows
                      "matvec_acc_v2",  # two groups, a seeded fold
                      Design("mvt_v4_1_1", (SEQUENTIAL,)),  # the DMA's transposed copy
                      Design("mvt_v5", (SEQUENTIAL,)),  # two accelerators
                      "mvt_v6",  # a seeded bank, a fork
                      "atax_v4",  # a held link into a bank, a fork's buffered branch
                      Design("vsum_v1_1", ("types=i8", "W=8")),
                      Design("dot_v3", ("types=i16", "W=4")),
                      Design("dot_v3", ("types=i32", "W=2")))],
    # TIERA3's points while it is open: atax's forms at 64 x 64, serial unless named
    "atax": designs(Design("atax_v1", (PADDED,)),
                    Design("atax_v1_1", (PADDED,)),
                    "atax_v2",
                    Design("atax_v1_2", (SEQUENTIAL,)),
                    "atax_v2_1",
                    Design("atax_v2_1", ("M=40",)),
                    Design("atax_v2_1", ("W=16",)),
                    Design("atax_v2_1", ("W=32",)),
                    # the halves on shared movers (share_ports)
                    Design("atax_v1_3", (PADDED,)),
                    "atax_v2_2",
                    Design("atax_v1_2_1", (SEQUENTIAL,)),
                    "atax_v2_1_1",
                    # one group: t linked and held, then A read once through a buffer
                    "atax_v3",
                    "atax_v4",
                    Design("atax_v4", ("D=8",)),
                    Design("atax_v4", ("M=40",)),
                    Design("atax_v4", ("W=16", "R=4", "D=6")),
                    Design("atax_v4", ("W=32", "R=2", "D=4"))),
}  # fmt: skip


def run_forge(design: Design, output: Path) -> int:
    """``pixi run forge recipes/<folder>/<recipe>.json [--set ...]`` (``find_recipe``), its
    terminal output into ``output``; its exit."""
    with output.open("w") as f:
        cmd = [
            sys.executable,
            "-m",
            "snax_forge.flow.forge",
            str(find_recipe(design.recipe)),
            *(a for s in design.sets for a in ("--set", s)),
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
    "MICAS servers the one docs/TOOL.md section 6 names."
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
    entries = SETS[name]
    designs = [e.name for e in entries]
    logs = OUT / name
    logs.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.md").unlink(missing_ok=True)
    warm_emit()
    print(f"forge-set {name}: {len(designs)} designs, {jobs} at a time; each forge's output in "
          f"{cwd_relative(logs)}/<design>.txt", flush=True)  # fmt: skip
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rcs = dict(zip(designs, pool.map(lambda e: run_forge(e, logs / f"{e.name}.txt"), entries),
                       strict=True))  # fmt: skip
    failed = [d for d in designs if rcs[d] != 0]
    width = max(16, *map(len, designs))
    for d in designs:
        print(f"  {d:<{width}} {'FAILED' if rcs[d] else 'passed'}", flush=True)
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
