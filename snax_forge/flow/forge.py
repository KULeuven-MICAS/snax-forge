"""A recipe to everything snax_cluster needs, in one command.

    python -m snax_forge.flow.forge RECIPE [--platform P] [--set NAME=VALUE ...]
                                    [--trace LEVEL[:SOURCES]]          (pixi forge)

Three steps, each the tool of that name called as its command line would be
(D136), nothing decided here:

    flow     ``python -m snax_forge.flow RECIPE --platform P [--set ...] [--trace ...]``:
             the design point and the run on SNAX-GVSoC      out/flow/<name>/
    bundle   ``python -m snax_forge.gen.bundle`` on the flow's design point, then the
             flow's expected lines beside it (``bundle.write_expected``: its
             ``forge counter`` lines and ``forge data: 0 errors``)
                                                             out/bundle/<name>/
    sw       the bundle's program, by snax-gvsoc's generic app (``make sw``), and its
             sha256 checked against the flow's ``elf_sha256`` (docs/RUN.md section 8)
                                                             out/forge/<name>/sw/forge.elf

``<name>`` is the flow's folder name (the recipe's with every ``--set``, then
the trace level above ``off``). The platform defaults to
``platforms/snax_forge.json``. Each step's whole output, as its own command
line prints it (sbt's and make's lines too), goes to ``out/forge/<name>/forge.log``;
the terminal gets one line per step, SNAX-GVSoC's four ``forge`` lines from
the flow's ``run/stdout.txt``, and the commands that run the bundle's ELF on
snax_cluster's Verilator build, ``B`` and ``E`` filled in by absolute path.
snax-forge never calls into snax_cluster (D130).

Exit 0 when every step passed, 1 when one failed (the step named, the last 30
lines of the log and its path), 2 on bad arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import IO, Any

from snax_forge.common import cwd_relative
from snax_forge.gen import GenError, bundle
from snax_forge.run import RunError
from snax_forge.run.sources import parse_trace
from snax_forge.sandbox import Recipe, SandboxError

from . import __main__ as flow_cli
from . import run as flow_run

OUT = flow_run.ROOT / "out" / "forge"
PLATFORM = flow_run.ROOT / "platforms" / "snax_forge.json"
LOG = "forge.log"
TAIL = 30
VLT = (
    "rtl-gen CFG_OVERRIDE=$B",
    "snax-forge-sim SIM=vlt CFG_OVERRIDE=$B",
    "snax-forge-run SIM=vlt CFG_OVERRIDE=$B ELF=$E",
)


class StepFailed(Exception):
    """A step that failed; the message names it."""


@contextmanager
def _into(log: IO[str]) -> Iterator[None]:
    """``sys.stdout``, ``sys.stderr`` and file descriptors 1 and 2 into ``log``, so the
    subprocesses' output (sbt's, make's) goes there too."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    try:
        with redirect_stdout(log), redirect_stderr(log):
            yield
    finally:
        log.flush()
        for fd, old in zip((1, 2), saved, strict=True):
            os.dup2(old, fd)
            os.close(old)


def _step(log: IO[str], step: str, argv: list[str], main) -> None:
    """``main(argv)`` with its output in ``log``, under a header naming the command; a
    StepFailed on a nonzero exit."""
    print(f"== {step}: {' '.join(argv)}", file=log, flush=True)
    with _into(log):
        try:
            rc = main(argv)
        except SystemExit as e:  # argparse's
            rc = e.code if isinstance(e.code, int) else 1
    if rc != 0:
        raise StepFailed(f"{step}: exit {rc}")


def _sw(log: IO[str], name: str, bundle_dir: Path, forge_dir: Path, flow_sha: str) -> Path:
    """The bundle's program by ``make sw`` (its output into ``log``); its ELF, checked to be the
    flow's (``flow_sha``)."""
    sw = forge_dir / "sw"
    variables = {"APPS": "forge", "FORGE": str(bundle_dir.resolve()), "DESIGN": name,
                 "FORGE_BUILD": str(sw.resolve())}  # fmt: skip
    print(f"== sw: make -C gvsoc sw {' '.join(f'{k}={v}' for k, v in variables.items())}",
          file=log, flush=True)  # fmt: skip
    rc = flow_run.run_gvsoc("sw", variables, log)
    elf = sw / flow_run.ELF
    if rc != 0 or not elf.is_file():
        raise StepFailed(f"sw: make sw exit {rc}" + ("" if elf.is_file() else f", no {elf}"))
    sha = hashlib.sha256(elf.read_bytes()).hexdigest()
    if sha != flow_sha:
        message = (f"sw: {cwd_relative(elf)} has sha256 {sha}, the flow's ELF {flow_sha}: "
                   "the bundle's program is not the flow's")  # fmt: skip
        print(message, file=log, flush=True)
        raise StepFailed(message)
    return elf


def _setting(text: str) -> tuple[str, tuple[str, str, Any]]:
    """The text, passed to the flow as it is, and the flow's reading of it."""
    return text, flow_cli.setting(text)


def forge(recipe: Path, platform: Path, sets: list[tuple[str, tuple[str, str, Any]]],
          trace: str) -> int:  # fmt: skip
    """Run the three steps and print what they wrote; see the module doc. ``sets`` are
    ``_setting``'s."""
    by_head = flow_run.split_sets([reading for _, reading in sets])
    recipe_sets = dict(by_head["recipe"])
    level, _ = parse_trace(trace)
    r = Recipe.load(recipe).with_params(recipe_sets)
    name = flow_run.with_level(
        flow_run.default_name(r, recipe_sets, by_head["platform"], by_head["memory"]), level
    )
    flow_dir, bundle_dir, forge_dir = flow_run.OUT / name, bundle.OUT / name, OUT / name
    shutil.rmtree(forge_dir, ignore_errors=True)
    forge_dir.mkdir(parents=True)
    log_path = forge_dir / LOG
    point = flow_dir / "design" / "design_point.json"
    print(f"forge {name}: {cwd_relative(recipe)} on {cwd_relative(platform)}; every step's "
          f"output in {cwd_relative(log_path)}", flush=True)  # fmt: skip
    flow_argv = [str(recipe), "--platform", str(platform),
                 *(a for text, _ in sets for a in ("--set", text)), "--trace", trace]  # fmt: skip
    with log_path.open("w", buffering=1) as log:
        try:
            _step(log, "flow", flow_argv, flow_cli.main)
            print(f"  flow      passed: design point {cwd_relative(point)}, run "
                  f"{cwd_relative(flow_dir)}/run/", flush=True)  # fmt: skip
            for x in (flow_dir / flow_run.LOG).read_text().splitlines():
                if x.startswith("  WARNING"):
                    print(x, flush=True)
            _step(log, "bundle", [str(point), "--out", str(bundle_dir)], bundle.main)
            made = bundle.Bundle.load(bundle_dir / bundle.BUNDLE_JSON)
            commit = (made.snax_forge_commit or "outside git")[:12]
            print(f"  bundle    {cwd_relative(bundle_dir)}/: {made.tag}_shell_wrapper, snax-forge "
                  f"{commit}{', dirty' if made.dirty else ''}", flush=True)  # fmt: skip
            try:
                expected = bundle.write_expected(
                    bundle_dir, (flow_dir / "run" / "stdout.txt").read_text()
                )
            except GenError as e:
                print(f"== expected: {e}", file=log, flush=True)
                raise StepFailed(f"expected: {e}") from None
            print(
                f"  expected  {cwd_relative(expected)}: the run's counter and data lines",
                flush=True,
            )
            sha = json.loads((flow_dir / "run" / "run.json").read_text())["elf_sha256"]
            elf = _sw(log, name, bundle_dir, forge_dir, sha)
            print(
                f"  sw        {cwd_relative(elf)} from the bundle (sha256 {sha[:16]}, the flow's)"
            )
        except StepFailed as e:
            log.flush()
            tail = log_path.read_text().splitlines()[-TAIL:]
            print(f"forge {name} FAILED at {e}; the last {len(tail)} lines of "
                  f"{cwd_relative(log_path)}:", file=sys.stderr)  # fmt: skip
            print("\n".join(tail), file=sys.stderr)
            return 1
    stdout = flow_dir / "run" / "stdout.txt"
    lines = [f"SNAX-GVSoC ({cwd_relative(stdout)}):"]
    lines += [f"  {x}" for x in stdout.read_text().splitlines() if x.startswith("forge ")]
    lines += [
        "snax_cluster on Verilator, from snax_cluster in its pixi environment:",
        f"  B={(bundle_dir / 'snax_forge.hjson').resolve()}",
        f"  E={elf.resolve()}",
    ]
    lines += [f"  pixi run make -C target/snitch_cluster {goal}" for goal in VLT]
    lines.append(
        "Questa: rtl-gen and vsim_preparation in snax_cluster's development image, then "
        "snax-forge-sim and snax-forge-run with SIM=vsim and the same B and E: docs/RUN.md "
        "section 8."
    )
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.flow.forge")
    ap.add_argument("recipe", type=Path, metavar="RECIPE")
    ap.add_argument("--platform", type=Path, default=PLATFORM,
                    help="platform (default platforms/snax_forge.json)")  # fmt: skip
    ap.add_argument("--set", type=_setting, action="append", default=[], metavar="NAME=VALUE",
                    help="passed to the flow")  # fmt: skip
    ap.add_argument("--trace", default="off", metavar="LEVEL[:SOURCES]",
                    help="passed to the flow")  # fmt: skip
    args = ap.parse_args(argv)
    try:
        parse_trace(args.trace)
    except RunError as e:
        ap.error(f"--trace: {e}")
    if args.recipe.suffix == flow_cli.BOUND:
        ap.error(f"{args.recipe}: forge takes a recipe (a bound graph goes to pixi run flow)")
    try:
        return forge(args.recipe, args.platform, args.set, args.trace)
    except (SandboxError, OSError) as e:
        print(f"FAILED {args.recipe}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
