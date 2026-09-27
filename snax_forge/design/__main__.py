"""SNAX-DESIGN command line (DP1a).

    python -m snax_forge.design GRAPH --platform P [--set platform.PATH=VALUE ...]
                                      [--name NAME] [--out DIR]
    python -m snax_forge.design check GRAPH --platform P [--set platform.PATH=VALUE ...]
    python -m snax_forge.design save SRC DST [--force]

The first form pairs a bound graph (the last ``.snaxdfg`` of a sandbox run)
with a platform, runs every design check and, when they pass, writes the
working copy of the platform to ``DIR/platform.json`` (default
``out/design/<name>/``) and prints the streamer shell. The name is the
sandbox folder of the graph (``out/sandbox/vecadd_w8/2_bind.snaxdfg`` ->
``vecadd_w8``), else the file's stem. Each run overwrites that folder.
Passing the working copy as ``--platform`` continues from it: new ``--set``
paths add to its ``changes``.

``check`` runs the checks only and writes nothing. ``save`` keeps a
platform, usually a working copy, under a new name: DST is a path, or a
bare name for ``platforms/<DST>.json``; an existing file is kept unless
``--force``.

Exit 0 when everything passed, 1 with every problem found (each with its
fix), 2 on bad arguments. pixi: ``design``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from snax_forge.sdfg.paths import _repo_root

from .check import Design, load, run_checks
from .platform import Platform, parse_value
from .problems import DesignError, report
from .streamers import resolve

ROOT = _repo_root()
OUT = ROOT / "out" / "design"
PLATFORMS = ROOT / "platforms"
STEP = re.compile(r"\d+_\w+\.snaxdfg")


def _setting(text: str) -> tuple[str, Any]:
    path, sep, value = text.partition("=")
    head, dot, rest = path.partition(".")
    if not sep or head != "platform" or not dot or not rest:
        raise argparse.ArgumentTypeError(f"{text!r} is not platform.PATH=VALUE")
    return rest, parse_value(value)


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def design_name(graph: Path) -> str:
    """The sandbox folder of a step's graph, else the file's stem."""
    return graph.parent.name if STEP.fullmatch(graph.name) else graph.stem


def _inputs(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("graph", type=Path, help="a bound .snaxdfg (the last step of a recipe)")
    ap.add_argument("--platform", type=Path, required=True, help="platform file or working copy")
    ap.add_argument(
        "--set", type=_setting, action="append", default=[], metavar="platform.PATH=VALUE"
    )


def _checked(args: argparse.Namespace) -> Design | None:
    design = load(args.graph, args.platform, args.set)
    problems = run_checks(design)
    if problems:
        print(report(problems, f"design check of {args.graph} on {args.platform}"), file=sys.stderr)
        return None
    return design


def _summary(design: Design) -> str:
    pf = design.platform
    changes = len(pf.changes)
    head = (
        f"{design.graph_path} on {pf.name} (base {pf.base}, "
        f"{changes} change{'' if changes == 1 else 's'}): checks passed"
    )
    lines = [head, "streamer shell:"]
    for s in resolve(pf, design.instances).values():
        side = "writer" if s.write else "reader"
        o = s.options
        lines.append(
            f"  {s.name:<12} {s.instance}.{s.port:<6} {side}  {s.n_ports} lanes "
            f"{list(s.spatial_bounds)}  temporal_dims {o.temporal_dims}  "
            f"fifo_depth {o.fifo_depth}  addr_depth {o.addr_depth}  prio {o.prio}"
        )
    return "\n".join(lines)


def _run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.design")
    _inputs(ap)
    ap.add_argument("--name", help="design name (default: the graph's sandbox folder)")
    ap.add_argument("--out", type=Path, help="output directory (default out/design/<name>)")
    args = ap.parse_args(argv)
    design = _checked(args)
    if design is None:
        return 1
    out = args.out or OUT / (args.name or design_name(args.graph))
    out.mkdir(parents=True, exist_ok=True)
    design.platform.save(out / "platform.json")
    print(_summary(design))
    print(f"wrote {_rel(out / 'platform.json')}")
    return 0


def _check(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.design check")
    _inputs(ap)
    args = ap.parse_args(argv)
    design = _checked(args)
    if design is None:
        return 1
    print(_summary(design))
    return 0


def _save(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.design save")
    ap.add_argument("src", type=Path, help="the platform to keep, usually a working copy")
    ap.add_argument("dst", help="a path, or a bare name for platforms/<name>.json")
    ap.add_argument("--force", action="store_true", help="replace an existing file")
    args = ap.parse_args(argv)
    dst = Path(args.dst)
    if dst.suffix != ".json" and dst.parent == Path("."):
        dst = PLATFORMS / f"{args.dst}.json"
    if dst.exists() and not args.force:
        print(f"{dst} exists; pass --force to replace it", file=sys.stderr)
        return 1
    try:
        platform = Platform.load(args.src).saved_as(dst.stem)
    except DesignError as e:
        print(e, file=sys.stderr)
        return 1
    problems = run_checks(Design(platform, None), ["platform"])
    if problems:
        print(report(problems, f"platform check of {args.src}"), file=sys.stderr)
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    platform.save(dst)
    print(f"wrote {_rel(dst)} (base {platform.base}, {len(platform.changes)} changes)")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "check":
        return _check(argv[1:])
    if argv and argv[0] == "save":
        return _save(argv[1:])
    return _run(argv)


if __name__ == "__main__":
    sys.exit(main())
