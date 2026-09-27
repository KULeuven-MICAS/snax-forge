"""SNAX-DESIGN command line (DP1a, DP1b).

    python -m snax_forge.design GRAPH --platform P [--memory M]
                                      [--set platform.PATH=VALUE ...]
                                      [--set memory.PATH=VALUE ...]
                                      [--name NAME] [--out DIR]
    python -m snax_forge.design check GRAPH --platform P [--memory M] [--set ...]
    python -m snax_forge.design save SRC DST [--force]

The first form pairs a bound graph (the last ``.snaxdfg`` of a sandbox run)
with a platform, makes the memory plan, runs every design check and, when
they pass, writes to ``DIR`` (default ``out/design/<name>/``):

    platform.json       the platform working copy (base, changes)
    memory.json         the memory working copy (passes, changes, layouts)
    design_point.json   graph, platform, streamers and memory (point.py)

and prints the streamer shell and the memory plan. The name is the sandbox
folder of the graph (``out/sandbox/vecadd_w8/2_bind.snaxdfg`` ->
``vecadd_w8``), else the file's stem. Each run overwrites that folder.
Passing a working copy (``--platform DIR/platform.json``, ``--memory
DIR/memory.json``) continues from it: new ``--set`` paths add to its
``changes``. ``--set memory.C.l1.base=1152`` pins a base,
``--set memory.passes.placement=NAME`` picks a registered pass.

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
from .point import DesignPoint
from .problems import DesignError, report
from .streamers import resolve

ROOT = _repo_root()
OUT = ROOT / "out" / "design"
PLATFORMS = ROOT / "platforms"
STEP = re.compile(r"\d+_\w+\.snaxdfg")


def _setting(text: str) -> tuple[str, str, Any]:
    path, sep, value = text.partition("=")
    head, dot, rest = path.partition(".")
    if not sep or head not in ("platform", "memory") or not dot or not rest:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not platform.PATH=VALUE or memory.PATH=VALUE"
        )
    return head, rest, parse_value(value)


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
    ap.add_argument("--memory", type=Path, help="memory working copy to continue from")
    ap.add_argument(
        "--set", type=_setting, action="append", default=[], metavar="platform|memory.PATH=VALUE"
    )


def _checked(args: argparse.Namespace) -> Design | None:
    sets = [(p, v) for h, p, v in args.set if h == "platform"]
    memory = [(p, v) for h, p, v in args.set if h == "memory"]
    design = load(args.graph, args.platform, sets, args.memory, memory)
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
    plan = design.memory
    n = len(plan.changes)
    passes = ", ".join(f"{k} {v}" for k, v in plan.passes.items())
    lines.append(f"memory plan ({passes}; {n} change{'' if n == 1 else 's'}):")
    ctx = design.context()
    pins = plan.spec.pins()
    for c, mems in plan.layouts.items():
        parts = []
        for m, lay in mems.items():
            word = ctx.word_bytes(m)
            lo, hi = lay.span()
            where = f"{m} [{lo}, {hi + word})"
            if m == "l1":
                where += f" bank {(lay.base - pf.l1.base_addr) // word % pf.l1.n_banks}"
            parts.append(f"{where + (' pinned' if (c, m) in pins else ''):<30}")
        lines.append(f"  {c:<12} " + "".join(parts).rstrip())
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
    name = args.name or design_name(args.graph)
    point = DesignPoint.of(design, name)
    out.mkdir(parents=True, exist_ok=True)
    design.platform.save(out / "platform.json")
    design.memory.save(out / "memory.json")
    point.save(out / "design_point.json")
    print(_summary(design))
    for f in ("platform.json", "memory.json", "design_point.json"):
        print(f"wrote {_rel(out / f)}")
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
