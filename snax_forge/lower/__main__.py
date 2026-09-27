"""SNAX-LOWER command line (LOW1c).

    python -m snax_forge.lower cluster DESIGN_POINT [--out FILE]

Reads a design point (``out/design/<name>/design_point.json``, checked again
on load), derives its cluster file (cluster.py), builds it in SNAX-MODEL
once to be sure it runs, and writes it to FILE (default ``cluster.json``
next to the design point). Prints the components in order. LOW1a adds
``tasks``. Exit 0 when written, 1 when the design point or the cluster is
rejected (every problem named), 2 on bad arguments. pixi: ``lower``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from snax_forge.design import DesignError
from snax_forge.design.point import DesignPoint
from snax_forge.snax_model.scenario import ClusterConfig, ScenarioError, build_cluster

from .cluster import cluster_file


def describe(cfg: ClusterConfig) -> str:
    """One line per component, in the file's order."""
    lines = []
    for c in cfg.components:
        if c.kind == "accel":
            attach = ", ".join(f"{p} <- {s}" for p, s in c.attach.items())
            lines.append(f"  {c.name:<10} accel {c.accel} {c.params}  attach {attach}")
        elif c.kind == "streamer":
            k = c.config
            side = "writer" if k["write"] else "reader"
            lines.append(
                f"  {c.name:<10} streamer {side} n_ports {k['n_ports']} "
                f"temporal_dims {k['temporal_dims']} fifo_depth {k['fifo_depth']}"
            )
        else:
            lines.append(f"  {c.name:<10} {c.kind}")
    return "\n".join(lines)


def _cluster(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.lower cluster")
    ap.add_argument("point", type=Path, help="a design point (out/design/<name>/design_point.json)")
    ap.add_argument("--out", type=Path, help="cluster file (default: cluster.json beside it)")
    args = ap.parse_args(argv)
    try:
        point = DesignPoint.load(args.point)
    except DesignError as e:
        print(e, file=sys.stderr)
        return 1
    cfg = cluster_file(point)
    try:
        build_cluster(cfg)
    except (ScenarioError, ValueError) as e:
        print(f"the cluster of {args.point} does not build: {e}", file=sys.stderr)
        return 1
    out = args.out or args.point.with_name("cluster.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg.save(out)
    print(f"cluster of {point.name} ({point.platform.name}, {len(cfg.components)} components):")
    print(describe(cfg))
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "cluster":
        return _cluster(argv[1:])
    print("usage: python -m snax_forge.lower cluster DESIGN_POINT [--out FILE]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
