"""SNAX-LOWER command line (LOW1c, LOW1a).

    python -m snax_forge.lower cluster DESIGN_POINT [--out FILE]
    python -m snax_forge.lower tasks DESIGN_POINT [--out FILE]

Both read a design point (``out/design/<name>/design_point.json``, checked
again on load). ``cluster`` derives its cluster file (cluster.py), builds it
in SNAX-MODEL once to be sure it runs, and writes it (default
``cluster.json`` next to the design point); it prints the components in
order. ``tasks`` derives its task list (derive.py), lowers it once to the
command list against that cluster, and writes it (default ``tasks.json``
next to the design point); it prints the steps. Exit 0 when written, 1 when
the design point, the cluster or the task list is rejected (named), 2 on
bad arguments. pixi: ``lower``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from snax_forge.design import DesignError
from snax_forge.design.point import DesignPoint
from snax_forge.snax_model.scenario import ClusterConfig, ScenarioError, build_cluster

from .cluster import cluster_file
from .commands import lower_program
from .derive import LowerError, task_list
from .tasks import Configure, Start, Sync, TaskList, TaskListError


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


def steps(tl: TaskList) -> str:
    """One line per step."""
    lines = []
    for s in tl.steps:
        if isinstance(s, Configure):
            after = f"  after {s.after}" if s.after else ""
            lines.append(f"  configure {s.task_name:<14} {s.component:<8} {s.values}{after}")
        elif isinstance(s, Start):
            lines.append(f"  start     {', '.join(s.tasks)}")
        elif isinstance(s, Sync):
            lines.append(f"  sync      {s.task} ({s.mode})")
        else:
            lines.append(f"  {s.op} {s.to_dict()}")
    return "\n".join(lines)


def _tasks(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.lower tasks")
    ap.add_argument("point", type=Path, help="a design point (out/design/<name>/design_point.json)")
    ap.add_argument("--out", type=Path, help="task list (default: tasks.json beside it)")
    args = ap.parse_args(argv)
    try:
        point = DesignPoint.load(args.point)
        tl = task_list(point)
        program = lower_program(tl, cluster_file(point))
    except (DesignError, LowerError, TaskListError, ValueError) as e:
        print(f"{args.point}: {e}", file=sys.stderr)
        return 1
    out = args.out or args.point.with_name("tasks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tl.save(out)
    print(f"task list of {point.name} ({len(tl.steps)} steps, {len(program)} commands):")
    print(steps(tl))
    print(f"wrote {out}")
    return 0


USAGE = "usage: python -m snax_forge.lower cluster|tasks DESIGN_POINT [--out FILE]"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"cluster": _cluster, "tasks": _tasks}
    if argv and argv[0] in commands:
        return commands[argv[0]](argv[1:])
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
