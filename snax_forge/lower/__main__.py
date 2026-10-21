"""SNAX-LOWER command line.

    python -m snax_forge.lower tasks DESIGN_POINT [--out FILE]

``tasks`` reads a design point (``out/design/<name>/design_point.json``,
checked again on load), derives its task list (derive.py), checks its values
against the design point (``check_values``) and writes it (default
``tasks.json`` next to the design point); it prints the steps. Exit 0 when
written, 1 when the design point or the task list is rejected (named), 2 on
bad arguments. pixi: ``lower``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from snax_forge.design import DesignError
from snax_forge.design.point import DesignPoint

from .derive import LowerError, task_list
from .tasks import Configure, Start, Sync, TaskList, TaskListError, check_values


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
        check_values(tl, point)
    except (DesignError, LowerError, TaskListError, ValueError) as e:
        print(f"{args.point}: {e}", file=sys.stderr)
        return 1
    out = args.out or args.point.with_name("tasks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tl.save(out)
    print(f"task list of {point.name} ({len(tl.steps)} steps):")
    print(steps(tl))
    print(f"wrote {out}")
    return 0


USAGE = "usage: python -m snax_forge.lower tasks DESIGN_POINT [--out FILE]"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"tasks": _tasks}
    if argv and argv[0] in commands:
        return commands[argv[0]](argv[1:])
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
