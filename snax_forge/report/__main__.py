"""The report command line (REP1, D99).

    python -m snax_forge.report DIR [--out OUT] [--tasks TASKS]

DIR is a flow folder (``out/flow/<name>``) or a run directory; the reports
go to ``DIR/report/`` or ``DIR_report/`` unless ``--out`` says otherwise
(files.py). Exit 0 when both were written, 1 when DIR cannot be read, 2 on bad
arguments. pixi: ``report``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .files import DESIGN_MD, RUN_MD, ReportError, write_reports


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m snax_forge.report")
    ap.add_argument("folder", type=Path, help="a flow folder or a run directory")
    ap.add_argument("--out", type=Path, help="where to write design.md and run.md")
    ap.add_argument(
        "--tasks", type=Path, help="the task list that names the tasks (a scenario's tasks.json)"
    )
    args = ap.parse_args(argv)
    try:
        r = write_reports(args.folder, args.out, args.tasks)
    except (ReportError, OSError, ValueError, KeyError) as e:
        print(f"FAILED {args.folder}: {e}", file=sys.stderr)
        return 1
    print(
        f"report {r.run.name}: {r.run.cycles} cycles, trace {r.run.trace_level} -> {r.out / DESIGN_MD}, {r.out / RUN_MD}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
