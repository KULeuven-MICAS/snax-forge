"""Command line for SNAX-MODEL (MOD9, D44).

    python -m snax_forge.snax_model run SCENARIO --out DIR
           [--trace off|task|beat] [--no-skip] [--max-cycles N]

Runs one scenario file (scenario.py) and writes the output files into DIR.
The skip mode only changes how fast the run is, never what is written
(D29, D38), so it is printed here and not stored in run.json.

Exit codes: 0 done, 1 the simulation failed (a model error or a program
that does not finish within max_cycles), 2 the scenario is invalid or
cannot be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .scenario import Scenario, ScenarioError, run, write_outputs
from .sched import SimulationError, SimulationTimeout
from .trace import LEVELS


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m snax_forge.snax_model", description=__doc__.split("\n\n")[0]
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a scenario and write profile, trace and final memory")
    r.add_argument("scenario", help="scenario JSON file")
    r.add_argument("--out", required=True, help="output directory (created if missing)")
    r.add_argument("--trace", choices=LEVELS, default="off", help="trace level (default: off)")
    r.add_argument("--no-skip", action="store_true", help="tick every cycle (same results, slower)")
    r.add_argument(
        "--max-cycles", type=int, default=None, help="overrides the scenario's max_cycles"
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    skip = not args.no_skip
    try:
        sc = Scenario.load(args.scenario)
    except (OSError, json.JSONDecodeError, ScenarioError) as e:
        print(f"error: {args.scenario}: {e}", file=sys.stderr)
        return 2
    try:
        res = run(sc, skip_idle=skip, trace_level=args.trace, max_cycles=args.max_cycles)
    except ScenarioError as e:
        print(f"error: {sc.name}: {e}", file=sys.stderr)
        return 2
    except (SimulationError, SimulationTimeout) as e:
        print(f"simulation failed: {sc.name}: {e}", file=sys.stderr)
        return 1
    out = write_outputs(res, args.out)
    print(
        f"{sc.name}: {res.total_cycles} cycles "
        f"(skip {'on' if skip else 'off'}, trace {args.trace}) -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
