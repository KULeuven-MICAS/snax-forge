"""Command line of the DFG viewer in view mode.

    python -m snax_forge.viz.dfg FILE|DIR [FILE|DIR ...] [--platform P] [--tech T] [--port 8766]
                                 [--verbose]

Shows the ``.snaxdfg`` files given (a directory as every one in it) as a read-only history on
127.0.0.1 until Ctrl-C, each step's estimate on ``--platform`` and its cost on ``--tech``
(default nangate45). Files from several folders are named ``<folder>/<stem>``, so Compare can
put two designs side by side. pixi: ``view-dfg``. Edit mode is edit.py.

Exit codes: 0 stopped with Ctrl-C, 2 a path or the platform does not exist or the port is taken.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from snax_forge.char import DEFAULT_TECH, TECHS
from snax_forge.design import DesignError

from .. import server as viewer
from . import server


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m snax_forge.viz.dfg", description=__doc__.split("\n\n")[0]
    )
    p.add_argument("paths", nargs="+", metavar="FILE|DIR", help=".snaxdfg file or directory")
    p.add_argument("--platform", help="the platform the estimate is on (default: none)")
    p.add_argument("--tech", choices=TECHS, default=DEFAULT_TECH,
                   help=f"the technology of the cost estimate (default {DEFAULT_TECH})")  # fmt: skip
    p.add_argument("--port", type=int, default=server.DEFAULT_PORT,
                   help=f"port on 127.0.0.1 (default {server.DEFAULT_PORT})")  # fmt: skip
    p.add_argument("--verbose", action="store_true", help="log every request")
    args = p.parse_args(argv)
    try:
        srv = server.make_dfg_server(args.paths, args.port, args.verbose, args.platform, args.tech)
    except (OSError, DesignError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return viewer.serve(srv, ", ".join(x["name"] for x in srv.loaded.summaries()))


if __name__ == "__main__":
    sys.exit(main())
