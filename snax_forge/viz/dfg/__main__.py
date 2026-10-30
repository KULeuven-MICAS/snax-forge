"""Command line of the DFG viewer in view mode.

    python -m snax_forge.viz.dfg FILE|DIR [FILE|DIR ...] [--platform P] [--port 8766] [--verbose]

Shows the ``.snaxdfg`` files given, a directory as every ``.snaxdfg`` in it (a recipe's
``out/sandbox/<name>/``), as a read-only history on 127.0.0.1 until Ctrl-C, each step's
estimate on ``--platform`` when given (server.py). The page's Reload button reads the files
again; nothing watches them. A file that does not load shows its error in its own panel. pixi:
``view-dfg``. Edit mode is ``python -m snax_forge.viz.dfg.edit`` (edit.py).

Exit codes: 0 stopped with Ctrl-C, 2 a path or the platform does not exist or the port is
taken.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from snax_forge.design import DesignError

from .. import server as viewer
from . import server


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m snax_forge.viz.dfg", description=__doc__.split("\n\n")[0]
    )
    p.add_argument("paths", nargs="+", metavar="FILE|DIR", help=".snaxdfg file or directory")
    p.add_argument("--platform", help="the platform the estimate is on (default: none)")
    p.add_argument("--port", type=int, default=server.DEFAULT_PORT,
                   help=f"port on 127.0.0.1 (default {server.DEFAULT_PORT})")  # fmt: skip
    p.add_argument("--verbose", action="store_true", help="log every request")
    args = p.parse_args(argv)
    try:
        srv = server.make_dfg_server(args.paths, args.port, args.verbose, args.platform)
    except (OSError, DesignError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return viewer.serve(srv, ", ".join(x["name"] for x in srv.loaded.summaries()))


if __name__ == "__main__":
    sys.exit(main())
