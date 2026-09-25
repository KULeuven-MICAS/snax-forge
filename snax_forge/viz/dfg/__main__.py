"""Command line of the DFG viewer (VIS5, D76, D81).

    python -m snax_forge.viz.dfg FILE|DIR [FILE|DIR ...] [--port 8766] [--verbose]

Shows the ``.snaxdfg`` files given, a directory as every ``.snaxdfg`` in it
(a recipe's ``out/sandbox/<name>/``), side by side on 127.0.0.1 until Ctrl-C.
The page's Reload button reads the files again; nothing watches them. A
file that does not load shows its error in its own panel. pixi:
``view-dfg``.

Exit codes: 0 stopped with Ctrl-C, 2 a path does not exist or the port is
taken.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .server import DEFAULT_PORT, make_dfg_server


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m snax_forge.viz.dfg", description=__doc__.split("\n\n")[0]
    )
    p.add_argument("paths", nargs="+", metavar="FILE|DIR", help=".snaxdfg file or directory")
    p.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"port on 127.0.0.1 (default {DEFAULT_PORT})"
    )
    p.add_argument("--verbose", action="store_true", help="log every request")
    args = p.parse_args(argv)
    try:
        srv = make_dfg_server(args.paths, args.port, args.verbose)
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    names = ", ".join(e["name"] for e in srv.graphs.summaries())
    print(f"serving {names} at {srv.url} (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
