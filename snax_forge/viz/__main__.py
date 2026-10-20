"""Command line of the visualiser.

    python -m snax_forge.viz DIR [DIR ...] [--port 8765] [--verbose]

Loads each flow run (``out/flow/<name>/`` or its ``run/``) once and serves the viewer on
127.0.0.1 until Ctrl-C. The page's Reload button reads the directories
again; nothing watches the files. ``--port 0`` takes any free port.

Exit codes: 0 stopped with Ctrl-C, 2 a directory cannot be read or the
port is taken.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .server import DEFAULT_PORT, READ_ERRORS, make_server


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m snax_forge.viz", description=__doc__.split("\n\n")[0]
    )
    p.add_argument(
        "dirs", nargs="+", metavar="DIR", help="a flow folder (out/flow/<name>) or its run/"
    )
    p.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"port on 127.0.0.1 (default {DEFAULT_PORT})"
    )
    p.add_argument("--verbose", action="store_true", help="log every request")
    args = p.parse_args(argv)
    try:
        srv = make_server(args.dirs, args.port, args.verbose)
    except READ_ERRORS as e:  # a folder that is not a flow run, or a taken port
        print(f"error: {e}", file=sys.stderr)
        return 2
    names = ", ".join(r["name"] for r in srv.runs.summaries())
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
