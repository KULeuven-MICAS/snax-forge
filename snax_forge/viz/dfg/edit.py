"""Command line of the DFG viewer in edit mode (D55, server.py).

    python -m snax_forge.viz.dfg.edit [--kernel K [--set NAME=VALUE ...] | --recipe R |
                                       --session DIR] [--platform P] [--port 8767] [--verbose]

Serves the editor on 127.0.0.1 until Ctrl-C with one session (sandbox/session.py): a new one
on kernel K (``--set`` gives its params, each symbol the param of its name), one replaying
recipe R, or the session in DIR (``out/sessions/<name>/``) resumed. Without a start option the
page shows its start dialog. Sessions go to ``out/sessions/``, saved recipes to ``recipes/``
and their steps to ``out/sandbox/``. pixi: ``edit-dfg``.

Exit codes: 0 stopped with Ctrl-C, 2 a start option that fails (its message) or a port taken.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from snax_forge.sandbox.__main__ import setting

from .. import server as viewer
from . import server


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m snax_forge.viz.dfg.edit",
                                description=__doc__.split("\n\n")[0])  # fmt: skip
    start = p.add_mutually_exclusive_group()
    start.add_argument("--kernel", help="start a session on this kernel's import")
    start.add_argument("--recipe", help="start a session replaying this recipe")
    start.add_argument("--session", help="resume the session in this folder")
    p.add_argument("--set", type=setting, action="append", default=[], metavar="NAME=VALUE",
                   help="a param of the --kernel session")  # fmt: skip
    p.add_argument("--platform", help="the session's platform (default platforms/snax_forge.json)")
    p.add_argument("--port", type=int, default=server.EDIT_PORT,
                   help=f"port on 127.0.0.1 (default {server.EDIT_PORT})")  # fmt: skip
    p.add_argument("--verbose", action="store_true", help="log every request")
    args = p.parse_args(argv)
    if args.set and not args.kernel:
        p.error("--set goes with --kernel")
    try:
        srv = server.make_edit_server(args.port, args.verbose, args.platform)
        state = srv.loaded
        if args.kernel:
            state.open({"kernel": args.kernel, "params": dict(args.set)})
        elif args.recipe:
            state.open({"recipe": args.recipe})
        elif args.session:
            state.open({"session": args.session})
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    what = state.session.name if state.session else "the editor (no session yet)"
    return viewer.serve(srv, what)


if __name__ == "__main__":
    sys.exit(main())
