"""Command line of the DFG viewer in edit mode (server.py).

    python -m snax_forge.viz.dfg.edit [--kernel K --set NAME=VALUE ... | --recipe R |
                                       --session DIR] [--platform P] [--tech T] [--port 8767]
                                       [--verbose]

Serves the editor on 127.0.0.1 until Ctrl-C with one session (sandbox/session.py): a new one on
kernel K (``--set types=i8`` its type preset, ``--set dtypes.C=int16`` a dtype, the rest its
params), one replaying recipe R (a path or a name, ``find_recipe``), or the session in DIR
resumed; without one the page shows its start dialog. The cost is on ``--tech`` (default
nangate45). The session's files are in CONTRACTS, "Recipe". pixi: ``edit-dfg``.

Exit codes: 0 stopped with Ctrl-C, 2 a start option that fails (its message) or a port taken.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from snax_forge.char import DEFAULT_TECH, TECHS
from snax_forge.sandbox.__main__ import setting

from .. import server as viewer
from . import server


def kernel_start(sets: dict[str, int | str]) -> dict[str, Any]:
    """A ``--kernel`` session's ``--set`` values as the start's body: ``types``, ``dtypes.<name>``
    (into ``dtypes``) and the params, as a recipe's ``--set`` takes them (``Recipe.with_params``)."""
    params = dict(sets)
    body: dict[str, Any] = {}
    if "types" in params:
        body["types"] = str(params.pop("types"))
    dtypes = {
        k.partition(".")[2]: str(params.pop(k))
        for k in [k for k in params if k.startswith("dtypes.")]
    }
    if dtypes:
        body["dtypes"] = dtypes
    return {**body, "params": params}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m snax_forge.viz.dfg.edit",
                                description=__doc__.split("\n\n")[0])  # fmt: skip
    start = p.add_mutually_exclusive_group()
    start.add_argument("--kernel", help="start a session on this kernel's import")
    start.add_argument("--recipe", help="start a session replaying this recipe (a path or a name)")
    start.add_argument("--session", help="resume the session in this folder")
    p.add_argument("--set", type=setting, action="append", default=[], metavar="NAME=VALUE",
                   help="a param of the --kernel session, or types=PRESET, dtypes.NAME=DTYPE")  # fmt: skip
    p.add_argument("--platform", help="the session's platform (default platforms/snax_forge.json)")
    p.add_argument("--tech", choices=TECHS, default=DEFAULT_TECH,
                   help=f"the technology of the cost estimate (default {DEFAULT_TECH})")  # fmt: skip
    p.add_argument("--port", type=int, default=server.EDIT_PORT,
                   help=f"port on 127.0.0.1 (default {server.EDIT_PORT})")  # fmt: skip
    p.add_argument("--verbose", action="store_true", help="log every request")
    args = p.parse_args(argv)
    if args.set and not args.kernel:
        p.error("--set goes with --kernel")
    try:
        srv = server.make_edit_server(args.port, args.verbose, args.platform, tech=args.tech)
        state = srv.loaded
        if args.kernel:
            state.open({"kernel": args.kernel, **kernel_start(dict(args.set))})
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
