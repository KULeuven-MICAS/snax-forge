"""Command line of the DFG viewer.

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

import sys
from collections.abc import Sequence

from .. import server as viewer
from . import server


def main(argv: Sequence[str] | None = None) -> int:
    return viewer.main(
        argv, "python -m snax_forge.viz.dfg", __doc__,
        ("paths", "FILE|DIR", ".snaxdfg file or directory"),
        server.DEFAULT_PORT, server.make_dfg_server, (OSError,),
    )  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
