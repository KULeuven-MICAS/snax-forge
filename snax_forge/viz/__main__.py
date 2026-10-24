"""Command line of the visualiser.

    python -m snax_forge.viz DIR [DIR ...] [--port 8765] [--verbose]

Loads each flow run (``out/flow/<name>/`` or its ``run/``) once and serves the viewer on
127.0.0.1 until Ctrl-C. The page's Reload button reads the directories
again; nothing watches the files. ``--port 0`` takes any free port.

Exit codes: 0 stopped with Ctrl-C, 2 a directory cannot be read or the
port is taken.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from . import server


def main(argv: Sequence[str] | None = None) -> int:
    return server.main(
        argv, "python -m snax_forge.viz", __doc__,
        ("dirs", "DIR", "a flow folder (out/flow/<name>) or its run/"),
        server.DEFAULT_PORT, server.make_server,
        server.READ_ERRORS,  # a folder that is not a flow run, or a taken port
    )  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
