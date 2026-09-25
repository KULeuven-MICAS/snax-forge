"""The DFG viewer's HTTP server (VIS5, D76, D81): the viewer's server in a second mode.

Same stdlib ``ThreadingHTTPServer`` on 127.0.0.1, same static files and
the same ``Handler`` for them (server.py), with the page ``dfg.html`` at
``/`` and these routes instead of the run routes:

    GET  /api/graphs          name, path, graph name and load error per file
    GET  /api/graph/<name>    the view of one file (api.graph_view)
    POST /api/reload          read every file again

A file that does not load is not an HTTP error: its entry carries the
message, so the page shows it in that file's panel.
"""

from __future__ import annotations

from collections.abc import Sequence
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from ..server import HOST, Handler
from . import api

DEFAULT_PORT = 8766  # one more than the run viewer's, so both can run at once


class DfgHandler(Handler):
    def do_POST(self) -> None:
        if urlsplit(self.path).path.rstrip("/") != "/api/reload":
            self._error(HTTPStatus.NOT_FOUND, f"no POST route {self.path}")
            return
        try:
            names = self.server.graphs.reload()
        except OSError as e:  # a file or directory gone: keep the old entries
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"reload failed: {e}")
            return
        self._json({"graphs": names})

    def _api_get(self, parts: list[str], q: dict[str, list[str]]) -> None:
        graphs = self.server.graphs
        if parts == ["graphs"]:
            self._json(graphs.summaries())
            return
        if len(parts) == 2 and parts[0] == "graph":
            entry = graphs.get(parts[1])
            if entry is None:
                self._error(HTTPStatus.NOT_FOUND, f"no graph named {parts[1]!r}")
                return
            self._json(api.graph_view(entry))
            return
        self._error(HTTPStatus.NOT_FOUND, "unknown API route /api/" + "/".join(parts))


class DfgServer(ThreadingHTTPServer):
    """A ThreadingHTTPServer carrying the GraphSet it serves."""

    daemon_threads = True
    index = "dfg.html"

    def __init__(self, graphs: api.GraphSet, port: int, verbose: bool = False) -> None:
        self.graphs = graphs
        self.verbose = verbose
        super().__init__((HOST, port), DfgHandler)

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.server_address[1]}/"


def make_dfg_server(paths: Sequence[str | Path], port: int = DEFAULT_PORT, verbose: bool = False):
    """Load the files and bind the server (port 0 = any free port); does not serve yet."""
    return DfgServer(api.GraphSet(paths), port, verbose)
