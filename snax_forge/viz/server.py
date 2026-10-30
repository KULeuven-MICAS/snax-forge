"""Local HTTP server of the visualiser (D55).

What it does
------------
Serves the static viewer (static/: index.html, plain JS modules, CSS) and
a small JSON API over the runs given on the command line. It is for a
human on the same machine: it binds to 127.0.0.1, uses only the standard
library (``http.server.ThreadingHTTPServer``) and needs no internet, since
the viewer loads no external file.

Routes (every answer is JSON except the static files):

    GET  /api/runs                          name, trace level, total cycles per run
    GET  /api/run/<name>                    run.json, the profile, the cluster,
                                            the trace's level and window
    GET  /api/run/<name>/memory[?marks=M]    per memory its geometry, regions and
                                            folded lines, the marks the run has
                                            and why it lacks the others; with
                                            marks, folded by mark M per word
                                            (index.py's registry: conflicts,
                                            arrival, use, wait)
    GET  /api/run/<name>/memory/<mem>/rows?from=A&to=B[&marks=M]
                                            rows A <= r < B in full, at most 256
    GET  /api/run/<name>/memory/cycle?t=T   the cycle overlay of cycle T: its accesses
                                            and the reads whose data comes back in it
    GET  /api/run/<name>/journey?region=R&index=I
                                            every hop of element R[I] (I a flat
                                            index, or indices separated by commas)
    GET  /api/run/<name>/conflicts?from=A&to=B
                                            the L1 conflicts with A <= t < B,
                                            placed on the layout
    GET  /api/run/<name>/schedule[?open=A,B&cycle=C]
                                            the lanes and the segments of the
                                            window, folds starting at A, B or
                                            holding C opened (schedule.py)
    GET  /api/run/<name>/schedule/detail?from=A&to=B
                                            the beat and stall lines of A <= t < B
                                            per lane
    GET  /api/run/<name>/cluster?t=T        the data side of the cluster at T
                                            (cluster.py)
    GET  /api/run/<name>/step?t=T&dir=D&by=B
                                            the next (D 1) or previous (D -1)
                                            cycle with an event (B event) or a
                                            task boundary (B boundary)
    POST /api/reload                        read every run directory again

Every POST, on this viewer and the DFG viewer, needs ``Content-Type: application/json``
(D55): another type is a 415, so a page of another origin cannot post a form to it. Its body,
when it has one, is a JSON object; another body is a 400.

All the work is in api.py; this module only parses paths and queries and
turns results and errors into responses: 404 for an unknown run, memory or
file, 400 for a bad query (a mark the run lacks included, with the reason), 500 with the message when a reload fails (the old runs stay
loaded). A movement or schedule route answers a run below the level it needs
(beat, or stall for the conflicts; task for the schedule, beat for its
detail) with ``available`` false and the reason.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from collections.abc import Callable, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from snax_forge.common import LoadedSet

from . import api

STATIC = Path(__file__).resolve().parent / "static"
HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# What reading a run directory can raise: a missing file, bad JSON, a key or
# field missing from a file of another version.
READ_ERRORS = (OSError, ValueError, KeyError, TypeError)

# Served types; .js must be a JavaScript type or the browser refuses the module.
_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


class _BadRequest(Exception):
    """A malformed query; answered with 400."""


def _int_arg(q: dict[str, list[str]], key: str, default: int | None) -> int | None:
    vals = q.get(key)
    if not vals:
        return default
    try:
        return int(vals[-1])
    except ValueError as e:
        raise _BadRequest(f"{key} must be an integer, got {vals[-1]!r}") from e


def _need_int(q: dict[str, list[str]], key: str, say: str) -> int:
    """``key``'s int; a 400 saying ``say`` when it is not given."""
    v = _int_arg(q, key, None)
    if v is None:
        raise _BadRequest(say)
    return v


def _ints_arg(q: dict[str, list[str]], key: str) -> list[int]:
    """A list of integers, repeated or separated by commas."""
    try:
        return [int(x) for v in q.get(key, []) for x in v.split(",") if x]
    except ValueError as e:
        raise _BadRequest(f"{key} must be integers separated by commas") from e


def _str_arg(q: dict[str, list[str]], key: str) -> str | None:
    vals = q.get(key)
    return vals[-1] if vals else None


class _NotFound(Exception):
    """A 404 with its text: a route part that names nothing in the run."""


def _memory(rv: api.RunView, q: dict[str, list[str]]) -> Any:
    try:
        return api.memory_view(rv, _str_arg(q, "marks"))
    except api.MemoryViewError as e:
        raise _BadRequest(str(e)) from e


def _memory_rows(rv: api.RunView, q: dict[str, list[str]], mem: str) -> Any:
    if mem not in api.memories(rv):
        raise _NotFound(f"no memory {mem!r}")
    a = _int_arg(q, "from", 0)
    b = _int_arg(q, "to", a + api.MAX_ROWS)
    try:
        return api.memory_rows(rv, mem, a, b, _str_arg(q, "marks"))
    except api.MemoryViewError as e:
        raise _BadRequest(str(e)) from e


def _journey(rv: api.RunView, q: dict[str, list[str]]) -> Any:
    region = _str_arg(q, "region")
    index = _str_arg(q, "index")
    if not region or index is None:
        raise _BadRequest("give region=R and index=I (a flat index or i,j,...)")
    try:
        idx = [int(x) for x in index.split(",")]
        return api.journey_view(rv, region, idx[0] if len(idx) == 1 else idx)
    except ValueError as e:
        raise _BadRequest(str(e)) from e


def _schedule_detail(rv: api.RunView, q: dict[str, list[str]]) -> Any:
    a, b = _int_arg(q, "from", None), _int_arg(q, "to", None)
    if a is None or b is None:
        raise _BadRequest("give from=A and to=B, the cycles")
    return api.schedule_detail(rv, a, b)


def _step(rv: api.RunView, q: dict[str, list[str]]) -> Any:
    t = _need_int(q, "t", "give t=T, dir=1 or -1 and by=event or boundary")
    try:  # cluster.step checks dir and by
        return api.step_view(rv, t, _int_arg(q, "dir", None), _str_arg(q, "by"))
    except ValueError as e:
        raise _BadRequest(str(e)) from e


# /api/run/<name>/<route>: route -> its answer from the run and the query; a ``*`` part is any
# one part, passed on after the query
ROUTES: dict[tuple[str, ...], Callable[..., Any]] = {
    (): lambda rv, q: api.run_detail(rv),
    ("memory",): _memory,
    ("memory", "cycle"): lambda rv, q: api.memory_cycle(rv, _need_int(q, "t", "give t=T, the cycle")),
    ("memory", "*", "rows"): _memory_rows,
    ("journey",): _journey,
    ("schedule",): lambda rv, q: api.schedule_view(rv, _ints_arg(q, "open"),
                                                   _int_arg(q, "cycle", None)),
    ("schedule", "detail"): _schedule_detail,
    ("cluster",): lambda rv, q: api.cluster_view(rv, _need_int(q, "t", "give t=T, the cycle")),
    ("step",): _step,
    ("conflicts",): lambda rv, q: api.conflicts_view(rv, _int_arg(q, "from", 0),
                                                     _int_arg(q, "to", None)),
}  # fmt: skip


def _route(route: list[str]) -> tuple[Callable[..., Any], list[str]] | None:
    """The answer for ``route`` and the parts its ``*`` matched, or None."""
    for key, answer in ROUTES.items():
        if len(key) == len(route) and all(k in ("*", r) for k, r in zip(key, route, strict=True)):
            return answer, [r for k, r in zip(key, route, strict=True) if k == "*"]
    return None


class Handler(BaseHTTPRequestHandler):
    """One request. ``server.loaded`` is the RunSet (see ``make_server``); the DFG viewer's
    handler (dfg/server.py) serves its GraphSet with its own routes. ``key`` names the set in
    the reload's answer and ``reload_errors`` are what its reload may raise."""

    server: ViewerServer  # narrowed for type checkers
    protocol_version = "HTTP/1.1"
    key = "runs"
    reload_errors: tuple[type[Exception], ...] = READ_ERRORS

    # -- plumbing -------------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.verbose:
            super().log_message(format, *args)

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")  # a reload must show new data
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _error(self, status: int, msg: str) -> None:
        self._json({"error": msg}, status)

    # -- routing --------------------------------------------------------------

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        parts = [unquote(p) for p in url.path.split("/") if p]
        try:
            if parts[:1] == ["api"]:
                self._api_get(parts[1:], parse_qs(url.query))
            else:
                self._static(parts)
        except _BadRequest as e:
            self._error(HTTPStatus.BAD_REQUEST, str(e))

    do_HEAD = do_GET

    def do_POST(self) -> None:
        """A POST with a JSON body (module doc), then ``_api_post`` on its route."""
        ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if ctype != "application/json":
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                        f"a POST needs Content-Type: application/json, not {ctype or 'none'!r}")  # fmt: skip
            return
        try:
            body = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as e:
            self._error(HTTPStatus.BAD_REQUEST, f"the body is not JSON: {e}")
            return
        if not isinstance(body, dict):
            self._error(HTTPStatus.BAD_REQUEST, "the body must be a JSON object")
            return
        parts = [unquote(p) for p in urlsplit(self.path).path.split("/") if p]
        self._api_post(parts[1:] if parts[:1] == ["api"] else [None, *parts], body)

    def _api_post(self, parts: list[Any], body: dict[str, Any]) -> None:
        if parts != ["reload"]:
            self._error(HTTPStatus.NOT_FOUND, f"no POST route {self.path}")
            return
        self._reload()

    def _reload(self) -> None:
        try:
            names = self.server.loaded.reload()
        except self.reload_errors as e:  # keep the old set, say why
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"reload failed: {e}")
            return
        self._json({self.key: names})

    def _api_get(self, parts: list[str], q: dict[str, list[str]]) -> None:
        runs = self.server.loaded
        if parts == ["runs"]:
            self._json(runs.summaries())
            return
        if len(parts) < 2 or parts[0] != "run":
            self._error(HTTPStatus.NOT_FOUND, "unknown API route /api/" + "/".join(parts))
            return
        rv = runs.get(parts[1])
        if rv is None:
            self._error(HTTPStatus.NOT_FOUND, f"no run named {parts[1]!r}")
            return
        found = _route(parts[2:])
        if found is None:
            self._error(HTTPStatus.NOT_FOUND, "unknown API route /api/" + "/".join(parts))
            return
        answer, args = found
        try:
            self._json(answer(rv, q, *args))
        except _NotFound as e:
            self._error(HTTPStatus.NOT_FOUND, str(e))

    def _static(self, parts: list[str]) -> None:
        """A file below static/, index.html for /; nothing outside it is served."""
        path = (STATIC / "/".join(parts or [self.server.index])).resolve()
        if STATIC not in path.parents or not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, f"no file {self.path}")
            return
        ctype = _TYPES.get(path.suffix) or mimetypes.guess_type(path.name)[0]
        self._send(HTTPStatus.OK, path.read_bytes(), ctype or "application/octet-stream")


class ViewerServer(ThreadingHTTPServer):
    """A ThreadingHTTPServer on 127.0.0.1 that carries the set it serves (``loaded``: the
    runs, or the DFG viewer's files) and the page at ``/`` (``index``)."""

    daemon_threads = True

    def __init__(self, loaded: LoadedSet, port: int, verbose: bool = False,
                 handler: type[Handler] = Handler, index: str = "index.html") -> None:  # fmt: skip
        self.loaded, self.verbose, self.index = loaded, verbose, index
        super().__init__((HOST, port), handler)

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.server_address[1]}/"


def make_server(dirs: list[str | Path], port: int = DEFAULT_PORT, verbose: bool = False):
    """Load the runs and bind the server (port 0 = any free port); does not serve yet."""
    return ViewerServer(api.RunSet(dirs), port, verbose)


def main(argv: Sequence[str] | None, prog: str, doc: str, paths: tuple[str, str, str],
         port: int, make: Callable[[list[str], int, bool], ViewerServer],
         errors: tuple[type[Exception], ...]) -> int:  # fmt: skip
    """A viewer's command line (``paths``: its argument's name, metavar and help): load, then
    serve until Ctrl-C. Exit 0 when stopped, 2 when ``make`` raises one of ``errors``."""
    p = argparse.ArgumentParser(prog=prog, description=doc.split("\n\n")[0])
    name, metavar, help_ = paths
    p.add_argument(name, nargs="+", metavar=metavar, help=help_)
    p.add_argument("--port", type=int, default=port, help=f"port on 127.0.0.1 (default {port})")
    p.add_argument("--verbose", action="store_true", help="log every request")
    args = p.parse_args(argv)
    try:
        srv = make(getattr(args, name), args.port, args.verbose)
    except errors as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return serve(srv, ", ".join(x["name"] for x in srv.loaded.summaries()))


def serve(srv: ViewerServer, what: str) -> int:
    """Serve until Ctrl-C, saying ``what`` is served where; 0."""
    print(f"serving {what} at {srv.url} (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


__all__ = ["DEFAULT_PORT", "HOST", "Handler", "ViewerServer", "main", "make_server", "serve"]
