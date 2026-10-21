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
    GET  /api/run/<name>                    run.json, profile, class intervals,
                                            tasks per block
    GET  /api/run/<name>/events?from=A&to=B&src=S[&src=T]&k=K[&k=L]
                                            events with A <= t < B, of these
                                            sources and kinds only if src / k is
                                            given (repeat it, or separate names
                                            with commas)
    GET  /api/run/<name>/fifo               FIFO busy window per streamer
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
    GET  /api/run/<name>/movement           residency, patterns and conflict
                                            counts (movement.py)
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

All the work is in api.py; this module only parses paths and queries and
turns results and errors into responses: 404 for an unknown run, memory or
file, 400 for a bad query (a mark the run lacks included, with the reason), 500 with the message when a reload fails (the old runs stay
loaded). A movement or schedule route answers a run below the level it needs
(beat, or stall for the conflicts; task for the schedule, beat for its
detail) with ``available`` false and the reason.
"""

from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

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


def _ints_arg(q: dict[str, list[str]], key: str) -> list[int]:
    """A list of integers, repeated or separated by commas."""
    try:
        return [int(x) for v in q.get(key, []) for x in v.split(",") if x]
    except ValueError as e:
        raise _BadRequest(f"{key} must be integers separated by commas") from e


def _str_arg(q: dict[str, list[str]], key: str) -> str | None:
    vals = q.get(key)
    return vals[-1] if vals else None


class Handler(BaseHTTPRequestHandler):
    """One request. ``server.runs`` is the RunSet (see ``make_server``)."""

    server: ViewerServer  # narrowed for type checkers
    protocol_version = "HTTP/1.1"

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
        if urlsplit(self.path).path.rstrip("/") != "/api/reload":
            self._error(HTTPStatus.NOT_FOUND, f"no POST route {self.path}")
            return
        try:
            names = self.server.runs.reload()
        except READ_ERRORS as e:  # keep the old runs, say why
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"reload failed: {e}")
            return
        self._json({"runs": names})

    def _api_get(self, parts: list[str], q: dict[str, list[str]]) -> None:
        runs = self.server.runs
        if parts == ["runs"]:
            self._json(runs.summaries())
            return
        if len(parts) == 4 and parts[0] == "run" and parts[2:] == ["memory", "cycle"]:
            rv = runs.get(parts[1])
            if rv is None:
                self._error(HTTPStatus.NOT_FOUND, f"no run named {parts[1]!r}")
                return
            t = _int_arg(q, "t", None)
            if t is None:
                raise _BadRequest("give t=T, the cycle")
            self._json(api.memory_cycle(rv, t))
            return
        if len(parts) == 4 and parts[0] == "run" and parts[2:] == ["schedule", "detail"]:
            rv = runs.get(parts[1])
            if rv is None:
                self._error(HTTPStatus.NOT_FOUND, f"no run named {parts[1]!r}")
                return
            a, b = _int_arg(q, "from", None), _int_arg(q, "to", None)
            if a is None or b is None:
                raise _BadRequest("give from=A and to=B, the cycles")
            self._json(api.schedule_detail(rv, a, b))
            return
        if len(parts) == 5 and parts[0] == "run" and parts[2:] == ["memory", parts[3], "rows"]:
            rv = runs.get(parts[1])
            if rv is None or parts[3] not in api.memories(rv):
                what = f"run {parts[1]!r}" if rv is None else f"memory {parts[3]!r}"
                self._error(HTTPStatus.NOT_FOUND, f"no {what}")
                return
            a = _int_arg(q, "from", 0) or 0
            b = _int_arg(q, "to", a + api.MAX_ROWS)
            try:
                self._json(api.memory_rows(rv, parts[3], a, b or 0, _str_arg(q, "marks")))
            except api.MemoryViewError as e:
                raise _BadRequest(str(e)) from e
            return
        if len(parts) in (2, 3) and parts[0] == "run":
            rv = runs.get(parts[1])
            if rv is None:
                self._error(HTTPStatus.NOT_FOUND, f"no run named {parts[1]!r}")
                return
            if len(parts) == 2:
                self._json(api.run_detail(rv))
                return
            if parts[2] == "events":
                srcs = [s for v in q.get("src", []) for s in v.split(",") if s]
                kinds = [k for v in q.get("k", []) for k in v.split(",") if k]
                a = _int_arg(q, "from", 0)
                b = _int_arg(q, "to", None)
                evs = api.events_window(rv, a or 0, b, srcs or None, kinds or None)
                self._json({"level": rv.trace_level, "from": a, "to": b, "events": evs})
                return
            if parts[2] == "fifo":
                self._json(api.fifo_windows(rv))
                return
            if parts[2] == "memory":
                try:
                    self._json(api.memory_view(rv, _str_arg(q, "marks")))
                except api.MemoryViewError as e:
                    raise _BadRequest(str(e)) from e
                return
            if parts[2] == "movement":
                self._json(api.movement_view(rv))
                return
            if parts[2] == "journey":
                region = _str_arg(q, "region")
                index = _str_arg(q, "index")
                if not region or index is None:
                    raise _BadRequest("give region=R and index=I (a flat index or i,j,...)")
                try:
                    idx = [int(x) for x in index.split(",")]
                    self._json(api.journey_view(rv, region, idx[0] if len(idx) == 1 else idx))
                except ValueError as e:
                    raise _BadRequest(str(e)) from e
                return
            if parts[2] == "schedule":
                self._json(api.schedule_view(rv, _ints_arg(q, "open"), _int_arg(q, "cycle", None)))
                return
            if parts[2] == "cluster":
                t = _int_arg(q, "t", None)
                if t is None:
                    raise _BadRequest("give t=T, the cycle")
                self._json(api.cluster_view(rv, t))
                return
            if parts[2] == "step":
                t, d, by = _int_arg(q, "t", None), _int_arg(q, "dir", None), _str_arg(q, "by")
                if t is None or d not in (1, -1) or by not in ("event", "boundary"):
                    raise _BadRequest("give t=T, dir=1 or -1 and by=event or boundary")
                self._json(api.step_view(rv, t, d, by))
                return
            if parts[2] == "conflicts":
                a = _int_arg(q, "from", 0)
                b = _int_arg(q, "to", None)
                self._json(api.conflicts_view(rv, a or 0, b))
                return
        self._error(HTTPStatus.NOT_FOUND, "unknown API route /api/" + "/".join(parts))

    def _static(self, parts: list[str]) -> None:
        """A file below static/, index.html for /; nothing outside it is served."""
        index = getattr(self.server, "index", "index.html")  # the DFG viewer's is dfg.html
        path = (STATIC / "/".join(parts or [index])).resolve()
        if STATIC not in path.parents or not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, f"no file {self.path}")
            return
        ctype = _TYPES.get(path.suffix) or mimetypes.guess_type(path.name)[0]
        self._send(HTTPStatus.OK, path.read_bytes(), ctype or "application/octet-stream")


class ViewerServer(ThreadingHTTPServer):
    """A ThreadingHTTPServer that carries the RunSet it serves."""

    daemon_threads = True

    def __init__(self, runs: api.RunSet, port: int, verbose: bool = False) -> None:
        self.runs = runs
        self.verbose = verbose
        super().__init__((HOST, port), Handler)

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.server_address[1]}/"


def make_server(dirs: list[str | Path], port: int = DEFAULT_PORT, verbose: bool = False):
    """Load the runs and bind the server (port 0 = any free port); does not serve yet."""
    return ViewerServer(api.RunSet(dirs), port, verbose)


__all__ = ["DEFAULT_PORT", "HOST", "Handler", "ViewerServer", "make_server"]
