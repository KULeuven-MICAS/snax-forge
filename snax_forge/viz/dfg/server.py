"""The DFG viewer's HTTP server (D55): the viewer's server for the graph page, in two modes.

Same stdlib ``ThreadingHTTPServer`` on 127.0.0.1, same static files and the same ``Handler``
for them (server.py), with the page ``dfg.html`` at ``/``. One handler serves both modes;
``GET /api/mode`` says which (``{"mode": "view"}`` or ``"edit"``). Every POST needs
``Content-Type: application/json`` (415 otherwise, server.py) and a JSON object body.

**View mode** (``make_dfg_server``, ``pixi run view-dfg``): the given ``.snaxdfg`` files, read
once and on Reload, as a read-only history, each file a step in the order given (a directory as
its files in natural order). The estimate is on ``--platform`` when given, else on none.

    GET  /api/steps           per step its name, label (``1 split_map``) and load error
    GET  /api/step/<k>        step k's view (api.step_view: the drawing, its compact texts and
                              Details, the marks and the estimate); a file that does not load
                              is its entry with the error, not an HTTP error
    POST /api/reload          read every file again
    GET  /api/graphs          the old page's routes (dfg.js), until R5a: name, path, graph name
    GET  /api/graph/<name>    and load error per file, and the view of one file

**Edit mode** (``make_edit_server``, ``pixi run edit-dfg``): one session (sandbox/session.py)
behind routes; the page opens or starts one, then edits it. Sessions, saved recipes and step
files go to the roots the server was given (sandbox/session.py's ``Roots``).

    GET  /api/kernels               the staged kernels (kernels/) and their symbols, each with
                                    its load error if it has one
    GET  /api/recipes               the recipes (name, kernel, params, steps)
    GET  /api/sessions              the sessions (name, kernel, steps, cursor, platform)
    POST /api/session               start or open the session: ``{kernel, params, symbols?,
                                    dtypes?, platform?}``, ``{recipe}`` (a path, or a name in
                                    the recipes) or ``{session}`` (a folder, or a name in the
                                    sessions); answers as GET
    GET  /api/session               its name, kernel, params, symbols, steps, cursor, platform
    GET  /api/step/<k>              step k's view, with its diff against step k − 1
    GET  /api/step/<k>/site/<id>    Session.sites on node or container ``id``, bind's with the
                                    bind list
    POST /api/apply                 ``{transform, params, drop?}``: Session.apply
    POST /api/cursor                ``{k}``: Session.move
    POST /api/params                ``{params}``: Session.set_params
    POST /api/save                  ``{name, force?}``: Session.save
    POST /api/export                ``{k}``: Session.export, the file's ``path``

A refusal is a 400 with ``{code, message}`` (and ``problems`` for a save); a route before a
session is open is a 400 ``session.none``; an unknown route or step is a 404, as in view mode.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from http import HTTPStatus
from pathlib import Path
from typing import Any

from ..server import Handler, ViewerServer
from . import api

DEFAULT_PORT = 8766  # one more than the run viewer's, so both can run at once
EDIT_PORT = 8767


class _Refused(Exception):
    """A 400 with ``{code, message}`` and more."""

    def __init__(self, code: str, message: str, **more: Any) -> None:
        super().__init__(message)
        self.body = {"code": code, "message": message, **more}


def _label(name: str) -> str:
    """A step's label: ``1_split_map`` -> ``1 split_map``."""
    head, _, tail = name.partition("_")
    return f"{head} {tail}" if head.isdigit() and tail else name


class EditState:
    """Edit mode's state: the session (None until one is opened), the roots and the default
    platform. ``reload`` and ``summaries`` make it stand where the view mode's GraphSet does."""

    def __init__(self, roots: Any = None, platform: str | None = None) -> None:
        from snax_forge.sandbox.session import PLATFORM, Roots

        self.roots = roots or Roots()
        self.platform = platform or PLATFORM
        self.session: Any = None
        self.lock = threading.Lock()  # opening a session; the session has its own
        self._kernels: list[dict[str, Any]] | None = None

    def summaries(self) -> list[dict[str, Any]]:
        return [] if self.session is None else [{"name": self.session.name}]

    def reload(self) -> list[str]:
        return [x["name"] for x in self.summaries()]

    def kernels(self) -> list[dict[str, Any]]:
        """Every staged kernel and its symbols (the names in its descriptors' shapes), once."""
        if self._kernels is None:
            from snax_forge.sdfg.loader import kernel_paths, load

            out = []
            for name in sorted(kernel_paths()):
                try:
                    spec = load(name)
                    symbols = sorted({str(s) for d in spec.descriptors.values()
                                      for dim in getattr(d, "shape", ())
                                      for s in getattr(dim, "free_symbols", ())})  # fmt: skip
                    out.append({"name": name, "symbols": symbols, "error": None})
                except Exception as e:  # noqa: BLE001 - a kernel that does not load says why
                    out.append({"name": name, "symbols": [], "error": f"{type(e).__name__}: {e}"})
            self._kernels = out
        return self._kernels

    def recipes(self) -> list[dict[str, Any]]:
        from snax_forge.sandbox import Recipe

        out = []
        for p in sorted(self.roots.recipes.glob("*.json")):
            try:
                r = Recipe.load(p)
                out.append({"name": r.name, "path": str(p), "kernel": r.kernel,
                            "params": dict(r.params), "steps": len(r.steps), "error": None})  # fmt: skip
            except (OSError, ValueError, KeyError, TypeError) as e:
                out.append({"name": p.stem, "path": str(p), "error": str(e)})
        return out

    def sessions(self) -> list[dict[str, Any]]:
        import json

        from snax_forge.sandbox import Recipe
        from snax_forge.sandbox.session import RECIPE_JSON, SESSION_JSON

        out = []
        for d in sorted(p for p in self.roots.sessions.glob("*") if (p / SESSION_JSON).is_file()):
            try:
                r = Recipe.load(d / RECIPE_JSON)
                state = json.loads((d / SESSION_JSON).read_text())
                out.append({"name": d.name, "kernel": r.kernel, "steps": len(r.steps),
                            "cursor": state["cursor"], "platform": state["platform"],
                            "error": None})  # fmt: skip
            except (OSError, ValueError, KeyError, TypeError) as e:
                out.append({"name": d.name, "error": str(e)})
        return out

    def open(self, body: dict[str, Any]) -> Any:
        """Start or open the session ``body`` asks for (module doc)."""
        from snax_forge.sandbox.session import Session

        platform = body.get("platform") or self.platform
        if "kernel" in body:
            session = Session.from_kernel(body["kernel"], dict(body.get("params") or {}),
                                          body.get("symbols"), body.get("dtypes"), platform,
                                          body.get("name"), self.roots)  # fmt: skip
        elif "recipe" in body:
            path = Path(body["recipe"])
            if not path.is_file():
                path = self.roots.recipes / f"{body['recipe']}.json"
            session = Session.from_recipe(path, platform, self.roots)
        elif "session" in body:
            folder = Path(body["session"])
            if not folder.is_dir():
                folder = self.roots.sessions / body["session"]
            session = Session.resume(folder, self.roots)
        else:
            raise _Refused("session.start", "give kernel (with params), recipe or session")
        with self.lock:
            self.session = session
        return session

    def need(self) -> Any:
        if self.session is None:
            raise _Refused("session.none", "no session is open: POST /api/session first")
        return self.session


def session_summary(s: Any) -> dict[str, Any]:
    """GET /api/session."""
    r = s.recipe
    return {"name": s.name, "kernel": r.kernel, "params": dict(r.params),
            "symbols": dict(r.symbols), "dtypes": dict(r.dtypes),
            "steps": [x.to_dict() for x in r.steps], "cursor": s.cursor,
            "platform": s.platform_path}  # fmt: skip


def edit_step(s: Any, k: int) -> dict[str, Any]:
    """GET /api/step/<k> in edit mode: the step's view with its diff."""
    from snax_forge.dfg import DfgError

    if not 0 <= k <= s.n:
        raise LookupError(f"no step {k}: the steps are 0..{s.n}")
    name = s.step_name(k)
    entry = api.Entry(name, s.folder / "export" / f"{name}.snaxdfg", s.graphs[k], None)
    try:
        est = s.estimate(k)
    except (ValueError, KeyError, TypeError, DfgError):
        est = None
    return api.step_view(entry, _label(name), s.platform, s.marks(k), s.diff(k), est)


class DfgHandler(Handler):
    key = "graphs"
    reload_errors = (OSError,)  # a file or directory gone: keep the old entries

    @property
    def _edit(self) -> bool:
        return isinstance(self.server.loaded, EditState)

    def _refuse(self, e: _Refused) -> None:
        self._json(e.body, HTTPStatus.BAD_REQUEST)

    def _api_get(self, parts: list[str], q: dict[str, list[str]]) -> None:
        if parts == ["mode"]:
            self._json({"mode": "edit" if self._edit else "view"})
            return
        try:
            found = self._edit_get(parts) if self._edit else self._view_get(parts)
        except _Refused as e:
            self._refuse(e)
            return
        except LookupError as e:
            self._error(HTTPStatus.NOT_FOUND, str(e).strip("'\""))
            return
        if found is None:
            self._error(HTTPStatus.NOT_FOUND, "unknown API route /api/" + "/".join(parts))
            return
        self._json(found)

    def _view_get(self, parts: list[str]) -> Any:
        graphs = self.server.loaded
        if parts == ["graphs"]:
            return graphs.summaries()
        if len(parts) == 2 and parts[0] == "graph":
            entry = graphs.get(parts[1])
            if entry is None:
                raise LookupError(f"no graph named {parts[1]!r}")
            return api.graph_view(entry, self.server.platform)
        entries = list(graphs.values())
        if parts == ["steps"]:
            return [{"name": e.name, "label": _label(e.name), "error": e.error} for e in entries]
        if len(parts) == 2 and parts[0] == "step":
            k = _index(parts[1], len(entries) - 1)
            return api.step_view(entries[k], _label(entries[k].name), self.server.platform)
        return None

    def _edit_get(self, parts: list[str]) -> Any:
        state = self.server.loaded
        if parts == ["kernels"]:
            return state.kernels()
        if parts == ["recipes"]:
            return state.recipes()
        if parts == ["sessions"]:
            return state.sessions()
        if parts == ["session"]:
            return session_summary(state.need())
        if len(parts) in (2, 4) and parts[0] == "step":
            s = state.need()
            k = _index(parts[1], s.n)
            if len(parts) == 2:
                return edit_step(s, k)
            if parts[2] == "site":
                return s.sites(k, parts[3])
        return None

    def _api_post(self, parts: list[Any], body: dict[str, Any]) -> None:
        if not self._edit:
            super()._api_post(parts, body)
            return
        state = self.server.loaded
        try:
            out = self._edit_post(state, parts, body)
        except _Refused as e:
            self._refuse(e)
            return
        except (ValueError, KeyError, TypeError, OSError) as e:  # SandboxError, DfgError, ...
            code = getattr(e, "code", None) or type(e).__name__
            self._refuse(_Refused(code, str(e).strip("'\"")))
            return
        if out is None:
            self._error(HTTPStatus.NOT_FOUND, f"no POST route {self.path}")
            return
        if isinstance(out, dict) and out.get("ok") is False:
            self._json({k: v for k, v in out.items() if k != "ok"}, HTTPStatus.BAD_REQUEST)
            return
        self._json(out)

    def _edit_post(self, state: EditState, parts: list[Any], body: dict[str, Any]) -> Any:
        if parts == ["session"]:
            return session_summary(state.open(body))
        if parts == ["reload"]:
            return {"graphs": state.reload()}
        if len(parts) != 1 or parts[0] not in ("apply", "cursor", "params", "save", "export"):
            return None
        s = state.need()
        route = parts[0]
        if route == "apply":
            _needs(body, "transform")
            return s.apply(
                body["transform"], dict(body.get("params") or {}), bool(body.get("drop"))
            )
        if route == "cursor":
            _needs(body, "k")
            return s.move(body["k"])
        if route == "params":
            _needs(body, "params")
            return s.set_params(dict(body["params"]))
        if route == "save":
            _needs(body, "name")
            return s.save(body["name"], bool(body.get("force")))
        _needs(body, "k")
        if type(body["k"]) is not int or not 0 <= body["k"] <= s.n:
            raise _Refused("session.move", f"no step {body['k']!r}: the steps are 0..{s.n}")
        return {"path": str(s.export(body["k"]))}


def _needs(body: dict[str, Any], key: str) -> None:
    if key not in body:
        raise _Refused("request.body", f"the body needs {key!r}")


def _index(text: str, last: int) -> int:
    """A step number 0..last from the path; a LookupError (404) otherwise."""
    if not text.isdigit() or int(text) > last:
        raise LookupError(f"no step {text}: the steps are 0..{last}")
    return int(text)


def make_dfg_server(paths: Sequence[str | Path], port: int = DEFAULT_PORT, verbose: bool = False,
                    platform: str | Path | None = None) -> ViewerServer:  # fmt: skip
    """View mode: load the files and bind the server (port 0 = any free port); does not serve
    yet. ``platform`` is the estimate's."""
    srv = ViewerServer(api.GraphSet(paths), port, verbose, DfgHandler, "dfg.html")
    srv.platform = _platform(platform)  # type: ignore[attr-defined]
    return srv


def make_edit_server(port: int = EDIT_PORT, verbose: bool = False, platform: str | None = None,
                     roots: Any = None) -> ViewerServer:  # fmt: skip
    """Edit mode with no session yet (``EditState.open`` starts one); does not serve yet."""
    state = EditState(roots, platform)
    srv = ViewerServer(state, port, verbose, DfgHandler, "dfg.html")  # type: ignore[arg-type]
    srv.platform = _platform(state.platform)  # type: ignore[attr-defined]
    return srv


def _platform(path: str | Path | None) -> Any:
    if path is None:
        return None
    from snax_forge.design import Platform
    from snax_forge.sandbox.session import ROOT

    p = Path(path)
    return Platform.load(p if p.is_absolute() or p.exists() else ROOT / p)
