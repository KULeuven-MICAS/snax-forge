"""The editor's session (D55, C§12): a kernel or a recipe opened, transforms applied one by one,
a cursor for undo and redo, and Save writing a recipe.

    Session.from_kernel(kernel, params, symbols=None, dtypes=None, platform=..., name=None)
    Session.from_recipe(path, platform=...)
    Session.resume(dir)

**Starting one.** ``from_kernel`` takes the kernel's symbols only, each the param of its name
unless ``symbols`` says otherwise, and no other param unless given (no W): a symbol without a
value is refused with one line naming it (``vecmul: give N, e.g. --set N=2048``). A recipe's
params come with it.

**The state.** A session is a working recipe (the kernel, its params, symbols and dtypes, and
every step applied, n of them), the graphs of steps 0..n in memory (step 0 the import with its
symbols bound), a cursor k in 0..n (the step shown, the one the next transform applies to) and
a platform. The graphs are replayed up to ``replayed`` (≥ k): after a param change the steps
past the cursor stay as redo steps, replayed and checked when the cursor reaches them (D156).
Only the recipe and the cursor go to disk, in ``<sessions>/<name>/``: ``recipe.json``, a recipe
holding every step (named after the session), and ``session.json``, ``{"cursor": k,
"platform": path, "own_params": [...]}`` (the params the session started with, the recipe's
or the kernel's, which are never renamed); ``resume`` replays the recipe, a step past the
cursor that refuses (and the ones after it) left as redo steps. The graphs are rebuilt
from the kernel's import, never stored. The seeded inputs (seed 0) and what step 0 computes on
them are drawn once and every step is checked against them (run.py's ``checked_step``). The
roots are injectable (``Roots``): sessions default to ``out/sessions/``, recipes to
``recipes/`` (its folders by category; Save writes ``custom/``), step files to ``out/sandbox/``. A new session takes its kernel's or its recipe's
name, ``_2``, ``_3``, ... when one is taken; an opened recipe file is never written.

**Changing it**, each a dict with ``ok``; a refusal changes nothing and gives its ``code`` and
``message``:

    apply(transform, params, drop=False,   the transform on step k, then the reference check;
          new_params=None)                 int params may be expressions over the session's
                                           params (Recipe.value) and are kept as given;
                                           ``new_params`` (``{"W_mul": 8}``) adds each name
                                           not a param yet to the params first, so each split
                                           can have its own; an int param naming a param the
                                           session lacks is ``session.param`` (``split: factor
                                           'W' is not a param; give its value``, with the
                                           ``param``'s name, for the page to ask). A param
                                           stays when the step that added it is undone. An
                                           int param given as a number (a split's factor, a
                                           link's depth) becomes a param of its own, the next
                                           free ``W``, ``W_2``, ... or ``D``, ... at that value,
                                           and the step stores its name (``fresh_param``), so
                                           it can change after the bind; a recipe opened from
                                           a file keeps its numbers. A ``bind`` renames a
                                           factor param the editor named (``W``, ``W_2``, not
                                           one of ``own_params``) that only this unit's splits
                                           use to ``W_<instance>`` through every step and the
                                           params (``rename_split_params``; dot 1's ``W`` to
                                           ``W_acc_1``), given back as ``renamed``. With
                                           k < n and no ``drop``: ``session.drop``, the server's
                                           guard (the page asks ``drop_question`` first). On success the
                                           steps past k go, the new one is step k + 1 and the
                                           cursor; returns its index, transform, params, the
                                           milliseconds it took, its ``diff`` and its ``checks``
    move(k)                                the cursor to any step 0..n (``session.move``); a
                                           redo step not replayed since a param change is
                                           replayed and checked on the way, and its refusal
                                           (with its ``index``) leaves the cursor where it was
    set_params(values)                     steps 1..k replayed with the new values, the steps
                                           after k left as redo steps (D156); a step that
                                           refuses keeps the old params and graphs and is named
                                           by its ``index``; returns the cursor's ``checks``
    save(name, force=False)                steps 1..k as ``<recipes>/custom/<name>.json``
                                           (``SAVED``, made when missing), with the
                                           params those steps or the symbols use and the
                                           session's own (``used_params``), and the step
                                           files with ``recipe.json`` in ``<sandbox>/<name>/``
                                           (run.py's ``write_steps``), after the design checks on
                                           step k's graph and the platform: ``save.name`` (not
                                           an identifier, or a recipe of that name exists in
                                           any folder of ``<recipes>``),
                                           ``save.graph`` (a problem of the ``graph`` stage, so
                                           the graph is not one SNAX-LOWER takes), ``save.design``
                                           (any other problem, unless ``force``); returns the
                                           ``pixi run sandbox``, ``flow`` and ``forge`` lines
                                           as ``commands``, each ``{command, what}``, ``what``
                                           one line on what it does (``COMMANDS``)

A transform's own refusal keeps its code (``link.*``), a plain SandboxError is ``sandbox``, a
DfgError ``dfg``, a failed reference check ``sandbox.check``.

**Asking it**, about step k:

    export(k)            the graph as ``<sessions>/<name>/export/<k>_<transform>.snaxdfg``
    sites(k, selected)   per registered transform its site's answer on the selection, one item
                         or a list (a node id, a container name, a group's or an accelerator's
                         name; transforms.py's ``site_of``: a group stands for its units, an
                         accelerator for its groups, and a transform that takes one item
                         refuses several, ``site.one``): ``{ok, params}`` or
                         ``{ok: false, code, message}``. ``bind``'s, when its site accepts, has
                         the bind list: every (BRM, implementation) of the library tried as
                         ``bind`` on the step, W read off the spatial bound and the instance
                         ``acc_N``, N the bind's order in the session (the binds before step k,
                         plus one), the next N while the graph has it as an instance,
                         accelerator or group (``default_instance``, D156), each ok (with
                         the params that bind it) or refused (with the message), with its
                         timing (latency, II, span) at that W where the implementation
                         resolves there
                         Each accepted transform also has its ``form`` (``form``): the fields the
                         editor shows, their types, defaults and choices
    marks(k)             the node ids and container names some site accepts
    diff(k)              step k against step k − 1: nodes added, removed and changed (attrs,
                         inputs or outputs, the body by its ids), containers added, removed and
                         changed (attrs, or the estimate's placement); empty at step 0
    estimate(k, tech)    the ideal estimate (estimate/, D155) on the session's platform, costed
                         on ``tech`` (default nangate45, C§20)
    checks(k)            Save's design checks on step k and the platform (D156): ``problems``,
                         each ``{code, where, message, fix}``, and ``notes``: ``graph.unbound``
                         while building (``before Save: bind it …``) as a note, not a
                         problem, which Save still refuses, and the design notes (check.py's
                         ``NOTES``, ``char.missing``) always, also beside problems

``sites``, ``marks``, ``estimate`` and ``checks`` are cached per step (``sites`` per step and selection,
``estimate`` per step and technology) and
the caches past a changed step dropped. Every public method holds the session's one lock, so
the server's threads see one change at a time.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from snax_forge import expr
from snax_forge.char import DEFAULT_TECH
from snax_forge.common import repo_root
from snax_forge.design import Design, Platform, run_checks
from snax_forge.design.check import CHECKS
from snax_forge.design.streamers import instance_of
from snax_forge.dfg import DfgError, Graph
from snax_forge.dfg.subset import parse_dim
from snax_forge.estimate import Estimate, estimate

from .recipe import Recipe, SandboxError, Step
from .run import (
    Result,
    StepError,
    bind_recipe_symbols,
    checked_step,
    reference,
    step_params,
    write_steps,
)
from .transforms import TRANSFORMS, _bind_site, _bound, bind, site_of

ROOT = repo_root()
PLATFORM = "platforms/snax_forge.json"
RECIPE_JSON, SESSION_JSON, EXPORT = "recipe.json", "session.json", "export"
SAVED = "custom"  # the folder of ``<recipes>`` Save writes into
SEED = 0

# The size a kernel session's refusal suggests for a symbol without a value
EXAMPLE_SIZE = 2048


@dataclass(frozen=True)
class Roots:
    """Where sessions, saved recipes and their step files go."""

    sessions: Path = field(default_factory=lambda: ROOT / "out" / "sessions")
    recipes: Path = field(default_factory=lambda: ROOT / "recipes")
    sandbox: Path = field(default_factory=lambda: ROOT / "out" / "sandbox")


def _refused(code: str, message: str, **more: Any) -> dict[str, Any]:
    return {"ok": False, "code": code, "message": message, **more}


def drop_question(k: int, n: int) -> str | None:
    """What the page asks before an apply at step k of n drops the steps after it; None at
    the last step."""
    return f"Drop steps {k + 1}..{n}?" if k < n else None


def _code(e: Exception) -> str:
    """A refusal's code (module doc)."""
    cause = e.cause if isinstance(e, StepError) else e
    if isinstance(e, StepError) and cause is None:
        return "sandbox.check"
    if getattr(cause, "code", None):
        return cause.code
    return "dfg" if isinstance(cause, DfgError) else "sandbox"


def _import(kernel: str, dtypes: dict[str, str]) -> Graph:
    from snax_forge.dfg.import_sdfg import import_kernel

    return import_kernel(kernel, dtypes or None)


def _unknown_param(transform: str, params: dict[str, Any],
                   known: dict[str, Any]) -> tuple[str, str] | None:  # fmt: skip
    """The first param name an int param uses that the session lacks, with the refusal's
    message; or None."""
    spec = TRANSFORMS.get(transform)
    for p, v in params.items():
        if spec is None or p not in spec.ints or not isinstance(v, str):
            continue
        try:
            names = expr.names(v)
        except expr.ExprError:
            continue  # the step's own check names it
        for x in sorted(names):
            if x not in known:
                shown = repr(v) if expr.is_name(v) else f"{v!r} names {x!r}, which"
                return x, f"{transform}: {p} {shown} is not a param; give its value"
    return None


class Session:
    """An editing session; see the module doc. Made by ``from_kernel``, ``from_recipe`` or
    ``resume``."""

    def __init__(self, name: str, recipe: Recipe, imported: Graph, platform: str,
                 roots: Roots, cursor: int | None = None,
                 own_params: list[str] | None = None) -> None:  # fmt: skip
        """``recipe``'s steps replayed on ``imported``: up to ``cursor`` (all without one) a
        SandboxError when one fails, past it a step that fails left a redo step with those
        after it; ``own_params`` the params never renamed (default: all of ``recipe``'s)."""
        self._lock = threading.Lock()
        self.name, self.roots, self.imported = name, roots, imported
        self.platform_path = platform
        self.platform = Platform.load(self._path(platform))
        self.recipe = replace(recipe, name=name)
        if cursor is not None and not 0 <= cursor <= len(recipe.steps):
            raise SandboxError(f"session {name}: cursor {cursor} outside 0..{self.n}")
        self.graphs, self.inputs, self.want = self._replay(self.recipe, cursor)
        try:  # the steps past the cursor too; one that refuses stays a redo step (D156)
            self._extend(self.recipe, self.graphs, self.inputs, self.want, len(recipe.steps))
        except StepError:
            pass
        self.cursor = len(self.graphs) - 1 if cursor is None else cursor
        self.own_params = list(recipe.params) if own_params is None else list(own_params)
        self._sites: dict[tuple[int, str], dict[str, Any]] = {}
        self._marks: dict[int, dict[str, list[str]]] = {}
        self._estimates: dict[tuple[int, str | None], Estimate] = {}
        self._checks: dict[int, dict[str, list[dict[str, Any]]]] = {}
        self._write()

    # -- making one -----------------------------------------------------------------------

    @classmethod
    def from_kernel(cls, kernel: str, params: dict[str, int | str],
                    symbols: dict[str, Any] | None = None, dtypes: dict[str, str] | None = None,
                    platform: str = PLATFORM, name: str | None = None,
                    roots: Roots | None = None) -> Session:  # fmt: skip
        """A new session on ``kernel``'s import, each symbol the param of its name by default."""
        roots = roots or Roots()
        imported = _import(kernel, dtypes or {})
        if symbols is None:
            symbols = {s: s for s in imported.symbols}
        missing = [x for s in imported.symbols if s in symbols
                   for x in sorted(expr.names(symbols[s])) if x not in params]  # fmt: skip
        if missing:
            missing = list(dict.fromkeys(missing))
            sets = " ".join(f"--set {x}={EXAMPLE_SIZE}" for x in missing)
            raise SandboxError(f"{kernel}: give {', '.join(missing)}, e.g. {sets}")
        recipe = Recipe(kernel, kernel, dict(params), dict(symbols), [], dict(dtypes or {}))
        return cls(cls._free(roots, name or kernel), recipe, imported, platform, roots, 0)

    @classmethod
    def from_recipe(cls, path: str | Path, platform: str = PLATFORM,
                    roots: Roots | None = None) -> Session:  # fmt: skip
        """A new session on a recipe, replayed from its kernel's import; the file stays as is."""
        roots = roots or Roots()
        recipe = Recipe.load(path)
        imported = _import(recipe.kernel, recipe.dtypes)
        return cls(cls._free(roots, recipe.name), recipe, imported, platform, roots)

    @classmethod
    def resume(cls, folder: str | Path, roots: Roots | None = None) -> Session:
        """A session from its folder: its recipe replayed, its cursor and platform restored."""
        folder = Path(folder)
        roots = roots or replace(Roots(), sessions=folder.parent)
        recipe = Recipe.load(folder / RECIPE_JSON)
        state = json.loads((folder / SESSION_JSON).read_text())
        imported = _import(recipe.kernel, recipe.dtypes)
        return cls(folder.name, recipe, imported, state["platform"], roots, state["cursor"],
                   state.get("own_params"))  # fmt: skip

    @staticmethod
    def _free(roots: Roots, base: str) -> str:
        """``base``, or ``base_2``, ``base_3``, ... whichever names no session yet."""
        name, k = base, 1
        while (roots.sessions / name).exists():
            k += 1
            name = f"{base}_{k}"
        return name

    # -- the state ------------------------------------------------------------------------

    @property
    def n(self) -> int:
        """The steps applied."""
        return len(self.recipe.steps)

    @property
    def replayed(self) -> int:
        """The last step replayed with the current params; the steps past it are redo steps."""
        return len(self.graphs) - 1

    @property
    def folder(self) -> Path:
        return self.roots.sessions / self.name

    def _path(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    def _replay(self, recipe: Recipe, upto: int | None = None,
                ) -> tuple[list[Graph], dict[str, Any], dict[str, Any]]:  # fmt: skip
        """Step 0 and the steps of ``recipe`` up to ``upto`` (all without it), checked; a
        StepError naming the step fails."""
        g = bind_recipe_symbols(recipe, self.imported)
        inputs, want = reference(g, SEED, recipe.name)
        graphs = [g]
        self._extend(recipe, graphs, inputs, want, len(recipe.steps) if upto is None else upto)
        return graphs, inputs, want

    @staticmethod
    def _extend(recipe: Recipe, graphs: list[Graph], inputs: dict[str, Any],
                want: dict[str, Any], upto: int) -> None:  # fmt: skip
        """``graphs`` extended in place with the steps after its last up to ``upto``, each
        checked; a StepError naming the step stops it, the steps before kept."""
        for i in range(len(graphs) - 1, upto):
            name, params = step_params(recipe, i)
            try:
                graphs.append(checked_step(graphs[-1], name, params, inputs, want,
                                           f"{recipe.name} step {i + 1} ({name})"))  # fmt: skip
            except StepError as e:
                e.index = i + 1  # type: ignore[attr-defined]
                raise

    def _write(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.recipe.save(self.folder / RECIPE_JSON)
        self._write_cursor()

    def _write_cursor(self) -> None:
        state = {"cursor": self.cursor, "platform": self.platform_path,
                 "own_params": self.own_params}  # fmt: skip
        (self.folder / SESSION_JSON).write_text(json.dumps(state, indent=1) + "\n")

    def _forget(self, after: int) -> None:
        """The cached answers of the steps past ``after``."""
        self._sites = {k: v for k, v in self._sites.items() if k[0] <= after}
        self._marks = {k: v for k, v in self._marks.items() if k <= after}
        self._estimates = {k: v for k, v in self._estimates.items() if k[0] <= after}
        self._checks = {k: v for k, v in self._checks.items() if k <= after}

    def _step(self, k: int) -> Graph:
        if not 0 <= k <= self.n:
            raise SandboxError(f"session {self.name}: no step {k} (0..{self.n})")
        if k > self.replayed:
            raise SandboxError(f"session {self.name}: step {k} is a redo step, not replayed "
                               "since the params changed: move the cursor to it")  # fmt: skip
        return self.graphs[k]

    def _transform(self, k: int) -> str:
        """Step ``k``'s transform, ``input`` for step 0 (run.py's file names)."""
        return "input" if k == 0 else self.recipe.steps[k - 1].transform

    def step_name(self, k: int) -> str:
        """Step ``k``'s name as its file has it: ``<k>_<transform>``."""
        return f"{k}_{self._transform(k)}"

    # -- changing it ----------------------------------------------------------------------

    def apply(self, transform: str, params: dict[str, Any], drop: bool = False,
              new_params: dict[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
        """The transform at the cursor (module doc)."""
        with self._lock:
            k, n = self.cursor, self.n
            if k < n and not drop:
                return _refused("session.drop", f"steps {k + 1}..{n} come after step {k}: drop "
                                "them to apply here", question=drop_question(k, n))  # fmt: skip
            added = {x: v for x, v in (new_params or {}).items() if x not in self.recipe.params}
            params = dict(params)
            spec = TRANSFORMS.get(transform)
            for p in spec.ints if spec else ():  # a number becomes a param of its own (C§12)
                if type(params.get(p)) is int:
                    name = fresh_param(p, {**self.recipe.params, **added})
                    added[name], params[p] = params[p], name
            unknown = _unknown_param(transform, params, {**self.recipe.params, **added})
            if unknown:
                return _refused("session.param", unknown[1], param=unknown[0])
            step = Step(transform, dict(params))
            try:
                base = replace(self.recipe, params={**self.recipe.params, **added})
            except SandboxError as e:
                return _refused("session.param", str(e))
            trial = replace(base, steps=[*self.recipe.steps[:k], step])
            start = time.perf_counter()
            try:
                name, values = step_params(trial, k)
                g = checked_step(self.graphs[k], name, values, self.inputs, self.want,
                                 f"{self.name} step {k + 1} ({name})")  # fmt: skip
            except (SandboxError, DfgError) as e:
                return _refused(_code(e), str(e))
            ms = (time.perf_counter() - start) * 1000
            renamed: dict[str, str] = {}
            if transform == "bind":  # W -> W_<instance> (D156)
                trial, renamed = rename_split_params(trial, g, step.params.get("instance"),
                                                     self.own_params)  # fmt: skip
                params = dict(trial.steps[k].params)
                added = {renamed.get(x, x): v for x, v in added.items()}
            self.recipe, self.graphs = trial, [*self.graphs[: k + 1], g]
            self.cursor = k + 1
            self._forget(k)
            self._write()
            return {"ok": True, "index": k + 1, "transform": transform, "params": dict(params),
                    "new_params": added, "renamed": renamed, "ms": ms,
                    "diff": self._diff(k + 1), "checks": self._checks_of(k + 1)}  # fmt: skip

    def move(self, k: int) -> dict[str, Any]:
        """The cursor to step ``k``, the steps kept."""
        with self._lock:
            if type(k) is not int or not 0 <= k <= self.n:
                return _refused("session.move", f"no step {k!r}: the steps are 0..{self.n}")
            replayed = []
            if k > self.replayed:  # redo steps since a param change: replayed and checked now
                first = self.replayed + 1
                try:
                    self._extend(self.recipe, self.graphs, self.inputs, self.want, k)
                except StepError as e:
                    return _refused(_code(e), str(e), index=e.index,  # type: ignore[attr-defined]
                                    cursor=self.cursor)  # fmt: skip
                replayed = list(range(first, k + 1))
            self.cursor = k
            self._write_cursor()
            return {"ok": True, "cursor": k, "replayed": replayed}

    def set_params(self, values: dict[str, int | str]) -> dict[str, Any]:
        """Every step replayed with the params ``values`` sets."""
        with self._lock:
            try:
                recipe = self.recipe.with_params(values)
                graphs, inputs, want = self._replay(recipe, self.cursor)
            except StepError as e:
                return _refused(_code(e), str(e), index=e.index)  # type: ignore[attr-defined]
            except (SandboxError, DfgError) as e:
                return _refused(_code(e), str(e), index=0)
            self.recipe, self.graphs, self.inputs, self.want = recipe, graphs, inputs, want
            self._forget(-1)
            self._write()
            return {"ok": True, "params": dict(recipe.params), "replayed": self.cursor,
                    "redo": list(range(self.cursor + 1, self.n + 1)),
                    "checks": self._checks_of(self.cursor)}  # fmt: skip

    def save(self, name: str, force: bool = False) -> dict[str, Any]:
        """Steps 1..cursor as a recipe of its own (module doc)."""
        with self._lock:
            if not isinstance(name, str) or not name.isidentifier():
                return _refused("save.name", f"{name!r} is not a name (an identifier)")
            path = self.roots.recipes / SAVED / f"{name}.json"
            taken = sorted(self.roots.recipes.glob(f"*/{name}.json"))
            taken += [p for p in [self.roots.recipes / f"{name}.json"] if p.exists()]
            if taken:
                return _refused("save.name", f"{_shown(taken[0])} exists; a recipe is never "
                                "overwritten")  # fmt: skip

            g = self.graphs[self.cursor]
            problems = run_checks(Design(self.platform, g))
            listed = [str(p) for p in problems]
            if any(_stage(p.code) == "graph" for p in problems):
                return _refused("save.graph", f"step {self.cursor} is not a graph the flow takes",
                                problems=listed)  # fmt: skip
            if problems and not force:
                return _refused("save.design", f"the design checks on {self.platform.name} "
                                f"report {len(problems)} problem(s)", problems=listed)  # fmt: skip
            steps = self.recipe.steps[: self.cursor]
            keep = used_params(self.recipe.params, steps, self.recipe.symbols, self.own_params)
            recipe = replace(self.recipe, name=name, steps=steps,
                             params={k: v for k, v in self.recipe.params.items() if k in keep})  # fmt: skip
            path.parent.mkdir(parents=True, exist_ok=True)
            recipe.save(path)
            results = [
                Result(k, self._transform(k), self.graphs[k]) for k in range(self.cursor + 1)
            ]
            write_steps(recipe, results, self.roots.sandbox / name)
            shown = _shown(path)
            lines = (f"pixi run sandbox {shown}",
                     f"pixi run flow {shown} --platform {self.platform_path}",
                     f"pixi run forge {shown}")  # fmt: skip
            return {"ok": True, "recipe": str(path), "shown": shown, "problems": listed,
                    "commands": [{"command": c, "what": w} for c, w in zip(lines, COMMANDS)]}  # fmt: skip

    # -- asking it ------------------------------------------------------------------------

    def export(self, k: int) -> Path:
        """Step ``k``'s graph as a file of its own."""
        with self._lock:
            g = self._step(k)
            out = self.folder / EXPORT / f"{self.step_name(k)}.snaxdfg"
            out.parent.mkdir(parents=True, exist_ok=True)
            g.save(out)
            return out

    def sites(self, k: int, selected: str | list[str]) -> dict[str, dict[str, Any]]:
        """Every transform's site on ``selected`` at step ``k``, one item or several (module
        doc), bind's with the bind list."""
        with self._lock:
            return self._sites_of(k, selected)

    def marks(self, k: int) -> dict[str, list[str]]:
        """The nodes and containers of step ``k`` some site accepts."""
        with self._lock:
            if k not in self._marks:
                self._marks[k] = marks_of(self._step(k))
            return self._marks[k]

    def diff(self, k: int) -> dict[str, Any]:
        """Step ``k`` against step ``k`` − 1."""
        with self._lock:
            return self._diff(k)

    def estimate(self, k: int, tech: str | None = DEFAULT_TECH) -> Estimate:
        """Step ``k``'s ideal estimate on the session's platform, costed on ``tech``."""
        with self._lock:
            return self._estimate(k, tech)

    def checks(self, k: int) -> dict[str, list[dict[str, Any]]]:
        """Save's design checks on step ``k`` (module doc)."""
        with self._lock:
            return self._checks_of(k)

    # -- the answers, under the lock --------------------------------------------------------

    def _sites_of(self, k: int, selected: str | list[str]) -> dict[str, dict[str, Any]]:
        items = [selected] if isinstance(selected, str) else list(dict.fromkeys(selected))
        key = (k, tuple(items))
        if key in self._sites:
            return self._sites[key]
        g = self._step(k)
        out: dict[str, dict[str, Any]] = {}
        for name, spec in TRANSFORMS.items():
            if spec.site is None and spec.multi is None:
                continue
            try:
                out[name] = {"ok": True, "params": site_of(name, g, items)}
            except (SandboxError, DfgError) as e:
                out[name] = _refused(_code(e), str(e))
        order = 1 + sum(st.transform == "bind" for st in self.recipe.steps[:k])
        if out.get("bind", {}).get("ok"):
            out["bind"]["list"] = bind_list(g, items[0], order)
        for name, answer in out.items():
            if answer["ok"]:
                answer["form"] = form(
                    name,
                    g,
                    answer,
                    self.recipe.params,
                    order,
                    self._kernel_inputs() if name == "bake_scalar" else None,
                )
        self._sites[key] = out
        return out

    def _kernel_inputs(self) -> dict[str, Any]:
        """The kernel's ``make_inputs`` at seed 0 on the input graph (axpy's alpha 3), else the
        session's seeded inputs (the reference check's)."""
        from snax_forge.dfg.reference import KernelError, kernel_inputs, kernel_spec

        try:
            return kernel_inputs(kernel_spec(self.recipe.kernel, "editor"), self.graphs[0], SEED)
        except (KernelError, KeyError, TypeError, ValueError):
            return self.inputs

    def _checks_of(self, k: int) -> dict[str, list[dict[str, Any]]]:
        if k not in self._checks:
            self._checks[k] = design_checks(self.platform, self._step(k))
        return self._checks[k]

    def _estimate(self, k: int, tech: str | None = DEFAULT_TECH) -> Estimate:
        if (k, tech) not in self._estimates:
            self._estimates[k, tech] = estimate(self._step(k), self.platform, tech)
        return self._estimates[k, tech]

    def _diff(self, k: int) -> dict[str, Any]:
        if k == 0:
            self._step(0)
            return graph_diff(None, self.graphs[0])
        return graph_diff(self._step(k - 1), self._step(k), self._estimate(k - 1).containers,
                          self._estimate(k).containers)  # fmt: skip


def graph_diff(old: Graph | None, new: Graph, placed_old: Any = None,
               placed_new: Any = None) -> dict[str, Any]:  # fmt: skip
    """``new`` against ``old`` (module doc's ``diff``), placements from the two estimates'
    containers when given; empty without ``old``. The view mode's compare uses it too."""
    out: dict[str, Any] = {"nodes": {"added": [], "removed": [], "changed": []},
                           "containers": {"added": [], "removed": [], "changed": []}}  # fmt: skip
    if old is None:
        return out
    a, b = _nodes(old), _nodes(new)
    out["nodes"] = {"added": [i for i in b if i not in a],
                    "removed": [i for i in a if i not in b],
                    "changed": [i for i in b if i in a and a[i] != b[i]]}  # fmt: skip
    pa, pb = placed_old or {}, placed_new or {}

    def placed(p: Any, c: str) -> Any:
        return (p[c].placement, p[c].depth) if c in p else None

    ca, cb = old.containers, new.containers
    out["containers"] = {
        "added": [c for c in cb if c not in ca],
        "removed": [c for c in ca if c not in cb],
        "changed": [c for c in cb if c in ca and (ca[c].attrs != cb[c].attrs
                                                   or placed(pa, c) != placed(pb, c))],
    }  # fmt: skip
    return out


def _accepted(g: Graph, selected: str) -> bool:
    """Whether some transform's site accepts ``selected``."""
    for spec in TRANSFORMS.values():
        if spec.site is None:
            continue
        try:
            spec.site(g, selected)
            return True
        except (SandboxError, DfgError):
            continue
    return False


def marks_of(g: Graph) -> dict[str, list[str]]:
    """The node ids and container names of ``g`` some site accepts (the view mode's marks)."""
    return {"nodes": [n.id for n, _ in g.walk() if _accepted(g, n.id)],
            "containers": [c for c in g.containers if _accepted(g, c)]}  # fmt: skip


def _stage(code: str) -> str:
    """The design-check stage a problem's code belongs to."""
    return CHECKS[code].stage if code in CHECKS else code.split(".")[0]


def _nodes(g: Graph) -> dict[str, Any]:
    """Each live node by id, as the diff compares it: kind, attrs, memlets, the body's ids."""
    return {n.id: (n.kind, json.dumps(n.to_dict()["attrs"], sort_keys=True),
                   {c: m.to_dict() for c, m in n.inputs.items()},
                   {c: m.to_dict() for c, m in n.outputs.items()},
                   [k.id for k in n.body or []]) for n, _ in g.walk()}  # fmt: skip


# What each of Save's commands does (C§12), in the order Save lists them
COMMANDS = (
    "writes the steps and checks each against the reference",
    "runs the recipe on SNAX-GVSoC and writes the reports",
    "the flow plus the bundle for snax_cluster's RTL",
)


def _shown(p: Path) -> str:
    """A path as a command line in the repository writes it."""
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def bind_list(g: Graph, selected: str, order: int = 1) -> list[dict[str, Any]]:
    """Every (BRM, implementation) of the library tried as ``bind`` on ``selected``, the
    instance the ``order``-th bind's default (module doc)."""
    from snax_forge.brm import BrmError, library_names, load_brm

    site = _bind_site(g, selected, f"bind {selected!r}")
    w = _bound(g, parse_dim(site.smap.attrs["range"], site.smap.id)[1], f"bind {selected!r}")
    instance = default_instance(_taken(g), order)
    out = []
    for b in library_names():
        brm = load_brm(b)
        for impl in brm.implementations:
            entry: dict[str, Any] = {"brm": b, "implementation": impl}
            params = {"node": selected, "brm": b, "implementation": impl, "instance": instance}
            try:
                bound = bind(g, **params)
                entry |= {"ok": True, "params": params,
                          "timing": _timing(instance_of(bound.node(site.tasklet.id)))}  # fmt: skip
            except (SandboxError, DfgError) as e:
                entry |= _refused(_code(e), str(e))
                lanes = {p.lanes for p in brm.interface.ports}
                given = {x: w for x in lanes if isinstance(x, str) and expr.is_name(x)}
                try:
                    entry["timing"] = _timing(brm.resolve(impl, given))
                except (BrmError, ValueError):
                    entry["timing"] = None
            out.append(entry)
    return out


def default_instance(taken: set[str], order: int = 1) -> str:
    """``bind``'s default instance (D156): ``acc_<order>``, ``order`` the bind's in the session
    (``acc_1``, ``acc_2``), the next number while ``taken`` (the graph's instances,
    accelerators and groups) has it."""
    k = order
    while f"{DEFAULT_INSTANCE}_{k}" in taken:
        k += 1
    return f"{DEFAULT_INSTANCE}_{k}"


def _taken(g: Graph) -> set[str]:
    """The graph's instances, accelerators and groups."""
    return {n.attrs[k] for n, _ in g.walk() if n.kind == "accelerated"
            for k in ("instance", "accelerator", "group") if k in n.attrs}  # fmt: skip


# bind's default instance's base, acc_1, acc_2, ... (C§12, D156)
DEFAULT_INSTANCE = "acc"

# The splits whose inner map a unit's replaced subtree holds (C§12's Bindings): per transform the
# suffixes of the maps it makes under the map it splits
SPLIT_SUFFIXES = {"split": ("_inner",), "split_map": ("_s",), "split_reduce": ("_s", "_final")}
# A factor param the editor names (fresh_param's W, W_2, ...), which a bind may rename (D156)
EDITOR_FACTOR = re.compile(r"^W(_\d+)?$")


def _map_ids(d: Any) -> set[str]:
    """Every map id in a ``replaced`` subtree (a node's dict)."""
    out: set[str] = set()
    if isinstance(d, dict):
        if d.get("kind") == "map" and d.get("id"):
            out.add(d["id"])
        for v in d.values():
            out |= _map_ids(v)
    elif isinstance(d, list):
        for v in d:
            out |= _map_ids(v)
    return out


def used_params(params: dict[str, Any], steps: list[Step], symbols: dict[str, Any],
                own: list[str]) -> set[str]:  # fmt: skip
    """The params a saved recipe keeps (D156): those its steps' or its symbols' expressions
    name, a kept string param's own names with them, and the session's own (a recipe opened
    from a file keeps its params); a link's ``D`` tried and stepped back from goes."""
    names: set[str] = set(own) & set(params)
    for st in steps:
        names |= _step_names(st)
    for v in symbols.values():
        if isinstance(v, str):
            try:
                names |= expr.names(v)
            except expr.ExprError:
                continue
    keep, todo = set(), [n for n in names if n in params]
    while todo:
        n = todo.pop()
        if n in keep:
            continue
        keep.add(n)
        v = params[n]
        if isinstance(v, str):
            try:
                todo += [x for x in expr.names(v) if x in params]
            except expr.ExprError:
                pass
    return keep


def _step_names(step: Step) -> set[str]:
    """The names a step's string params use (an int param's expression over the params)."""
    out: set[str] = set()
    for v in step.params.values():
        if isinstance(v, str):
            try:
                out |= expr.names(v)
            except expr.ExprError:
                continue
    return out


def rename_split_params(recipe: Recipe, g: Graph, instance: str | None,
                        own: list[str]) -> tuple[Recipe, dict[str, str]]:  # fmt: skip
    """After a bind of ``instance`` (``g`` the bound graph), each factor param the editor named
    (``W``, ``W_2``: not in ``own``) that only the splits of this unit name, renamed to
    ``W_<instance>`` (the next free ``_2``, ... if taken) in every step and in the params; the
    recipe and ``{old: new}`` (D156). A param a symbol, another step or another unit's split
    names is shared and stays."""
    if instance is None:
        return recipe, {}
    ids: set[str] = set()
    for n, _ in g.walk():
        if n.kind == "accelerated" and n.attrs.get("instance") == instance:
            ids |= _map_ids(n.attrs.get("replaced"))
    mine = []  # the split steps that configure this unit
    for i, st in enumerate(recipe.steps):
        m = st.params.get("map")
        if st.transform in SPLIT_SUFFIXES and isinstance(m, str):
            suffix = "|".join(SPLIT_SUFFIXES[st.transform])
            inner = re.compile(rf"^{re.escape(m)}({suffix})(_\d+)?$")
            if any(inner.match(x) for x in ids):
                mine.append(i)
    symbols: set[str] = set()
    for v in recipe.symbols.values():
        symbols |= expr.names(v) if isinstance(v, str) else set()
    renamed: dict[str, str] = {}
    taken = set(recipe.params) | symbols
    for i in mine:
        x = recipe.steps[i].params.get("factor")
        if not (isinstance(x, str) and EDITOR_FACTOR.match(x) and x in recipe.params):
            continue
        if x in own or x in symbols or x in renamed:
            continue
        users = {j for j, st in enumerate(recipe.steps) if x in _step_names(st)}
        if not users <= set(mine):
            continue  # shared with another unit or step
        base, k, new = f"W_{instance}", 1, f"W_{instance}"
        while new in taken:
            k += 1
            new = f"{base}_{k}"
        taken.add(new)
        renamed[x] = new
    if not renamed:
        return recipe, {}
    steps = [Step(st.transform, {p: _renamed(v, renamed) for p, v in st.params.items()})
             for st in recipe.steps]  # fmt: skip
    params = {renamed.get(p, p): v for p, v in recipe.params.items()}
    return replace(recipe, params=params, steps=steps), renamed


def _renamed(v: Any, renamed: dict[str, str]) -> Any:
    """A step param with the renamed params in place (a bare name; an expression rewritten)."""
    if not isinstance(v, str):
        return v
    if v in renamed:
        return renamed[v]
    try:
        names = expr.names(v)
    except expr.ExprError:
        return v
    if not names & set(renamed):
        return v
    return expr.canonical(re.sub(r"[A-Za-z_]\w*", lambda m: renamed.get(m[0], m[0]), v))


# The problems that only say a graph is still being built (D156): notes "before Save", which
# Save still refuses (save.graph)
BEFORE_SAVE = ("graph.unbound",)


def design_checks(platform: Platform, g: Graph) -> dict[str, list[dict[str, Any]]]:
    """Save's design checks on ``g`` (``Session.checks``): ``problems`` and ``notes``, each
    ``{code, where, message, fix, before_save}``. ``graph.unbound`` is a "before Save" note
    (``before_save`` true), not a problem, and the design notes (check.py's ``NOTES``, ``char.missing``) are
    listed whatever else is wrong, each one that cannot read the graph left out (D156)."""
    from snax_forge.design.check import NOTES

    design = Design(platform, g)
    found = run_checks(design)
    problems, notes = [], []
    for p in found:
        d = {"code": p.code, "where": p.where, "message": p.message, "fix": p.fix,
             "before_save": p.code in BEFORE_SAVE}  # fmt: skip
        if p.code == "graph.unbound":
            d["message"] = f"before Save: bind it ({p.message})"
            notes.append(d)
        else:
            problems.append(d)
    for fn in NOTES.values():
        try:
            notes += [{"code": p.code, "where": p.where, "message": p.message, "fix": p.fix,
                       "before_save": False} for p in fn(design)]  # fmt: skip
        except (ValueError, KeyError, TypeError, AttributeError):  # a graph a note cannot read
            continue
    return {"problems": problems, "notes": notes}


# The defaults of the params a site does not fill (the user's choices): an int param's value,
# and the name the editor gives it, a param of its own (C§12): every factor and depth the editor
# writes names a param, W, W_2, ... and D, D_2, ..., so it can change after the bind
DEFAULT_INT = {"factor": 4, "depth": 1}
PARAM_BASE = {"factor": "W", "depth": "D"}


def fresh_param(field: str, taken: Any) -> str:
    """The next free param name for int field ``field``: its base (``W`` for a factor, ``D``
    for a depth), then ``_2``, ``_3``, ... while ``taken`` has it."""
    base = PARAM_BASE.get(field, field.upper())
    name, k = base, 1
    while name in taken:
        k += 1
        name = f"{base}_{k}"
    return name


DEFAULT_KIND = "temporal"


# bake_scalar's value in the editor's form for a container with no seeded input
BAKE_DEFAULT = 1


def form(
    name: str,
    g: Graph,
    answer: dict[str, Any],
    params: dict[str, Any],
    order: int = 1,
    inputs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The fields the editor shows to apply transform ``name`` where its site accepted
    (``answer``, with the bind list for ``bind``), in the transform's param order: each
    ``name``, ``type`` (``int``, ``str``, ``choice``, ``checklist``), ``sweepable`` (one of the
    transform's ``ints``, so a number or an expression over the session's ``params``),
    ``value`` (the site's, else a default: an int field a param name of its own, ``W``, ``W_2``,
    ... for a factor, ``D``, ... for a depth, ``fresh_param``, with ``new_value`` the value a new
    name starts at, 4 and 1),
    ``readonly`` (the selection's own field) and ``choices`` where they apply: ``tag_map``'s
    kinds, ``bind``'s (BRM, implementation) pairs from the bind list with their timing and
    message (one field, ``binding``, whose ``value`` gives both params), ``encapsulate``'s
    units (the instances in no accelerator, in the graph's order, the selection's ticked),
    ``bake_scalar``'s value (an int literal, or ``csr``, its ``hint`` says; the container's in ``inputs``: the session gives
    the kernel's ``make_inputs`` at seed 0, axpy's alpha 3; else ``BAKE_DEFAULT``)."""

    from snax_forge.common import fresh

    from .transforms import TAGS

    spec = TRANSFORMS[name]
    filled = answer["params"]
    selection = next(iter(filled), None)
    fields: list[dict[str, Any]] = []
    for p in spec.param_names():
        field: dict[str, Any] = {"name": p, "type": "str", "sweepable": p in spec.ints,
                                 "value": filled.get(p), "readonly": p == selection}  # fmt: skip
        if p in spec.ints:
            field["type"] = "int"
            if field["value"] is None:  # a param of its own, its value beside it (C§12)
                field["value"] = fresh_param(p, params)
            field["new_value"] = DEFAULT_INT.get(p, 1)
        elif name == "bake_scalar" and p == "value":  # an int literal, not a param
            seeded = (inputs or {}).get(filled.get("container"))
            value = BAKE_DEFAULT if seeded is None else int(np.asarray(seeded).reshape(-1)[0])
            field |= {
                "type": "int",
                "value": value,
                "hint": "an int baked into the hardware, or csr: read at run time from a CSR",
            }
        elif name == "interchange_map" and p == "with":  # every map above and below it
            field |= _partner_field(g, filled["map"])
        elif name == "tag_map" and p == "kind":
            field |= {"type": "choice", "value": DEFAULT_KIND,
                      "choices": [{"value": t, "label": t} for t in TAGS]}  # fmt: skip
        elif name == "bind" and p in ("brm", "implementation", "params"):
            if p != "brm":
                continue  # one field for the pair; design params come off the graph
            entries = answer.get("list", [])
            ok = [e for e in entries if e["ok"]]
            field |= {"name": "binding", "type": "choice", "readonly": False,
                      "value": {"brm": ok[0]["brm"], "implementation": ok[0]["implementation"]} if ok else None,
                      "choices": [{"value": {"brm": e["brm"], "implementation": e["implementation"]},
                                   "label": f"{e['brm']} / {e['implementation']}", "ok": e["ok"],
                                   "timing": e["timing"], "message": e.get("message")}
                                  for e in entries]}  # fmt: skip
        elif name == "bind" and p == "instance":  # the bind list's default, made unique
            ok = [e for e in answer.get("list", []) if e["ok"]]
            field["value"] = (ok[0]["params"]["instance"] if ok
                              else default_instance(_taken(g), order))  # fmt: skip
        elif name == "encapsulate" and p == "units":
            nodes = [n for n, _ in g.walk() if n.kind == "accelerated"]
            free = list(
                dict.fromkeys(n.attrs["instance"] for n in nodes if "accelerator" not in n.attrs)
            )
            # the selection's units ticked (one unit's box: that unit; two group outlines: both)
            field |= {"type": "checklist", "readonly": False,
                      "choices": [{"value": u, "label": u} for u in free]}  # fmt: skip
        elif name == "encapsulate" and p == "accelerator":
            taken = {n.attrs[k] for n, _ in g.walk() if n.kind == "accelerated"
                     for k in ("instance", "accelerator", "group") if k in n.attrs}  # fmt: skip
            field["value"] = fresh(g.name, taken)
        fields.append(field)
    return fields


def _partner_field(g: Graph, map: str) -> dict[str, Any]:
    """interchange_map's ``with``: a choice per map above and below ``map`` in its nest, top to
    bottom, one it cannot trade with not ``ok`` with its refusal (as bind's list marks a BRM
    that does not fit); the default the nearest that can, the one below on a tie."""
    from .transforms import _locate, partners

    _, _, around = _locate(g, map)
    above = {o.id: len(around) - k for k, o in enumerate(around) if o.kind == "map"}
    depth = {n.id: len(outer) - len(around) for n, outer in g.walk()}
    choices = []
    for pid, why in partners(g, map):
        where = "above" if pid in above else "below"
        choices.append({"value": pid, "label": f"{pid} ({where})", "ok": why is None,
                        "message": why})  # fmt: skip
    ok = [c["value"] for c in choices if c["ok"]]
    dist = {pid: above[pid] if pid in above else depth[pid] for pid in ok}
    best = min(ok, key=lambda pid: (dist[pid], pid in above), default=None)
    return {"type": "choice", "readonly": False, "value": best, "choices": choices}


def _timing(inst: Any) -> dict[str, int]:
    return {"latency": inst.latency, "ii": inst.initiation_interval, "span": inst.span}
