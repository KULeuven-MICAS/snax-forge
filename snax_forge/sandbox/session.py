"""The editor's session (D55, C§12): a kernel or a recipe opened, transforms applied one by one,
a cursor for undo and redo, and Save writing a recipe.

    Session.from_kernel(kernel, params, symbols=None, dtypes=None, platform=..., name=None)
    Session.from_recipe(path, platform=...)
    Session.resume(dir)

**The state.** A session is a working recipe (the kernel, its params, symbols and dtypes, and
every step applied, n of them), the graphs of steps 0..n in memory (step 0 the import with its
symbols bound), a cursor k in 0..n (the step shown, the one the next transform applies to) and
a platform. Only the recipe and the cursor go to disk, in ``<sessions>/<name>/``:
``recipe.json``, a recipe holding every step (named after the session), and ``session.json``,
``{"cursor": k, "platform": path}``; ``resume`` replays the recipe. The graphs are rebuilt
from the kernel's import, never stored. The seeded inputs (seed 0) and what step 0 computes on
them are drawn once and every step is checked against them (run.py's ``checked_step``). The
roots are injectable (``Roots``): sessions default to ``out/sessions/``, recipes to
``recipes/``, step files to ``out/sandbox/``. A new session takes its kernel's or its recipe's
name, ``_2``, ``_3``, ... when one is taken; an opened recipe file is never written.

**Changing it**, each a dict with ``ok``; a refusal changes nothing and gives its ``code`` and
``message``:

    apply(transform, params, drop=False)   the transform on step k, then the reference check;
                                           int params may be expressions over the session's
                                           params (Recipe.value) and are kept as given. With
                                           k < n and no ``drop``: ``session.drop`` (the page
                                           asks whether to drop steps k+1..n). On success the
                                           steps past k go, the new one is step k + 1 and the
                                           cursor; returns its index, transform, params, the
                                           milliseconds it took and its ``diff``
    move(k)                                the cursor to any step 0..n (``session.move``)
    set_params(values)                     every step replayed with the new values; a step that
                                           refuses keeps the old params and graphs and is named
                                           by its ``index``
    save(name, force=False)                steps 1..k as ``<recipes>/<name>.json``, and the step
                                           files with ``recipe.json`` in ``<sandbox>/<name>/``
                                           (run.py's ``write_steps``), after the design checks on
                                           step k's graph and the platform: ``save.name`` (not
                                           an identifier, or a recipe of that name exists),
                                           ``save.graph`` (a problem of the ``graph`` stage, so
                                           the graph is not one SNAX-LOWER takes), ``save.design``
                                           (any other problem, unless ``force``: dot 1 without
                                           ``encapsulate`` is ``gen.instances``); returns the
                                           ``pixi run sandbox``, ``flow`` and ``forge`` lines

A transform's own refusal keeps its code (``link.*``), a plain SandboxError is ``sandbox``, a
DfgError ``dfg``, a failed reference check ``sandbox.check``.

**Asking it**, about step k:

    export(k)            the graph as ``<sessions>/<name>/export/<k>_<transform>.snaxdfg``
    sites(k, selected)   per registered transform its site's answer on the selection (a node id
                         or a container name, transforms.py): ``{ok, params}`` or
                         ``{ok: false, code, message}``. ``bind``'s, when its site accepts, has
                         the bind list: every (BRM, implementation) of the library tried as
                         ``bind`` on the step, W read off the spatial bound and the instance the
                         tasklet's id (a fused nest's first tasklet's) made unique against the
                         instances and accelerators, each ok (with the params that bind it) or
                         refused (with the message), with its timing (latency, II, span) at that
                         W where the implementation resolves there
                         Each accepted transform also has its ``form`` (``form``): the fields the
                         editor shows, their types, defaults and choices
    marks(k)             the node ids and container names some site accepts
    diff(k)              step k against step k − 1: nodes added, removed and changed (attrs,
                         inputs or outputs, the body by its ids), containers added, removed and
                         changed (attrs, or the estimate's placement); empty at step 0
    estimate(k)          the ideal estimate (estimate/, D151) on the session's platform

``sites``, ``marks`` and ``estimate`` are cached per step (``sites`` per step and selection) and
the caches past a changed step dropped. Every public method holds the session's one lock, so
the server's threads (R4) see one change at a time.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from snax_forge import expr
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
from .transforms import TRANSFORMS, _bind_site, _bound, bind

ROOT = repo_root()
PLATFORM = "platforms/snax_forge.json"
RECIPE_JSON, SESSION_JSON, EXPORT = "recipe.json", "session.json", "export"
SEED = 0


@dataclass(frozen=True)
class Roots:
    """Where sessions, saved recipes and their step files go."""

    sessions: Path = field(default_factory=lambda: ROOT / "out" / "sessions")
    recipes: Path = field(default_factory=lambda: ROOT / "recipes")
    sandbox: Path = field(default_factory=lambda: ROOT / "out" / "sandbox")


def _refused(code: str, message: str, **more: Any) -> dict[str, Any]:
    return {"ok": False, "code": code, "message": message, **more}


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


class Session:
    """An editing session; see the module doc. Made by ``from_kernel``, ``from_recipe`` or
    ``resume``."""

    def __init__(self, name: str, recipe: Recipe, imported: Graph, platform: str,
                 roots: Roots, cursor: int | None = None) -> None:  # fmt: skip
        """``recipe``'s steps replayed on ``imported`` (a SandboxError when one fails)."""
        self._lock = threading.Lock()
        self.name, self.roots, self.imported = name, roots, imported
        self.platform_path = platform
        self.platform = Platform.load(self._path(platform))
        self.recipe = replace(recipe, name=name)
        self.graphs, self.inputs, self.want = self._replay(self.recipe)
        self.cursor = len(self.graphs) - 1 if cursor is None else cursor
        if not 0 <= self.cursor < len(self.graphs):
            raise SandboxError(f"session {name}: cursor {self.cursor} outside 0..{self.n}")
        self._sites: dict[tuple[int, str], dict[str, Any]] = {}
        self._marks: dict[int, dict[str, list[str]]] = {}
        self._estimates: dict[int, Estimate] = {}
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
        return cls(folder.name, recipe, imported, state["platform"], roots, state["cursor"])

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
    def folder(self) -> Path:
        return self.roots.sessions / self.name

    def _path(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    def _replay(self, recipe: Recipe) -> tuple[list[Graph], dict[str, Any], dict[str, Any]]:
        """Step 0 and every step of ``recipe``, checked; a StepError naming the step fails."""
        g = bind_recipe_symbols(recipe, self.imported)
        inputs, want = reference(g, SEED, recipe.name)
        graphs = [g]
        for i in range(len(recipe.steps)):
            name, params = step_params(recipe, i)
            try:
                graphs.append(checked_step(graphs[-1], name, params, inputs, want,
                                           f"{recipe.name} step {i + 1} ({name})"))  # fmt: skip
            except StepError as e:
                e.index = i + 1  # type: ignore[attr-defined]
                raise
        return graphs, inputs, want

    def _write(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.recipe.save(self.folder / RECIPE_JSON)
        self._write_cursor()

    def _write_cursor(self) -> None:
        state = {"cursor": self.cursor, "platform": self.platform_path}
        (self.folder / SESSION_JSON).write_text(json.dumps(state, indent=1) + "\n")

    def _forget(self, after: int) -> None:
        """The cached answers of the steps past ``after``."""
        self._sites = {k: v for k, v in self._sites.items() if k[0] <= after}
        self._marks = {k: v for k, v in self._marks.items() if k <= after}
        self._estimates = {k: v for k, v in self._estimates.items() if k <= after}

    def _step(self, k: int) -> Graph:
        if not 0 <= k <= self.n:
            raise SandboxError(f"session {self.name}: no step {k} (0..{self.n})")
        return self.graphs[k]

    def _transform(self, k: int) -> str:
        """Step ``k``'s transform, ``input`` for step 0 (run.py's file names)."""
        return "input" if k == 0 else self.recipe.steps[k - 1].transform

    def step_name(self, k: int) -> str:
        """Step ``k``'s name as its file has it: ``<k>_<transform>``."""
        return f"{k}_{self._transform(k)}"

    # -- changing it ----------------------------------------------------------------------

    def apply(self, transform: str, params: dict[str, Any], drop: bool = False) -> dict[str, Any]:
        """The transform at the cursor (module doc)."""
        with self._lock:
            k, n = self.cursor, self.n
            if k < n and not drop:
                return _refused("session.drop", f"steps {k + 1}..{n} come after step {k}: drop "
                                "them to apply here", question=f"Drop steps {k + 1}..{n}?")  # fmt: skip
            step = Step(transform, dict(params))
            trial = replace(self.recipe, steps=[*self.recipe.steps[:k], step])
            start = time.perf_counter()
            try:
                name, values = step_params(trial, k)
                g = checked_step(self.graphs[k], name, values, self.inputs, self.want,
                                 f"{self.name} step {k + 1} ({name})")  # fmt: skip
            except (SandboxError, DfgError) as e:
                return _refused(_code(e), str(e))
            ms = (time.perf_counter() - start) * 1000
            self.recipe, self.graphs = trial, [*self.graphs[: k + 1], g]
            self.cursor = k + 1
            self._forget(k)
            self._write()
            return {"ok": True, "index": k + 1, "transform": transform, "params": dict(params),
                    "ms": ms, "diff": self._diff(k + 1)}  # fmt: skip

    def move(self, k: int) -> dict[str, Any]:
        """The cursor to step ``k``, the steps kept."""
        with self._lock:
            if type(k) is not int or not 0 <= k <= self.n:
                return _refused("session.move", f"no step {k!r}: the steps are 0..{self.n}")
            self.cursor = k
            self._write_cursor()
            return {"ok": True, "cursor": k}

    def set_params(self, values: dict[str, int | str]) -> dict[str, Any]:
        """Every step replayed with the params ``values`` sets."""
        with self._lock:
            try:
                recipe = self.recipe.with_params(values)
                graphs, inputs, want = self._replay(recipe)
            except StepError as e:
                return _refused(_code(e), str(e), index=e.index)  # type: ignore[attr-defined]
            except (SandboxError, DfgError) as e:
                return _refused(_code(e), str(e), index=0)
            self.recipe, self.graphs, self.inputs, self.want = recipe, graphs, inputs, want
            self._forget(-1)
            self._write()
            return {"ok": True, "params": dict(recipe.params)}

    def save(self, name: str, force: bool = False) -> dict[str, Any]:
        """Steps 1..cursor as a recipe of its own (module doc)."""
        with self._lock:
            path = self.roots.recipes / f"{name}.json"
            if not isinstance(name, str) or not name.isidentifier():
                return _refused("save.name", f"{name!r} is not a name (an identifier)")
            if path.exists():
                return _refused("save.name", f"{path} exists; a recipe is never overwritten")
            g = self.graphs[self.cursor]
            problems = run_checks(Design(self.platform, g))
            listed = [str(p) for p in problems]
            if any(_stage(p.code) == "graph" for p in problems):
                return _refused("save.graph", f"step {self.cursor} is not a graph the flow takes",
                                problems=listed)  # fmt: skip
            if problems and not force:
                return _refused("save.design", f"the design checks on {self.platform.name} "
                                f"report {len(problems)} problem(s)", problems=listed)  # fmt: skip
            recipe = replace(self.recipe, name=name, steps=self.recipe.steps[: self.cursor])
            path.parent.mkdir(parents=True, exist_ok=True)
            recipe.save(path)
            results = [
                Result(k, self._transform(k), self.graphs[k]) for k in range(self.cursor + 1)
            ]
            write_steps(recipe, results, self.roots.sandbox / name)
            shown = _shown(path)
            return {"ok": True, "recipe": str(path), "problems": listed, "commands": [
                f"pixi run sandbox {shown}",
                f"pixi run flow {shown} --platform {self.platform_path}",
                f"pixi run forge {shown}",
            ]}  # fmt: skip

    # -- asking it ------------------------------------------------------------------------

    def export(self, k: int) -> Path:
        """Step ``k``'s graph as a file of its own."""
        with self._lock:
            g = self._step(k)
            out = self.folder / EXPORT / f"{self.step_name(k)}.snaxdfg"
            out.parent.mkdir(parents=True, exist_ok=True)
            g.save(out)
            return out

    def sites(self, k: int, selected: str) -> dict[str, dict[str, Any]]:
        """Every transform's site on ``selected`` at step ``k``, bind's with the bind list."""
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

    def estimate(self, k: int) -> Estimate:
        """Step ``k``'s ideal estimate on the session's platform."""
        with self._lock:
            return self._estimate(k)

    # -- the answers, under the lock --------------------------------------------------------

    def _sites_of(self, k: int, selected: str) -> dict[str, dict[str, Any]]:
        key = (k, selected)
        if key in self._sites:
            return self._sites[key]
        g = self._step(k)
        out: dict[str, dict[str, Any]] = {}
        for name, spec in TRANSFORMS.items():
            if spec.site is None:
                continue
            try:
                out[name] = {"ok": True, "params": spec.site(g, selected)}
            except (SandboxError, DfgError) as e:
                out[name] = _refused(_code(e), str(e))
        if out.get("bind", {}).get("ok"):
            out["bind"]["list"] = bind_list(g, selected)
        for name, answer in out.items():
            if answer["ok"]:
                answer["form"] = form(name, g, answer, self.recipe.params)
        self._sites[key] = out
        return out

    def _estimate(self, k: int) -> Estimate:
        if k not in self._estimates:
            self._estimates[k] = estimate(self._step(k), self.platform)
        return self._estimates[k]

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


def _shown(p: Path) -> str:
    """A path as a command line in the repository writes it."""
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def bind_list(g: Graph, selected: str) -> list[dict[str, Any]]:
    """Every (BRM, implementation) of the library tried as ``bind`` on ``selected`` (module
    doc)."""
    from snax_forge.brm import BrmError, library_names, load_brm
    from snax_forge.common import fresh

    site = _bind_site(g, selected, f"bind {selected!r}")
    w = _bound(g, parse_dim(site.smap.attrs["range"], site.smap.id)[1], f"bind {selected!r}")
    taken = {n.attrs["instance"] for n, _ in g.walk() if n.kind == "accelerated"}
    taken |= {n.attrs[k] for n, _ in g.walk() if n.kind == "accelerated"
              for k in ("accelerator", "group") if k in n.attrs}  # fmt: skip
    instance = fresh(site.tasklet.id, taken)
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


# The defaults of the params a site does not fill (the user's choices)
DEFAULT_INT = {"factor": 4, "depth": 1}
DEFAULT_KIND = "temporal"


def form(
    name: str, g: Graph, answer: dict[str, Any], params: dict[str, Any]
) -> list[dict[str, Any]]:
    """The fields the editor shows to apply transform ``name`` where its site accepted
    (``answer``, with the bind list for ``bind``), in the transform's param order: each
    ``name``, ``type`` (``int``, ``str``, ``choice``, ``checklist``), ``sweepable`` (one of the
    transform's ``ints``, so a number or an expression over the session's ``params``),
    ``value`` (the site's, else a default: a factor is ``W`` when the session has that param),
    ``readonly`` (the selection's own field) and ``choices`` where they apply: ``tag_map``'s
    kinds, ``bind``'s (BRM, implementation) pairs from the bind list with their timing and
    message (one field, ``binding``, whose ``value`` gives both params), ``encapsulate``'s
    units (the instances in no accelerator, in the graph's order)."""
    import inspect

    from snax_forge.common import fresh

    from .transforms import TAGS

    spec = TRANSFORMS[name]
    filled = answer["params"]
    selection = next(iter(filled), None)
    fields: list[dict[str, Any]] = []
    for p in list(inspect.signature(spec.fn).parameters)[1:]:
        field: dict[str, Any] = {"name": p, "type": "str", "sweepable": p in spec.ints,
                                 "value": filled.get(p), "readonly": p == selection}  # fmt: skip
        if p in spec.ints:
            field["type"] = "int"
            if field["value"] is None:
                field["value"] = "W" if p == "factor" and "W" in params else DEFAULT_INT.get(p, 1)
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
            field["value"] = ok[0]["params"]["instance"] if ok else filled.get("node")
        elif name == "encapsulate" and p == "units":
            nodes = [n for n, _ in g.walk() if n.kind == "accelerated"]
            free = list(
                dict.fromkeys(n.attrs["instance"] for n in nodes if "accelerator" not in n.attrs)
            )
            field |= {"type": "checklist", "readonly": False,
                      "choices": [{"value": u, "label": u} for u in free]}  # fmt: skip
        elif name == "encapsulate" and p == "accelerator":
            taken = {n.attrs[k] for n, _ in g.walk() if n.kind == "accelerated"
                     for k in ("instance", "accelerator", "group") if k in n.attrs}  # fmt: skip
            field["value"] = fresh(g.name, taken)
        fields.append(field)
    return fields


def _timing(inst: Any) -> dict[str, int]:
    return {"latency": inst.latency, "ii": inst.initiation_interval, "span": inst.span}
