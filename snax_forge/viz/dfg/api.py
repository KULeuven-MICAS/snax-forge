"""What the DFG viewer asks for, as plain Python functions; server.py turns them into JSON.

Loading: ``expand`` takes ``.snaxdfg`` files and directories (a directory's files in natural
order), ``load_entry`` reads each into an ``Entry`` named after its stem (``<folder>/<stem>``
when the files come from several folders) and ``GraphSet`` holds them, read at start-up or on
Reload. A file that does not load keeps its error, so the other steps still draw. ``find_step``
reads a step back from the URL hash.

``graph_view`` lays a graph out in rows in execution order, containers between top-level nodes,
a container versioned after each write (``A@1``) so no edge goes up; independent chains side by
side (``columns``), blocks folded into one box (``fold``), groups and accelerators boxed. Every
item has a compact ``text`` and a ``details`` entry keyed ``<kind>:<id>``, all made here.
``step_view`` adds the marks, the estimate, its cost and the left column's sections;
``compare_view`` gives two steps' diff as rows ``{label, under, a, b, differs}``; ``partners``
lists the maps a map can be fused, seeded or interchanged with, from the transforms' own sites.
The page only lays these out. Its behaviour is in CONTRACTS, "Recipe"; the estimate and the cost
in "Estimate" and "Cost model".
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

from snax_forge import expr
from snax_forge.brm import SEED, BrmError
from snax_forge.brm.dtypes import bits
from snax_forge.char import DEFAULT_TECH
from snax_forge.common import LoadedSet, unique_names
from snax_forge.design.streamers import (
    SHARE_PORTS,
    accelerator_of,
    firing_loops,
    group_of,
    instance_of,
    port_loops,
)
from snax_forge.dfg import DfgError, Graph, Memlet, Node
from snax_forge.dfg.graph import BAKED, SCALAR, TRANSPOSE
from snax_forge.dfg.subset import dim_names, map_range, parse_dim
from snax_forge.estimate.cost import L1_LABEL, PARTS, L1Cost, caveat, dec3, heading, shares, sig3
from snax_forge.expr import ExprError
from snax_forge.sandbox.blocks import BLOCK
from snax_forge.sandbox.session import map_ids

# A bank word, in bits: num_slc = WORD_BITS / b slices of a b-bit dtype
WORD_BITS = 64

# =============================================================================
# Files
# =============================================================================


def _natural(p: Path) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def expand(paths: Sequence[str | Path]) -> list[Path]:
    """Files as given, directories as their ``.snaxdfg`` files in natural order."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files = sorted(p.glob("*.snaxdfg"), key=_natural)
            if not files:
                raise FileNotFoundError(f"{p}: no .snaxdfg file")
            out.extend(files)
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"{p}: no such file or directory")
    return out


def entry_folders(files: Sequence[Path]) -> list[str | None]:
    """Each file's folder, its parent directory's name, when the files come from more than one
    folder; else None for every file."""
    parents = [Path(os.path.abspath(f)).parent for f in files]
    if len(set(parents)) < 2:
        return [None] * len(files)
    return [d.name for d in parents]


def entry_names(files: Sequence[Path]) -> list[str]:
    """File stems, ``<folder>/<stem>`` when the files come from more than one folder
    (``entry_folders``), made unique with -2, -3, ... in the order given."""
    folders = entry_folders(files)
    return unique_names(f.stem if d is None else f"{d}/{f.stem}"
                        for f, d in zip(files, folders, strict=True))  # fmt: skip


def find_step(names: Sequence[str], ref: str) -> int | None:
    """The step a URL hash names (``step=``, ``cmp=``): a step's full name; else a bare stem
    exactly one step has (a hash from before the names had folders); else a number below
    ``len(names)`` (a hash from before it held names); else None."""
    if ref in names:
        return list(names).index(ref)
    by_stem = [i for i, n in enumerate(names) if n.rpartition("/")[2] == ref]
    if len(by_stem) == 1:
        return by_stem[0]
    if ref.isdigit() and int(ref) < len(names):
        return int(ref)
    return None


@dataclass
class Entry:
    name: str
    path: Path
    graph: Graph | None
    error: str | None
    folder: str | None = None  # its folder when the files come from more than one


def load_entry(name: str, path: Path, folder: str | None = None) -> Entry:
    """One file; a file that does not load keeps its error instead of a graph."""
    try:
        return Entry(name, path, Graph.load(path), None, folder)
    except (OSError, ValueError, KeyError, TypeError) as e:
        msg = f"invalid JSON: {e}" if isinstance(e, json.JSONDecodeError) else str(e)
        return Entry(name, path, None, msg, folder)


class GraphSet(LoadedSet):
    """The files the viewer was given, read once and again on ``reload`` (``LoadedSet``)."""

    def __init__(self, paths: Sequence[str | Path]):
        self.paths = [Path(p) for p in paths]
        super().__init__()

    def load(self) -> dict[str, Entry]:
        files = expand(self.paths)
        return {n: load_entry(n, f, d)
                for n, f, d in zip(entry_names(files), files, entry_folders(files), strict=True)}  # fmt: skip

    def summaries(self) -> list[dict[str, Any]]:
        return [entry_summary(e) for e in self.values()]


def entry_summary(e: Entry) -> dict[str, Any]:
    """A step's entry: its name, folder (None from one folder), path, graph name and load
    error."""
    return {
        "name": e.name,
        "folder": e.folder,
        "path": str(e.path),
        "graph": e.graph.name if e.graph else None,
        "error": e.error,
    }


# =============================================================================
# The view of one graph
# =============================================================================


def graph_view(e: Entry, platform: Any = None, est: Any = None,
               error: str | None = None, tech: str = DEFAULT_TECH,
               expanded: Sequence[str] = ()) -> dict[str, Any]:  # fmt: skip
    """The rows, nodes and edges a panel draws, with their compact texts and Details (module
    doc); ``est`` the graph's estimate (or ``error`` why there is none), else made on
    ``platform`` and costed on ``tech``; the blocks not ``expanded`` drawn as one box each."""
    out: dict[str, Any] = entry_summary(e)
    if e.graph is None:
        return out
    g = e.graph
    if est is None and error is None:
        est, error = _estimated(g, platform, tech)
    out["estimate_error"] = error
    placed = est.containers if est is not None else {}
    out.update(
        symbols=dict(g.symbols),
        containers={k: _container(g, k, placed.get(k)) for k in g.containers},
    )
    tops, out["blocks"] = fold(g, expanded)
    out.update(_layout(g, tops))
    out["partners"] = partners(g, out["boxes"])
    out["shared"] = shared_units(g)
    out["accelerators"] = accelerators(g)
    seen = _drawn_units(tops)
    for acc in out["accelerators"]:  # a group no unit of which is drawn (in collapsed blocks)
        for grp in acc["groups"]:
            if not {u["instance"] for u in grp["units"]} & seen:
                grp["drawn"] = False
    for acc in out["accelerators"]:
        acc["text"] = acc["name"]
        for grp in acc["groups"]:
            grp["text"] = grp["name"]
    out["details"] = details(g, out, est)
    for b in out["blocks"]:
        out["details"][f"block:{b['name']}"] = block_details(g, b)
    for edge in out["edges"]:  # their memlet and packing are in Details
        edge.pop("text")
        edge.pop("packing", None)
    return out


def step_view(e: Entry, label: str, platform: Any = None, marks: dict[str, list[str]] | None = None,
              diff: dict[str, Any] | None = None, est: Any = None, tech: str = DEFAULT_TECH,
              recipe: Any = None, editable: bool = False,
              expanded: Sequence[str] = ()) -> dict[str, Any]:  # fmt: skip
    """One step (module doc): the graph's view, its label, marks, estimate and its summary,
    and diff; the cost on ``tech``; ``recipe`` the steps up to this one (a session's, or a
    sandbox folder's ``recipe.json``), whose params the left column's sections name, inputs
    where ``editable`` (edit mode); the blocks ``expanded`` drawn as their nodes."""
    error = None
    if e.graph is not None and est is None:
        est, error = _estimated(e.graph, platform, tech)
    out = graph_view(e, platform, est, error, tech, expanded)
    out["label"] = label
    out["tech"] = tech
    if e.graph is None:
        return out
    out["estimate"] = est.to_dict() if est is not None else None
    out["summary"] = estimate_summary(est, e.graph) if est is not None else None
    out["containers_section"] = containers_section(e.graph, est)
    out["symbols_section"] = symbols_section(e.graph, recipe, editable)
    out["bindings_section"] = bindings_section(e.graph, recipe, editable)
    if recipe is not None:  # the recipe's param names in the boxes and Details
        _name_params(out, e.graph, recipe)
    if marks is None:
        from snax_forge.sandbox.session import marks_of

        marks = marks_of(e.graph)
    out["marks"] = marks
    if diff is not None:
        out["diff"] = diff
    return out


NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STEP = re.compile(r"^(\d+)_")


def folder_recipe(e: Entry) -> Any:
    """View mode: the ``recipe.json`` beside a step's file (a sandbox folder's), its steps up to
    that step (the stem's number), or None when there is none, it does not load, or the stem
    has no number."""
    from snax_forge.sandbox import Recipe, SandboxError

    m = STEP.match(Path(e.path).stem)
    path = Path(e.path).parent / "recipe.json"
    if m is None or not path.is_file():
        return None
    try:
        r = Recipe.load(path)
    except (SandboxError, OSError, ValueError):
        return None
    from dataclasses import replace

    return replace(r, steps=r.steps[: int(m[1])])


def symbols_section(g: Graph, recipe: Any = None, editable: bool = False) -> list[dict[str, Any]]:
    """The left column's Symbols: each graph symbol, its ``value`` (None while unbound),
    the recipe ``param`` its value is (the recipe's symbol bound to a param's name, ``N`` ->
    ``N``; None for a number or an expression), ``editable`` (edit mode and a param: an input,
    ``POST /api/params``) and the ``containers`` whose shape uses it."""
    users: dict[str, list[str]] = {s: [] for s in g.symbols}
    for c in containers_section(g):
        for x in c["symbols"]:
            users.setdefault(x, []).append(c["name"])
    out = []
    for name, value in g.symbols.items():
        given = recipe.symbols.get(name) if recipe is not None else None
        param = (
            given
            if isinstance(given, str) and expr.is_name(given) and given in recipe.params
            else None
        )
        out.append({"name": name, "value": value, "param": param,
                    "editable": bool(editable and param), "containers": users.get(name, [])})  # fmt: skip
    return out


# The transforms that split a map by a factor, and the inner maps each makes, by suffix
SPLITS = {"split": ("_inner",), "split_map": ("_s",), "split_reduce": ("_s", "_final")}


def _param_names(v: Any, params: dict[str, Any]) -> list[str]:
    """The recipe params an int step param names (a factor ``W``, ``W * 2``)."""
    if not isinstance(v, str):
        return []
    try:
        return [x for x in sorted(expr.names(v)) if x in params]
    except ExprError:
        return []


def _splits(recipe: Any) -> list[tuple[Any, str, list[str], re.Pattern[str], Any]]:
    """The recipe's split steps: map, transform, the params its factor names, the pattern of
    the inner maps it made (``M_inner``, ``M_s``, ``M_final``) and the factor as written."""
    if recipe is None:
        return []
    params, out = dict(recipe.params), []
    for st in recipe.steps:
        p = st.params
        if st.transform in SPLITS:
            m = p.get("map")
            inner = re.compile(rf"^{re.escape(str(m))}({'|'.join(SPLITS[st.transform])})(_\d+)?$")
            out.append((m, st.transform, _param_names(p.get("factor"), params), inner,
                        p.get("factor")))  # fmt: skip
    return out


def unit_params(g: Graph, recipe: Any) -> dict[str, list[str]]:
    """Per bound instance, the recipe params of the splits whose inner map it replaced (the recipe's
    Bindings rule: ``bind`` reads W off the spatial map)."""
    splits, out = _splits(recipe), {}
    for n, _ in g.walk():
        if n.kind == "accelerated" and n.attrs["instance"] not in out:
            ids = map_ids(n.attrs.get("replaced"))
            names: list[str] = []
            for _, _, ps, inner, _ in splits:
                if any(inner.match(x) for x in ids):
                    names += [x for x in ps if x not in names]
            out[n.attrs["instance"]] = names
    return out


def map_ranges(g: Graph, recipe: Any) -> dict[str, str]:
    """Per map a split made, its range with the factor's param name and its iterations beside
    it (``0:N // W_acc_1 = 256``, the inner ``0:W_acc_1 = 8``); a factor given as a
    number keeps the range as the graph has it."""
    out: dict[str, str] = {}
    maps = {n.id: n for n, _ in g.walk() if n.kind == "map"}
    for m, _, _, inner, factor in _splits(recipe):
        if not isinstance(factor, str) or m not in maps or "loop.split" not in maps[m].attrs:
            continue
        f = factor if expr.is_name(factor) else f"({factor})"
        node = maps[m]
        split = node.attrs["loop.split"]
        begin, end, _ = parse_dim(split["range"], m)
        length = expr.canonical(f"({end}) - ({begin})") if str(begin) != "0" else str(end)
        rng = f"0:({length} + {f} - 1) // {f}" if "valid" in split else f"0:{length} // {f}"
        its = _iterations(g, node.attrs["range"])
        out[m] = f"{rng} = {'?' if its is None else its}"
        for kid in node.body or []:
            if kid.kind == "map" and inner.match(kid.id):
                its = _iterations(g, kid.attrs["range"])
                out[kid.id] = f"0:{factor} = {'?' if its is None else its}"
    return out


def _name_params(view: dict[str, Any], g: Graph, recipe: Any) -> None:
    """The step view's unit boxes and map Details with the recipe's param names: a
    unit's lanes under its split's param (``acc_1 · in1 * in2 · W_acc_1 8``), a split map's
    ``range`` (``0:N // W_acc_1 = 256``, the graph's ``0:N // 8`` named)."""
    names = unit_params(g, recipe)
    nodes = {n.id: n for n, _ in g.walk()}

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            n = nodes.get(x.get("id")) if x.get("kind") == "accelerated" else None
            if n is not None and "text" in x:
                ps = names.get(n.attrs["instance"]) or []
                x["text"] = compact_text(g, n, ps[0] if ps else None)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(view.get("rows"))
    walk(view.get("nodes"))
    for m, text in map_ranges(g, recipe).items():
        d = view.get("details", {}).get(f"node:{m}")
        if d is not None:
            d["range"] = text


def bindings_section(g: Graph, recipe: Any = None, editable: bool = False) -> dict[str, Any]:
    """The left column's Bindings: each param shown with what it configures. The param
    stays the split's (``bind`` reads W off the spatial map); a split step on map M
    configures the unit whose replaced subtree holds the map it made (``M_inner``, ``M_s``,
    ``split_reduce``'s ``M_final``). ``entries``, in the graph's order:

        unit    per bound instance: its BRM, the lanes its design params give (``W = 8``) and
                the params of the splits that set them, then its slots, read only
                (``acc · simd · W = 8``, relu's ``acc · simd · W = 4 · k = 0``); an instance
                several nodes share names them (``vmul_x1 · simd · tmp0_mult, tmp2_mult``),
                the splits of every one its own
        split   a split whose map is not bound yet: its factor's params (``not bound``)
        link    a link whose depth a step gives as a param

    Each param ``{name, value, editable, shared_with}``: an input in edit mode
    (``POST /api/params``), ``shared_with`` the other entries naming it (dot 2–6's ``W``, the
    multiply's and the sum's). ``unused``: the session's params neither a symbol nor in an
    entry. Without a recipe (view mode on bare files) the units come from the graph alone,
    their design params read only."""
    params = dict(recipe.params) if recipe is not None else {}
    splits = _splits(recipe)  # (map, transform, factor params, inner ids)
    links = {}  # container -> depth params
    for st in list(recipe.steps) if recipe is not None else []:
        p = st.params
        if st.transform == "link":
            names = _param_names(p.get("depth"), params)
            if names:
                links[p.get("container")] = names
    entries: list[dict[str, Any]] = []
    used: set[int] = set()
    seen: set[str] = set()
    nodes_of: dict[str, list[Node]] = {}  # an instance's nodes, several when shared
    for n, _ in g.walk():
        if n.kind == "accelerated":
            nodes_of.setdefault(n.attrs["instance"], []).append(n)
    for n, _ in g.walk():
        if n.kind == "accelerated" and n.attrs["instance"] not in seen:
            a = n.attrs
            seen.add(a["instance"])
            ids = {x for m in nodes_of[a["instance"]] for x in map_ids(m.attrs.get("replaced"))}
            names: list[str] = []
            for i, (_, _, ps, inner, _) in enumerate(splits):
                if any(inner.match(x) for x in ids):
                    used.add(i)
                    names += [x for x in ps if x not in names]
            design = dict(a.get("params") or {})
            slots = slot_params(n)
            lanes = {k: v for k, v in design.items() if isinstance(v, int) and k not in slots
                     and k in _lanes_names(n)}  # fmt: skip
            shared = [m.id for m in nodes_of[a["instance"]]]
            label = f"{a['instance']} · {a['brm']}" + (
                f" · {', '.join(shared)}" if len(shared) > 1 else ""
            )
            entries.append({"kind": "unit", "id": a["instance"], "label": label, "names": names,
                            "lanes": lanes, "slots": slots, "note": None})  # fmt: skip
        elif n.kind == "link" and n.attrs["container"] in links:
            c = n.attrs["container"]
            entries.append({"kind": "link", "id": c, "label": f"link {c} · depth",
                            "names": links[c], "lanes": {}, "note": None})  # fmt: skip
    for i, (m, transform, ps, _, _) in enumerate(splits):
        node = next((x for x, _ in g.walk() if x.id == m and x.kind == "map"), None)
        if i in used or node is None or "loop.split" not in node.attrs:
            continue
        entries.append({"kind": "split", "id": m, "label": f"{m} {transform}", "names": ps,
                        "lanes": {}, "note": "not bound"})  # fmt: skip
    by_name: dict[str, list[str]] = {}
    for e in entries:
        for x in e["names"]:
            by_name.setdefault(x, []).append(e["label"])
    for e in entries:
        names, slots = e.pop("names"), e.pop("slots", {})
        e["params"] = [{"name": x, "value": params.get(x), "editable": bool(editable and x in params),
                        "shared_with": [lbl for lbl in by_name[x] if lbl != e["label"]]}
                       for x in names]  # fmt: skip
        if not e["params"] and e["lanes"]:  # no recipe param: the design params, read only
            e["params"] = [{"name": k, "value": v, "editable": False, "shared_with": []}
                           for k, v in e["lanes"].items()]  # fmt: skip
        e["params"] += [{"name": k, "value": v, "editable": False, "shared_with": []}
                        for k, v in slots.items()]  # a slot, baked in: read only  # fmt: skip
    symbol_params = {x["param"] for x in symbols_section(g, recipe) if x["param"]}
    unused = [{"name": k, "value": v, "editable": editable} for k, v in params.items()
              if k not in symbol_params and k not in by_name]  # fmt: skip
    return {"entries": entries, "unused": unused, "tree": bindings_tree(g, entries)}


def slot_params(n: Node) -> dict[str, int | str]:
    """An accelerated node's slots and their values: the int design params its code reads, baked
    in (``simd``'s ``k``, brm.py's ``Code``), then the slots read from a scalar, each an input
    connector that is no port, as ``csr(<container>)`` (axpy's ``k1``: ``csr(alpha)``), in the
    order the code reads them."""
    params = n.attrs.get("params") or {}
    read = [x for _, e in expr.statements(n.attrs["code"], n.id) for x in expr.ordered_names(e)]
    ports = _port_names(n)
    out: dict[str, int | str] = {}
    for x in dict.fromkeys(read):
        if type(params.get(x)) is int:
            out[x] = params[x]
        elif x in n.inputs and x not in ports:
            out[x] = f"csr({n.inputs[x].data})"
    return out


def _port_names(n: Node) -> set[str]:
    """Its BRM's port names; its connectors when the BRM does not load."""
    from snax_forge.brm import load_brm

    try:
        return {p.name for p in load_brm(n.attrs["brm"]).interface.ports}
    except (BrmError, KeyError, OSError):
        return set(n.inputs) | set(n.outputs)


def _lanes_names(n: Node) -> set[str]:
    """The design params its BRM's ports' lanes name; every int param when it does not
    resolve."""
    try:
        inst = instance_of(n)
    except (BrmError, ExprError, KeyError, ValueError):
        return set(n.attrs.get("params") or {})
    return {x for p in inst.ports for x in expr.names(p.lanes)}


def bindings_tree(g: Graph, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The Bindings as a tree, in the entries' order: an encapsulated accelerator
    ``{kind: accelerator, name, children}`` holds its groups ``{kind: group, name, children}``,
    each its units' and links' entries ``{kind: entry, index}`` (``index`` into ``entries``); a
    linked group not in an accelerator is a group at the top; a unit alone, a split not bound
    yet and a link of no group are entries at the top."""
    where: dict[tuple[str, str], tuple[str | None, str | None]] = {}  # entry -> (acc, group)
    for acc in accelerators(g):
        units = [u["instance"] for grp in acc["groups"] for u in grp["units"]]
        encapsulated = any("accelerator" in n.attrs for n, _ in g.walk()
                           if n.kind == "accelerated" and n.attrs["instance"] in units)  # fmt: skip
        for grp in acc["groups"]:
            several = len(grp["units"]) > 1
            group = grp["name"] if encapsulated or several else None
            for u in grp["units"]:
                where[("unit", u["instance"])] = (acc["name"] if encapsulated else None, group)
            for link in grp["links"]:
                node = next((n for n, _ in g.walk() if n.id == link), None)
                if node is not None:
                    where[("link", node.attrs["container"])] = (
                        acc["name"] if encapsulated else None,
                        grp["name"],
                    )
    top: list[dict[str, Any]] = []
    index: dict[tuple[str, ...], dict[str, Any]] = {}

    def node(path: tuple[str, ...], kind: str, name: str, into: list[dict[str, Any]]) -> list:
        if path not in index:
            index[path] = {"kind": kind, "name": name, "children": []}
            into.append(index[path])
        return index[path]["children"]

    for i, e in enumerate(entries):
        acc, group = where.get((e["kind"], e["id"]), (None, None))
        into = top
        if acc is not None:
            into = node(("acc", acc), "accelerator", acc, into)
        if group is not None:
            into = node(("acc", acc or "", "group", group), "group", group, into)
        into.append({"kind": "entry", "index": i})
    return top


@dataclass(frozen=True)
class _Local:
    """A fused nest's local value as the Containers and Details show it: in no memory, so no
    bytes, its placement ``local to <fused map>, no memory``."""

    fused: str
    bytes: None = None
    depth: None = None

    @property
    def placement(self) -> str:
        return f"local to {self.fused}, no memory"


def _placed_text(placed: Any) -> str | None:
    """A container's placement as its line shows it: ``L1``, ``wire``, ``FIFO 1``, a fused
    nest's local value ``local to <fused map>`` (no bytes after it: it has no memory; Details
    say so in full, so the line fits the column at 1280 px)."""
    if placed is None:
        return None
    if isinstance(placed, _Local):
        return f"local to {placed.fused}"
    if placed.placement == "l1":
        return "L1"
    if placed.placement == "link":
        return "wire" if placed.depth == 0 else f"FIFO {placed.depth}"
    return placed.placement


def containers_section(g: Graph, est: Any = None) -> list[dict[str, Any]]:
    """The left column's Containers, per container in the graph's order: its ``name``,
    ``shape`` as written (``shape_text``, its dims by " × ", and ``dims``, each dim's ``parts``,
    ``{text, symbol}`` with ``symbol`` the graph symbol a part names or None, so the page puts
    an input on each), ``sizes`` at the graph's symbols (None while one is unbound), ``dtype``,
    padded ``bytes`` and ``placement`` as the estimate has them (None without one), and the
    ``symbols`` its shape uses, in order; ``line``, what the page shows after its name: the shape,
    the placement and the padded bytes (``N · L1 · 16 KiB``)."""
    out = []
    for name, c in g.containers.items():
        placed = est.containers.get(name) if est is not None else None
        if "loop.local" in c.attrs:  # no memory: no bytes, its placement the fused map's
            placed = _Local(c.attrs["loop.local"])
        try:
            sizes = [expr.evaluate(x, g.bound_symbols) for x in c.shape]
        except ExprError:
            sizes = None
        dims, used = [], []
        for d in c.shape:
            text, parts, at = str(d), [], 0
            for m in NAME.finditer(text):
                if m[0] not in g.symbols:
                    continue
                if m.start() > at:
                    parts.append({"text": text[at : m.start()], "symbol": None})
                parts.append({"text": m[0], "symbol": m[0]})
                used.append(m[0])
                at = m.end()
            if at < len(text) or not parts:
                parts.append({"text": text[at:], "symbol": None})
            dims.append({"parts": parts})
        scalar = bool(c.attrs.get(SCALAR))
        out.append({
            "name": name, "shape": list(c.shape), "scalar": scalar,
            "shape_text": " × ".join(str(d) for d in c.shape) or "scalar", "dims": dims,
            "sizes": sizes, "sizes_text": " × ".join(map(str, sizes)) if sizes else None,
            "dtype": c.dtype, "transient": c.transient,
            "bytes": placed.bytes if placed else None,
            "bytes_text": size_text(placed.bytes) if placed and placed.bytes is not None else None,
            "placement": placed.placement if placed else None,
            "symbols": list(dict.fromkeys(used)),
            "line": f"scalar · {c.dtype}" if scalar else
                    " · ".join(x for x in (" × ".join(str(d) for d in c.shape) or "scalar",
                                           _placed_text(placed),
                                           size_text(placed.bytes)
                                           if placed and placed.bytes is not None else None)
                               if x),
        })  # fmt: skip
    return out


def compare_view(a: Entry, b: Entry, platform: Any = None, tech: str = DEFAULT_TECH,
                 labels: tuple[str, str] | None = None) -> dict[str, Any]:  # fmt: skip
    """Two steps compared (``a`` on the left, ``b`` on the right, ``labels`` their headings,
    else their names): the session's diff of their graphs (sandbox/session.py's
    ``graph_diff``), per Details key in either the fields that differ (every field of an item
    in one only), and the two ``columns`` (module doc)."""
    from snax_forge.sandbox.session import graph_diff

    if a.graph is None or b.graph is None:
        return {"diff": None, "changed": {}, "error": a.error or b.error}
    (ea, _), (eb, _) = _estimated(a.graph, platform, tech), _estimated(b.graph, platform, tech)
    da = graph_view(a, platform, ea, None if ea else "", tech)["details"]
    db = graph_view(b, platform, eb, None if eb else "", tech)["details"]
    changed = {}
    for key in dict.fromkeys([*da, *db]):
        x, y = da.get(key, {}), db.get(key, {})
        fields = [f for f in dict.fromkeys([*x, *y]) if x.get(f) != y.get(f)]
        if fields:
            changed[key] = fields
    diff = graph_diff(
        a.graph, b.graph, ea.containers if ea else None, eb.containers if eb else None
    )
    sa, sb = (estimate_summary(e, x.graph) if e is not None else None
              for e, x in ((ea, a), (eb, b)))  # fmt: skip
    columns = {
        "labels": list(labels or (a.name, b.name)),
        "details": {key: {"where": "added" if key not in da else "removed" if key not in db else "",
                          "rows": detail_rows(da.get(key), db.get(key))}
                    for key in dict.fromkeys([*da, *db])},
        "estimate": estimate_rows(sa, sb),
        "cost_tables": compare_cost_tables(sa, sb),
        "cost_bars": compare_cost_bars(list(labels or (a.name, b.name)), sa, sb),
        "caveat": ((sb or {}).get("cost") or (sa or {}).get("cost") or {}).get("caveat"),
    }  # fmt: skip
    return {"diff": diff, "changed": changed, "columns": columns}


def compare_cost_bars(labels: list[str], sa: dict[str, Any] | None,
                      sb: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]] | None:  # fmt: skip
    """Compare's cost bars: for ``area`` and for ``energy``, one bar per side (the left
    step first), both on one scale, the larger total the whole width, so the shorter bar is the
    cheaper design; each its ``label``, ``total`` and ``total_text`` and its ``segments``, per
    part costed its ``name``, ``label``, ``width`` (0..1 of the scale) and ``text``. None when
    neither side has a cost."""
    sides = [
        (lbl, ((s or {}).get("cost") or {}).get("parts") or [])
        for lbl, s in zip(labels, (sa, sb), strict=True)
    ]
    if not any(p for _, p in sides):
        return None
    out: dict[str, list[dict[str, Any]]] = {}
    for key, unit in (("area", _um2), ("energy", _nj)):
        totals = [sum(p[key] or 0.0 for p in parts) for _, parts in sides]
        scale = max(totals) or 1.0
        out[key] = [{"label": lbl, "total": t, "total_text": unit(t) if parts else "–",
                     "segments": [{"name": p["name"], "label": p["label"],
                                   "width": p[key] / scale, "text": p[f"{key}_text"]}
                                  for p in parts if p[key]]}
                    for (lbl, parts), t in zip(sides, totals, strict=True)]  # fmt: skip
    return out


def value_text(v: Any) -> str:
    """A Details value as the page shows one (dfg.js's ``valueText``): None "–", a list its
    items by ", ", an object its JSON."""
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return ", ".join(json.dumps(x, separators=(",", ":"), ensure_ascii=False)
                         if isinstance(x, (dict, list)) else value_text(x) for x in v) or "–"  # fmt: skip
    if isinstance(v, dict):
        return json.dumps(v, separators=(",", ":"), ensure_ascii=False)
    return str(v)


def _row(label: str, a: Any, b: Any, under: str | None = None) -> dict[str, Any]:
    x, y = value_text(a), value_text(b)
    return {"label": label, "under": under, "a": x, "b": y, "differs": x != y}


def detail_rows(x: dict[str, Any] | None, y: dict[str, Any] | None) -> list[dict[str, Any]]:
    """One item's Details in two columns (module doc): a row per field, its ``cost`` a row per
    text under ``cost``; "–" on a side without the field or the item."""
    x, y = x or {}, y or {}
    rows = []
    for f in dict.fromkeys([*x, *y]):
        if f == "kind":
            continue
        u, w = x.get(f), y.get(f)
        if f == "cost" and (isinstance(u, dict) or isinstance(w, dict)):
            cu, cw = u or {}, w or {}
            rows += [_row(k, cu.get(k), cw.get(k), "cost") for k in dict.fromkeys([*cu, *cw])]
        else:
            rows.append(_row(f, u, w))
    return rows


def estimate_rows(sa: dict[str, Any] | None, sb: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The Estimate panel in two columns (module doc), from each side's summary (None: no
    estimate): under ``accelerator`` its cycles, the unmapped cycles and the total (the
    one unmapped count, ``not_mapped`` staying in the estimate); under ``group <name>``,
    every group of either side, its cycles, expected BUSY_COUNTER (a guide), channels, area and
    energy, and its bank
    note where a side has one; under ``memory`` L1, the words read and written, the lanes; under the
    cost heading the
    accelerator's area, energy, L1 accesses and the blocks not characterized. "–" where a
    side has none."""

    def both(fn: Any) -> tuple[Any, Any]:
        def one(s: dict[str, Any] | None) -> Any:
            if s is None:
                return None
            try:
                return fn(s)
            except (KeyError, TypeError, StopIteration):
                return None

        return one(sa), one(sb)

    def l1(s: dict[str, Any]) -> str:
        x = s["l1"]
        return f"{x['footprint_text']} of {x['size_text']} ({x['share_text']})" if x["size"] else (
            x["footprint_text"])  # fmt: skip

    rows = [_row("cycles", *both(lambda s: s["accelerator_cycles"]), "accelerator"),
            _row("unmapped cycles", *both(lambda s: s["unmapped_cycles"]), "accelerator"),
            _row("total cycles", *both(lambda s: s["total_cycles"]), "accelerator"),
            _row("streamers", *both(lambda s: s["streamers"]["text"]), "accelerator")]  # fmt: skip
    names = dict.fromkeys(g["name"] for s in (sa, sb) if s for g in s["groups"])
    for name in names:

        def group(s: dict[str, Any], field: str, name: str = name) -> Any:
            return next(g for g in s["groups"] if g["name"] == name)[field]

        under = f"group {name}"
        rows += [_row(label, *both(lambda s, f=field: group(s, f)), under)
                 for label, field in (("cycles", "formula"), ("expected BUSY_COUNTER", "counter"),
                                      ("channels", "channels"), ("area", "area_text"),
                                      ("energy", "energy_text"))]  # fmt: skip
        notes = both(lambda s: group(s, "bank_note"))
        if any(notes):  # L1's banks allow fewer cycles than the ideal on a side
            rows.append(_row("bank bound", *notes, under))
    rows += [_row("L1", *both(l1), "memory"),
             _row("reads", *both(lambda s: f"{s['reads']} words"), "memory"),
             _row("writes", *both(lambda s: f"{s['writes']} words"), "memory")]  # fmt: skip
    if any(s and s["splits"] for s in (sa, sb)):
        rows.append(_row("lanes", *both(lambda s: "; ".join(f"{x['map']} {x['text']}"
                                                            for x in s["splits"]) or None),
                         "memory"))  # fmt: skip
    costs = [s["cost"] for s in (sb, sa) if s and s["cost"]]
    if costs:
        under = costs[0]["heading"]

        def missing(s: dict[str, Any]) -> str:
            return ", ".join(s["cost"]["missing"]) or "none"

        def part(s: dict[str, Any], name: str) -> str:
            p = next(x for x in s["cost"]["parts"] if x["name"] == name)
            return f"{p['area_text']} · {p['energy_text']}"

        rows += [_row("area", *both(lambda s: s["cost"]["area_text"]), under),
                 _row("area without the banks", *both(lambda s: s["cost"]["area_design_text"]),
                      under),
                 _row("energy per run", *both(lambda s: s["cost"]["energy_text"]), under),
                 *[_row(PART_LABELS[k], *both(lambda s, k=k: part(s, k)), under) for k in PARTS],
                 _row("L1 accesses", *both(lambda s: s["cost"]["l1_text"]), under),
                 _row("not characterized", *both(missing), under)]  # fmt: skip
    return rows


def estimate_summary(est: Any, graph: Graph | None = None) -> dict[str, Any]:
    """The Estimate panel's figures, every sum and share worked out here: the headline,
    ``cycles`` (the accelerator's, the unmapped work's at one cycle per iteration with that
    assumption as its ``title``, and the total),
    per group ``cycles = bottleneck + ramp [+ stall]`` and its expected BUSY_COUNTER (a guide, not a
    check), the L1
    footprint against the L1 (its share as a CSS width), the words read and written, and the
    lane use of each tailed split; per group its ``bank_bound`` and, where it exceeds the
    cycles, its ``bank_note`` (R2 shows them); with a technology per group its area and
    energy and the accelerator's ``cost`` (module doc)."""
    groups = []
    for g in est.groups:
        terms = f"{g.bottleneck} + {g.ramp}" + (f" + {g.stall}" if g.stall else "")
        groups.append({"name": g.name, "cycles": g.cycles, "bottleneck": g.bottleneck,
                       "ramp": g.ramp, "stall": g.stall, "counter": g.counter,
                       "channels": g.channels, "formula": f"{g.cycles} = {terms}",
                       "bank_bound": g.bank_bound, "bank_note": g.bank_bound_note,
                       "area_text": _um2(g.cost.area) if g.cost else None,
                       "energy_text": _nj(g.cost.energy) if g.cost else None})  # fmt: skip
    share = est.l1_footprint / est.l1_size if est.l1_size else None
    l1 = {
        "footprint": est.l1_footprint,
        "footprint_text": size_text(est.l1_footprint),
        "size": est.l1_size,
        "size_text": size_text(est.l1_size) if est.l1_size else None,
        "share_text": f"{100 * share:.1f}%" if share is not None else None,
        "share_css": f"{min(100.0, 100 * share):.2f}%" if share is not None else None,
    }
    splits = [
        {"map": k, "n": x.n, "w": x.w, "lane_use": x.lane_use,
         "text": f"{x.n} of {-(-x.n // x.w) * x.w} lanes ({100 * x.lane_use:.1f}%)"}
        for k, x in est.splits.items()
    ]  # fmt: skip
    from snax_forge.estimate import UNMAPPED_CYCLES_PER_ITERATION as PER

    assumption = (f"{est.unmapped_iterations} iterations no unit runs, at {PER} cycle each: an "
                  "assumption (a core takes several)")  # fmt: skip
    return {
        "accelerator_cycles": est.accelerator_cycles,
        "not_mapped": est.not_mapped,
        "unmapped_cycles": est.unmapped_cycles,
        "total_cycles": est.total_cycles,
        "cycles": [
            {
                "label": "accelerator",
                "value": est.accelerator_cycles,
                "title": "the groups' cycles, adding through L1",
            },
            {"label": "unmapped", "value": est.unmapped_cycles, "title": assumption},
            {
                "label": "total",
                "value": est.total_cycles,
                "title": "the accelerator's cycles plus the unmapped ones",
            },
        ],
        "unmapped_note": assumption if est.unmapped_iterations else None,
        "groups": groups,
        "l1": l1,
        "reads": sum(c.reads for c in est.containers.values()),
        "writes": sum(c.writes for c in est.containers.values()),
        "splits": splits,
        "streamers": streamers_summary(est),
        "cost": cost_summary(est, graph),
    }


def streamers_summary(est: Any) -> dict[str, Any]:
    """The Estimate panel's ``streamers`` (module doc): one streamer set per accelerator."""
    n = est.streamers
    text = f"{n} streamer set{'' if n == 1 else 's'}"
    if est.csrs is not None:
        text += f" · {est.csrs} CSRs"
    rows = [{"name": a.name, "groups": list(a.groups),
             "csrs_text": None if a.csrs is None else f"{a.csrs[0]}–{a.csrs[1]}",
             "movers": a.streamer.movers, "channels": a.streamer.channels} for a in est.accelerators]  # fmt: skip
    return {"count": n, "csrs": est.csrs, "text": text, "accelerators": rows}


def cost_summary(est: Any, g: Graph | None = None) -> dict[str, Any] | None:
    """The Estimate panel's ``cost`` (module doc): None with no technology or no group; with
    ``tables``, the cost table per accelerator (``cost_tables``)."""
    if est.tech is None or not est.groups:
        return None
    l1 = "; ".join(f"{a}: {x.words} words, {_nj(x.energy_nj)}" for a, x in est.l1.items())
    return {"tech": est.tech, "heading": heading(est.tech), "caveat": caveat(),
            "area_text": _um2(est.area), "energy_text": _nj(est.energy),
            "area_design_text": _um2(est.area_design),
            "l1_text": l1 or None, "missing": list(est.missing),
            "not_costed": list(est.not_costed), "extrapolated": list(est.extrapolated),
            "parts": cost_parts(est), "tables": cost_tables(est, g)}  # fmt: skip


PART_LABELS = {"datapath": "datapath", "control": "control", "streamers": "streamers",
               "crossbar": "crossbar", "memory": "memory (banks)"}  # fmt: skip


def cost_parts(est: Any) -> list[dict[str, Any]]:
    """The cost by part, in ``PARTS``' order: each its ``name``, ``label``, ``area`` and
    ``energy`` (None where not costed), their texts and their ``share`` of the totals (0..1)
    with its text; what the page's two bars draw to scale."""
    parts = est.parts or {}
    share = shares(parts)
    out = []
    for k, p in parts.items():
        sa, se = share[k]
        out.append({"name": k, "label": PART_LABELS[k], "area": p.area_um2, "energy": p.energy_nj,
                    "area_text": "not costed" if p.area_um2 is None else _um2(p.area_um2),
                    "energy_text": "not costed" if p.energy_nj is None else _nj(p.energy_nj),
                    "area_share": sa, "energy_share": se,
                    "area_share_text": "–" if sa is None else f"{100 * sa:.1f}%",
                    "energy_share_text": "–" if se is None else f"{100 * se:.1f}%"})  # fmt: skip
    return out


def cost_tables(est: Any, g: Graph | None = None) -> list[dict[str, Any]]:
    """The cost as tables, one per accelerator of ``g`` (its groups, in order; without
    ``g`` one of every group, named ``accelerator``): a row per unit, buffer link and control
    unit, each its ``block`` (the instance, the link's container, ``<group> ctrl``), ``kind``,
    ``datapath`` (``· not characterized`` where the table has no point), area (µm², 3
    significant figures) and energy (nJ, 3 decimals) with their texts, and ``share`` of the
    accelerator's energy; then a row per port's streamer (kind ``streamer``, its FIFO depth, loops
    and channels in ``datapath``, marked when extrapolated), then one row of its words through
    the crossbar and the banks (``L1_LABEL``, kind ``l1``, its reads and writes in
    ``datapath``, no area: the crossbar's and banks' areas are the design's, in ``parts``);
    then the ``total``."""
    by_name = {x.name: x for x in est.groups if x.cost is not None}
    accs = [(a["name"], [grp["name"] for grp in a["groups"]]) for a in accelerators(g)] if g else []
    l1 = est.l1
    streamers: dict[str, list[Any]] = {}
    for x in est.streamer_costs:
        streamers.setdefault(x.counts.accelerator, []).append(x)
    if not accs:
        accs = [("accelerator", list(by_name))]
        streamers = {"accelerator": [x for v in streamers.values() for x in v]}
        l1 = {"accelerator": L1Cost(sum(x.words for x in l1.values()),
                                    sum(x.energy_nj for x in l1.values()),
                                    sum(x.reads for x in l1.values()),
                                    sum(x.writes for x in l1.values()),
                                    sum(x.xbar_nj for x in l1.values()),
                                    sum(x.banks_nj for x in l1.values()))} if l1 else {}  # fmt: skip
    out = []
    for name, groups in accs:
        raw = []
        for gname in groups:
            c = by_name.get(gname)
            if c is None:
                continue
            c = c.cost
            for kind, blocks in (("unit", c.units), ("link", c.links)):
                for k, b in blocks.items():
                    dp = b.datapath or "–"
                    raw.append((k, kind, dp if b.found else f"{dp} · not characterized",
                                b.found is not None, b.total_um2, b.task_nj))  # fmt: skip
            raw.append((f"{gname} ctrl", "ctrl", "control unit", True, c.ctrl_um2, c.ctrl_nj))
        if not raw:
            continue
        for x in streamers.get(name, []):  # each port's streamer
            c = x.counts
            loops = f"{c.loops} loop{'' if c.loops == 1 else 's'}"
            dp = f"streamer · FIFO {c.depth}, {loops}, {c.channels} ch"
            raw.append((c.name, "streamer", dp + (" · extrapolated" if x.extrapolated else ""),
                        True, x.area_um2, x.energy_nj))  # fmt: skip
        if name in l1:  # its words through the crossbar and the banks
            x = l1[name]
            rw = f"{_thousands(x.reads)} reads, {_thousands(x.writes)} writes"
            raw.append((L1_LABEL, "l1", rw, True, None, x.energy_nj))
        area, energy = sum(r[4] or 0.0 for r in raw), sum(r[5] for r in raw)
        out.append({"accelerator": name, "rows": [_cost_row(*r, energy) for r in raw],
                    "total": _cost_row("total", "total", "", True, area, energy, energy)})  # fmt: skip
    return out


def _cost_row(block: str, kind: str, dp: str, found: bool, a: float | None, e: float,
              whole: float) -> dict[str, Any]:  # fmt: skip
    """A cost table's row, its share of ``whole``, the accelerator's energy; ``a`` None for a
    row with no area (the L1 accesses)."""
    share = e / whole if whole else None
    return {"block": block, "kind": kind, "datapath": dp, "found": found, "area": a, "energy": e,
            "share": share, "area_text": "–" if a is None else sig3(a), "energy_text": dec3(e),
            "share_text": "–" if share is None else f"{100 * share:.1f}%"}  # fmt: skip


COST_COLUMNS = (("block", "block"), ("datapath", "datapath"), ("area_text", "area (µm²)"),
                ("energy_text", "energy (nJ)"), ("share_text", "share"))  # fmt: skip


def compare_cost_tables(
    sa: dict[str, Any] | None, sb: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Compare's cost tables: every accelerator and block of either side, each cell ``before →
    after`` where the two differ, as it is where they do not, "–" on a side without it; each
    row's ``differs``."""

    def tables(s: dict[str, Any] | None) -> dict[str, Any]:
        cost = (s or {}).get("cost") or {}
        return {t["accelerator"]: t for t in cost.get("tables") or []}

    ta, tb = tables(sa), tables(sb)
    out = []
    for name in dict.fromkeys([*ta, *tb]):
        a, b = ta.get(name, {}), tb.get(name, {})
        ra = {r["block"]: r for r in a.get("rows", [])}
        rb = {r["block"]: r for r in b.get("rows", [])}

        def cells(x: dict[str, Any] | None, y: dict[str, Any] | None) -> dict[str, Any]:
            row: dict[str, Any] = {"differs": False}
            for key, _ in COST_COLUMNS:
                u = "–" if x is None else (x.get(key) or "–")
                w = "–" if y is None else (y.get(key) or "–")
                row[key] = u if u == w else f"{u} → {w}"
                row["differs"] |= u != w
            return row

        out.append({"accelerator": name,
                    "rows": [cells(ra.get(k), rb.get(k)) for k in dict.fromkeys([*ra, *rb])],
                    "total": cells(a.get("total"), b.get("total"))})  # fmt: skip
    return out


def _thousands(n: int) -> str:
    """An exact count, thousands split by a space: 393 216."""
    return f"{n:,}".replace(",", " ")


def _um2(x: float) -> str:
    return f"{sig3(x)} µm²"


def _nj(x: float) -> str:
    return f"{dec3(x)} nJ"


def cost_details(c: Any, bits: int, tech: str) -> dict[str, str]:
    """A unit's or buffer link's Details ``cost`` (module doc), from its ``BlockCost``."""
    return {
        "found": f"{c.found} on {tech}" if c.found else f"not characterized on {tech}",
        "area": f"{_um2(c.total_um2)} (comb {sig3(c.comb_um2)}, flops {sig3(c.flops_um2)})",
        "per fire": f"{dec3(c.e_fire_pj)} pJ",
        "holding": f"{dec3(c.e_hold_pj)} pJ per cycle" if c.e_hold_pj else "–",  # no flops
        "per task": f"{_nj(c.task_nj)} ({c.fires} fires, {bits} bits)",
    }


def group_cost_details(c: Any) -> dict[str, str]:
    """A group's Details ``cost``: its area and energy and its control unit's."""
    return {"area": _um2(c.area), "energy": _nj(c.energy),
            "ctrl": (f"{c.ctrl_bits} bits, {_um2(c.ctrl_um2)} (flops {sig3(c.ctrl_flops_um2)}, "
                     f"comb {sig3(c.ctrl_comb_um2)}), {_nj(c.ctrl_nj)}")}  # fmt: skip


def _estimated(g: Graph, platform: Any, tech: str | None = DEFAULT_TECH) -> tuple[Any, str | None]:
    """The graph's estimate on ``platform``, costed on ``tech``, or None and why not."""
    from snax_forge.estimate import estimate

    try:
        return estimate(g, platform, tech), None
    except (ValueError, KeyError, TypeError) as e:  # DfgError, LowerError, BrmError, ...
        return None, str(e)


# A fold's op as the compact text writes it
FOLD = {"add": "+", "mul": "*", "min": "min", "max": "max"}


def _rhs(code: str) -> str:
    """The right-hand sides of ``code``'s assignments, ``; `` between them."""
    return "; ".join(line.split("=", 1)[1].strip() for line in code.split("\n") if "=" in line)


def _operation(n: Node, code: str, ins: Sequence[str]) -> str:
    """What a tasklet or a unit computes: ``fold <op>`` for a copy of one input into a fold,
    else its code's right-hand side."""
    from snax_forge.dfg.pad import fold_op

    op = fold_op(n)
    if op is not None and _rhs(code) in ins:
        return f"fold {FOLD.get(op, op)}"
    if n.kind == "tasklet":  # a scalar read as CSR by name: csr(alpha) (module doc)
        csr = {r["name"] for r in n.attrs.get(BAKED, []) if r["value"] == "csr"} & set(ins)
        if csr:
            return re.sub(r"\b(" + "|".join(map(re.escape, sorted(csr))) + r")\b", r"csr(\1)",
                          _rhs(code))  # fmt: skip
    return _rhs(code)


def compact_text(g: Graph, n: Node, param: str | None = None) -> str:
    """A node's text in the compact drawing (module doc); a unit's lanes under ``param``, the
    recipe param its split names (``W_acc_1 8``), else ``W``."""
    a = n.attrs
    if n.kind == "map":
        its = _iterations(g, a["range"])
        return (
            f"{n.id} · {a['var']} · {'?' if its is None else its} · "
            f"{a.get('loop.kind') or 'untagged'}"
        )
    if n.kind == "tasklet":
        return f"{n.id} · {_operation(n, a['code'], list(n.inputs))}"
    if n.kind == "accelerated":
        text = f"{a['instance']} · {_operation(n, a['code'], list(n.inputs))}"
        try:
            inst = instance_of(n)
            first = next(p.name for p in inst.ports if p.direction == "in")  # a port, not a slot
            slots = "".join(f" · {k} {v}" for k, v in slot_params(n).items())
            return f"{text} · {param or 'W'} {inst.lanes(first)}{slots}"
        except (BrmError, ExprError, KeyError, StopIteration):
            return text
    if n.kind == "link":
        depth = a["depth"]
        return f"{a['container']} · {'wire' if depth == 0 else f'FIFO {depth}'}"
    return n.id


def badge(placed: Any) -> str | None:
    """A container's badge from its estimate (module doc)."""
    if placed is None:
        return None
    if placed.placement == "l1":
        return f"L1 · {size_text(placed.bytes)}"
    if placed.placement == "link":
        return "wire" if placed.depth == 0 else f"FIFO {placed.depth}"
    return placed.placement


def size_text(n: int) -> str:
    """Bytes as ``8 B``, ``4 KiB``, ``8200 B`` (under 10 KiB and not whole), ``12.5 KiB``."""
    for unit, size in (("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= size:
            if n % size == 0:
                return f"{n // size} {unit}"
            if n >= 10 * size or unit == "MiB":
                return f"{n / size:.1f} {unit}"
    return f"{n} B"


def accelerators(g: Graph) -> list[dict[str, Any]]:
    """Accelerator ⊃ groups ⊃ units (module doc)."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    where: dict[str, dict[str, Any]] = {}  # instance -> its group's entry
    found_links: list[tuple[Node, str]] = []
    for top in g.body:
        stack = [top]
        while stack:
            n = stack.pop(0)
            if n.kind == "accelerated":
                groups = out.setdefault(accelerator_of(n), {})
                inst, name = n.attrs["instance"], group_of(n)
                grp = groups.setdefault(name, {"name": name, "units": {}, "links": [], "tops": []})
                grp["units"].setdefault(inst, []).append(n.id)
                where[inst] = grp
                if top.id not in grp["tops"]:
                    grp["tops"].append(top.id)
            elif n.kind == "link":
                found_links.append((n, top.id))
            stack += list(n.body or [])
    order = [t.id for t in g.body]
    for n, top in found_links:
        grp = where.get(n.attrs["from"])
        if grp is not None:
            grp["links"].append(n.id)
            if top not in grp["tops"]:
                grp["tops"] = sorted([*grp["tops"], top], key=order.index)
    return [{"name": acc, "groups": [
        {"name": grp["name"], "tops": grp["tops"], "links": grp["links"],
         "units": [{"instance": u, "nodes": ids} for u, ids in grp["units"].items()]}
        for grp in groups.values()]} for acc, groups in out.items()]  # fmt: skip


def details(g: Graph, view: dict[str, Any], est: Any) -> dict[str, dict[str, Any]]:
    """Every item's Details, keyed by its id (module doc)."""
    out: dict[str, dict[str, Any]] = {}
    users: dict[str, dict[str, list[str]]] = {
        c: {"writers": [], "readers": []} for c in g.containers
    }
    nodes: dict[str, Node] = {}
    for n, _ in g.walk():
        nodes[n.id] = n
        seed = seed_memlet(n)
        for m in [
            *n.inputs.values(),
            *([seed] if seed is not None and n.kind == "tasklet" else []),
        ]:
            users[m.data]["readers"].append(n.id)
        for m in n.outputs.values():
            users[m.data]["writers"].append(n.id)
        out[f"node:{n.id}"] = _node_details(n)
    for c, cont in g.containers.items():
        placed = est.containers.get(c) if est is not None else None
        if "loop.local" in cont.attrs:  # as the Containers line has it
            placed = _Local(cont.attrs["loop.local"])
        try:
            sizes = [expr.evaluate(x, g.bound_symbols) for x in cont.shape]
        except ExprError:
            sizes = None
        out[f"container:{c}"] = {"kind": "container", "shape": list(cont.shape), "sizes": sizes,
                                 "dtype": cont.dtype,
                  "transient": cont.transient, "attrs": dict(cont.attrs),
                  "bytes": placed.bytes if placed else None,
                  "placement": placed.placement if placed else None, **users[c]}  # fmt: skip
    groups = {x.name: x for x in est.groups} if est is not None else {}
    for acc in view["accelerators"]:
        out[f"accelerator:{acc['name']}"] = {"kind": "accelerator",
                                             "groups": [x["name"] for x in acc["groups"]]}  # fmt: skip
        for grp in acc["groups"]:
            e = groups.get(grp["name"])
            shown = {k: v for k, v in e.to_dict().items() if k != "cost"} if e is not None else None
            out[f"group:{grp['name']}"] = {
                "kind": "group", "accelerator": acc["name"],
                "units": [u["instance"] for u in grp["units"]], "links": list(grp["links"]),
                "estimate": shown,
                "cost": group_cost_details(e.cost) if e is not None and e.cost else None,
            }  # fmt: skip
    for x in groups.values():  # a unit's and a buffer link's cost, after its timing
        if x.cost is None:
            continue
        for n in nodes.values():
            d = out[f"node:{n.id}"]
            if n.kind == "accelerated" and n.attrs["instance"] in x.cost.units:
                d["cost"] = cost_details(x.cost.units[n.attrs["instance"]],
                                         x.bits.get(n.attrs["instance"], 0), est.tech)  # fmt: skip
            elif n.kind == "link" and n.attrs["container"] in x.cost.links:
                c = n.attrs["container"]
                d["cost"] = cost_details(x.cost.links[c], x.bits.get(c, 0), est.tech)
    ports = est.ports if est is not None else {}
    loops_of: dict[str, Any] = {}
    for edge in view["edges"]:
        d: dict[str, Any] = {"kind": "edge", "data": edge["data"], "memlet": edge["text"],
                             "packing": edge.get("packing")}  # fmt: skip
        for end in (edge["from"], edge["to"]):
            node, _, port = end.partition(".")
            n = nodes.get(node)
            if n is not None and n.kind == "accelerated":
                m = (n.inputs | n.outputs)[port]
                d["subset"] = list(m.subset)
                d["port"] = f"{n.attrs['instance']}.{port}"
                d["streamer"] = d["port"] in ports  # an outer port; a linked one has none
                try:
                    inst = instance_of(n)
                    if n.id not in loops_of:
                        loops_of[n.id] = firing_loops(g, n)
                    mine = port_loops(inst, port, m, loops_of[n.id])
                except (BrmError, ExprError, KeyError, ValueError):
                    continue
                d["loops"] = [[lp.var, lp.count] for lp in mine]
                d["beats"] = prod(lp.count for lp in mine)
            elif n is not None and port in (n.inputs | n.outputs):
                d["subset"] = list((n.inputs | n.outputs)[port].subset)
        out[f"edge:{edge['id']}"] = d
    return out


def block_details(g: Graph, b: dict[str, Any]) -> dict[str, Any]:
    """A block's Details: its kernel, recipe and params, where it sits, the units in it."""
    rec = block_records(g)[b["name"]]
    inner = [x["name"] for x in block_records(g).values() if x["parent"] == b["name"]]
    return {"kind": "block", "kernel": rec["kernel"], "recipe": rec.get("recipe"),
            "params": rec.get("params") or {}, "heads": list(rec["heads"]),
            "in": rec["parent"], "blocks": inner, "units": rec["instances"],
            "expands with": b["together"],
            "drawn": "one box" if b["collapsed"] else "its nodes"}  # fmt: skip


def _node_details(n: Node) -> dict[str, Any]:
    a = n.attrs
    if n.kind == "map":
        keys = ("loop.kind", "loop.split", "loop.fused", "loop.split_reduce")
        return {
            "kind": "map",
            "var": a["var"],
            "range": a["range"],
            **{k: a[k] for k in keys if k in a},
        }
    io = {"inputs": {c: memlet_text(m) for c, m in n.inputs.items()},
          "outputs": {c: memlet_text(m) for c, m in n.outputs.items()}}  # fmt: skip
    seed = seed_memlet(n)
    if seed is not None:  # the seed (its edge's label)
        io["seed"] = memlet_text(seed)
    if n.kind == "tasklet":
        return {"kind": "tasklet", "code": a["code"], **io}
    if n.kind == "accelerated":
        d = {"kind": "unit", "instance": a["instance"], "brm": a["brm"],
             "implementation": a["implementation"], "params": dict(a["params"]),
             "code": a["code"], "replaced": _replaced_text(a.get("replaced")),
             "group": group_of(n), "accelerator": accelerator_of(n), **io}  # fmt: skip
        try:
            inst = instance_of(n)
            d["timing"] = {"latency": inst.latency, "ii": inst.initiation_interval,
                           "span": inst.span}  # fmt: skip
        except (BrmError, ExprError, KeyError):
            d["timing"] = None
        return d
    if n.kind == "link":
        return {"kind": "link", "container": a["container"], "depth": a["depth"],
                "from": a["from"], "to": a["to"]}  # fmt: skip
    return {"kind": n.kind, "attrs": dict(a), **io}


def container_text(g: Graph, name: str) -> str:
    """A container's text in its box: its name and shape, as the graph spells it over the
    kernel's symbols (``A[M, N]``, ``y[M]``, a convolution's ``X[B, C, H, W]``); a kernel's
    scalar argument by its name alone (``alpha``)."""
    c = g.containers[name]
    if c.attrs.get(SCALAR):
        return name
    return f"{name}[{', '.join(str(d) for d in c.shape)}]"


def _container(g: Graph, name: str, placed: Any = None) -> dict[str, Any]:
    """A container as the drawing shows it: its name and shape, its badge, whether it is
    transient, and on a ``share_ports`` container its marker ``share_ports · <name>``."""
    c = g.containers[name]
    shared = {"share": f"share_ports · {name}"} if SHARE_PORTS in c.attrs else {}
    return {"name": name, "text": container_text(g, name), "badge": badge(placed),
            "transient": c.transient, "scalar": bool(c.attrs.get(SCALAR))} | shared  # fmt: skip


def memlet_text(m: Memlet) -> str:
    text = f"{m.data}[{', '.join(str(d) for d in m.subset)}]"
    if m.reduce is not None and m.reduce.seed is not None:
        sm = m.reduce.seed
        return f"{text} (reduce {m.reduce.op}, seed {sm.data}[{', '.join(map(str, sm.subset))}])"
    return text if m.reduce is None else f"{text} (reduce {m.reduce.op})"


def seed_memlet(n: Node) -> Memlet | None:
    """The element a node's fold starts from (module doc, "The seed"): a seeded tasklet's
    reduce's seed, or a bound unit's ``seed`` port; None for any other node."""
    if n.kind == "accelerated":
        return n.inputs.get(SEED)
    if n.kind != "tasklet":
        return None
    for m in n.outputs.values():
        if m.reduce is not None and m.reduce.seed is not None:
            return Memlet(m.reduce.seed.data, list(m.reduce.seed.subset))
    return None


def _iterations(g: Graph, rng: str) -> int | None:
    try:
        return len(map_range(rng, g.bound_symbols, "range"))
    except (ExprError, DfgError):
        return None


def csr_connectors(g: Graph, n: Node) -> set[str]:
    """The input connectors ``n`` reads a kernel's scalar argument through at run time, from a
    CSR (module doc): a tasklet's that ``bake_scalar`` named after its container (value
    ``csr``), a unit's slots (inputs that are no BRM port)."""
    if n.kind == "tasklet":
        return {r["name"] for r in n.attrs.get(BAKED, [])
                if r["value"] == "csr" and r["name"] in n.inputs}  # fmt: skip
    if n.kind == "accelerated":
        ports = _port_names(n)
        return {c for c, m in n.inputs.items() if c not in ports
                and g.containers.get(m.data) is not None
                and g.containers[m.data].attrs.get(SCALAR)}  # fmt: skip
    return set()


def _node(g: Graph, n: Node) -> dict[str, Any]:
    """A node as the compact drawing shows it: its text, a map's tag, its connectors (where
    the edges meet it) and its body; the rest is in Details."""
    if n.kind == "block":  # a collapsed block (fold)
        return {"id": n.id, "kind": "block", "text": block_text(n.attrs["block"]),
                "inputs": [_conn(c, m) for c, m in n.inputs.items()],
                "outputs": [_conn(c, m) for c, m in n.outputs.items()]}  # fmt: skip
    csr = csr_connectors(g, n)
    d: dict[str, Any] = {
        "id": n.id,
        "kind": n.kind,
        "text": compact_text(g, n),
        "inputs": [_conn(c, m) | ({"csr": True} if c in csr else {})
                   | ({"seed": True} if c == SEED and n.kind == "accelerated" else {})
                   for c, m in n.inputs.items()],
        "outputs": [_conn(c, m) for c, m in n.outputs.items()],
    }  # fmt: skip
    seed = seed_memlet(n) if n.kind == "tasklet" else None
    if seed is not None:  # a seeded fold's seed, a connector the drawing gives it
        d["inputs"].append({"connector": SEED, "data": seed.data, "seed": True})
    if n.kind == "map":
        d["loop_kind"] = n.attrs.get("loop.kind")
        merged = n.attrs.get("view.merged") or []
        if len(merged) > 1:  # a block's firing loop, one per nest it spans (fold)
            d["text"] = d["text"].replace(n.id, " + ".join(merged), 1)
    if folds(n):  # drawn as an upside-down trapezoid (module doc, "Folds")
        d["fold"] = True
    if n.kind == "link":  # its container's edges meet it here (module doc)
        c = n.attrs["container"]
        d["inputs"], d["outputs"] = (
            [{"connector": "in", "data": c}],
            [{"connector": "out", "data": c}],
        )
    if n.body is not None:
        d["body"] = [_node(g, c) for c in n.body]
    return d


def shared_units(g: Graph) -> dict[str, list[str]]:
    """Each accelerated node whose instance other nodes share (mvt's variant 1: one ``vmul_x1``
    for both halves) -> those other nodes' ids, in the graph's order; the page highlights
    them with a selected one. Nodes of an instance alone are left out."""
    by: dict[str, list[str]] = {}
    for n, _ in g.walk():
        if n.kind == "accelerated":
            by.setdefault(n.attrs["instance"], []).append(n.id)
    return {i: [j for j in ids if j != i] for ids in by.values() if len(ids) > 1 for i in ids}


# The transforms a map's partners are found by, in the order a map's list gives them
PARTNER_TRANSFORMS = ("fuse_map", "fuse_seed", "interchange_map")


def partners(g: Graph, boxes: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Each map's partners (module doc, "Partners"): map id -> ``{"map", "transform"}`` per map
    it fuses, seeds or interchanges with, by the transforms' own sites, and for a fuse the
    ``marker`` on the transient between the two nests, ``{"box", "container", "text"}``."""
    from snax_forge.sandbox.compose import fuse_link, fuse_near, fuse_pairs, link_text
    from snax_forge.sandbox.recipe import SandboxError
    from snax_forge.sandbox.seed import seed_link, seed_pairs
    from snax_forge.sandbox.transforms import partners as swaps

    out: dict[str, list[dict[str, Any]]] = {}
    found = {"fuse_map": (fuse_pairs, fuse_link), "fuse_seed": (seed_pairs, seed_link)}

    def add(a: str, b: str, transform: str, marker: dict[str, Any] | None = None) -> None:
        out.setdefault(a, []).append({"map": b, "transform": transform}
                                     | ({"marker": marker} if marker else {}))  # fmt: skip

    for transform in PARTNER_TRANSFORMS:
        try:
            if transform == "interchange_map":
                maps = [n.id for n, _ in g.walk() if n.kind == "map"]
                for m in maps:
                    for other, why in swaps(g, m):
                        if why is None:
                            add(m, other, transform)
                continue
            pairs, link = found[transform]
            for a, b in pairs(g):
                ln = link(g, a, b)
                box = next((k for k, x in boxes.items() if x["container"] == ln["container"]
                            and ln["writer"] in x["written_by"] and not x["again"]), None)  # fmt: skip
                marker = {"box": box, "container": ln["container"], "text": link_text(ln)}
                add(a, b, transform, marker)
                add(b, a, transform, marker)
        except (SandboxError, DfgError, ExprError, KeyError):
            continue  # a graph the transform cannot judge: no partners by it
    try:  # near partners: refused for different ranges only, its refusal naming the split
        for a, b, why in fuse_near(g):
            for x, y in ((a, b), (b, a)):
                out.setdefault(x, []).append({"map": y, "transform": "fuse_map", "near": why})
    except (SandboxError, DfgError, ExprError, KeyError):
        pass
    return out


def folds(n: Node) -> bool:
    """Whether ``n``'s output folds (module doc, "Folds"): a tasklet's output memlet has a
    ``reduce`` (an identity or a seed), a bound unit's replaced nest holds such a tasklet."""
    if n.kind == "tasklet":
        return any(m.reduce is not None for m in n.outputs.values())
    if n.kind == "accelerated" and n.attrs.get("replaced"):
        return _folds_in(n.attrs["replaced"])
    return False


def _folds_in(d: dict[str, Any]) -> bool:
    """A stored subtree (``replaced``) holds a tasklet whose output folds."""
    if d.get("kind") == "tasklet" and any("reduce" in m for m in (d.get("outputs") or {}).values()):
        return True
    return any(_folds_in(c) for c in d.get("body") or [])


def _replaced_text(r: dict[str, Any] | None) -> str | None:
    """What an accelerated node replaced, as its chain of ids (``add_map_s › add``)."""
    if not r:
        return None
    ids = []
    while r:
        ids.append(r["id"])
        r = r["body"][0] if r.get("body") and len(r["body"]) == 1 else None
    return " › ".join(ids)


def _conn(c: str, m: Memlet) -> dict[str, Any]:
    return {"connector": c, "data": m.data}


def _walk(n: Node) -> list[Node]:
    """``n`` and every node under it."""
    return [n, *(x for child in n.body or [] for x in _walk(child))]


def _uses(n: Node) -> list[tuple[str, Node, str, Memlet]]:
    """Every memlet under ``n`` in execution order: (``read`` / ``write``, node, connector, memlet)."""
    out: list[tuple[str, Node, str, Memlet]] = []
    out += [("read", n, c, m) for c, m in n.inputs.items()]
    seed = seed_memlet(n) if n.kind == "tasklet" else None
    if seed is not None:  # a seeded fold reads its seed (no connector of its own)
        out.append(("read", n, SEED, seed))
    for child in n.body or []:
        out += _uses(child)
    out += [("write", n, c, m) for c, m in n.outputs.items()]
    return out


def packing(g: Graph, n: Node, port: str, data: str) -> str | None:
    """``int32 · num_slc 2 · num_ch 2`` for an accelerated node's port (module doc); None for any
    other
    node, or an instance the BRM library does not resolve."""
    if n.kind != "accelerated":
        return None
    try:
        lanes = instance_of(n).lanes(port)
    except (BrmError, ExprError, KeyError):
        return None
    dtype = g.containers[data].dtype
    num_slc = max(1, WORD_BITS // bits(dtype))
    return f"{dtype} · num_slc {num_slc} · num_ch {-(-lanes // num_slc)}"


def _touches(top: Node) -> tuple[set[str], set[str]]:
    """The containers a top-level node reads and writes, every memlet under it counted; a
    link node its container, both ways."""
    if top.kind == "link":
        return {top.attrs["container"]}, {top.attrs["container"]}
    r: set[str] = set()
    w: set[str] = set()
    for side, _, _, m in _uses(top):
        (r if side == "read" else w).add(m.data)
    return r, w


def columns(tops: Sequence[Node]) -> tuple[list[int], list[dict[str, Any]]]:
    """The top-level nodes' independent chains, each a column: each node's column and the
    columns, ``{"id", "tops", "joins"}`` in the order made, ``tops`` indices into ``tops``.
    Two nodes are in one chain when one writes a container the other reads or writes; a node
    linked so to two columns starts a new one, ``joins`` those columns, and is its first top."""
    touch = [_touches(t) for t in tops]
    cols: list[dict[str, Any]] = []
    into: dict[int, int] = {}  # a joined column -> the column that joined it
    of: list[int] = []
    for k, (r, w) in enumerate(touch):
        deps: list[int] = []
        for j in range(k):
            rj, wj = touch[j]
            if w & (rj | wj) or r & wj:
                c = of[j]
                while c in into:
                    c = into[c]
                if c not in deps:
                    deps.append(c)
        if len(deps) == 1:
            c = deps[0]
        else:
            c = len(cols)
            cols.append({"id": c, "tops": [], "joins": sorted(deps)})
            into.update(dict.fromkeys(deps, c))
        cols[c]["tops"].append(k)
        of.append(c)
    return of, cols


def drawn_once(g: Graph) -> set[str]:
    """The containers drawn once, above the columns that read them, when the graph has more
    than one column (the inputs no node writes among them): those ``mark_transpose`` marked in
    any column (a tasklet's or a bound unit's ``sandbox.transpose``); every other input is
    drawn in each column that reads it."""
    return {n.inputs[c].data for n, _ in g.walk() for c in n.attrs.get(TRANSPOSE, {})
            if c in n.inputs}  # fmt: skip


def transpose_label(data: str, perm: Sequence[int]) -> str:
    """A marked edge's label (sandbox/transpose.py): ``Aᵀ`` for a matrix, else
    ``A · transpose(2, 0, 1)``."""
    if len(perm) == 2:
        return f"{data}ᵀ"
    return f"{data} · transpose({', '.join(map(str, perm))})"


# -- blocks (sandbox/blocks.py): a collapsed block drawn as one box --------------------

LEAVES = ("tasklet", "accelerated", "link")


def block_records(g: Graph) -> dict[str, dict[str, Any]]:
    """Every block of ``g`` by name, in the graph's order: its record (sandbox/blocks.py) with
    ``parent`` (the block holding it, or None), ``depth`` and ``instances`` (the units bound
    in it)."""
    out: dict[str, dict[str, Any]] = {}
    for n, _ in g.walk():
        path = n.attrs.get(BLOCK) or []
        for i, b in enumerate(path):
            rec = out.setdefault(b["name"], {**b, "parent": path[i - 1]["name"] if i else None,
                                             "depth": i, "instances": []})  # fmt: skip
            if n.kind == "accelerated" and n.attrs["instance"] not in rec["instances"]:
                rec["instances"].append(n.attrs["instance"])
    return out


def _held(top: Node) -> list[str]:
    """The blocks holding every leaf under ``top`` (tasklets, units, links), outer first."""
    paths = [[b["name"] for b in n.attrs.get(BLOCK) or []] for n in _walk(top) if n.kind in LEAVES]
    if not paths:
        return []
    out = []
    for names in zip(*paths, strict=False):
        if len(set(names)) != 1:
            break
        out.append(names[0])
    return out


def block_text(rec: dict[str, Any]) -> str:
    """A block's text: ``dot · dot/dot_3 · W = 4``, a lifted one ``dot · dot · no recipe``, one
    named by hand ``half1 · by hand``."""
    if rec.get("kernel") is None:
        return f"{rec['name']} · by hand"
    if "recipe" not in rec:
        return f"{rec['name']} · {rec['kernel']} · no recipe"
    params = "".join(f" · {k} = {v}" for k, v in (rec.get("params") or {}).items())
    return f"{rec['name']} · {rec['recipe']}{params}"


def fold(g: Graph, expanded: Sequence[str] = ()) -> tuple[list[Node], list[dict[str, Any]]]:
    """The top-level nodes as drawn, a block not ``expanded`` as one node (module doc,
    "Blocks"), and the blocks' list."""
    recs = block_records(g)
    tops = list(g.body)
    held = [_held(t) for t in tops]
    touching: dict[str, list[int]] = {}
    for k, t in enumerate(tops):
        for n in _walk(t):
            for b in n.attrs.get(BLOCK) or []:
                if k not in touching.setdefault(b["name"], []):
                    touching[b["name"]].append(k)
    # a block whose top-level nodes it holds only in part stays drawn as its nodes
    whole = {b for b, ks in touching.items() if all(b in held[k] for k in ks)}
    open_ = set(expanded)
    into: list[str | None] = []
    for k in range(len(tops)):
        into.append(next((b for b in held[k] if b in whole and b not in open_), None))
    drawn: list[Node] = []
    made: dict[str, Node] = {}
    for k, t in enumerate(tops):
        b = into[k]
        if b is None:
            drawn.append(t)
        elif b not in made:
            made[b] = _block_node(g, recs[b], [tops[j] for j in range(len(tops)) if into[j] == b])
            drawn.append(made[b])
    hidden = {b for b in recs if any(a in made for a in _ancestors(recs, b))}
    shown_tops: dict[str, list[str]] = {}
    for k, t in enumerate(tops):
        for b in touching_of(t):
            shown_tops.setdefault(b, [])
            ident = made[into[k]].id if into[k] is not None else t.id
            if ident not in shown_tops[b]:
                shown_tops[b].append(ident)
    together = _together(recs)
    out = []
    for name, rec in recs.items():
        out.append({"name": name, "kernel": rec["kernel"], "recipe": rec.get("recipe"),
                    "params": rec.get("params") or {}, "text": block_text(rec),
                    "parent": rec["parent"], "depth": rec["depth"],
                    "collapsed": name in made, "hidden": name in hidden,
                    "foldable": name in whole, "tops": shown_tops.get(name, []),
                    "together": together.get(name, [])})  # fmt: skip
    return drawn, out


def _drawn_units(tops: list[Node]) -> set[str]:
    """The instances of the units drawn (not inside a collapsed block)."""
    return {n.attrs["instance"] for t in tops for n in _walk(t) if n.kind == "accelerated"}


def touching_of(top: Node) -> list[str]:
    """The blocks some node under ``top`` is in, in the order met."""
    out: list[str] = []
    for n in _walk(top):
        for b in n.attrs.get(BLOCK) or []:
            if b["name"] not in out:
                out.append(b["name"])
    return out


def _ancestors(recs: dict[str, dict[str, Any]], name: str) -> list[str]:
    out = []
    p = recs[name]["parent"]
    while p is not None:
        out.append(p)
        p = recs[p]["parent"]
    return out


def _together(recs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Each block -> the blocks of its depth whose units it shares (mvt's halves on one
    vmul): they expand and collapse together."""
    out: dict[str, list[str]] = {}
    for a, ra in recs.items():
        for b, rb in recs.items():
            if (
                a != b
                and ra["depth"] == rb["depth"]
                and set(ra["instances"]) & set(rb["instances"])
            ):
                out.setdefault(a, []).append(b)
    return out


def _outer_chains(rec: dict[str, Any], tops: list[Node]) -> list[list[Node]] | None:
    """Each top-level nest's maps above the block (its firing loops, sandbox/blocks.py's R2),
    each holding one node; None when one holds more, or when the nests' loops differ level by
    level (range or tag), so they cannot be drawn as one."""
    chains = []
    for t in tops:
        if t.kind == "link":
            continue
        chain, n = [], t
        while n.kind == "map" and rec["name"] not in [b["name"] for b in n.attrs.get(BLOCK, [])]:
            if len(n.body or []) != 1:
                return None
            chain.append(n)
            n = n.body[0]
        chains.append(chain)
    if not chains or any(len(c) != len(chains[0]) for c in chains):
        return None
    for level in zip(*chains, strict=True):
        if len({(m.attrs["range"], m.attrs.get("loop.kind")) for m in level}) != 1:
            return None
    return chains


def _projected(g: Graph, m: Memlet, outer: set[str]) -> Memlet:
    """``m`` as the block takes it on one firing: a dimension indexed by the outer loops'
    variables (and symbols) only kept, any other the container's whole dimension (``A[m_t,
    4 * n_1_t + n_1_s]`` -> ``A[m_t, 0:N]``)."""
    shape = g.containers[m.data].shape
    known = outer | set(g.symbols)
    dims = []
    for i, d in enumerate(m.subset):
        if not dim_names(d) <= known:
            dims.append(f"0:{shape[i]}")
            continue
        parts = parse_dim(d, "subset")
        one = len(parts) >= 2 and expr.canonical(f"({parts[0]}) + 1") == parts[1]
        dims.append(parts[0] if one and (len(parts) == 2 or parts[2] == 1) else d)  # m:m + 1 -> m
    return Memlet(m.data, dims, m.reduce)


def _block_node(g: Graph, rec: dict[str, Any], tops: list[Node]) -> Node:
    """A collapsed block as one node (module doc, "Blocks"): an input per container read from
    outside it (``in:A``), an output per container written to outside it (``out:x``), each
    memlet as one firing takes it (``_projected``); its transients (written and read only
    inside it) left out; a read ``mark_transpose`` marked keeps its mark. Its firing loops,
    the maps above it in each nest when they are alike (``_outer_chains``), stay drawn around
    it, one map per level (``view.merged`` the maps it stands for)."""
    inside = {n.id for t in tops for n in _walk(t)}
    users: dict[str, set[str]] = {}
    for n, _ in g.walk():
        for m in n.memlets():
            users.setdefault(m.data, set()).add(n.id)
    internal = {c for c, ids in users.items() if ids <= inside and g.containers[c].transient}
    chains = _outer_chains(rec, tops)
    outer = {m.attrs["var"] for c in chains or [] for m in c}
    ins: dict[str, Memlet] = {}
    outs: dict[str, Memlet] = {}
    marks: dict[str, Any] = {}
    for t in tops:
        for side, n, c, m in _uses(t):
            if m.data in internal or (side == "read" and c in csr_connectors(g, n)):
                continue
            if side == "read" and m.data not in ins:
                ins[m.data] = _projected(g, m, outer)
                perm = n.attrs.get(TRANSPOSE, {}).get(c)
                if perm is not None:
                    marks[f"in:{m.data}"] = perm
            elif side == "write" and m.data not in outs:
                outs[m.data] = _projected(g, m, outer)
    attrs: dict[str, Any] = {"block": dict(rec)}
    if marks:
        attrs[TRANSPOSE] = marks
    node = Node(rec["name"], "block", {f"in:{c}": m for c, m in ins.items()},
                {f"out:{c}": m for c, m in outs.items()}, attrs)  # fmt: skip
    if not chains:
        return node
    for level in reversed(list(zip(*chains, strict=True))):  # innermost first
        first = level[0]
        node = Node(first.id, "map", {}, {}, {**first.attrs, "view.merged": [m.id for m in level]},
                    [node])  # fmt: skip
    return node


def _layout(g: Graph, tops: list[Node] | None = None) -> dict[str, Any]:
    """Rows, nodes and edges (module doc), the rows in columns, one per independent chain
    (``columns``); ``tops`` the top-level nodes as drawn (``fold``), else the graph's."""
    tops = list(g.body) if tops is None else tops
    # A link's container gets no box: its writer's edge goes into the link, the link's out of it
    linked = {n.attrs["container"]: n.id for n, _ in g.walk() if n.kind == "link"}
    # nor does a fused nest's local value: its writer's connector joins its reader's
    local = {c for c, cont in g.containers.items() if "loop.local" in cont.attrs}
    # a kernel's scalar argument read as one element: its edge says it is broadcast
    scalars = {c for c, cont in g.containers.items() if cont.attrs.get(SCALAR)}
    reads, writes = [], []
    for top in tops:
        r: list[str] = []
        w: list[str] = []
        for side, node, c, m in _uses(top):
            if m.data in linked or m.data in local:
                continue
            if side == "read" and c in csr_connectors(g, node):  # no box: csr(alpha) in the text
                continue
            if side == "read" and m.data not in w and m.data not in r:
                r.append(m.data)
            if side == "write" and m.data not in w:
                w.append(m.data)
        reads.append(r)
        writes.append(w)
    used = {c for rw in (*reads, *writes) for c in rw}
    of, cols = columns(tops)
    joined = {d for col in cols for d in col["joins"]}
    written = {c for w in writes for c in w}
    once = [c for r in reads for c in r if c not in written] if len(cols) > 1 else []
    once = [c for c in dict.fromkeys(once) if c in drawn_once(g)]
    # the containers each column's nodes touch, a join's its columns' too
    touched: dict[int, set[str]] = {col["id"]: set() for col in cols}
    for k, c in enumerate(of):
        touched[c] |= {*reads[k], *writes[k]}
    for col in cols:
        for d in col["joins"]:
            touched[col["id"]] |= touched[d]

    version = dict.fromkeys(g.containers, 0)
    boxes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []  # the rows above every column
    rows_of: dict[int, list[dict[str, Any]]] = {col["id"]: [] for col in cols}
    seen: dict[int | None, set[str]] = {None: set()} | {col["id"]: set() for col in cols}
    writer: dict[str, list[str]] = {}  # container -> the nodes that wrote its latest version
    last: dict[int, tuple[int, dict[str, list[tuple[Node, str, Memlet]]]]] = {}

    def box_row(into: list[dict[str, Any]], col: int | None, names: list[str],
                fresh: dict[str, list[str]], k: int) -> dict[str, str]:  # fmt: skip
        """A row of container boxes in column ``col``; ``fresh`` ones are new versions, the
        rest shown again (``again`` when the column showed them already)."""
        keys = {}
        for c in names:
            if c in fresh:
                version[c] += 1
                writer[c] = fresh[c]
            base = f"{c}@{version[c]}"
            key = base if base not in boxes else f"{base}~{k}"  # the same data again
            boxes[key] = {
                "key": key,
                "container": c,
                "version": version[c],
                "written_by": writer.get(c, []),
                "again": base in seen[col],
                "column": col,
            }
            seen[col].add(base)
            keys[c] = key
        if keys:
            into.append({"kind": "containers", "column": col, "boxes": list(keys.values())})
        return keys

    def close(col: int, nxt: list[str], k: int) -> dict[str, str]:
        """The row after column ``col``'s last node: what it wrote and ``nxt``, the containers
        node ``k`` reads from this column; the last node's write edges into it."""
        kl, wrote = last.pop(col)
        names = nxt + [c for c in writes[kl] if c not in nxt]
        fresh = {c: list(dict.fromkeys(n.id for n, _, _ in wrote[c])) for c in writes[kl]}
        below = box_row(rows_of[col], col, names, fresh, k)
        for c in writes[kl]:
            for n, cn, m in wrote[c]:
                edges.append(
                    _edge(
                        f"{n.id}.{cn}",
                        below[c],
                        c,
                        memlet_text(m),
                        "write",
                        packing(g, n, cn, c),
                        slice_=memlet_text(m) if n.kind == "block" else None,
                    )
                )
        return below

    csr_read = {n.inputs[c].data for n, _ in g.walk() for c in csr_connectors(g, n)}
    unused = [c for c in g.containers
              if c not in used and c not in linked and c not in local and c not in csr_read]  # fmt: skip
    if not tops:
        box_row(top_rows, None, list(g.containers), {}, 0)
    shared = box_row(top_rows, None, once, {}, 0)  # drawn once, above the columns
    for col in cols:
        seen[col["id"]] |= seen[None]
    for k, top in enumerate(tops):
        col = of[k]
        joins = cols[col]["joins"]
        if k != cols[col]["tops"][0]:  # the row between this column's last node and this one
            above = close(col, reads[k], k)
        elif joins:  # a join: its columns side by side above it, each's last row what it reads
            rows_of[col].append({"kind": "columns", "column": col,
                                 "columns": [{"id": d, "rows": rows_of[d]} for d in joins]})  # fmt: skip
            above = {}
            claimed = set(shared)
            for d in joins:
                mine = [c for c in reads[k] if c in touched[d] and c not in claimed]
                claimed |= set(mine)
                above |= close(d, mine, k)
            for d in joins:
                seen[col] |= seen[d]
            above |= box_row(rows_of[col], col, [c for c in reads[k] if c not in claimed], {}, k)
        else:  # a new column: what its first node reads (the first column also what nothing uses)
            first = [c for c in reads[k] if c not in shared] + (unused if k == 0 else [])
            above = box_row(rows_of[col], col, first, {}, k)
        if k == cols[col]["tops"][0]:  # an input drawn once: its edge from above the columns
            above |= {c: key for c, key in shared.items() if c in reads[k]}
        rows_of[col].append({"kind": "node", "column": col, "node": _node(g, top)})
        writers: dict[str, list[str]] = {}  # container -> writer connectors inside this node
        wrote: dict[str, list[tuple[Node, str, Memlet]]] = {}  # container -> its writes here
        for side, n, c, m in _uses(top):
            port, text = f"{n.id}.{c}", memlet_text(m)
            if m.data in linked:  # straight into or out of the link node
                link = linked[m.data]
                if side == "write":
                    edges.append(
                        _edge(port, f"{link}.in", m.data, text, "write", packing(g, n, c, m.data))
                    )
                else:
                    edges.append(
                        _edge(f"{link}.out", port, m.data, text, "read", packing(g, n, c, m.data))
                    )
                continue
            if m.data in local:  # one iteration's value, from the writer straight to the reader
                if side == "write":
                    writers.setdefault(m.data, []).append(port)
                else:
                    edges.extend(_edge(w, port, m.data, text, "read", packing(g, n, c, m.data),
                                       local=True) for w in writers.get(m.data, []))  # fmt: skip
                continue
            if side == "write":
                writers.setdefault(m.data, []).append(port)
                wrote.setdefault(m.data, []).append((n, c, m))
            elif m.data in writers:  # written earlier inside this node: straight from the writer
                edges.extend(_edge(w, port, m.data, text, "read", packing(g, n, c, m.data))
                             for w in writers[m.data])  # fmt: skip
            elif c in csr_connectors(g, n):  # read at run time from a CSR: no edge (module doc)
                continue
            else:
                label = f"{text} · broadcast" if m.data in scalars and not m.names else None
                seeded = c == SEED and seed_memlet(n) is not None
                if seeded:  # the fold's seed, labelled with its memlet
                    label, text = text, f"{text} (seed)"
                perm = n.attrs.get(TRANSPOSE, {}).get(c)
                if perm is not None:  # read across the loop order (mark_transpose): Aᵀ
                    label = transpose_label(m.data, perm)
                    text = f"{text} (transpose {', '.join(map(str, perm))})"
                edges.append(_edge(above[m.data], port, m.data, text, "read",
                                   packing(g, n, c, m.data), label=label, seed=seeded,
                                   transpose=perm is not None,
                                   slice_=memlet_text(m) if n.kind == "block" else None))  # fmt: skip
        last[col] = (k, wrote)
    ends = [col["id"] for col in cols if col["id"] not in joined]
    for col in ends:  # the last row of each column: what its last node wrote
        close(col, [], len(tops))
    if len(ends) <= 1:
        rows = top_rows + (rows_of[ends[0]] if ends else [])
    else:
        rows = top_rows + [{"kind": "columns", "column": None,
                            "columns": [{"id": c, "rows": rows_of[c]} for c in ends]}]  # fmt: skip
    out = [{"id": col["id"], "tops": [tops[k].id for k in col["tops"]], "joins": col["joins"]}
           for col in cols]  # fmt: skip
    return {"rows": rows, "boxes": boxes, "edges": edges, "columns": out}


def _edge(
    src: str,
    dst: str,
    data: str,
    text: str,
    side: str,
    pack: str | None = None,
    local: bool = False,
    label: str | None = None,
    seed: bool = False,
    transpose: bool = False,
    slice_: str | None = None,
) -> dict[str, Any]:
    """One memlet; ``pack`` the port's packing when it meets an accelerated node; ``local`` a
    fused nest's local value between its two tasklets, labelled with the container's name;
    ``label`` any other text drawn beside it (a scalar's ``alpha[0] · broadcast``, a fold's
    seed ``q[m_t]``); ``seed`` a fold's seed; ``transpose`` a read ``mark_transpose`` marked,
    its label ``Aᵀ``; ``slice_`` at a collapsed block's port, the part one firing takes
    (``A[m_t, 0:N]``), drawn beside the edge."""
    out = {"id": f"{src}->{dst}", "from": src, "to": dst, "data": data, "text": text, "dir": side}
    if slice_ is not None:
        out["slice"] = slice_
    if pack is not None:
        out["packing"] = pack
    if seed:
        out["seed"] = True
    if transpose:
        out["transpose"] = True
    if local:
        out |= {"local": True, "label": data}
    elif label is not None:
        out |= {"label": label}
    return out
