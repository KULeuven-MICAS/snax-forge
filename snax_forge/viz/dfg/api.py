"""What the DFG viewer asks for, as plain Python functions (D55).

The server (server.py) only turns these results into JSON; the tests call
them directly, and the drawing is checked by eye.

Loading
-------
The viewer is given ``.snaxdfg`` files and directories. A directory stands
for every ``.snaxdfg`` in it, in natural order (``0_input``,
``1_split_map``, ``2_bind``, ... ``10_x``), so a recipe's output directory
shows every step of the recipe. Each file is one entry named after its stem;
when the files come from more than one folder (two designs to compare), after
``<folder>/<stem>``, the folder its parent directory's name (``dot4_w8/6_link``),
and its entry keeps the folder. Names are made unique with ``-2``, ``-3``, ...
in the order given. ``find_step`` reads a step back from the URL hash: its
name, a bare stem one step has, or a number. A file that does
not load (bad JSON, a ``DfgError``) is kept as an entry with its error, so
one broken hand edit shows its message in its own panel and the others
still draw. Everything is read once, at start-up or on Reload.

The view of a graph
-------------------
The graph is drawn top to bottom in execution order, in rows:

    containers top-level node 1 reads    (and containers nothing uses)
    top-level node 1
    containers node 1 writes, and those node 2 reads
    top-level node 2
    ...
    containers the last node writes

So every edge joins two neighbouring rows and never passes behind a node. A
container gets a new version (``A@1``, ``A@2``, ...) in the row after each
top-level node that writes it, so a graph that reads ``A`` and later writes
it (jacobi1d) has no upward edge, and a transient sits between the node that
writes it and the node that reads it. A container read again further down is
shown again in the row above its reader, with the same version and a
``~<k>`` key, k the number of top-level nodes above it (``B@0~1``), as SDFG
shows an array once per access. Inside
one top-level node, a read of a container written earlier in that same node
is an edge straight from the writer's connector. Within a row, containers
come in the order the next node's connectors read them, then the rest, so
edges cross as little as possible.

Each edge is one memlet: from a container box (or a writer's connector) to
an input connector, or from an output connector to a container box, with its ``id``
``<from>-><to>``; it carries no text. Its Details hold the memlet, for a folding write
with the op (``out[0] (reduce add)``), and at an accelerated node's port its ``packing`` (D142):
``int32 · num_slc 2 · num_ch 2``, the container's dtype of b bits, num_slc =
64 / b slices (elements) per bank word, and num_ch = W / num_slc channels, W
the instance's lanes for that port (the wire slice, the only value
``connect.num_ch`` takes), rounded up so a fold's one-lane output is one word
(``int64 · num_slc 1 · num_ch 1``). All of it is read
from the bound graph and the BRM library; nothing is stored in the graph.
An edge to or from a connector inside a top-level node runs to that connector,
under the boxes it passes (a map's bar hides it, its body shows it). The
nodes are the graph's own tree, each with its compact text, a map's tag and its connectors.
Positions are the browser's business: the boxes are HTML laid out by CSS,
and the edges an SVG layer the viewer draws over them (static/dfg.js).

**Accelerators** (D148). A bound graph nests accelerator ⊃ groups ⊃ units:
``accelerators`` lists each accelerator in the order the graph first uses
it, its groups in the graph's order (a unit alone, or the units ``link``
joined, by their ``group``), each with its units (the instance and its
accelerated nodes), its links (each link node's id) and ``tops``, the
top-level nodes that hold them and their link nodes. The viewer draws a box
around each group's top-level nodes and one around each accelerator's groups,
so dot 1's ``mul`` and ``sum`` sit in the accelerator ``dot``, and dot 4's
``mac``, its link and ``sum`` in one group box; an accelerated node's Details
name its group and accelerator.

**The compact texts and Details** (VIEW2). Every node, container, group and accelerator also
carries ``text``, what the compact drawing shows, and the view a ``details`` map of everything
else, keyed ``<kind>:<id>`` (``node:``, ``container:``, ``group:``, ``accelerator:``, ``edge:``;
a link node and its container share a name, so the page selects by name and shows both), all
worked out here:

    map          ``<id> · <var> · <iterations> · <kind or untagged>``
                                                       (``mult_map · i · 2048 · untagged``)
    tasklet      ``<id> · <operation>``: the code's right-hand side on its connector names,
                 ``fold <op>`` for a copy into a ``reduce``        (``mult · in1 * in2``, ``sum · fold +``)
    unit         ``<instance> · <what one lane computes> · W <lanes>``, a fold's as a
                 tasklet's                                          (``mul · a * b · W 4``)
    link         ``<container> · wire`` or ``<container> · FIFO <depth>``
    container    its name, and a ``badge`` from the estimate's placement (``L1 · 4 KiB``,
                 ``local``, ``wire``, ``FIFO 1``)
    group, accelerator   their name

Details: a map's range as written, ``loop.split``, ``loop.fused`` and ``loop.split_reduce``; a
tasklet's code and memlets; a unit's BRM, implementation, params, timing, what it replaced, its
group and accelerator; a link's container, depth and units; a container's shape, dtype, padded
bytes, writers and readers; a group's units, links, estimated cycles and its cost; an accelerator's
groups; an edge's (keyed ``<from>-><to>``, its ``id``) subset, packing and, at a unit's port,
the firing loops its memlet runs over and its beats (design/streamers.py's ``port_loops``, as the
estimate counts them, D155) and whether it has a streamer (a linked port has none). The estimate is the
step's on the viewer's platform, or none; a graph it cannot judge keeps its message in
``estimate_error`` and has no badges.

**The cost** (C§20) is the estimate's on the technology asked for (``tech``, default nangate45),
worked out here as text, an area to 3 significant figures and an energy to 3 decimals
(estimate/cost.py's ``sig3`` and ``dec3``):
a unit's and a buffer link's Details gain ``cost`` after ``timing`` (``found``, its area total,
comb and flops, its energy per fire, holding per cycle and per task, or "not characterized"), a
group's its area, energy and control unit; the Estimate panel's ``summary`` per group
``area_text`` and ``energy_text``, and ``cost`` (the heading, ``CAVEAT``, the accelerator's
area and energy, the blocks not characterized, and ``tables``, the cost as a table per
accelerator, ``cost_tables``, with its L1 accesses' row, a placeholder (D154); None when nothing
is bound or there is no technology); a block with no flops holds "–".

**A step** (``step_view``) is a graph's view with its ``label``, its ``marks`` (the nodes and
containers some transform's site accepts), its ``estimate`` (``to_dict``) with its ``summary``
(the Estimate panel's figures, every sum and share made here, D57), its ``containers_section``
(the left column's Containers, C§12: each container's shape, sizes, dtype, bytes, placement,
symbols and its one ``line``), its ``symbols_section`` (each kernel symbol, its value and its
param) and ``bindings_section`` (each param with the unit, split or link it configures; the
params come from the session's recipe in edit mode, a sandbox folder's ``recipe.json`` in view
mode, ``folder_recipe``) and, in edit mode, its
``diff`` against the step before (sandbox/session.py). ``compare_view`` gives two steps' diff
(the session's ``graph_diff``), per Details key the fields that changed, and ``columns``, what
Compare lays out as two columns (the left step, ``a``, and the right, ``b``): their
``labels``; per Details key its ``rows`` and ``where`` (``added`` or ``removed`` for an item on
one side only); the Estimate's ``rows`` and the ``caveat``. A row is ``{label, under, a, b,
differs}``: ``under`` the section it sits in (``cost``, ``group <name>``, the cost heading) or
None, ``a`` and ``b`` each side's text as a single step shows it ("–" where a side has none),
``differs`` whether they differ. Every text is made here; the page only lays them out.

**Links**. A link node is drawn in the accelerated node's style, its text
``<container> · wire`` or ``<container> · FIFO <depth>``, its units and the transient it
replaced in its Details. The container it replaced gets no box: the writer's edge runs from
its output connector into the link (connector ``in``) and the link's out of it (``out``) into
the reader's input connector; the container stays in Details. A transient in L1 (dot 1's
``tmp0``), a local value and the kernel's inputs and outputs keep their boxes.
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
from snax_forge.brm import BrmError
from snax_forge.brm.dtypes import bits
from snax_forge.char import DEFAULT_TECH
from snax_forge.common import LoadedSet, unique_names
from snax_forge.design.streamers import (
    accelerator_of,
    firing_loops,
    group_of,
    instance_of,
    port_loops,
)
from snax_forge.dfg import DfgError, Graph, Memlet, Node
from snax_forge.dfg.subset import map_range
from snax_forge.estimate.cost import CAVEAT, L1_LABEL, L1Cost, dec3, heading, sig3
from snax_forge.expr import ExprError

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
               error: str | None = None, tech: str = DEFAULT_TECH) -> dict[str, Any]:  # fmt: skip
    """The rows, nodes and edges a panel draws, with their compact texts and Details (module
    doc); ``est`` the graph's estimate (or ``error`` why there is none), else made on
    ``platform`` and costed on ``tech``."""
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
    out.update(_layout(g))
    out["accelerators"] = accelerators(g)
    for acc in out["accelerators"]:
        acc["text"] = acc["name"]
        for grp in acc["groups"]:
            grp["text"] = grp["name"]
    out["details"] = details(g, out, est)
    for edge in out["edges"]:  # their memlet and packing are in Details
        edge.pop("text")
        edge.pop("packing", None)
    return out


def step_view(e: Entry, label: str, platform: Any = None, marks: dict[str, list[str]] | None = None,
              diff: dict[str, Any] | None = None, est: Any = None, tech: str = DEFAULT_TECH,
              recipe: Any = None, editable: bool = False) -> dict[str, Any]:  # fmt: skip
    """One step (module doc): the graph's view, its label, marks, estimate and its summary,
    and diff; the cost on ``tech``; ``recipe`` the steps up to this one (a session's, or a
    sandbox folder's ``recipe.json``), whose params the left column's sections name, inputs
    where ``editable`` (edit mode)."""
    error = None
    if e.graph is not None and est is None:
        est, error = _estimated(e.graph, platform, tech)
    out = graph_view(e, platform, est, error, tech)
    out["label"] = label
    out["tech"] = tech
    if e.graph is None:
        return out
    out["estimate"] = est.to_dict() if est is not None else None
    out["summary"] = estimate_summary(est, e.graph) if est is not None else None
    out["containers_section"] = containers_section(e.graph, est)
    out["symbols_section"] = symbols_section(e.graph, recipe, editable)
    out["bindings_section"] = bindings_section(e.graph, recipe, editable)
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
    """The left column's Symbols (C§12): each graph symbol, its ``value`` (None while unbound),
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


def _param_names(v: Any, params: dict[str, Any]) -> list[str]:
    """The recipe params an int step param names (a factor ``W``, ``W * 2``)."""
    if not isinstance(v, str):
        return []
    try:
        return [x for x in sorted(expr.names(v)) if x in params]
    except ExprError:
        return []


def bindings_section(g: Graph, recipe: Any = None, editable: bool = False) -> dict[str, Any]:
    """The left column's Bindings (C§12): each param shown with what it configures. The param
    stays the split's (D68, D73: ``bind`` reads W off the spatial map); a split step on map M
    configures the unit whose replaced subtree holds the map it made (``M_inner``, ``M_s``,
    ``split_reduce``'s ``M_final``). ``entries``, in the graph's order:

        unit    per bound instance: its BRM, the lanes its design params give (``W = 8``) and
                the params of the splits that set them (``acc · elementwise_mul · W = 8``)
        split   a split whose map is not bound yet: its factor's params (``not bound``)
        link    a link whose depth a step gives as a param

    Each param ``{name, value, editable, shared_with}``: an input in edit mode
    (``POST /api/params``), ``shared_with`` the other entries naming it (dot 2–6's ``W``, the
    multiply's and the sum's). ``unused``: the session's params neither a symbol nor in an
    entry. Without a recipe (view mode on bare files) the units come from the graph alone,
    their design params read only."""
    params = dict(recipe.params) if recipe is not None else {}
    steps = list(recipe.steps) if recipe is not None else []
    splits = []  # (map, transform, factor params, inner ids)
    links = {}  # container -> depth params
    for st in steps:
        p = st.params
        if st.transform in SPLITS:
            m = p.get("map")
            inner = re.compile(rf"^{re.escape(str(m))}({'|'.join(SPLITS[st.transform])})(_\d+)?$")
            splits.append((m, st.transform, _param_names(p.get("factor"), params), inner))
        elif st.transform == "link":
            names = _param_names(p.get("depth"), params)
            if names:
                links[p.get("container")] = names
    entries: list[dict[str, Any]] = []
    used: set[int] = set()
    seen: set[str] = set()
    for n, _ in g.walk():
        if n.kind == "accelerated" and n.attrs["instance"] not in seen:
            a = n.attrs
            seen.add(a["instance"])
            ids = _map_ids(a.get("replaced"))
            names: list[str] = []
            for i, (_, _, ps, inner) in enumerate(splits):
                if any(inner.match(x) for x in ids):
                    used.add(i)
                    names += [x for x in ps if x not in names]
            design = dict(a.get("params") or {})
            lanes = {k: v for k, v in design.items() if isinstance(v, int)}
            entries.append({"kind": "unit", "id": a["instance"], "label": f"{a['instance']} · {a['brm']}",
                            "names": names, "lanes": lanes, "note": None})  # fmt: skip
        elif n.kind == "link" and n.attrs["container"] in links:
            c = n.attrs["container"]
            entries.append({"kind": "link", "id": c, "label": f"link {c} · depth",
                            "names": links[c], "lanes": {}, "note": None})  # fmt: skip
    for i, (m, transform, ps, _) in enumerate(splits):
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
        names = e.pop("names")
        e["params"] = [{"name": x, "value": params.get(x), "editable": bool(editable and x in params),
                        "shared_with": [lbl for lbl in by_name[x] if lbl != e["label"]]}
                       for x in names]  # fmt: skip
        if not e["params"] and e["lanes"]:  # no recipe param: the design params, read only
            e["params"] = [{"name": k, "value": v, "editable": False, "shared_with": []}
                           for k, v in e["lanes"].items()]  # fmt: skip
    symbol_params = {x["param"] for x in symbols_section(g, recipe) if x["param"]}
    unused = [{"name": k, "value": v, "editable": editable} for k, v in params.items()
              if k not in symbol_params and k not in by_name]  # fmt: skip
    return {"entries": entries, "unused": unused}


def _placed_text(placed: Any) -> str | None:
    """A container's placement as its line shows it: ``L1``, ``local``, ``wire``, ``FIFO 1``."""
    if placed is None:
        return None
    if placed.placement == "l1":
        return "L1"
    if placed.placement == "link":
        return "wire" if placed.depth == 0 else f"FIFO {placed.depth}"
    return placed.placement


def containers_section(g: Graph, est: Any = None) -> list[dict[str, Any]]:
    """The left column's Containers (C§12), per container in the graph's order: its ``name``,
    ``shape`` as written (``shape_text``, its dims by " × ", and ``dims``, each dim's ``parts``,
    ``{text, symbol}`` with ``symbol`` the graph symbol a part names or None, so the page puts
    an input on each), ``sizes`` at the graph's symbols (None while one is unbound), ``dtype``,
    padded ``bytes`` and ``placement`` as the estimate has them (None without one), and the
    ``symbols`` its shape uses, in order; ``line``, what the page shows after its name: the shape,
    the placement and the padded bytes (``N · L1 · 16 KiB``)."""
    out = []
    for name, c in g.containers.items():
        placed = est.containers.get(name) if est is not None else None
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
        out.append({
            "name": name, "shape": list(c.shape),
            "shape_text": " × ".join(str(d) for d in c.shape) or "scalar", "dims": dims,
            "sizes": sizes, "sizes_text": " × ".join(map(str, sizes)) if sizes else None,
            "dtype": c.dtype, "transient": c.transient,
            "bytes": placed.bytes if placed else None,
            "bytes_text": size_text(placed.bytes) if placed else None,
            "placement": placed.placement if placed else None,
            "symbols": list(dict.fromkeys(used)),
            "line": " · ".join(x for x in (" × ".join(str(d) for d in c.shape) or "scalar",
                                           _placed_text(placed),
                                           size_text(placed.bytes) if placed else None) if x),
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
        "caveat": ((sb or {}).get("cost") or (sa or {}).get("cost") or {}).get("caveat"),
    }  # fmt: skip
    return {"diff": diff, "changed": changed, "columns": columns}


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
    estimate): under ``accelerator`` its cycles, the unmapped cycles, the total (D155) and the
    not-mapped count; under ``group <name>``,
    every group of either side, its cycles, COUNTER, channels, area and energy, and its bank
    note where a side has one; under ``memory`` L1, the words read and written, the lanes; under the cost heading the
    accelerator's area, energy, L1 accesses (D154) and the blocks not characterized. "–" where a
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
            _row("not mapped", *both(lambda s: s["not_mapped"]), "accelerator")]  # fmt: skip
    names = dict.fromkeys(g["name"] for s in (sa, sb) if s for g in s["groups"])
    for name in names:

        def group(s: dict[str, Any], field: str, name: str = name) -> Any:
            return next(g for g in s["groups"] if g["name"] == name)[field]

        under = f"group {name}"
        rows += [_row(label, *both(lambda s, f=field: group(s, f)), under)
                 for label, field in (("cycles", "formula"), ("COUNTER", "counter"),
                                      ("channels", "channels"), ("area", "area_text"),
                                      ("energy", "energy_text"))]  # fmt: skip
        notes = both(lambda s: group(s, "bank_note"))
        if any(notes):  # L1's banks allow fewer cycles than the ideal on a side (D155)
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

        rows += [_row("area", *both(lambda s: s["cost"]["area_text"]), under),
                 _row("energy per run", *both(lambda s: s["cost"]["energy_text"]), under),
                 _row("L1 accesses", *both(lambda s: s["cost"]["l1_text"]), under),
                 _row("not characterized", *both(missing), under)]  # fmt: skip
    return rows


def estimate_summary(est: Any, graph: Graph | None = None) -> dict[str, Any]:
    """The Estimate panel's figures, every sum and share worked out here (D57): the headline,
    ``cycles`` (the accelerator's, the unmapped work's at one cycle per iteration with that
    assumption as its ``title``, and the total, D155),
    per group ``cycles = bottleneck + ramp [+ stall]`` and its expected COUNTER, the L1
    footprint against the L1 (its share as a CSS width), the words read and written, and the
    lane use of each tailed split; per group its ``bank_bound`` and, where it exceeds the
    cycles, its ``bank_note`` (D155; R2 shows them); with a technology per group its area and
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
                  "assumption (a core takes several), D155")  # fmt: skip
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
        "cost": cost_summary(est, graph),
    }


def cost_summary(est: Any, g: Graph | None = None) -> dict[str, Any] | None:
    """The Estimate panel's ``cost`` (module doc): None with no technology or no group; with
    ``tables``, the cost table per accelerator (``cost_tables``)."""
    if est.tech is None or not est.groups:
        return None
    l1 = "; ".join(f"{a}: {x.words} words, {_nj(x.energy_nj)}" for a, x in est.l1.items())
    return {"tech": est.tech, "heading": heading(est.tech), "caveat": CAVEAT,
            "area_text": _um2(est.area), "energy_text": _nj(est.energy),
            "l1_text": l1 or None, "missing": list(est.missing),
            "tables": cost_tables(est, g)}  # fmt: skip


def cost_tables(est: Any, g: Graph | None = None) -> list[dict[str, Any]]:
    """The cost as tables (C§12), one per accelerator of ``g`` (its groups, in order; without
    ``g`` one of every group, named ``accelerator``): a row per unit, buffer link and control
    unit, each its ``block`` (the instance, the link's container, ``<group> ctrl``), ``kind``,
    ``datapath`` (``· not characterized`` where the table has no point), area (µm², 3
    significant figures) and energy (nJ, 3 decimals) with their texts, and ``share`` of the
    accelerator's energy; then one row of its L1 accesses (``L1_LABEL``, kind ``l1``, its
    ``words`` and ``bits`` in ``datapath``, no area; a placeholder, D154); then the ``total``."""
    by_name = {x.name: x for x in est.groups if x.cost is not None}
    accs = [(a["name"], [grp["name"] for grp in a["groups"]]) for a in accelerators(g)] if g else []
    l1 = est.l1
    if not accs:
        accs = [("accelerator", list(by_name))]
        l1 = {"accelerator": L1Cost(sum(x.words for x in l1.values()),
                                    sum(x.energy_nj for x in l1.values()))} if l1 else {}  # fmt: skip
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
        if name in l1:  # its streamers' words in L1, a placeholder (D154)
            x = l1[name]
            raw.append((L1_LABEL, "l1", f"{x.words} words, {_thousands(x.bits)} bits", True, None,
                        x.energy_nj))  # fmt: skip
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
    return _rhs(code)


def compact_text(g: Graph, n: Node) -> str:
    """A node's text in the compact drawing (module doc)."""
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
            first = next(p.name for p in inst.brm.interface.ports if p.direction == "in")
            return f"{text} · W {inst.lanes(first)}"
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
        for m in n.inputs.values():
            users[m.data]["readers"].append(n.id)
        for m in n.outputs.values():
            users[m.data]["writers"].append(n.id)
        out[f"node:{n.id}"] = _node_details(n)
    for c, cont in g.containers.items():
        placed = est.containers.get(c) if est is not None else None
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


def _container(g: Graph, name: str, placed: Any = None) -> dict[str, Any]:
    """A container as the drawing shows it: its name, its badge, whether it is transient."""
    c = g.containers[name]
    return {"name": name, "text": name, "badge": badge(placed), "transient": c.transient}


def memlet_text(m: Memlet) -> str:
    text = f"{m.data}[{', '.join(str(d) for d in m.subset)}]"
    return text if m.reduce is None else f"{text} (reduce {m.reduce.op})"


def _iterations(g: Graph, rng: str) -> int | None:
    try:
        return len(map_range(rng, g.bound_symbols, "range"))
    except (ExprError, DfgError):
        return None


def _node(g: Graph, n: Node) -> dict[str, Any]:
    """A node as the compact drawing shows it: its text, a map's tag, its connectors (where
    the edges meet it) and its body; the rest is in Details."""
    d: dict[str, Any] = {
        "id": n.id,
        "kind": n.kind,
        "text": compact_text(g, n),
        "inputs": [_conn(c, m) for c, m in n.inputs.items()],
        "outputs": [_conn(c, m) for c, m in n.outputs.items()],
    }
    if n.kind == "map":
        d["loop_kind"] = n.attrs.get("loop.kind")
    if n.kind == "link":  # its container's edges meet it here (module doc)
        c = n.attrs["container"]
        d["inputs"], d["outputs"] = (
            [{"connector": "in", "data": c}],
            [{"connector": "out", "data": c}],
        )
    if n.body is not None:
        d["body"] = [_node(g, c) for c in n.body]
    return d


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
    for child in n.body or []:
        out += _uses(child)
    out += [("write", n, c, m) for c, m in n.outputs.items()]
    return out


def packing(g: Graph, n: Node, port: str, data: str) -> str | None:
    """``int32 · num_slc 2 · num_ch 2`` for an accelerated node's port (module doc); None for any other
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


def _layout(g: Graph) -> dict[str, Any]:
    """Rows, nodes and edges (module doc)."""
    tops = list(g.body)
    # A link's container gets no box: its writer's edge goes into the link, the link's out of it
    linked = {n.attrs["container"]: n.id for n, _ in g.walk() if n.kind == "link"}
    reads, writes = [], []
    for top in tops:
        r: list[str] = []
        w: list[str] = []
        for side, _, _, m in _uses(top):
            if m.data in linked:
                continue
            if side == "read" and m.data not in w and m.data not in r:
                r.append(m.data)
            if side == "write" and m.data not in w:
                w.append(m.data)
        reads.append(r)
        writes.append(w)
    used = {c for rw in (*reads, *writes) for c in rw}
    version = dict.fromkeys(g.containers, 0)
    boxes: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    writer: dict[str, list[str]] = {}  # container -> the nodes that wrote its latest version

    def box_row(names: list[str], fresh: dict[str, list[str]], k: int) -> dict[str, str]:
        """A row of container boxes; ``fresh`` ones are new versions, the rest shown again."""
        keys = {}
        for c in names:
            if c in fresh:
                version[c] += 1
                writer[c] = fresh[c]
            key = f"{c}@{version[c]}"
            if key in boxes:  # the same data again, in a later row
                key = f"{key}~{k}"
            boxes[key] = {
                "key": key,
                "container": c,
                "version": version[c],
                "written_by": writer.get(c, []),
                "again": "~" in key,
            }
            keys[c] = key
        if keys:
            rows.append({"kind": "containers", "boxes": list(keys.values())})
        return keys

    unused = [c for c in g.containers if c not in used and c not in linked]
    before = reads[0] + unused if tops else list(g.containers)
    above = box_row(before, {}, 0)
    for k, top in enumerate(tops):
        rows.append({"kind": "node", "node": _node(g, top)})
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
            if side == "write":
                writers.setdefault(m.data, []).append(port)
                wrote.setdefault(m.data, []).append((n, c, m))
            elif m.data in writers:  # written earlier inside this node: straight from the writer
                edges.extend(_edge(w, port, m.data, text, "read", packing(g, n, c, m.data))
                             for w in writers[m.data])  # fmt: skip
            else:
                edges.append(
                    _edge(above[m.data], port, m.data, text, "read", packing(g, n, c, m.data))
                )
        nxt = reads[k + 1] if k + 1 < len(tops) else []
        names = nxt + [c for c in writes[k] if c not in nxt]
        fresh = {c: list(dict.fromkeys(n.id for n, _, _ in wrote[c])) for c in writes[k]}
        below = box_row(names, fresh, k + 1)
        for c in writes[k]:
            for n, cn, m in wrote[c]:
                edges.append(
                    _edge(
                        f"{n.id}.{cn}", below[c], c, memlet_text(m), "write", packing(g, n, cn, c)
                    )
                )
        above = below
    return {"rows": rows, "boxes": boxes, "edges": edges}


def _edge(
    src: str, dst: str, data: str, text: str, side: str, pack: str | None = None
) -> dict[str, Any]:
    """One memlet; ``pack`` the port's packing when it meets an accelerated node."""
    out = {"id": f"{src}->{dst}", "from": src, "to": dst, "data": data, "text": text, "dir": side}
    if pack is not None:
        out["packing"] = pack
    return out
