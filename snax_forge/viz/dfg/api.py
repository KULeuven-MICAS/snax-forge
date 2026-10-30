"""What the DFG viewer asks for, as plain Python functions (D55).

The server (server.py) only turns these results into JSON; the tests call
them directly, and the drawing is checked by eye.

Loading
-------
The viewer is given ``.snaxdfg`` files and directories. A directory stands
for every ``.snaxdfg`` in it, in natural order (``0_input``,
``1_split_map``, ``2_bind``, ... ``10_x``), so a recipe's output directory
shows every step of the recipe. Each file is one entry named after its stem,
made unique with ``-2``, ``-3``, ... in the order given. A file that does
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
An edge between a container and a connector inside a top-level node is drawn
only up to that node's outer box, above or below the connector it belongs
to (``stop``), so it never crosses the text inside; an edge between
two connectors inside one node is drawn in full. The
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

    map          ``<var> · <iterations> · <kind or untagged>``      (``i_t · 128 · temporal``)
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
bytes, writers and readers; a group's units, links and estimated cycles; an accelerator's
groups; an edge's (keyed ``<from>-><to>``, its ``id``) subset, packing and, at a unit's port,
the firing loops its memlet runs over and its beats (design/streamers.py's ``port_loops``, as the
estimate counts them, D151) and whether it has a streamer (a linked port has none). The estimate is the
step's on the viewer's platform, or none; a graph it cannot judge keeps its message in
``estimate_error`` and has no badges.

**A step** (``step_view``) is a graph's view with its ``label``, its ``marks`` (the nodes and
containers some transform's site accepts), its ``estimate`` (``to_dict``) with its ``summary``
(the Estimate panel's figures, every sum and share made here, D57) and, in edit mode, its
``diff`` against the step before (sandbox/session.py). ``compare_view`` gives two steps' diff
(the session's ``graph_diff``) and, per Details key, the fields that changed.

**Links**. A link node is drawn in the accelerated node's style, its text
``<container> · wire`` or ``<container> · FIFO <depth>``, its units and the transient it
replaced in its Details. The container still shows between the writer and the reader: they
name it, with the link's badge.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

from snax_forge import expr
from snax_forge.brm import BrmError
from snax_forge.brm.dtypes import bits
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


def entry_names(files: Sequence[Path]) -> list[str]:
    """File stems, made unique with -2, -3, ... in the order given."""
    return unique_names(f.stem for f in files)


@dataclass
class Entry:
    name: str
    path: Path
    graph: Graph | None
    error: str | None


def load_entry(name: str, path: Path) -> Entry:
    """One file; a file that does not load keeps its error instead of a graph."""
    try:
        return Entry(name, path, Graph.load(path), None)
    except (OSError, ValueError, KeyError, TypeError) as e:
        msg = f"invalid JSON: {e}" if isinstance(e, json.JSONDecodeError) else str(e)
        return Entry(name, path, None, msg)


class GraphSet(LoadedSet):
    """The files the viewer was given, read once and again on ``reload`` (``LoadedSet``)."""

    def __init__(self, paths: Sequence[str | Path]):
        self.paths = [Path(p) for p in paths]
        super().__init__()

    def load(self) -> dict[str, Entry]:
        files = expand(self.paths)
        return {n: load_entry(n, f) for n, f in zip(entry_names(files), files, strict=True)}

    def summaries(self) -> list[dict[str, Any]]:
        return [entry_summary(e) for e in self.values()]


def entry_summary(e: Entry) -> dict[str, Any]:
    """A step's entry: its name, path, graph name and load error."""
    return {
        "name": e.name,
        "path": str(e.path),
        "graph": e.graph.name if e.graph else None,
        "error": e.error,
    }


# =============================================================================
# The view of one graph
# =============================================================================


def graph_view(e: Entry, platform: Any = None, est: Any = None,
               error: str | None = None) -> dict[str, Any]:  # fmt: skip
    """The rows, nodes and edges a panel draws, with their compact texts and Details (module
    doc); ``est`` the graph's estimate (or ``error`` why there is none), else made on
    ``platform``."""
    out: dict[str, Any] = entry_summary(e)
    if e.graph is None:
        return out
    g = e.graph
    if est is None and error is None:
        est, error = _estimated(g, platform)
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
              diff: dict[str, Any] | None = None, est: Any = None) -> dict[str, Any]:  # fmt: skip
    """One step (module doc): the graph's view, its label, marks, estimate and its summary,
    and diff."""
    error = None
    if e.graph is not None and est is None:
        est, error = _estimated(e.graph, platform)
    out = graph_view(e, platform, est, error)
    out["label"] = label
    if e.graph is None:
        return out
    out["estimate"] = est.to_dict() if est is not None else None
    out["summary"] = estimate_summary(est) if est is not None else None
    if marks is None:
        from snax_forge.sandbox.session import marks_of

        marks = marks_of(e.graph)
    out["marks"] = marks
    if diff is not None:
        out["diff"] = diff
    return out


def compare_view(a: Entry, b: Entry, platform: Any = None) -> dict[str, Any]:
    """Two steps compared (``a`` before, ``b`` after): the session's diff of their graphs
    (sandbox/session.py's ``graph_diff``) and, per Details key in either, the fields that
    differ (every field of an item in one only)."""
    from snax_forge.sandbox.session import graph_diff

    if a.graph is None or b.graph is None:
        return {"diff": None, "changed": {}, "error": a.error or b.error}
    (ea, _), (eb, _) = _estimated(a.graph, platform), _estimated(b.graph, platform)
    da = graph_view(a, platform, ea, None if ea else "")["details"]
    db = graph_view(b, platform, eb, None if eb else "")["details"]
    changed = {}
    for key in dict.fromkeys([*da, *db]):
        x, y = da.get(key, {}), db.get(key, {})
        fields = [f for f in dict.fromkeys([*x, *y]) if x.get(f) != y.get(f)]
        if fields:
            changed[key] = fields
    diff = graph_diff(
        a.graph, b.graph, ea.containers if ea else None, eb.containers if eb else None
    )
    return {"diff": diff, "changed": changed}


def estimate_summary(est: Any) -> dict[str, Any]:
    """The Estimate panel's figures, every sum and share worked out here (D57): the headline,
    per group ``cycles = bottleneck + ramp [+ stall]`` and its expected COUNTER, the L1
    footprint against the L1 (its share as a CSS width), the words read and written, and the
    lane use of each tailed split."""
    groups = []
    for g in est.groups:
        terms = f"{g.bottleneck} + {g.ramp}" + (f" + {g.stall}" if g.stall else "")
        groups.append({"name": g.name, "cycles": g.cycles, "bottleneck": g.bottleneck,
                       "ramp": g.ramp, "stall": g.stall, "counter": g.counter,
                       "channels": g.channels, "formula": f"{g.cycles} = {terms}"})  # fmt: skip
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
    return {
        "accelerator_cycles": est.accelerator_cycles,
        "not_mapped": est.not_mapped,
        "groups": groups,
        "l1": l1,
        "reads": sum(c.reads for c in est.containers.values()),
        "writes": sum(c.writes for c in est.containers.values()),
        "splits": splits,
    }


def _estimated(g: Graph, platform: Any) -> tuple[Any, str | None]:
    """The graph's estimate on ``platform``, or None and why not."""
    from snax_forge.estimate import estimate

    try:
        return estimate(g, platform), None
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
        return f"{a['var']} · {'?' if its is None else its} · {a.get('loop.kind') or 'untagged'}"
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
        out[f"node:{n.id}"] = _node_details(n, est)
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
            out[f"group:{grp['name']}"] = {
                "kind": "group", "accelerator": acc["name"],
                "units": [u["instance"] for u in grp["units"]], "links": list(grp["links"]),
                "estimate": e.to_dict() if e is not None else None,
            }  # fmt: skip
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
            elif n is not None:
                d["subset"] = list((n.inputs | n.outputs)[port].subset)
        out[f"edge:{edge['id']}"] = d
    return out


def _node_details(n: Node, est: Any) -> dict[str, Any]:
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
    reads, writes = [], []
    for top in tops:
        r: list[str] = []
        w: list[str] = []
        for side, _, _, m in _uses(top):
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

    before = reads[0] + [c for c in g.containers if c not in used] if tops else list(g.containers)
    above = box_row(before, {}, 0)
    for k, top in enumerate(tops):
        rows.append({"kind": "node", "node": _node(g, top)})
        writers: dict[str, list[str]] = {}  # container -> writer connectors inside this node
        wrote: dict[str, list[tuple[Node, str, Memlet]]] = {}  # container -> its writes here
        for side, n, c, m in _uses(top):
            port, text = f"{n.id}.{c}", memlet_text(m)
            if side == "write":
                writers.setdefault(m.data, []).append(port)
                wrote.setdefault(m.data, []).append((n, c, m))
            elif m.data in writers:  # written earlier inside this node: straight from the writer
                edges.extend(_edge(w, port, m.data, text, "read", None, packing(g, n, c, m.data))
                             for w in writers[m.data])  # fmt: skip
            else:
                edges.append(_edge(above[m.data], port, m.data, text, "read", top.id,
                                   packing(g, n, c, m.data)))  # fmt: skip
        nxt = reads[k + 1] if k + 1 < len(tops) else []
        names = nxt + [c for c in writes[k] if c not in nxt]
        fresh = {c: list(dict.fromkeys(n.id for n, _, _ in wrote[c])) for c in writes[k]}
        below = box_row(names, fresh, k + 1)
        for c in writes[k]:
            for n, cn, m in wrote[c]:
                edges.append(_edge(f"{n.id}.{cn}", below[c], c, memlet_text(m), "write", top.id,
                                   packing(g, n, cn, c)))  # fmt: skip
        above = below
    return {"rows": rows, "boxes": boxes, "edges": edges}


def _edge(
    src: str, dst: str, data: str, text: str, side: str, stop: str | None = None,
    pack: str | None = None,
) -> dict[str, Any]:  # fmt: skip
    """One memlet; ``stop`` is the top-level node whose box the drawn edge ends at, ``pack``
    the port's packing when it meets an accelerated node."""
    out = {"id": f"{src}->{dst}", "from": src, "to": dst, "data": data, "text": text,
           "dir": side, "stop": stop}  # fmt: skip
    if pack is not None:
        out["packing"] = pack
    return out
