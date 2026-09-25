"""What the DFG viewer asks for, as plain Python functions (VIS5, D76, D81).

The server (server.py) only turns these results into JSON; the tests call
them directly, and the drawing is checked by eye (D55).

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
an input connector, or from an output connector to a container box. An
edge between a container and a connector inside a top-level node is drawn
only up to that node's outer box, above or below the connector it belongs
to (``stop``, D82), so it never crosses the text inside; an edge between
two connectors inside one node is drawn in full. The
nodes are the graph's own tree, with what a panel shows about each one
already worked out: a map's iteration count when its range evaluates with
the bound symbols, a tasklet's code, an accelerated node's instance line.
Positions are the browser's business: the boxes are HTML laid out by CSS,
and the edges an SVG layer the viewer draws over them (static/dfg.js).
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snax_forge import expr
from snax_forge.dfg import DfgError, Graph, Memlet, Node, parse_dim
from snax_forge.expr import ExprError

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
    names: list[str] = []
    for f in files:
        name, i = f.stem, 1
        while name in names:
            i += 1
            name = f"{f.stem}-{i}"
        names.append(name)
    return names


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


class GraphSet:
    """The files the viewer was given, read once and again on ``reload``."""

    def __init__(self, paths: Sequence[str | Path]):
        self.paths = [Path(p) for p in paths]
        self._lock = threading.Lock()
        self.entries: dict[str, Entry] = {}
        self.reload()

    def reload(self) -> list[str]:
        files = expand(self.paths)
        entries = {n: load_entry(n, f) for n, f in zip(entry_names(files), files, strict=True)}
        with self._lock:
            self.entries = entries
        return list(entries)

    def get(self, name: str) -> Entry | None:
        with self._lock:
            return self.entries.get(name)

    def summaries(self) -> list[dict[str, Any]]:
        with self._lock:
            return [entry_summary(e) for e in self.entries.values()]


def entry_summary(e: Entry) -> dict[str, Any]:
    """One entry of /api/graphs."""
    return {
        "name": e.name,
        "path": str(e.path),
        "graph": e.graph.name if e.graph else None,
        "error": e.error,
    }


# =============================================================================
# The view of one graph
# =============================================================================


def graph_view(e: Entry) -> dict[str, Any]:
    """/api/graph/<name>: the rows, nodes and edges a panel draws (module doc)."""
    out: dict[str, Any] = entry_summary(e)
    if e.graph is None:
        return out
    g = e.graph
    out.update(
        symbols=dict(g.symbols),
        containers={k: _container(g, k) for k in g.containers},
    )
    out.update(_layout(g))
    return out


def _container(g: Graph, name: str) -> dict[str, Any]:
    c = g.containers[name]
    env = {k: v for k, v in g.symbols.items() if v is not None}
    try:
        sizes = [expr.evaluate(s, env) for s in c.shape]
    except ExprError:
        sizes = None
    return {
        "name": name,
        "dtype": c.dtype,
        "shape": list(c.shape),
        "sizes": sizes,
        "transient": c.transient,
        "attrs": dict(c.attrs),
    }


def memlet_text(m: Memlet) -> str:
    return f"{m.data}[{', '.join(str(d) for d in m.subset)}]"


def _iterations(g: Graph, rng: str) -> int | None:
    env = {k: v for k, v in g.symbols.items() if v is not None}
    try:
        begin, end, step = (expr.evaluate(p, env) for p in parse_dim(rng, "range"))
    except (ExprError, DfgError):
        return None
    return max(0, -(-(end - begin) // step))


def _node(g: Graph, n: Node, top: str) -> dict[str, Any]:
    """A node as a panel shows it; ``top`` is the id of its top-level node."""
    a = n.attrs
    d: dict[str, Any] = {
        "id": n.id,
        "kind": n.kind,
        "top": top,
        "attrs": dict(a),
        "inputs": [_conn(c, m) for c, m in n.inputs.items()],
        "outputs": [_conn(c, m) for c, m in n.outputs.items()],
    }
    if n.kind == "map":
        d["title"] = f"{a['var']} in {a['range']}"
        d["loop_kind"] = a.get("loop.kind")
        d["iterations"] = _iterations(g, a["range"])
        d["lines"] = []
    elif n.kind == "tasklet":
        d["title"] = "tasklet"
        d["lines"] = a["code"].split("\n")
    elif n.kind == "accelerated":
        params = ", ".join(f"{k} = {v}" for k, v in a["params"].items())
        d["title"] = f"{a['instance']} = {a['brm']}"
        d["heading"] = [f"impl = {a['implementation']}"] + ([params] if params else [])
        d["lines"] = a["code"].split("\n")
        d["replaced"] = _replaced_text(a.get("replaced"))
    else:
        d["title"] = n.kind
        d["lines"] = [f"{k} = {v}" for k, v in a.items() if "." not in k]
    if n.body is not None:
        d["body"] = [_node(g, c, top) for c in n.body]
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
    return {"connector": c, "data": m.data, "subset": list(m.subset), "text": memlet_text(m)}


def _uses(n: Node) -> list[tuple[str, Node, str, Memlet]]:
    """Every memlet under ``n`` in execution order: (``read`` / ``write``, node, connector, memlet)."""
    out: list[tuple[str, Node, str, Memlet]] = []
    out += [("read", n, c, m) for c, m in n.inputs.items()]
    for child in n.body or []:
        out += _uses(child)
    out += [("write", n, c, m) for c, m in n.outputs.items()]
    return out


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
        rows.append({"kind": "node", "node": _node(g, top, top.id)})
        writers: dict[str, list[str]] = {}  # container -> writer connectors inside this node
        for side, n, c, m in _uses(top):
            port, text = f"{n.id}.{c}", memlet_text(m)
            if side == "write":
                writers.setdefault(m.data, []).append(port)
            elif m.data in writers:  # written earlier inside this node: straight from the writer
                edges.extend(_edge(w, port, m.data, text, "read") for w in writers[m.data])
            else:
                edges.append(_edge(above[m.data], port, m.data, text, "read", top.id))
        nxt = reads[k + 1] if k + 1 < len(tops) else []
        names = nxt + [c for c in writes[k] if c not in nxt]
        fresh = {c: list(dict.fromkeys(n.id for n, _, _ in _writes(top, c))) for c in writes[k]}
        below = box_row(names, fresh, k + 1)
        for c in writes[k]:
            for n, cn, m in _writes(top, c):
                edges.append(_edge(f"{n.id}.{cn}", below[c], c, memlet_text(m), "write", top.id))
        above = below
    return {"rows": rows, "boxes": boxes, "edges": edges}


def _writes(top: Node, c: str) -> list[tuple[Node, str, Memlet]]:
    return [(n, cn, m) for side, n, cn, m in _uses(top) if side == "write" and m.data == c]


def _edge(
    src: str, dst: str, data: str, text: str, side: str, stop: str | None = None
) -> dict[str, Any]:
    """One memlet; ``stop`` is the top-level node whose box the drawn edge ends at (D82)."""
    return {"from": src, "to": dst, "data": data, "text": text, "dir": side, "stop": stop}
