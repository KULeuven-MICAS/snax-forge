"""Loop variables named after their dimension (B2 R4b): one rule, applied after every change.

Each map carries two namespaced attrs: ``loop.dim``, its dimension's base name, and
``loop.piece``, its number among the pieces of that dimension, 1 outer to inner. The pieces of a
dimension are the maps of one ``loop.dim`` nested in one another (a map and the maps of its
dimension around it and inside it): dot's two maps over N are siblings, so each is a dimension
of its own, both ``n``. A variable's name is derived, never set by hand (``var_name``): the
base, then ``_<piece>`` when the dimension has two or more pieces, then the tag's suffix
(``_t`` temporal, ``_s`` spatial, ``_tile`` tile, none untagged). So ``n``; split, ``n_1``,
``n_2``; split again, ``n_1``, ``n_2``, ``n_3``; tagged, ``n_1_t``, ``n_2_t``, ``n_3_s``; an
unsplit tagged map ``n_t``.

``rename(g)`` gives every map that carries ``loop.dim`` its derived name, in place, and
rewrites every memlet subset, map range and split record (``loop.split``'s ``range`` and
``valid``, its ``prior``) that names the old one, in the map's body, a bound unit's
``replaced`` subtree included; the renames are simultaneous, so pieces may trade names
(``n_3`` to ``n_4`` while ``n_2`` becomes ``n_3``). A map without ``loop.dim`` (a graph
written by hand) keeps its name until a transform gives it the attrs (``ensure``: its
variable as its base, piece 1). The sandbox's transforms that make, merge or tag pieces
(sandbox/transforms.py, compose.py) set the pieces and call ``rename``; the importer names
the imported maps (``import_names``).

**Import** (``import_names``). A map over ``0:X``, X one symbol, gets the base ``lower(X)``:
rowsum's ``m`` over M and ``n`` over N. It keeps the importer's name (``i``, ``j``, ...)
when its range is anything else (``0:i``, ``1:N - 1``), or when ``lower(X)`` is a container,
a symbol, the base of an enclosing map or the variable of a map inside it: the second map over
N in one nest (mvt's ``A[n, j]``). That name is then its base.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from snax_forge import expr

from .graph import Graph, Node
from .subset import dim_names, format_dim, parse_dim

DIM = "loop.dim"
PIECE = "loop.piece"
# A tag's suffix on the variable (loop.kind), none untagged
SUFFIX = {None: "", "temporal": "_t", "spatial": "_s", "tile": "_tile"}


def var_name(dim: str, piece: int, pieces: int, kind: str | None) -> str:
    """The variable of piece ``piece`` of ``pieces`` of dimension ``dim``, tagged ``kind``."""
    return f"{dim}{f'_{piece}' if pieces > 1 else ''}{SUFFIX[kind]}"


def ensure(node: Node) -> None:
    """Give a map without ``loop.dim`` its variable as its base and piece 1, right after its
    range (where the importer writes them)."""
    if DIM in node.attrs:
        return
    attrs: dict[str, Any] = {}
    for k, v in node.attrs.items():
        attrs[k] = v
        if k == "range":
            attrs[DIM], attrs[PIECE] = node.attrs["var"], 1
    node.attrs = attrs


def _opened(body: list[Node]) -> Iterator[tuple[Node, Node]]:
    """Each bound unit under ``body`` with its ``replaced`` subtree as a node, outer first."""
    for n in body:
        if n.kind == "accelerated" and n.attrs.get("replaced"):
            r = Node.from_dict(n.attrs["replaced"], n.id)
            yield n, r
            yield from _opened([r])
        if n.body:
            yield from _opened(n.body)


def _maps(body: list[Node], outer: tuple[Node, ...], opened: Mapping[int, Node],
          out: list[tuple[Node, tuple[Node, ...]]]) -> None:  # fmt: skip
    """Every map under ``body`` with the maps around it, a replaced subtree in its unit's
    place."""
    for n in body:
        if id(n) in opened:
            _maps([opened[id(n)]], outer, opened, out)
        if n.kind == "map":
            out.append((n, outer))
            _maps(n.body or [], (*outer, n), opened, out)
        elif n.body:
            _maps(n.body, outer, opened, out)


def _groups(maps: list[tuple[Node, tuple[Node, ...]]]) -> dict[int, list[Node]]:
    """id(map) -> the pieces of its dimension (the maps of its ``loop.dim`` nested in one
    another), for every map with ``loop.dim``."""
    group: dict[int, list[Node]] = {}
    for n, outer in maps:
        if DIM not in n.attrs:
            continue
        around = [o for o in outer if o.attrs.get(DIM) == n.attrs[DIM]]
        g = group[id(around[-1])] if around else []
        g.append(n)
        group[id(n)] = g
    return group


def renumber(g: Graph, node: Node, above: int, by: int) -> None:
    """Add ``by`` to the piece of every other piece of ``node``'s dimension numbered above
    ``above``, bound units' replaced subtrees included; in place (``split`` makes room below a
    piece, ``join_map`` closes the gap)."""
    pairs = list(_opened(g.body))
    maps: list[tuple[Node, tuple[Node, ...]]] = []
    _maps(g.body, (), {id(n): r for n, r in pairs}, maps)
    for m in _groups(maps).get(id(node), []):
        if m is not node and m.attrs[PIECE] > above:
            m.attrs[PIECE] += by
    for n, r in reversed(pairs):
        n.attrs["replaced"] = r.to_dict()


def dims(d: Any, fn: Callable[[Any], Any]) -> Any:
    """``fn`` on each part of a dimension (index or range), formatted again."""
    return format_dim(tuple(fn(p) for p in parse_dim(d, "dimension")))


def split_record(split: Mapping[str, Any], fn: Callable[[Any], Any]) -> dict[str, Any]:
    """A ``loop.split`` record with ``fn`` on its ``range`` and ``valid``, its ``prior`` too."""
    out = {**split, "range": fn(split["range"])}
    if "valid" in split:
        out["valid"] = fn(split["valid"])
    if "prior" in split:
        out["prior"] = split_record(split["prior"], fn)
    return out


def _substitute(d: Any, env: Mapping[str, str]) -> Any:
    if not env or not (dim_names(d) & env.keys()):
        return d
    return dims(d, lambda p: expr.substitute(p, env))


def rename(g: Graph) -> None:
    """Every map with ``loop.dim`` named by ``var_name``, every use of its old name rewritten
    (module doc); in place."""
    pairs = list(_opened(g.body))
    opened = {id(n): r for n, r in pairs}
    maps: list[tuple[Node, tuple[Node, ...]]] = []
    _maps(g.body, (), opened, maps)
    groups = _groups(maps)
    names = {}
    for n, _ in maps:
        if id(n) in groups:
            group = groups[id(n)]
            names[id(n)] = var_name(n.attrs[DIM], n.attrs[PIECE], len(group),
                                    n.attrs.get("loop.kind"))  # fmt: skip
    changed = any(names.get(id(n), n.attrs["var"]) != n.attrs["var"] for n, _ in maps)
    if not changed:
        return

    def go(body: list[Node], env: Mapping[str, str]) -> None:
        for n in body:
            for m in (*n.inputs.values(), *n.outputs.values()):
                m.subset = [_substitute(d, env) for d in m.subset]
            if id(n) in opened:
                go([opened[id(n)]], env)
            if n.kind == "map":
                n.attrs["range"] = _substitute(n.attrs["range"], env)
                if n.attrs.get("loop.split"):
                    n.attrs["loop.split"] = split_record(n.attrs["loop.split"],
                                                         lambda v: _substitute(v, env))  # fmt: skip
                inner = env
                new = names.get(id(n), n.attrs["var"])
                if new != n.attrs["var"]:
                    inner = {**env, n.attrs["var"]: new}
                    n.attrs["var"] = new
                go(n.body or [], inner)
            elif n.body:
                go(n.body, env)

    go(g.body, {})
    for n, r in reversed(pairs):  # inner subtrees first, so an outer one holds them
        n.attrs["replaced"] = r.to_dict()


def import_names(g: Graph) -> None:
    """The imported maps' bases and pieces (module doc), every variable renamed; in place."""
    taken = set(g.symbols) | set(g.containers)
    maps: list[tuple[Node, tuple[Node, ...]]] = []
    _maps(g.body, (), {}, maps)
    inside: dict[int, set[str]] = {}
    for n, outer in maps:
        for o in outer:
            inside.setdefault(id(o), set()).add(n.attrs["var"])
    for n, outer in maps:  # outer first: an enclosing map's base is set before its body's
        base = n.attrs["var"]
        b, e, s = parse_dim(n.attrs["range"], n.id)
        if (b, s) == (0, 1) and isinstance(e, str) and e in g.symbols:
            low = e.lower()
            if (low not in taken and low not in inside.get(id(n), set())
                    and all(o.attrs[DIM] != low for o in outer)):  # fmt: skip
                base = low
        attrs = {k: v for k, v in n.attrs.items() if k in ("var", "range")}
        n.attrs = {**attrs, DIM: base, PIECE: 1,
                   **{k: v for k, v in n.attrs.items() if k not in attrs}}  # fmt: skip
    rename(g)
