"""The SNAX-DFG: a workload graph as plain data, one ``.snaxdfg`` JSON file (DFG1, D71, D77).

A graph borrows SDFG's concepts, not its format:

    {
     "name":       the kernel
     "symbols":    {name: int or null}      null as imported, set by a recipe
     "containers": {name: {shape, dtype, transient, attrs}}
     "body":       [node, ...]              in execution order
    }

A node is ``id``, ``kind``, ``inputs``, ``outputs``, ``attrs`` and, for a
kind that has one, ``body`` (kinds.py). ``inputs`` and ``outputs`` map a
connector name to a memlet ``{"data": container, "subset": [dim, ...]}``,
one dimension per container dimension (subset.py). The graph is a tree:
scopes hold ordered bodies, memlets sit on the connectors of the nodes that
use the data, and there are no map entry/exit nodes, access nodes or outer
memlets, which are derived (D77).

**Names.** Every name in an expression is bound by exactly one thing: a
symbol, or the variable of an enclosing map. Container shapes use symbols
only. Node ids are unique in the graph and are identifiers, since derived
task names are built from them (D75). Symbols, containers and variables
never share a name.

**Stored form.** Every field is written; missing keys take defaults
(``inputs``, ``outputs``, ``attrs`` and a scope's ``body`` empty,
``transient`` false) and unknown keys are errors (D26, D41). ``body`` is
written exactly for kinds that have one. Expressions are kept canonical
(``expr.canonical``, subset.py), so a transform's output and a hand-written
file diff cleanly. A graph is validated when it is made, from a file or in
Python.

**Not checked here.** An accelerated node's connectors and params against
its BRM: that is ``bind``'s job (SBX1) and the reference executor's (REF1),
which load the BRM.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from snax_forge import expr
from snax_forge.expr import ExprError, Value
from snax_forge.snax_model.config import check_keys, plain, to_json

from .kinds import KINDS, DfgError, Scope, _wrap, ident, kind_of
from .subset import canonical_dim, dim_names, parse_dim


def _keys(d: Any, required: tuple[str, ...], optional: tuple[str, ...], what: str) -> None:
    if not isinstance(d, Mapping):
        raise DfgError(f"{what}: must be an object, got {d!r}")
    try:
        check_keys(d, [*required, *optional], what)
    except ValueError as e:
        raise DfgError(str(e)) from None
    missing = [k for k in required if k not in d]
    if missing:
        raise DfgError(f"{what}: missing key {missing[0]!r}")


def _namespaced(key: Any) -> bool:
    """``ns.name``: an attr no kind owns, passed through untouched (D19)."""
    if not isinstance(key, str) or key.count(".") < 1:
        return False
    return all(p.isidentifier() for p in key.split("."))


# =============================================================================
# Containers and memlets
# =============================================================================


@dataclass
class Container:
    """A data container: logical shape and element type. Its layout is the memory plan's (D74)."""

    shape: list[Value]
    dtype: str
    transient: bool = False
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.shape, (list, tuple)) or not self.shape:
            raise DfgError(f"shape: must be a non-empty list, got {self.shape!r}")
        self.shape = [
            _wrap(lambda s=s: (expr.check(s, "shape"), expr.canonical(s))[1]) for s in self.shape
        ]
        self.attrs = plain(self.attrs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "transient": self.transient,
            "attrs": plain(self.attrs),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Container:
        _keys(d, ("shape", "dtype"), ("transient", "attrs"), what)
        try:
            return cls(list(d["shape"]), d["dtype"], d.get("transient", False), d.get("attrs", {}))
        except DfgError as e:
            raise DfgError(f"{what}.{e}") from None


@dataclass
class Memlet:
    """Data movement on one connector: which container and which subset of it."""

    data: str
    subset: list[Value]

    def __post_init__(self) -> None:
        if not isinstance(self.subset, (list, tuple)) or not self.subset:
            raise DfgError(f"subset: must be a non-empty list, got {self.subset!r}")
        self.subset = [_wrap(lambda d=d: canonical_dim(d, "subset")) for d in self.subset]

    def to_dict(self) -> dict[str, Any]:
        return {"data": self.data, "subset": list(self.subset)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Memlet:
        _keys(d, ("data", "subset"), (), what)
        try:
            return cls(d["data"], list(d["subset"]))
        except (DfgError, TypeError) as e:
            raise DfgError(f"{what}: {e}") from None


def _memlets(d: Any, what: str) -> dict[str, Memlet]:
    if not isinstance(d, Mapping):
        raise DfgError(f"{what}: must be an object")
    return {c: Memlet.from_dict(m, f"{what}.{c}") for c, m in d.items()}


# =============================================================================
# Nodes
# =============================================================================


@dataclass
class Node:
    """One element of the graph: the core schema of D19 plus a scope's body."""

    id: str
    kind: str
    inputs: dict[str, Memlet] = field(default_factory=dict)
    outputs: dict[str, Memlet] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
    body: list[Node] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "kind": self.kind,
            "inputs": {c: m.to_dict() for c, m in self.inputs.items()},
            "outputs": {c: m.to_dict() for c, m in self.outputs.items()},
            "attrs": plain(self.attrs),
        }
        if KINDS[self.kind].body:
            d["body"] = [n.to_dict() for n in self.body or []]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], what: str) -> Node:
        _keys(d, ("id", "kind"), ("inputs", "outputs", "attrs", "body"), what)
        what = f"node {d['id']!r}"
        kind = kind_of(d["kind"], what)
        body = d.get("body")
        if body is not None and not kind.body:
            raise DfgError(f"{what}: a {d['kind']!r} has no body")
        if kind.body:
            if not isinstance(body if body is not None else [], (list, tuple)):
                raise DfgError(f"{what}.body: must be a list")
            body = [cls.from_dict(n, f"{what}.body[{i}]") for i, n in enumerate(body or [])]
        attrs = d.get("attrs", {})
        if not isinstance(attrs, Mapping):
            raise DfgError(f"{what}.attrs: must be an object")
        return cls(
            d["id"],
            d["kind"],
            _memlets(d.get("inputs", {}), f"{what}.inputs"),
            _memlets(d.get("outputs", {}), f"{what}.outputs"),
            dict(attrs),
            body,
        )


# =============================================================================
# The graph
# =============================================================================


@dataclass
class Graph:
    """A ``.snaxdfg``: symbols, containers and an ordered body of nodes."""

    name: str
    symbols: dict[str, int | None] = field(default_factory=dict)
    containers: dict[str, Container] = field(default_factory=dict)
    body: list[Node] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate(self)

    # --- access ---

    def walk(self) -> Iterator[tuple[Node, tuple[Node, ...]]]:
        """Every node in execution order (depth first), with its enclosing nodes."""

        def go(
            nodes: list[Node], outer: tuple[Node, ...]
        ) -> Iterator[tuple[Node, tuple[Node, ...]]]:
            for n in nodes:
                yield n, outer
                if n.body:
                    yield from go(n.body, (*outer, n))

        yield from go(self.body, ())

    def node(self, node_id: str) -> Node:
        """The node with id ``node_id``."""
        for n, _ in self.walk():
            if n.id == node_id:
                return n
        raise DfgError(f"{self.name}: no node {node_id!r}")

    # --- files ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "symbols": dict(self.symbols),
            "containers": {k: c.to_dict() for k, c in self.containers.items()},
            "body": [n.to_dict() for n in self.body],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Graph:
        _keys(d, ("name",), ("symbols", "containers", "body"), "graph")
        name = d["name"]
        containers = d.get("containers", {})
        if not isinstance(containers, Mapping):
            raise DfgError(f"{name}.containers: must be an object")
        body = d.get("body", [])
        if not isinstance(body, (list, tuple)):
            raise DfgError(f"{name}.body: must be a list")
        symbols = d.get("symbols", {})
        if not isinstance(symbols, Mapping):
            raise DfgError(f"{name}.symbols: must be an object")
        return cls(
            name,
            dict(symbols),
            {k: Container.from_dict(c, f"container {k!r}") for k, c in containers.items()},
            [Node.from_dict(n, f"{name}.body[{i}]") for i, n in enumerate(body)],
        )

    def to_json(self) -> str:
        return to_json(self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> Graph:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())


# =============================================================================
# Validation (and the stored form of kind-owned attrs)
# =============================================================================


def _validate(g: Graph) -> None:
    if not isinstance(g.name, str) or not g.name:
        raise DfgError(f"graph name: must be a non-empty string, got {g.name!r}")
    for s, v in g.symbols.items():
        ident(s, f"{g.name}.symbols")
        if v is not None and type(v) is not int:
            raise DfgError(f"{g.name}.symbols.{s}: must be an int or null, got {v!r}")
    for c, cont in g.containers.items():
        _check_container(g, c, cont)
    scope = Scope(frozenset(g.symbols), g.containers)
    ids: set[str] = set()
    instances: dict[str, tuple[Any, ...]] = {}
    _check_body(g.body, scope, ids, instances)


def _check_container(g: Graph, name: str, c: Container) -> None:
    what = f"container {name!r}"
    ident(name, what)
    if name in g.symbols:
        raise DfgError(f"{what}: {name!r} is also a symbol")
    try:
        ok = isinstance(c.dtype, str) and np.dtype(c.dtype).name == c.dtype
    except TypeError:
        ok = False
    if not ok:
        raise DfgError(f"{what}.dtype: {c.dtype!r} is not a NumPy dtype name (e.g. 'int64')")
    if type(c.transient) is not bool:
        raise DfgError(f"{what}.transient: must be true or false")
    for s in c.shape:
        unbound = sorted(expr.names(s) - set(g.symbols))
        if unbound:
            raise DfgError(f"{what}.shape: {unbound[0]!r} is not a symbol")
        if type(s) is int and s < 1:
            raise DfgError(f"{what}.shape: {s} is not a positive size")
    for k in c.attrs:
        if not _namespaced(k):
            raise DfgError(f"{what}.attrs: {k!r} must be namespaced (mem.*, hw.*, user.*, ...)")


def _check_body(
    body: list[Node], scope: Scope, ids: set[str], instances: dict[str, tuple[Any, ...]]
) -> None:
    for node in body:
        what = f"node {node.id!r}"
        ident(node.id, what)
        if node.id in ids:
            raise DfgError(f"{what}: the id is used twice")
        ids.add(node.id)
        kind = kind_of(node.kind, what)
        _check_attrs(node, kind, what)
        _check_memlets(node, scope, what)
        if kind.check:
            kind.check(node, scope, what)
        if node.kind == "accelerated":
            _check_instance(node, instances, what)
            _check_replaced(node, scope, what)
        if kind.body:
            node.body = list(node.body or [])
            inner = scope.inner(kind.binds(node) if kind.binds else [])
            _check_body(node.body, inner, ids, instances)
        elif node.body is not None:
            raise DfgError(f"{what}: a {node.kind!r} has no body")


def _check_attrs(node: Node, kind: Any, what: str) -> None:
    if not isinstance(node.attrs, Mapping):
        raise DfgError(f"{what}.attrs: must be an object")
    for k in node.attrs:
        if not kind.owns(k) and not _namespaced(k):
            raise DfgError(
                f"{what}.attrs: {k!r} is not an attr of a {node.kind!r} and not namespaced"
            )
    missing = [k for k in kind.required if k not in node.attrs]
    if missing:
        raise DfgError(f"{what}.attrs: missing {missing[0]!r}")
    # kind-owned attrs first, in registration order, then the namespaced ones as given
    own = [*kind.required, *kind.defaults]
    attrs = {k: node.attrs.get(k, kind.defaults.get(k)) for k in own}
    attrs.update({k: v for k, v in node.attrs.items() if k not in attrs})
    attrs = plain(attrs)
    node.attrs = kind.normalize(attrs, f"{what}.attrs") if kind.normalize else attrs


def _check_memlets(node: Node, scope: Scope, what: str) -> None:
    if set(node.inputs) & set(node.outputs):
        raise DfgError(
            f"{what}: {min(set(node.inputs) & set(node.outputs))!r} "
            "is both an input and an output connector"
        )
    for side, conns in (("inputs", node.inputs), ("outputs", node.outputs)):
        for c, m in conns.items():
            w = f"{what}.{side}.{c}"
            ident(c, w)
            if m.data not in scope.containers:
                raise DfgError(f"{w}: unknown container {m.data!r}")
            rank = len(scope.containers[m.data].shape)
            if len(m.subset) != rank:
                raise DfgError(f"{w}: {len(m.subset)} dimensions for {m.data!r}, which has {rank}")
            for d in m.subset:
                unbound = sorted(dim_names(d) - scope.names())
                if unbound:
                    raise DfgError(f"{w}: {unbound[0]!r} is not a symbol or an enclosing variable")
                try:
                    parse_dim(d, w)
                except ExprError as e:
                    raise DfgError(str(e)) from None


def _check_replaced(node: Node, scope: Scope, what: str) -> None:
    """The subtree an accelerated node replaced: a valid node in the node's own scope (D82).

    It is history, not part of the graph: its ids may repeat live ones (the
    tasklet ``bind`` replaced has the accelerated node's id) and it is never
    run. It is kept in its stored form.
    """
    r = node.attrs.get("replaced")
    if r is None:
        return
    w = f"{what}.attrs.replaced"
    try:
        old = Node.from_dict(r, w)
        _check_body([old], scope, set(), {})
    except DfgError as e:
        raise DfgError(f"{w}: {e}") from None
    node.attrs["replaced"] = old.to_dict()


def _check_instance(node: Node, instances: dict[str, tuple[Any, ...]], what: str) -> None:
    """Two accelerated nodes on one instance agree on its BRM, implementation and params."""
    a = node.attrs
    key = (a["brm"], a["implementation"], plain(a["params"]))
    seen = instances.setdefault(a["instance"], key)
    if seen != key:
        raise DfgError(f"{what}: instance {a['instance']!r} is bound elsewhere as {seen}")
