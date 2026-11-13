"""Node kinds of the SNAX-DFG: registered entries, never subclasses.

Every node has the core fields of graph.py; what a kind adds is a ``Kind``, registered with
``register_kind`` and looked up with ``kind_of``: the attrs it owns (``required``, ``defaults``,
``optional`` ones written only when set), whether it holds a ``body``, whether its outputs may
carry a ``reduce``, the variables it ``binds`` for its body, ``normalize`` (attrs to their
stored form) and ``check`` (everything else about one node, given its ``Scope``). Kind-owned
attrs have no namespace; an unknown one without a namespace is an error, and a namespaced one
(``loop.*``, ``mem.*``, ``hw.*``, ``user.*``) passes through for its owner to check.

The built-in kinds, ``map``, ``tasklet``, ``accelerated`` and ``link``, and their attrs are
CONTRACTS, "SNAX-DFG". ``loop`` and ``branch`` come with the kernels that need them.
"""

from __future__ import annotations

import keyword
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from snax_forge import expr
from snax_forge.expr import ExprError

from .subset import canonical_dim, is_range, parse_dim

if TYPE_CHECKING:
    from .graph import Container, Node

LOOP_KINDS = ("tile", "temporal", "spatial")  # values of loop.kind


class DfgError(ValueError):
    """A SNAX-DFG that is malformed or inconsistent."""


def ident(name: Any, what: str) -> None:
    """A name usable in expressions and derived names: an identifier, not a keyword."""
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise DfgError(f"{what}: {name!r} is not a valid name")


@dataclass(frozen=True)
class Scope:
    """The names bound where a node sits: symbols, containers and enclosing variables."""

    symbols: frozenset[str]
    containers: Mapping[str, Container]
    variables: tuple[str, ...] = ()

    def names(self) -> set[str]:
        return set(self.symbols) | set(self.variables)

    def inner(self, variables: list[str]) -> Scope:
        return Scope(self.symbols, self.containers, (*self.variables, *variables))

    def uses(self, v: Any, what: str) -> None:
        """Every name in value ``v`` is bound here."""
        unbound = sorted(expr.names(v) - self.names())
        if unbound:
            raise DfgError(f"{what}: {unbound[0]!r} is not a symbol or an enclosing variable")


Normalize = Callable[[dict[str, Any], str], dict[str, Any]]
Check = Callable[["Node", Scope, str], None]


@dataclass(frozen=True)
class Kind:
    required: tuple[str, ...] = ()
    defaults: Mapping[str, Any] = field(default_factory=dict)
    body: bool = False
    binds: Callable[[Node], list[str]] | None = None
    normalize: Normalize | None = None
    check: Check | None = None
    reduce: bool = False
    optional: tuple[str, ...] = ()

    def owns(self, key: str) -> bool:
        return key in self.required or key in self.defaults or key in self.optional


KINDS: dict[str, Kind] = {}


def register_kind(
    name: str,
    *,
    required: tuple[str, ...] = (),
    defaults: Mapping[str, Any] | None = None,
    body: bool = False,
    binds: Callable[[Node], list[str]] | None = None,
    normalize: Normalize | None = None,
    check: Check | None = None,
    reduce: bool = False,
    optional: tuple[str, ...] = (),
) -> None:
    """Make ``"kind": name`` usable in a ``.snaxdfg`` (principle 6)."""
    if name in KINDS:
        raise ValueError(f"node kind {name!r} already registered")
    KINDS[name] = Kind(
        required, dict(defaults or {}), body, binds, normalize, check, reduce, tuple(optional)
    )


def kind_of(name: Any, what: str) -> Kind:
    if name not in KINDS:
        raise DfgError(f"{what}: unknown kind {name!r} (registered: {sorted(KINDS)})")
    return KINDS[name]


def _wrap(f: Callable[[], Any]) -> Any:
    """Run ``f`` and report an expression error as a DfgError."""
    try:
        return f()
    except ExprError as e:
        raise DfgError(str(e)) from None


# =============================================================================
# map
# =============================================================================


def _normalize_map(attrs: dict[str, Any], what: str) -> dict[str, Any]:
    attrs["range"] = _wrap(lambda: canonical_dim(attrs["range"], f"{what}.range"))
    return attrs


def _check_map(node: Node, scope: Scope, what: str) -> None:
    if node.inputs or node.outputs:
        raise DfgError(f"{what}: a map has no connectors; its memlets are its body's")
    var, rng = node.attrs["var"], node.attrs["range"]
    ident(var, f"{what}.attrs.var")
    if var in scope.symbols or var in scope.containers or var in scope.variables:
        raise DfgError(f"{what}.attrs.var: {var!r} is already a symbol, container or variable")
    if not is_range(rng):
        raise DfgError(f"{what}.attrs.range: {rng!r} is not a range (begin:end or begin:end:step)")
    for part in parse_dim(rng, f"{what}.attrs.range"):
        scope.uses(part, f"{what}.attrs.range")
    kind = node.attrs.get("loop.kind")
    if kind is not None and kind not in LOOP_KINDS:
        raise DfgError(f"{what}.attrs.loop.kind: {kind!r} is not one of {list(LOOP_KINDS)}")
    split = node.attrs.get("loop.split")
    w = f"{what}.attrs.loop.split"
    while split is not None:
        if not isinstance(split, Mapping) or set(split) - {"valid", "prior"} != {"var", "range"}:
            raise DfgError(
                f"{w}: must be {{var, range}} of the map before split_map, its valid count "
                "(valid) when the split left a tail, and the map's record before it (prior) when "
                "it was split before"
            )
        ident(split["var"], f"{w}.var")
        if not is_range(split["range"]):
            raise DfgError(f"{w}.range: {split['range']!r} is not a range")
        for part in parse_dim(split["range"], f"{w}.range"):
            scope.uses(part, f"{w}.range")
        if "valid" in split:
            scope.uses(split["valid"], f"{w}.valid")
        split, w = split.get("prior"), f"{w}.prior"


register_kind(
    "map",
    required=("var", "range"),
    body=True,
    binds=lambda n: [n.attrs["var"]],
    normalize=_normalize_map,
    check=_check_map,
)


# =============================================================================
# tasklet
# =============================================================================


def _normalize_code(attrs: dict[str, Any], what: str) -> dict[str, Any]:
    attrs["code"] = _wrap(lambda: expr.canonical_code(attrs["code"], f"{what}.code"))
    return attrs


def _check_code(node: Node, what: str, slots: Iterable[str] = ()) -> None:
    """One ``output = expression`` per output connector, reading input connectors and
    ``slots`` only."""
    reads = [*node.inputs, *slots]
    _wrap(lambda: expr.check_code(node.attrs["code"], reads, node.outputs, f"{what}.attrs.code"))


def _check_tasklet(node: Node, scope: Scope, what: str) -> None:
    _check_code(node, what)
    for side, conns in (("inputs", node.inputs), ("outputs", node.outputs)):
        for c, m in conns.items():
            for d in m.subset:
                if is_range(d):
                    raise DfgError(f"{what}.{side}.{c}: a tasklet takes one element, got {d!r}")


register_kind(
    "tasklet", required=("code",), normalize=_normalize_code, check=_check_tasklet, reduce=True
)


# =============================================================================
# accelerated
# =============================================================================


def _check_accelerated(node: Node, scope: Scope, what: str) -> None:
    a = node.attrs
    ident(a["instance"], f"{what}.attrs.instance")
    for key in ("brm", "implementation"):
        if not isinstance(a[key], str) or not a[key]:
            raise DfgError(f"{what}.attrs.{key}: must be a name, got {a[key]!r}")
    if not isinstance(a["params"], Mapping):
        raise DfgError(f"{what}.attrs.params: must be an object")
    for k, v in a["params"].items():
        ident(k, f"{what}.attrs.params")
        if type(v) not in (int, str):
            raise DfgError(f"{what}.attrs.params.{k}: must be an int or a string, got {v!r}")
    # the code may read an int design param, a slot the BRM bakes in (simd's k, brm.py's Code),
    # or an input connector that is no port: a slot read from a scalar (axpy's k1: alpha[0])
    _check_code(node, what, [k for k, v in a["params"].items() if type(v) is int])
    if a["replaced"] is not None and not isinstance(a["replaced"], Mapping):
        raise DfgError(f"{what}.attrs.replaced: must be a node or null")
    for key in ("accelerator", "group"):
        if key in a:
            ident(a[key], f"{what}.attrs.{key}")
    if "locals" in a:
        _check_locals(a["locals"], f"{what}.attrs.locals")
    # the replaced subtree itself is checked in graph.py, in this node's scope, with its locals


def _check_locals(locals_: Any, what: str) -> None:
    """``locals``: a list of ``{name, index, container}``, the names distinct, each index an int
    from 0, the container an object (checked as one in graph.py)."""
    if not isinstance(locals_, list):
        raise DfgError(f"{what}: must be a list")
    names = []
    for k, rec in enumerate(locals_):
        w = f"{what}[{k}]"
        if not isinstance(rec, Mapping) or set(rec) != {"name", "index", "container"}:
            raise DfgError(f"{w}: must be an object of name, index and container")
        ident(rec["name"], f"{w}.name")
        if type(rec["index"]) is not int or rec["index"] < 0:
            raise DfgError(f"{w}.index: must be an int from 0, got {rec['index']!r}")
        if not isinstance(rec["container"], Mapping):
            raise DfgError(f"{w}.container: must be an object")
        names.append(rec["name"])
    if len(set(names)) != len(names):
        raise DfgError(f"{what}: a name is given twice")


register_kind(
    "accelerated",
    required=("instance", "brm", "implementation", "code"),
    defaults={"params": {}, "replaced": None},
    optional=("accelerator", "group", "locals"),
    body=True,
    normalize=_normalize_code,
    check=_check_accelerated,
)


# =============================================================================
# link
# =============================================================================


def _check_link(node: Node, scope: Scope, what: str) -> None:
    a = node.attrs
    if node.inputs or node.outputs:
        raise DfgError(f"{what}: a link has no connectors; its units' memlets name the container")
    if a["container"] not in scope.containers or not scope.containers[a["container"]].transient:
        raise DfgError(f"{what}.attrs.container: {a['container']!r} is not a transient container")
    if type(a["depth"]) is not int or a["depth"] < 0:
        raise DfgError(f"{what}.attrs.depth: {a['depth']!r}; 0 is a wire, >= 1 a buffer")
    for key in ("from", "to"):
        ident(a[key], f"{what}.attrs.{key}")


register_kind("link", required=("container", "depth", "from", "to"), check=_check_link)
