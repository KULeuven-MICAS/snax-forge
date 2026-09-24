"""Node kinds of the SNAX-DFG (D19, D77): registered entries, never subclasses.

Every node has the same core fields (graph.py): ``id``, ``kind``, ``inputs``,
``outputs``, ``attrs`` and, for a kind that has one, ``body``. What a kind
adds is registered here with ``register_kind``:

* the attrs it owns: ``required`` names and ``defaults`` for the optional
  ones. Kind-owned attrs have no namespace and every one is written. An attr
  without a namespace that the kind does not own is an error; a namespaced
  attr (``loop.kind``, ``mem.*``, ``hw.*``, ``user.*``) is passed through
  untouched, and whoever knows the namespace checks it;
* ``body``: whether the node holds an ordered body of nodes (a scope);
* ``binds``: the variables the node binds for its body (a map's ``var``);
* ``normalize(attrs, what)``: the attrs in their stored form (canonical
  expressions), run before ``check``;
* ``check(node, scope, what)``: everything else about one node, given the
  names its scope binds. The memlet rules every kind shares (known
  container, one dimension per container dimension, bound names) are
  checked in graph.py before it.

Built-in kinds (DFG1):

    map          one loop variable over a range, parallel iterations; a body.
                 ``loop.kind`` (tile, temporal, spatial) says how SNAX-SANDBOX
                 mapped it (D73); absent as imported. No connectors: the
                 memlets are its body's, and the outer ones SDFG draws on a map
                 are derived by propagation.
    tasklet      ``code``: one assignment per output connector over the input
                 connectors, in the expression grammar. One element per
                 connector: its memlets are indices, never ranges.
    accelerated  a bound BRM instance (``instance``, ``brm``, ``implementation``,
                 design ``params``) doing one beat: its connectors are the BRM's
                 ports, its memlets give each port's lanes as a range. It sits
                 inside the temporal maps it runs over. A body is allowed for a
                 nested block (D71); a leaf block has an empty one.

``loop`` (a sequential loop) and ``branch`` come with the kernels that need
them (open item 36).
"""

from __future__ import annotations

import ast
import keyword
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from snax_forge import expr
from snax_forge.expr import ExprError

from .subset import canonical_dim, is_range, parse_dim

if TYPE_CHECKING:
    from .graph import Container, Node

LOOP_KINDS = ("tile", "temporal", "spatial")  # values of loop.kind (D73, D77)


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

    def owns(self, key: str) -> bool:
        return key in self.required or key in self.defaults


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
) -> None:
    """Make ``"kind": name`` usable in a ``.snaxdfg`` (principle 6)."""
    if name in KINDS:
        raise ValueError(f"node kind {name!r} already registered")
    KINDS[name] = Kind(required, dict(defaults or {}), body, binds, normalize, check)


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


def _statements(code: Any, what: str) -> list[ast.Assign]:
    if not isinstance(code, str):
        raise DfgError(f"{what}: must be a string, got {code!r}")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise DfgError(f"{what}: cannot parse {code!r}") from e
    out = []
    for st in tree.body:
        if not (
            isinstance(st, ast.Assign)
            and len(st.targets) == 1
            and isinstance(st.targets[0], ast.Name)
        ):
            raise DfgError(f"{what}: {ast.unparse(st)!r} is not 'output = expression'")
        _wrap(lambda st=st: expr.check(ast.unparse(st.value), what))
        out.append(st)
    if not out:
        raise DfgError(f"{what}: no statement")
    return out


def _normalize_tasklet(attrs: dict[str, Any], what: str) -> dict[str, Any]:
    attrs["code"] = "\n".join(ast.unparse(s) for s in _statements(attrs["code"], f"{what}.code"))
    return attrs


def _check_tasklet(node: Node, scope: Scope, what: str) -> None:
    stmts = _statements(node.attrs["code"], f"{what}.attrs.code")
    targets = [s.targets[0].id for s in stmts]
    for t in targets:
        if t not in node.outputs:
            raise DfgError(f"{what}.attrs.code: assigns {t!r}, which is not an output connector")
    for o in node.outputs:
        if targets.count(o) != 1:
            raise DfgError(f"{what}.attrs.code: output {o!r} must be assigned exactly once")
    for s in stmts:
        for name in sorted(expr.names(ast.unparse(s.value))):
            if name not in node.inputs:
                raise DfgError(f"{what}.attrs.code: {name!r} is not an input connector")
    for side, conns in (("inputs", node.inputs), ("outputs", node.outputs)):
        for c, m in conns.items():
            for d in m.subset:
                if is_range(d):
                    raise DfgError(f"{what}.{side}.{c}: a tasklet takes one element, got {d!r}")


register_kind("tasklet", required=("code",), normalize=_normalize_tasklet, check=_check_tasklet)


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


register_kind(
    "accelerated",
    required=("instance", "brm", "implementation"),
    defaults={"params": {}},
    body=True,
    check=_check_accelerated,
)
