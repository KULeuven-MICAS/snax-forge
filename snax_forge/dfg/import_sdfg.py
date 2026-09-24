"""SDFG importer (IMP1, D71, D77): a simplified DaCe SDFG -> a ``.snaxdfg`` graph.

The importer derives and decides nothing (principle 4). It reads the
simplified SDFG that ``pixi run forge <kernel>`` writes (or builds the same
SDFG in-process from the kernel's ``SPEC``) and maps it onto the format of
D77:

    SDFG                              .snaxdfg
    ------------------------------    ---------------------------------------
    free symbols                      symbols, each null
    arrays                            containers: shape, dtype, transient
    one state                         the graph's body, in topological order
    MapEntry / MapExit pair           one map node (var, range); its body the
                                      map's scope, in topological order
    Tasklet                           tasklet node: code, connector memlets
    edges into / out of a tasklet     memlets on its connectors
    access nodes, map connectors,     not stored: derived (D77)
    outer memlets on a map

What DaCe's simplify already did is kept as it is: for vecadd it folded the
transient-plus-copy of ``C[:] = A + B`` into a direct write into ``C``, so
there is nothing left to fold. A copy between containers that simplify left
is an error here.

**Names** (D75, D77). Containers keep the kernel's names (``A``, ``B``,
``C``); a transient loses its underscores (``__tmp0`` -> ``tmp0``). Maps and
tasklets take their DaCe label without underscores, lower case
(``_Add__map`` -> ``add_map``, ``_Add_`` -> ``add``), with ``_1``, ``_2``
added on a clash. A map variable is named by its depth (``i``, ``j``, ``k``,
``l``), a tasklet connector loses its underscores (``__in1`` -> ``in1``),
and the tasklet's code is rewritten to match, always skipping a name
already taken.

**Supported** is what vecadd needs, plus nothing that would need a
decision: one state; arrays with the default strides and no offset; maps
over one parameter with a positive constant step; Python tasklets whose
code is one ``output = expression`` per output in the expression grammar;
plain memlets. Everything else raises ``SdfgImportError`` naming the
construct and, where one exists, the open item or task that adds it:
several states (open item 36, M9), library nodes such as dot's ``Reduce``
(DFG3), write-conflict resolution (DFG3), maps over several parameters,
scalars, nested SDFGs, code outside the grammar (open item 37). FE1 and FE2
extend this list (M7).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from typing import Any

import dace
from dace.sdfg import nodes as dn
from dace.sdfg.utils import dfs_topological_sort

from snax_forge import expr
from snax_forge.expr import ExprError, Value

from .graph import Container, Graph, Memlet, Node
from .kinds import DfgError
from .subset import format_dim

VARIABLES = ("i", "j", "k", "l")


class SdfgImportError(DfgError):
    """An SDFG construct the importer does not support (yet)."""


# =============================================================================
# Names
# =============================================================================


def clean(label: str) -> str:
    """A DaCe label as a readable name: no leading, trailing or doubled underscores, lower case."""
    name = re.sub(r"_+", "_", label.strip("_")).lower()
    return name if name.isidentifier() else f"n_{re.sub(r'[^0-9a-z_]', '_', name)}"


def fresh(base: str, taken: set[str]) -> str:
    """``base``, or ``base_1``, ``base_2``, ... whichever is not taken; the result is taken."""
    name, k = base, 0
    while name in taken:
        k += 1
        name = f"{base}_{k}"
    taken.add(name)
    return name


def _variable(depth: int, taken: set[str]) -> str:
    base = VARIABLES[depth] if depth < len(VARIABLES) else f"i{depth}"
    return fresh(base, taken)


# =============================================================================
# Expressions: DaCe/sympy text -> the expression grammar
# =============================================================================


def _value(v: Any, rename: dict[str, str], what: str) -> Value:
    """A sympy value or string as a canonical value, DaCe names replaced by ours."""
    text = str(v)
    try:
        expr.check(text, what)
    except ExprError as e:
        raise SdfgImportError(f"{e} (outside the expression grammar, open item 37)") from None
    return expr.substitute(text, rename)


def _range(r: tuple[Any, Any, Any], rename: dict[str, str], what: str) -> tuple[Value, ...]:
    """A DaCe range entry (begin, end inclusive, step) as (begin, end exclusive, step).

    The arithmetic is done by sympy before anything is written, so ``0:N``
    comes back as ``0:N``, not ``0:N - 1 + 1``.
    """
    begin, end, step = (dace.symbolic.pystr_to_symbolic(str(x)) for x in r)
    s = _value(step, rename, what)
    if type(s) is not int or s < 1:
        raise SdfgImportError(f"{what}: step {step} is not a positive constant")
    return (_value(begin, rename, what), _value(end + 1, rename, what), s)


def _subset(m: dace.Memlet, rename: dict[str, str], what: str) -> list[Value]:
    """A memlet's subset, one dimension per container dimension (subset.py)."""
    if not isinstance(m.subset, (dace.subsets.Range, dace.subsets.Indices)):
        raise SdfgImportError(f"{what}: subset {m.subset} is not a range or index")
    dims = []
    for r in m.subset.ndrange():
        begin, end, _ = (dace.symbolic.pystr_to_symbolic(str(x)) for x in r)
        if (end - begin) == 0:
            dims.append(_value(begin, rename, what))
        else:
            dims.append(format_dim(_range(r, rename, what)))
    return dims


# =============================================================================
# The importer
# =============================================================================


class _Importer:
    def __init__(self, sdfg: dace.SDFG, name: str):
        self.sdfg = sdfg
        self.name = name
        self.ids: set[str] = set()
        self.containers: dict[str, str] = {}  # DaCe array name -> container name

    # --- graph ---

    def run(self) -> Graph:
        states = list(self.sdfg.states())
        if len(states) != 1:
            labels = [s.label for s in states]
            raise SdfgImportError(
                f"{self.name}: {len(states)} states {labels}; only one is imported "
                "(control flow is open item 36)"
            )
        (self.state,) = states
        symbols = sorted(str(s) for s in self.sdfg.free_symbols)
        for s in symbols:
            if not s.isidentifier():
                raise SdfgImportError(f"{self.name}: symbol {s!r} is not a name")
        self._check_nodes()
        containers = self._containers(set(symbols))
        self.outer_names = set(symbols) | set(containers)  # never a map variable
        body = self._scope(None, {}, 0)
        return Graph(self.name, dict.fromkeys(symbols), containers, body)

    def _containers(self, symbols: set[str]) -> dict[str, Container]:
        out: dict[str, Container] = {}
        taken = set(symbols)
        for name, d in self.sdfg.arrays.items():
            what = f"{self.name}: array {name!r}"
            if type(d) is not dace.data.Array:
                raise SdfgImportError(f"{what}: a {type(d).__name__} is not imported (only arrays)")
            new = fresh(clean(name) if d.transient else name, taken)
            if not new.isidentifier():
                raise SdfgImportError(f"{what}: not a usable container name")
            shape = [_value(s, {}, f"{what} shape") for s in d.shape]
            expected = [expr.canonical(_c_stride(shape, i)) for i in range(len(shape))]
            strides = [_value(s, {}, f"{what} strides") for s in d.strides]
            if strides != expected:
                raise SdfgImportError(f"{what}: strides {strides} are not the default {expected}")
            if any(_value(o, {}, f"{what} offset") != 0 for o in d.offset):
                raise SdfgImportError(f"{what}: a nonzero offset {list(d.offset)} is not imported")
            out[new] = Container(shape, d.dtype.as_numpy_dtype().name, bool(d.transient))
            self.containers[name] = new
        return out

    def _check_nodes(self) -> None:
        st = self.state
        for n in st.nodes():
            what = f"{self.name}: node {n.label!r}"
            if isinstance(n, dn.LibraryNode):
                raise SdfgImportError(
                    f"{what}: library node {type(n).__name__} is not imported "
                    "(dot's Reduce comes with DFG3)"
                )
            if isinstance(n, dn.NestedSDFG):
                raise SdfgImportError(f"{what}: nested SDFGs are not imported")
            if not isinstance(n, (dn.AccessNode, dn.MapEntry, dn.MapExit, dn.Tasklet)):
                raise SdfgImportError(f"{what}: a {type(n).__name__} is not imported")
            if isinstance(n, dn.AccessNode) and st.entry_node(n) is not None:
                raise SdfgImportError(f"{what}: an access node inside a map is not imported")
        for e in st.edges():
            if isinstance(e.src, dn.AccessNode) and isinstance(e.dst, dn.AccessNode):
                raise SdfgImportError(
                    f"{self.name}: copy {e.src.data!r} -> {e.dst.data!r} is not imported "
                    "(simplify leaves none for vecadd)"
                )
            if e.data.wcr is not None:
                raise SdfgImportError(
                    f"{self.name}: write-conflict resolution on {e.data} is not imported (DFG3)"
                )
            if e.data.dynamic:
                raise SdfgImportError(f"{self.name}: dynamic memlet {e.data} is not imported")

    # --- scopes ---

    def _scope(self, entry: dn.MapEntry | None, rename: dict[str, str], depth: int) -> list[Node]:
        children = set(self.state.scope_children()[entry])
        body = []
        for n in dfs_topological_sort(self.state):
            if n not in children:
                continue
            if isinstance(n, dn.MapEntry):
                body.append(self._map(n, rename, depth))
            elif isinstance(n, dn.Tasklet):
                body.append(self._tasklet(n, rename))
        return body

    def _map(self, entry: dn.MapEntry, rename: dict[str, str], depth: int) -> Node:
        m = entry.map
        what = f"{self.name}: map {m.label!r}"
        if len(m.params) != 1:
            raise SdfgImportError(
                f"{what}: a map over {len(m.params)} parameters {m.params} is not imported yet "
                "(nested maps, one per parameter, D77)"
            )
        (param,), (rng,) = m.params, m.range.ranges
        var = _variable(depth, self.outer_names | set(rename.values()))
        begin, end, step = _range(rng, rename, f"{what} range")
        inner = {**rename, param: var}
        node_id = fresh(clean(m.label), self.ids)
        body = self._scope(entry, inner, depth + 1)
        attrs = {"var": var, "range": format_dim((begin, end, step))}
        return Node(node_id, "map", attrs=attrs, body=body)

    def _tasklet(self, t: dn.Tasklet, rename: dict[str, str]) -> Node:
        what = f"{self.name}: tasklet {t.label!r}"
        if t.code.language != dace.Language.Python:
            raise SdfgImportError(f"{what}: {t.code.language} code is not imported (only Python)")
        conns = fresh_all([*t.in_connectors, *t.out_connectors], set())
        inputs = self._memlets((e.dst_conn, e.data) for e in self.state.in_edges(t))
        outputs = self._memlets((e.src_conn, e.data) for e in self.state.out_edges(t))
        code = _rename_code(t.code.as_string, conns, what)
        return Node(
            fresh(clean(t.label), self.ids),
            "tasklet",
            inputs={
                conns[c]: Memlet(self.containers[m.data], _subset(m, rename, f"{what} {c}"))
                for c, m in inputs.items()
            },
            outputs={
                conns[c]: Memlet(self.containers[m.data], _subset(m, rename, f"{what} {c}"))
                for c, m in outputs.items()
            },
            attrs={"code": code},
        )

    def _memlets(self, edges: Iterable[tuple[str | None, dace.Memlet]]) -> dict[str, dace.Memlet]:
        out = {}
        for conn, m in edges:
            if conn is None or m.is_empty():
                raise SdfgImportError(
                    f"{self.name}: an empty or unconnected memlet is not imported"
                )
            out[conn] = m
        return out


def fresh_all(names: list[str], taken: set[str]) -> dict[str, str]:
    """Readable names for DaCe connectors (``__in1`` -> ``in1``), none clashing."""
    return {n: fresh(clean(n), taken) for n in names}


def _c_stride(shape: list[Value], i: int) -> Value:
    """The default (row-major) stride of dimension ``i``, in elements."""
    rest = shape[i + 1 :]
    return "1" if not rest else " * ".join(f"({s})" for s in rest)


def _rename_code(code: str, conns: dict[str, str], what: str) -> str:
    """Tasklet code with its connectors renamed, checked against the expression grammar."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SdfgImportError(f"{what}: cannot parse its code {code!r}") from e
    for st in tree.body:
        ok = isinstance(st, ast.Assign) and len(st.targets) == 1
        if not ok or not isinstance(st.targets[0], ast.Name):
            raise SdfgImportError(f"{what}: {ast.unparse(st)!r} is not 'output = expression'")
        try:
            expr.check(ast.unparse(st.value), what)
        except ExprError as e:
            raise SdfgImportError(f"{e} (outside the expression grammar, open item 37)") from None
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in conns:
            n.id = conns[n.id]
    return "\n".join(ast.unparse(st) for st in tree.body)


# =============================================================================
# Entry points
# =============================================================================


def import_sdfg(sdfg: dace.SDFG, name: str | None = None) -> Graph:
    """The ``.snaxdfg`` graph of a simplified SDFG; ``name`` defaults to the SDFG's."""
    return _Importer(sdfg, name or sdfg.name).run()


def import_kernel(kernel: str) -> Graph:
    """Build the kernel's simplified SDFG (as ``pixi run forge`` does) and import it."""
    from snax_forge.sdfg.build import build
    from snax_forge.sdfg.loader import load

    spec = load(kernel)
    return import_sdfg(build(spec, simplify=True), spec.name)
