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
    Reduce library node               nested maps, one per input dimension,
                                      around a tasklet ``out = in1`` whose
                                      output memlet has a ``reduce`` (D102)
    access nodes, map connectors,     not stored: derived (D77)
    outer memlets on a map

What DaCe's simplify already did is kept as it is: for vecadd it folded the
transient-plus-copy of ``C[:] = A + B`` into a direct write into ``C``, so
there is nothing left to fold. A copy between containers that simplify left
is an error here.

**Reductions** (DFG3, D102). DaCe writes ``np.sum(A * B)`` as a map into a
transient, a ``Reduce`` library node, and, for a result stored with
``out[0] = ...``, a scalar transient copied into ``out`` by a tasklet
``__out = __inp``; simplify leaves that copy. The importer folds it, as
simplify folded vecadd's (the one fold it does itself): the Reduce writes
straight into the copy's target, and the scalar and the copy are gone. It
is folded only when the scalar has that one writer and that one copy as its
only reader; any other use of a Reduce's scalar is an error. The Reduce
then becomes a map per input dimension (``sum_map``, ``prod_map``,
``min_map``, ``max_map`` after its reduction type), variables by depth as
for any map, around a tasklet (``sum``, ...) with code ``out = in1``. The
input memlet takes one element per iteration; the output memlet drops the
reduced dimensions and carries ``reduce``, the SNAX-DFG's own name for
what DaCe calls write-conflict resolution: the op (``add``, ``mul``, ``min``,
``max``) and the Reduce's identity (the op's for the dtype when DaCe gives
none). dot becomes ``mult_map`` then ``sum_map``, with ``tmp0`` between
them.

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
plain memlets; the ``Reduce`` library node at the top level, over
unit-step ranges. Everything else raises ``SdfgImportError`` naming the
construct and, where one exists, the open item or task that adds it:
several states (open item 36, M9), other library nodes, a ``wcr`` memlet
DaCe wrote itself (a fold into the element's contents, FE1), maps over
several parameters, scalars and views (``np.sum(A, axis=1)`` stored with
``B[:] = ...`` goes through a view, FE1), nested SDFGs, code outside the
grammar (open item 37). FE1 and FE2 extend this list (M7).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from typing import Any

import dace
import numpy as np
from dace.frontend.operations import detect_reduction_type
from dace.libraries.standard import Reduce
from dace.sdfg import nodes as dn
from dace.sdfg.utils import dfs_topological_sort

from snax_forge import expr
from snax_forge.expr import ExprError, Value

from .graph import Container, Graph, Memlet, Node
from .kinds import DfgError
from .reduce import REDUCE_OPS, Reduction
from .subset import format_dim

VARIABLES = ("i", "j", "k", "l")
# DaCe's reduction type -> (reduce op, base name of the map and tasklet)
REDUCTIONS = {
    dace.dtypes.ReductionType.Sum: ("add", "sum"),
    dace.dtypes.ReductionType.Product: ("mul", "prod"),
    dace.dtypes.ReductionType.Min: ("min", "min"),
    dace.dtypes.ReductionType.Max: ("max", "max"),
}


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
        self._fold_reduce_copies()
        containers = self._containers(set(symbols))
        self.outer_names = set(symbols) | set(containers)  # never a map variable
        body = self._scope(None, {}, 0)
        return Graph(self.name, dict.fromkeys(symbols), containers, body)

    def _containers(self, symbols: set[str]) -> dict[str, Container]:
        out: dict[str, Container] = {}
        taken = set(symbols)
        for name, d in self.sdfg.arrays.items():
            if name in self.gone:
                continue
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
            if isinstance(n, Reduce):
                if st.entry_node(n) is not None:
                    raise SdfgImportError(f"{what}: a Reduce inside a map is not imported")
                continue
            if isinstance(n, dn.LibraryNode):
                raise SdfgImportError(
                    f"{what}: library node {type(n).__name__} is not imported (only Reduce, D102)"
                )
            if isinstance(n, dn.NestedSDFG):
                raise SdfgImportError(f"{what}: nested SDFGs are not imported")
            if not isinstance(n, (dn.AccessNode, dn.MapEntry, dn.MapExit, dn.Tasklet)):
                raise SdfgImportError(f"{what}: a {type(n).__name__} is not imported")
            if isinstance(n, dn.AccessNode) and st.entry_node(n) is not None:
                raise SdfgImportError(f"{what}: an access node inside a map is not imported")
        for e in st.edges():
            views = [
                n.data
                for n in (e.src, e.dst)
                if isinstance(n, dn.AccessNode)
                and isinstance(self.sdfg.arrays[n.data], dace.data.View)
            ]
            if views:
                raise SdfgImportError(
                    f"{self.name}: view {views[0]!r} is not imported (a result stored through a "
                    "view, as B[:] = np.sum(A, axis=1) is, comes with FE1)"
                )
            if isinstance(e.src, dn.AccessNode) and isinstance(e.dst, dn.AccessNode):
                raise SdfgImportError(
                    f"{self.name}: copy {e.src.data!r} -> {e.dst.data!r} is not imported "
                    "(simplify leaves none for vecadd)"
                )
            if e.data.wcr is not None:
                raise SdfgImportError(
                    f"{self.name}: write-conflict resolution on {e.data} not from a Reduce is "
                    "not imported (a fold into the element's contents, FE1)"
                )
            if e.data.dynamic:
                raise SdfgImportError(f"{self.name}: dynamic memlet {e.data} is not imported")

    def _fold_reduce_copies(self) -> None:
        """Every Reduce's output: its own memlet, or the target of the copy of its scalar."""
        st = self.state
        self.reduce_out: dict[Reduce, dace.Memlet] = {}
        self.skip: set[Any] = set()  # the folded copy tasklets
        self.gone: set[str] = set()  # the folded scalars
        for red in (n for n in st.nodes() if isinstance(n, Reduce)):
            what = f"{self.name}: reduce {red.label!r}"
            ins, outs = st.in_edges(red), st.out_edges(red)
            if len(ins) != 1 or len(outs) != 1:
                raise SdfgImportError(
                    f"{what}: {len(ins)} inputs and {len(outs)} outputs, not 1 and 1"
                )
            tmp = outs[0].dst
            desc = self.sdfg.arrays[tmp.data]
            if not desc.transient or type(desc) is not dace.data.Scalar:
                self.reduce_out[red] = outs[0].data  # written as it is, no fold
                continue
            readers = st.out_edges(tmp)
            copy = readers[0].dst if len(readers) == 1 else None
            ok = (
                isinstance(copy, dn.Tasklet)
                and len(st.in_edges(tmp)) == 1
                and [n for n in st.data_nodes() if n.data == tmp.data] == [tmp]
                and _is_copy(copy)
                and len(st.out_edges(copy)) == 1
                and isinstance(st.out_edges(copy)[0].dst, dn.AccessNode)
            )
            if not ok:
                raise SdfgImportError(
                    f"{what}: its result {tmp.data!r} is used other than by one copy into a "
                    "container; only that copy is folded (D102)"
                )
            self.reduce_out[red] = st.out_edges(copy)[0].data
            self.skip.add(copy)
            self.gone.add(tmp.data)

    # --- scopes ---

    def _scope(self, entry: dn.MapEntry | None, rename: dict[str, str], depth: int) -> list[Node]:
        children = set(self.state.scope_children()[entry])
        body = []
        for n in dfs_topological_sort(self.state):
            if n not in children:
                continue
            if isinstance(n, dn.MapEntry):
                body.append(self._map(n, rename, depth))
            elif isinstance(n, Reduce):
                body.append(self._reduce(n))
            elif isinstance(n, dn.Tasklet) and n not in self.skip:
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

    def _reduce(self, red: Reduce) -> Node:
        """A top-level Reduce as nested maps around ``out = in1`` with a ``reduce`` output (D102)."""
        what = f"{self.name}: reduce {red.label!r}"
        rtype = detect_reduction_type(red.wcr)
        if rtype not in REDUCTIONS:
            raise SdfgImportError(
                f"{what}: a {rtype.name} reduction is not imported (Sum, Product, Min, Max)"
            )
        op, base = REDUCTIONS[rtype]
        src = self.state.in_edges(red)[0].data
        dst = self.reduce_out[red]
        in_ranges, out_ranges = src.subset.ndrange(), dst.subset.ndrange()
        rank = len(in_ranges)
        axes = list(range(rank)) if red.axes is None else sorted(red.axes)

        variables: list[str] = []
        taken = set(self.outer_names)
        ranges = []
        for d, r in enumerate(in_ranges):
            b, e, s = _range(r, {}, f"{what} input")
            if s != 1:
                raise SdfgImportError(f"{what}: input dimension {d} has step {s}, not 1")
            variables.append(_variable(d, taken))
            ranges.append((b, e, s))
        out_subset = _reduce_out_subset(in_ranges, out_ranges, axes, variables, what)

        target = self.containers[dst.data]
        dtype = np.dtype(self.sdfg.arrays[dst.data].dtype.as_numpy_dtype())
        identity = red.identity
        if identity is None:
            identity = REDUCE_OPS[op].identity(dtype)
        elif int(identity) != identity:
            raise SdfgImportError(f"{what}: identity {identity!r} is not an integer (D28)")
        tasklet = Node(
            fresh(base, self.ids),
            "tasklet",
            inputs={"in1": Memlet(self.containers[src.data], list(variables))},
            outputs={"out": Memlet(target, out_subset, Reduction(op, int(identity)))},
            attrs={"code": "out = in1"},
        )
        ids = [fresh(f"{base}_map", self.ids) for _ in variables]
        node = tasklet
        for d in reversed(range(rank)):
            node = Node(
                ids[d],
                "map",
                attrs={"var": variables[d], "range": format_dim(ranges[d])},
                body=[node],
            )
        return node

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


def _length(r: tuple[Any, Any, Any]) -> Any:
    """The number of elements of a DaCe range entry (begin, end inclusive, step 1)."""
    b, e, _ = (dace.symbolic.pystr_to_symbolic(str(x)) for x in r)
    return e - b + 1


def _reduce_out_subset(
    in_ranges: list[Any], out_ranges: list[Any], axes: list[int], variables: list[str], what: str
) -> list[Value]:
    """A Reduce's output memlet over the maps of its input dimensions (``variables``).

    Each output dimension is either a kept input dimension, indexed by that
    dimension's variable, or (reduced) a single element. DaCe writes the
    output with the reduced dimensions dropped, kept with length 1, or, for a
    reduction to one element, as any one-element subset.
    """
    rank = len(in_ranges)
    kept = [d for d in range(rank) if d not in axes]
    if not kept and all(_length(r) == 1 for r in out_ranges):  # to one element
        pairs: dict[int, int] = {}
    elif len(out_ranges) == len(kept):
        pairs = dict(zip(range(len(out_ranges)), kept, strict=True))
    elif len(out_ranges) == rank:  # the reduced dimensions kept with length 1
        pairs = {k: k for k in kept}
        for k in axes:
            if _length(out_ranges[k]) != 1:
                raise SdfgImportError(f"{what}: reduced dimension {k} of the output is not 1")
    else:
        raise SdfgImportError(
            f"{what}: an output of {len(out_ranges)} dimensions for {rank} input dimensions "
            f"reduced over {axes}"
        )
    out: list[Value] = []
    for k, r in enumerate(out_ranges):
        ob = dace.symbolic.pystr_to_symbolic(str(r[0]))
        if k not in pairs:
            out.append(_value(ob, {}, f"{what} output"))
            continue
        d = pairs[k]
        if r[2] != 1 or (_length(r) - _length(in_ranges[d])) != 0:
            raise SdfgImportError(
                f"{what}: output dimension {k} does not match input dimension {d} one to one"
            )
        ib = dace.symbolic.pystr_to_symbolic(str(in_ranges[d][0]))
        index = ob - ib + dace.symbolic.pystr_to_symbolic(variables[d])
        out.append(_value(index, {}, f"{what} output"))
    return out


def _is_copy(t: dn.Tasklet) -> bool:
    """A tasklet ``out = in`` with one input and one output: what a store of a scalar leaves."""
    if t.code.language != dace.Language.Python or len(t.in_connectors) != 1:
        return False
    if len(t.out_connectors) != 1:
        return False
    try:
        (st,) = ast.parse(t.code.as_string).body
    except (SyntaxError, ValueError):
        return False
    return (
        isinstance(st, ast.Assign)
        and [ast.unparse(x) for x in st.targets] == list(t.out_connectors)
        and isinstance(st.value, ast.Name)
        and st.value.id in t.in_connectors
    )


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
