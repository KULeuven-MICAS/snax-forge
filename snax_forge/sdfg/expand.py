"""Library nodes into maps, in dot's shape, before the importer.

``expand(sdfg)`` rewrites a simplified SDFG in place (and returns it), with no simplify after
it. It brings DaCe's ``MatMul`` and ``Transpose`` into the shape ``np.sum(A * B)`` has: a
multiply map into a transient, then a ``Reduce`` (sum, identity 0) over the contracted axis,
innermost. It is one fixed form and decides nothing else.

A MatMul ``c = a @ b`` (``a`` of rank >= 2 and ``b`` of rank 1, or both of one rank >= 2 with
the same batch dimensions, outermost) gives ``prod[i, k] = a[i, k] * b[k]`` or ``prod[i, j, k]
= a[i, k] * b[k, j]``, then ``c = sum_k prod``. The multiply map runs over ``c``'s indices, then
the contracted one, into a transient ``__<c>_prod`` of ``c``'s dtype, labelled ``<c>_mult``
(bicg's ``q_prod``, ``q_mult_map``). A MatMul input written only by a 2-D Transpose and read
only by MatMuls is read through the swapped indices, and the Transpose goes. Refused
(``ExpandError``, naming both shapes): a rank-1 ``a``, ranks or dimensions that do not match,
an output memlet of another shape, an operand not read whole from an access node, a MatMul
inside a map. Other library nodes are left to the importer.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import dace
from dace.libraries.blas import MatMul
from dace.libraries.standard import Transpose
from dace.sdfg import nodes as dn

from snax_forge.dfg.import_sdfg import SdfgImportError

SUM = "lambda a, b: a + b"


class ExpandError(SdfgImportError):
    """A library node ``expand`` does not take."""


def _sym(x: Any) -> Any:
    return dace.symbolic.pystr_to_symbolic(str(x))


def _ranges(m: dace.Memlet, what: str) -> list[tuple[Any, Any]]:
    """A memlet's subset as (begin, size) per dimension, step 1."""
    out = []
    for b, e, s in m.subset.ndrange():
        if _sym(s) != 1:
            raise ExpandError(f"{what}: {m} has step {s}, not 1")
        out.append((_sym(b), _sym(e) - _sym(b) + 1))
    return out


@dataclass
class _Operand:
    """One MatMul input: the node and array read, its shape, and its index map."""

    node: dn.AccessNode
    data: str
    shape: list[Any]
    index: Callable[[list[str]], list[str]]  # logical indices -> the array's indices


def _same(a: list[Any], b: list[Any]) -> bool:
    return len(a) == len(b) and all((x - y) == 0 for x, y in zip(a, b, strict=True))


def _show(shape: list[Any]) -> str:
    return "[" + ", ".join(str(s) for s in shape) + "]"


# =============================================================================
# Transpose fold
# =============================================================================


def _foldable_transposes(sdfg: dace.SDFG, state: dace.SDFGState) -> dict[str, Any]:
    """Each transient written only by a 2-D Transpose and read only by MatMuls: its in-edge."""
    out = {}
    for tr in state.nodes():
        if not isinstance(tr, Transpose):
            continue
        ins, outs = state.in_edges(tr), state.out_edges(tr)
        if len(ins) != 1 or len(outs) != 1 or not isinstance(outs[0].dst, dn.AccessNode):
            continue
        name = outs[0].dst.data
        desc = sdfg.arrays[name]
        if not desc.transient or len(desc.shape) != 2 or len(ins[0].data.subset) != 2:
            continue
        if not isinstance(ins[0].src, dn.AccessNode) or ins[0].data.data != ins[0].src.data:
            continue
        nodes = [n for n in state.data_nodes() if n.data == name]
        elsewhere = any(
            n.data == name for s in sdfg.states() if s is not state for n in s.data_nodes()
        )
        writers = [e for n in nodes for e in state.in_edges(n)]
        readers = [e for n in nodes for e in state.out_edges(n)]
        if elsewhere or len(writers) != 1 or not readers:
            continue
        if all(isinstance(e.dst, MatMul) for e in readers):
            out[name] = ins[0]
    return out


def _drop_transposes(sdfg: dace.SDFG, state: dace.SDFGState, folds: dict[str, Any]) -> None:
    """Remove each folded Transpose once nothing reads its transient."""
    for name, src_edge in folds.items():
        nodes = [n for n in state.data_nodes() if n.data == name]
        if any(state.out_edges(n) for n in nodes):
            continue
        (writer,) = [e.src for n in nodes for e in state.in_edges(n)]
        source = src_edge.src
        state.remove_nodes_from([*nodes, writer])
        if state.degree(source) == 0:
            state.remove_node(source)
        sdfg.remove_data(name)


# =============================================================================
# MatMul
# =============================================================================


def _operand(
    sdfg: dace.SDFG, state: dace.SDFGState, e: Any, folds: dict[str, Any], what: str
) -> _Operand:
    node = e.src
    if not isinstance(node, dn.AccessNode) or e.data.data != node.data:
        raise ExpandError(f"{what}: input {e.dst_conn} is not read from an access node")
    ranges = _ranges(e.data, f"{what} {e.dst_conn}")
    shape = [n for _, n in ranges]
    if node.data not in folds:
        return _Operand(
            node,
            node.data,
            shape,
            lambda idx, r=ranges: [str(b + _sym(i)) for (b, _), i in zip(r, idx, strict=True)],
        )
    src = folds[node.data]  # read the Transpose's source, the two indices swapped
    s0, s1 = (b for b, _ in _ranges(src.data, f"{what} transpose"))
    (b0, _), (b1, _) = ranges

    def swapped(idx: list[str]) -> list[str]:
        i0, i1 = (_sym(i) for i in idx)
        return [str(s0 + b1 + i1), str(s1 + b0 + i0)]

    return _Operand(src.src, src.src.data, shape, swapped)


def _matmul(sdfg: dace.SDFG, state: dace.SDFGState, mm: MatMul, folds: dict[str, Any]) -> None:
    what = f"{sdfg.name}: MatMul {mm.label!r}"
    if state.entry_node(mm) is not None:
        raise ExpandError(f"{what}: a MatMul inside a map is not expanded")
    ins = {e.dst_conn: e for e in state.in_edges(mm)}
    outs = state.out_edges(mm)
    if sorted(ins) != ["_a", "_b"] or len(outs) != 1 or not isinstance(outs[0].dst, dn.AccessNode):
        raise ExpandError(f"{what}: not one _a, one _b and one output to an access node")
    a = _operand(sdfg, state, ins["_a"], folds, what)
    b = _operand(sdfg, state, ins["_b"], folds, what)
    ra, rb = len(a.shape), len(b.shape)
    shapes = f"a {_show(a.shape)} @ b {_show(b.shape)}"
    if ra < 2:
        raise ExpandError(
            f"{what}: {shapes} is not expanded, a has rank 1 (vec-mat or vec-vec; "
            "only a of rank >= 2)"
        )
    if rb not in (1, ra):
        raise ExpandError(f"{what}: {shapes} do not match (b of rank 1 or of a's rank)")
    k = a.shape[-1]
    if rb == 1:
        ok, out_shape = _same([k], b.shape), a.shape[:-1]
    else:
        ok = _same(a.shape[:-2], b.shape[:-2]) and _same([k], [b.shape[-2]])
        out_shape = [*a.shape[:-1], b.shape[-1]]
    if not ok:
        raise ExpandError(f"{what}: {shapes} do not match")
    c_edge = outs[0]
    c_shape = [n for _, n in _ranges(c_edge.data, f"{what} output")]
    if not _same(c_shape, out_shape):
        raise ExpandError(
            f"{what}: {shapes} gives {_show(out_shape)}, the output is {_show(c_shape)}"
        )

    out_params = [f"__i{d}" for d in range(len(out_shape))]
    params = [*out_params, "__k"]
    if rb == 1:
        a_idx, b_idx = params, ["__k"]
    else:
        *batch, i, j = out_params
        a_idx, b_idx = [*batch, i, "__k"], [*batch, "__k", j]

    c = c_edge.data.data
    base = c.lstrip("_")
    prod, desc = sdfg.add_transient(
        f"__{base}_prod", [*out_shape, k], sdfg.arrays[c].dtype, find_new_name=True
    )
    prod_node = state.add_access(prod)
    nodes = {a.data: a.node, b.data: b.node}
    state.add_mapped_tasklet(
        f"{base}_mult",
        {p: f"0:{n}" for p, n in zip(params, [*out_shape, k], strict=True)},
        {
            "__in1": dace.Memlet(f"{a.data}[{', '.join(a.index(a_idx))}]"),
            "__in2": dace.Memlet(f"{b.data}[{', '.join(b.index(b_idx))}]"),
        },
        "__out = __in1 * __in2",
        {"__out": dace.Memlet(f"{prod}[{', '.join(params)}]")},
        input_nodes=nodes,
        output_nodes={prod: prod_node},
        external_edges=True,
    )
    red = state.add_reduce(SUM, axes=[len(params) - 1], identity=0)
    state.add_edge(prod_node, None, red, None, dace.Memlet.from_array(prod, desc))
    state.add_edge(red, None, c_edge.dst, c_edge.dst_conn, copy.deepcopy(c_edge.data))
    state.remove_node(mm)


# =============================================================================
# Entry point
# =============================================================================


def expand(sdfg: dace.SDFG) -> dace.SDFG:
    """``sdfg`` with every MatMul in dot's shape and the foldable Transposes folded (in place)."""
    for state in sdfg.states():
        matmuls = [n for n in state.nodes() if isinstance(n, MatMul)]
        if not matmuls:
            continue
        folds = _foldable_transposes(sdfg, state)
        for mm in matmuls:
            _matmul(sdfg, state, mm, folds)
        _drop_transposes(sdfg, state, folds)
    sdfg.validate()
    return sdfg
