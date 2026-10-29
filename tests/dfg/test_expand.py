"""MatMul and Transpose expanded into dot's shape before import (sdfg/expand.py, D71).

Accepted when bicg, mvt, gesummv and atax have no MatMul or Transpose left
after ``expand``, their expanded SDFGs compile in DaCe and match NumPy, and
they import (``import_kernel``) and pass the reference check, at two sizes
each. Also: a mat-mat and a batched ``A @ B`` expand, compile and match; a
rank-1 ``a`` and mismatched shapes are refused by name; a Transpose with a
reader that is not a MatMul is left for the importer to refuse; and a kernel
without one imports as it is, untouched by ``expand``.
"""

from __future__ import annotations

import json

import dace
import numpy as np
import pytest
from dace.libraries.blas import MatMul
from dace.libraries.standard import Transpose
from dace.sdfg import nodes as dn

from snax_forge.dfg.execute import against_reference, execute
from snax_forge.dfg.import_sdfg import SdfgImportError, import_kernel, import_sdfg
from snax_forge.sdfg.build import build, verify
from snax_forge.sdfg.expand import ExpandError, expand
from snax_forge.sdfg.loader import load

from . import sdfg_programs as progs

pytestmark = pytest.mark.slow  # DaCe builds and compiles every SDFG here

# the products each MatMul writes, named after its output
PRODUCTS = {
    "bicg": ["__q_prod", "__s_prod"],
    "mvt": ["__tmp0_prod", "__tmp2_prod"],
    "gesummv": ["__tmp1_prod", "__tmp3_prod"],
    "atax": ["__tmp1_prod", "__y_prod"],
}
SIZES = [
    ("bicg", {"m": 8, "n": 8}),
    ("bicg", {"m": 6, "n": 10}),
    ("mvt", {"m": 8, "n": 8}),
    ("mvt", {"m": 6, "n": 10}),
    ("atax", {"m": 8, "n": 8}),
    ("atax", {"m": 6, "n": 10}),
    ("gesummv", {"n": 8}),
    ("gesummv", {"n": 12}),
]
UNCHANGED = ["vecadd", "vecmul", "vsum", "dot", "relu", "axpy", "matadd", "rowsum"]


def _expanded(kernel: str) -> dace.SDFG:
    return expand(build(load(kernel), simplify=True))


def _nodes(sdfg: dace.SDFG) -> list[dn.Node]:
    return [n for st in sdfg.states() for n in st.nodes()]


# =============================================================================
# Structure
# =============================================================================


@pytest.mark.parametrize("kernel", sorted(PRODUCTS))
def test_no_matmul_or_transpose_left(kernel):
    sdfg = _expanded(kernel)
    assert not [n for n in _nodes(sdfg) if isinstance(n, (MatMul, Transpose))]
    assert sorted(n for n in sdfg.arrays if n.endswith("_prod")) == PRODUCTS[kernel]
    assert all(sdfg.arrays[p].transient for p in PRODUCTS[kernel])


def test_bicg_s_reads_a_swapped_output_index_outer():
    """s = A.T @ r: s_prod[j, i] = A[i, j] * r[i], the map over s's index (N) outer."""
    sdfg = _expanded("bicg")
    assert "__tmp1" not in sdfg.arrays  # the Transpose's transient is gone
    (st,) = sdfg.states()
    (entry,) = [n for n in st.nodes() if isinstance(n, dn.MapEntry) and n.label == "s_mult_map"]
    assert entry.map.params == ["__i0", "__k"]
    assert str(entry.map.range) == "0:N, 0:M"
    (t,) = [n for n in st.nodes() if isinstance(n, dn.Tasklet) and n.label == "s_mult"]
    ins = {e.dst_conn: str(e.data) for e in st.in_edges(t)}
    assert ins == {"__in1": "A[__k, __i0]", "__in2": "r[__k]"}
    assert t.code.as_string.replace("(", "").replace(")", "") == "__out = __in1 * __in2"


def test_imported_names():
    """The products and maps as the importer names them: underscores gone, no clash."""
    g = import_kernel("bicg")
    assert list(g.containers) == ["A", "p", "q", "r", "s", "q_prod", "s_prod"]
    assert [n.id for n in g.body] == ["q_mult_map", "sum_map", "s_mult_map", "sum_map_2"]
    g = import_kernel("atax")
    assert [c for c, x in g.containers.items() if x.transient] == ["tmp1", "tmp1_prod", "y_prod"]


# =============================================================================
# DaCe's check and the reference check, at two sizes each
# =============================================================================


@pytest.mark.parametrize(("kernel", "sizes"), SIZES)
def test_expanded_compiles_and_matches_numpy(kernel, sizes):
    assert verify(load(kernel), _expanded(kernel), **sizes)


@pytest.mark.parametrize(("kernel", "sizes"), SIZES)
def test_imports_and_matches_the_reference(kernel, sizes):
    spec, g = load(kernel), import_kernel(kernel)
    inputs = spec.make_inputs(np.random.default_rng(0), **sizes)
    pairs = against_reference(spec, inputs, lambda x: execute(g, x))
    for got, ref in pairs.values():
        assert np.array_equal(got, ref)


# =============================================================================
# Mat-mat and batched
# =============================================================================


@pytest.mark.parametrize(
    ("prog", "a_shape", "b_shape", "symbols"),
    [
        (progs.matmat, (3, 4), (4, 5), {"M": 3, "K": 4, "N": 5}),
        (progs.batched, (2, 3, 4), (2, 4, 5), {"M": 3, "K": 4, "N": 5}),
    ],
)
def test_mat_mat_expands_compiles_and_matches(prog, a_shape, b_shape, symbols):
    """prod[.., i, j, k] = A[.., i, k] * B[.., k, j], the batch dimension outermost."""
    sdfg = expand(prog.to_sdfg(simplify=True))
    assert not [n for n in _nodes(sdfg) if isinstance(n, MatMul)]
    rank = len(a_shape)
    assert len(sdfg.arrays["__C_prod"].shape) == rank + 1
    rng = np.random.default_rng(0)
    A = rng.integers(-8, 8, size=a_shape, dtype=np.int64)
    B = rng.integers(-8, 8, size=b_shape, dtype=np.int64)
    C = np.zeros(np.matmul(A, B).shape, dtype=np.int64)
    sdfg.compile()(A=A, B=B, C=C, **symbols)
    assert np.array_equal(C, A @ B)
    g = import_sdfg(sdfg, "mm")
    got = execute(g, {"A": A, "B": B, "C": np.zeros_like(C)})["C"]
    assert np.array_equal(got, A @ B)


# =============================================================================
# Refusals
# =============================================================================


def test_vec_vec_is_refused():
    with pytest.raises(ExpandError, match=r"a \[N\] @ b \[N\] is not expanded, a has rank 1"):
        expand(progs.vecvec.to_sdfg(simplify=True))


def test_shapes_that_do_not_match_are_refused():
    with pytest.raises(ExpandError, match=r"a \[4, 6\] @ b \[5\] do not match"):
        expand(progs.unsupported_matmul_shapes())


def test_a_transpose_alone_is_left_for_the_importer():
    """B[:] = A.T: the Transpose writes B, no MatMul reads it, so it stays and import fails."""
    sdfg = expand(progs.transpose_only.to_sdfg(simplify=True))
    assert [type(n) for n in _nodes(sdfg) if isinstance(n, dn.LibraryNode)] == [Transpose]
    with pytest.raises(SdfgImportError, match="library node Transpose is not imported"):
        import_sdfg(sdfg, "t")


# =============================================================================
# Unchanged
# =============================================================================


@pytest.mark.parametrize("kernel", UNCHANGED)
def test_kernels_without_matmul_import_as_before(kernel):
    """import_kernel (now through expand) gives the same .snaxdfg as a plain import."""
    plain = import_sdfg(build(load(kernel), simplify=True), kernel)
    assert json.loads(import_kernel(kernel).to_json()) == json.loads(plain.to_json())
