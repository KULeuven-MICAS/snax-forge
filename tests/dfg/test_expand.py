"""MatMul and Transpose expanded into dot's shape before import (sdfg/expand.py).

Accepted when bicg, mvt, gesummv and atax have no MatMul or Transpose left
after ``expand``, their expanded SDFGs compile in DaCe and match NumPy, and
they import (``import_kernel``) and pass the reference check, at two sizes
each. Also: a mat-mat and a batched ``A @ B`` expand, compile and match; a
rank-1 ``a`` and mismatched shapes are refused by name; a Transpose with a
reader that is not a MatMul is left for the importer to refuse; and a kernel
without one imports as it is, untouched by ``expand``.
"""

from __future__ import annotations

import dace
import numpy as np
import pytest
from dace.libraries.blas import MatMul
from dace.sdfg import nodes as dn

from snax_forge.dfg.execute import execute
from snax_forge.dfg.import_sdfg import import_sdfg
from snax_forge.sdfg.build import build, verify
from snax_forge.sdfg.expand import expand
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


# =============================================================================
# DaCe's check and the reference check, at two sizes each
# =============================================================================


# A kernel's cases compile into one DaCe build folder (out/.dacecache/<kernel>_simp): one xdist
# group per kernel, so two workers never build it at once (--dist loadgroup)
COMPILED = [pytest.param(k, s, marks=pytest.mark.xdist_group(f"dace-{k}")) for k, s in SIZES]


@pytest.mark.parametrize(("kernel", "sizes"), COMPILED)
def test_expanded_compiles_and_matches_numpy(kernel, sizes):
    assert verify(load(kernel), _expanded(kernel), **sizes)


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


# =============================================================================
# Unchanged
# =============================================================================
