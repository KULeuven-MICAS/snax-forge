"""Small SDFGs for the importer tests (IMP1).

DaCe programs have to live in a file (DaCe reads their source), so the ones
the tests import are here. ``two`` keeps a transient between two maps in one
state; ``shift`` has offset subsets and a shortened range, like one half-step
of jacobi1d; ``amax`` is a Max reduction stored through a scalar and a copy,
as dot's Sum is (DFG3), and ``total`` a Sum over both dimensions of a matrix. ``row_sum`` is a Reduce over one axis of two, built by
hand since NumPy's ``B[:] = np.sum(A, axis=1)`` goes through a view
(``row_sum_view``, rejected). The ``unsupported_*`` builders make one
construct each that the importer must reject by name.

No ``from __future__ import annotations`` here: DaCe reads the type hints of
a program as objects, and postponed annotations turn every array into a
scalar.
"""

import dace
import numpy as np

N = dace.symbol("N")


@dace.program
def two(A: dace.int64[N], B: dace.int64[N], C: dace.int64[N]):
    C[:] = (A + B) * B


@dace.program
def amax(A: dace.int64[N], out: dace.int64[1]):
    out[0] = np.max(A)


@dace.program
def total(A: dace.int64[N, N], out: dace.int64[1]):
    out[0] = np.sum(A)


@dace.program
def row_sum_view(A: dace.int64[N, N], B: dace.int64[N]):
    B[:] = np.sum(A, axis=1)


@dace.program
def shift(A: dace.int64[N], B: dace.int64[N]):
    B[1:-1] = A[:-2] + A[2:]


def _one_state(arrays: dict[str, tuple[list, dict]]) -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG("t")
    for name, (shape, kw) in arrays.items():
        sdfg.add_array(name, shape, dace.int64, **kw)
    return sdfg, sdfg.add_state()


def unsupported_wcr() -> dace.SDFG:
    sdfg, st = _one_state({"A": ([8], {}), "s": ([1], {})})
    st.add_mapped_tasklet(
        "r",
        {"i": "0:8"},
        {"a": dace.Memlet("A[i]")},
        "o = a",
        {"o": dace.Memlet("s[0]", wcr="lambda x, y: x + y")},
        external_edges=True,
    )
    return sdfg


def unsupported_two_params() -> dace.SDFG:
    sdfg, st = _one_state({"A": ([4, 4], {}), "B": ([4, 4], {})})
    st.add_mapped_tasklet(
        "c",
        {"i": "0:4", "j": "0:4"},
        {"a": dace.Memlet("A[i, j]")},
        "o = a",
        {"o": dace.Memlet("B[i, j]")},
        external_edges=True,
    )
    return sdfg


def unsupported_cast() -> dace.SDFG:
    sdfg, st = _one_state({"A": ([8], {}), "B": ([8], {})})
    st.add_mapped_tasklet(
        "c",
        {"i": "0:8"},
        {"a": dace.Memlet("A[i]")},
        "o = dace.int64(a) // 3",
        {"o": dace.Memlet("B[i]")},
        external_edges=True,
    )
    return sdfg


def unsupported_copy() -> dace.SDFG:
    sdfg, st = _one_state({"A": ([8], {}), "B": ([8], {})})
    st.add_nedge(st.add_read("A"), st.add_write("B"), dace.Memlet("A[0:8]"))
    return sdfg


def unsupported_strides() -> dace.SDFG:
    sdfg, st = _one_state({"A": ([8], {"strides": [2]}), "B": ([8], {})})
    st.add_mapped_tasklet(
        "c",
        {"i": "0:8"},
        {"a": dace.Memlet("A[i]")},
        "o = a",
        {"o": dace.Memlet("B[i]")},
        external_edges=True,
    )
    return sdfg


def row_sum() -> dace.SDFG:
    """B[i] = sum over j of A[i, j]: a Reduce over axis 1, writing B directly."""
    sdfg, st = _one_state({"A": ([4, 6], {}), "B": ([4], {})})
    red = st.add_reduce("lambda a, b: a + b", axes=[1], identity=0)
    st.add_edge(st.add_read("A"), None, red, "_in", dace.Memlet("A[0:4, 0:6]"))
    st.add_edge(red, "_out", st.add_write("B"), None, dace.Memlet("B[0:4]"))
    return sdfg


def unsupported_reduce_used_twice() -> dace.SDFG:
    """A Reduce into a scalar that two copies read: not the one-copy store that is folded."""
    sdfg, st = _one_state({"A": ([8], {}), "B": ([1], {}), "C": ([1], {})})
    sdfg.add_scalar("s", dace.int64, transient=True)
    red = st.add_reduce("lambda a, b: a + b", axes=None, identity=0)
    s = st.add_access("s")
    st.add_edge(st.add_read("A"), None, red, "_in", dace.Memlet("A[0:8]"))
    st.add_edge(red, "_out", s, None, dace.Memlet("s[0]"))
    for out in ("B", "C"):
        t = st.add_tasklet(f"copy_{out}", {"i"}, {"o"}, "o = i")
        st.add_edge(s, None, t, "i", dace.Memlet("s[0]"))
        st.add_edge(t, "o", st.add_write(out), None, dace.Memlet(f"{out}[0]"))
    return sdfg
