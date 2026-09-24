"""Small SDFGs for the importer tests (IMP1).

DaCe programs have to live in a file (DaCe reads their source), so the ones
the tests import are here. ``two`` keeps a transient between two maps in one
state; ``shift`` has offset subsets and a shortened range, like one half-step
of jacobi1d. The ``unsupported_*`` builders make one construct each that the
importer must reject by name.

No ``from __future__ import annotations`` here: DaCe reads the type hints of
a program as objects, and postponed annotations turn every array into a
scalar.
"""

import dace

N = dace.symbol("N")


@dace.program
def two(A: dace.int64[N], B: dace.int64[N], C: dace.int64[N]):
    C[:] = (A + B) * B


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
