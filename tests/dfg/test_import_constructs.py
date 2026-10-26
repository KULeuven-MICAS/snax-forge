"""The importer's constructs for M13 B0: scalars, the view fold, constant casts, nested maps.

axpy's ``alpha`` (a DaCe ``Scalar``) is a one-element container with attr
``sdfg.scalar``, read as ``alpha[0]``, and the reference executor takes it
as a NumPy scalar. rowsum's Reduce writes ``y`` directly, its view and the
view's copy folded. relu built at int32 reads ``max(in1, 0)``: DaCe's
``dace.int32(0)`` is a constant cast to the output's type, folded. matadd's
map over two parameters is two nested maps, and a map over three is three.
Each is checked on small numbers in the reference executor.
"""

from __future__ import annotations

import dataclasses

import dace
import numpy as np
import pytest

from snax_forge.dfg.execute import ExecutionError, execute
from snax_forge.dfg.import_sdfg import SdfgImportError, import_kernel, import_sdfg
from snax_forge.sdfg.build import build
from snax_forge.sdfg.loader import load

from . import sdfg_programs as progs


def _loops(g):
    """Each map as (id, var, range), outermost first, and the tasklet's code."""
    return [
        (n.id, n.attrs["var"], n.attrs["range"]) if n.kind == "map" else (n.id, n.attrs["code"])
        for n, _ in g.walk()
    ]


# =============================================================================
# Scalars: axpy
# =============================================================================


@pytest.fixture(scope="module")
def axpy():
    return import_kernel("axpy")


def test_axpy_alpha_is_a_scalar_container(axpy):
    a = axpy.containers["alpha"]
    assert (a.shape, a.dtype, a.transient, a.attrs) == ([1], "int64", False, {"sdfg.scalar": True})
    assert axpy.node("mult").inputs["in1"].to_dict() == {"data": "alpha", "subset": [0]}
    assert _loops(axpy) == [
        ("mult_map", "i", "0:N"),
        ("mult", "out = in1 * in2"),
        ("augassign_19_4_map", "i", "0:N"),
        ("augassign_19_4", "out = in1 + in2"),
    ]


def test_executor_takes_a_numpy_scalar(axpy):
    """alpha = 3, x = (1, 2), y = (10, 20) gives y = (13, 26), as np.int64(3) or as [3]."""
    x, y = np.array([1, 2], dtype=np.int64), np.array([10, 20], dtype=np.int64)
    for alpha in (np.int64(3), np.array([3], dtype=np.int64)):
        out = execute(axpy, {"alpha": alpha, "x": x, "y": y})
        assert out["y"].tolist() == [13, 26]
    with pytest.raises(ExecutionError, match="input 'alpha'"):
        execute(axpy, {"alpha": np.array([3, 4], dtype=np.int64), "x": x, "y": y})


# =============================================================================
# The view fold: rowsum
# =============================================================================


def test_rowsum_view_is_folded():
    """The Reduce writes y directly: no view, no copy, a sum map per dimension."""
    g = import_kernel("rowsum")
    assert list(g.containers) == ["A", "y"]
    assert _loops(g) == [
        ("sum_map", "i", "0:M"),
        ("sum_map_1", "j", "0:N"),
        ("sum", "out = in1"),
    ]
    out = g.node("sum").outputs["out"].to_dict()
    assert out == {"data": "y", "subset": ["i"], "reduce": {"op": "add", "identity": 0}}
    A = np.array([[1, 2], [3, 4]], dtype=np.int64)
    assert execute(g, {"A": A, "y": np.zeros(2, np.int64)})["y"].tolist() == [3, 7]


def test_a_view_into_a_transient_is_not_folded():
    sdfg = build(load("rowsum"), simplify=True)
    sdfg.arrays["y"].transient = True
    with pytest.raises(SdfgImportError, match="view '__tmp0' is used other than by one copy"):
        import_sdfg(sdfg, "rowsum")


# =============================================================================
# Constant casts: relu
# =============================================================================


def _relu_int32() -> dace.SDFG:
    """relu's SPEC at int32, as it was in KER0: DaCe writes max(__in1, dace.int32(0))."""
    spec = load("relu")
    N = dace.symbol("N")
    spec = dataclasses.replace(spec, descriptors={"A": dace.int32[N], "B": dace.int32[N]})
    return build(spec, simplify=True)


def _tasklet(sdfg: dace.SDFG) -> dace.sdfg.nodes.Tasklet:
    (t,) = [n for n in sdfg.states()[0].nodes() if isinstance(n, dace.sdfg.nodes.Tasklet)]
    return t


def test_relu_at_int64_and_int32_reads_max():
    for g, dtype in (
        (import_kernel("relu"), "int64"),
        (import_sdfg(_relu_int32(), "relu"), "int32"),
    ):
        assert _loops(g) == [
            ("numpy_maximum_map", "i", "0:N"),
            ("numpy_maximum", "out = max(in1, 0)"),
        ]
        assert {c.dtype for c in g.containers.values()} == {dtype}
        A = np.array([-3, 5], dtype=dtype)
        assert execute(g, {"A": A, "B": np.zeros(2, dtype)})["B"].tolist() == [0, 5]


def test_int32_relu_casts_dace_writes():
    assert "dace.int32(0)" in _tasklet(_relu_int32()).code.as_string


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("__out = max(__in1, dace.int32(3000000000))", "3000000000 does not fit in int32"),
        ("__out = max(__in1, dace.int32(-1))", None),
        ("__out = max(__in1, dace.int64(0))", "cast to int64, not the output's type int32"),
        ("__out = max(dace.int32(__in1), 0)", "a cast of a non-constant"),
    ],
)
def test_cast_rule(code, message):
    """Only a constant cast to the output's type that fits is folded; the rest name open item 37."""
    sdfg = _relu_int32()
    _tasklet(sdfg).code = dace.properties.CodeBlock(code)
    if message is None:
        g = import_sdfg(sdfg, "relu")
        assert g.node("numpy_maximum").attrs["code"] == "out = max(in1, -1)"
        return
    with pytest.raises(SdfgImportError, match=f"{message} .*open item 37"):
        import_sdfg(sdfg, "relu")


# =============================================================================
# Maps over several parameters: matadd
# =============================================================================


def test_matadd_is_two_nested_maps():
    """add_map keeps the label's id, the inner map takes a fresh one; i outer, j inner."""
    g = import_kernel("matadd")
    assert _loops(g) == [
        ("add_map", "i", "0:M"),
        ("add_map_1", "j", "0:N"),
        ("add", "out = in1 + in2"),
    ]
    assert [m.subset for m in g.node("add").inputs.values()] == [["i", "j"], ["i", "j"]]
    A = np.arange(6, dtype=np.int64).reshape(2, 3)
    B = 10 * np.arange(6, dtype=np.int64).reshape(2, 3)
    C = execute(g, {"A": A, "B": B, "C": np.zeros((2, 3), np.int64)})["C"]
    assert C.tolist() == [[0, 11, 22], [33, 44, 55]]


def test_a_map_over_three_parameters():
    """Three maps in DaCe's parameter order, variables by depth."""
    g = import_sdfg(progs.three_params(), "t")
    assert _loops(g) == [
        ("c_map", "i", "0:2"),
        ("c_map_1", "j", "0:3"),
        ("c_map_2", "k", "0:4"),
        ("c", "o = a + 1"),
    ]
    assert g.node("c").inputs["a"].subset == ["i", "j", "k"]
    A = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    assert np.array_equal(execute(g, {"A": A, "B": np.zeros_like(A)})["B"], A + 1)
