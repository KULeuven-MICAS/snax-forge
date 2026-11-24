"""The importer's constructs: scalars, the view fold, constant casts, nested maps.

axpy's ``alpha`` (a DaCe ``Scalar``) is a one-element container with attr
``sdfg.scalar``, read as ``alpha[0]``, and the reference executor takes it
as a NumPy scalar. rowsum's Reduce writes ``y`` directly, its view and the
view's copy folded. relu built at int32 reads ``max(in1, 0)``: DaCe's
``dace.int32(0)`` is a constant cast to the output's type, folded. matadd's
map over two parameters is two nested maps, and a map over three is three.
Each is checked on small numbers in the reference executor.
"""

from __future__ import annotations

import dace
import pytest

from snax_forge.dfg.import_sdfg import SdfgImportError, import_kernel, import_sdfg
from snax_forge.sdfg.build import build
from snax_forge.sdfg.loader import load

# =============================================================================
# Scalars: axpy
# =============================================================================


@pytest.fixture(scope="module")
def axpy():
    return import_kernel("axpy")


# =============================================================================
# The view fold: rowsum
# =============================================================================


# =============================================================================
# Constant casts: relu
# =============================================================================


def _relu_int32() -> dace.SDFG:
    """relu's SPEC, int32: DaCe writes
    max(__in1, dace.int32(0))."""
    return build(load("relu"), simplify=True)


def _tasklet(sdfg: dace.SDFG) -> dace.sdfg.nodes.Tasklet:
    (t,) = [n for n in sdfg.states()[0].nodes() if isinstance(n, dace.sdfg.nodes.Tasklet)]
    return t


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("__out = max(__in1, dace.int32(3000000000))", "3000000000 does not fit in int32"),
        ("__out = max(__in1, dace.int32(-1))", "out = max(in1, -1)"),
        ("__out = max(__in1, dace.int64(0))", "cast to int64, not the output's or the op's type"),
        ("__out = max(dace.int32(__in1), 0)", "out = max(int32(in1), 0)"),
        ("__out = max(dace.float32(__in1), 0)", "cast to float32, not an integer type"),
    ],
)
def test_cast_rule(code, message):
    """A constant cast to the output's or the op's type that fits is folded, a cast of a value
    kept in the grammar's form; the rest are refused."""
    sdfg = _relu_int32()
    _tasklet(sdfg).code = dace.properties.CodeBlock(code)
    if message.startswith("out = "):
        g = import_sdfg(sdfg, "relu")
        assert g.node("numpy_maximum").attrs["code"] == message
        return
    with pytest.raises(SdfgImportError, match=message):
        import_sdfg(sdfg, "relu")


# =============================================================================
# Maps over several parameters: matadd
# =============================================================================
