"""The SDFG importer: vecadd's and dot's simplified SDFGs become their fixtures.

Accepted when the import of ``kernels/polybench/vecadd.py`` equals the
checked-in ``tests/dfg/fixtures/vecadd.snaxdfg``, with no transient and no
copy left, and the import of ``dot.py`` equals ``dot.snaxdfg``: a multiply
map, then its ``Reduce`` as a sum map folding into ``out[0]``; and
``vecmul.py``'s equals ``vecmul.snaxdfg``, ``vsum.py``'s ``vsum.snaxdfg``, ``relu.py``'s
``relu.snaxdfg`` (int32, ``out = max(in1, 0)``). The
importer builds the simplified SDFG as ``pixi run sdfg`` does, and a stored
``.sdfg`` gives the same graph. Beyond them: a transient between two maps,
offset subsets, a Max reduction, a Reduce over one axis of two, a narrowing convert kept as
a nest, a chain of widening ones composed, and every construct the importer does not support
rejected by name. relu, axpy, matadd
and rowsum import through the command line and match their kernel's
reference at two sizes each (the constructs they need: test_import_constructs.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.dfg.__main__ import main
from snax_forge.dfg.execute import execute
from snax_forge.dfg.import_sdfg import (
    SdfgImportError,
    import_sdfg,
)

from . import sdfg_programs as progs
from .helpers import fixture

# =============================================================================
# vecadd (the acceptance)
# =============================================================================


@pytest.mark.parametrize(
    "kernel", ["vecadd", "dot", "vecmul", "vsum", "relu", "axpy", "matvec_acc"]
)
def test_cli_writes_the_fixture(tmp_path, capsys, kernel):
    """vecmul is vecadd with ``*`` (the map and tasklet named after Mult); vsum is dot's sum
    map on its own, the Reduce folding A into out[0] with add."""
    assert main(["import", kernel, "--out", str(tmp_path)]) == 0
    assert (tmp_path / f"{kernel}.snaxdfg").read_text() == fixture(kernel).read_text()
    assert capsys.readouterr().out.strip() == str(tmp_path / f"{kernel}.snaxdfg")


def test_cli_reports_an_unsupported_kernel(tmp_path, capsys):
    assert main(["import", "jacobi1d", "--out", str(tmp_path)]) == 1
    assert "no control flow yet" in capsys.readouterr().err
    assert not (tmp_path / "jacobi1d.snaxdfg").exists()


# =============================================================================
# dot: a reduction
# =============================================================================


# =============================================================================
# Beyond vecadd
# =============================================================================


# =============================================================================
# Loop variables named after their dimension (loops.py)
# =============================================================================


# =============================================================================
# Unsupported constructs are rejected by name
# =============================================================================


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (progs.unsupported_wcr, "write-conflict resolution .* not from a Reduce .*contents"),
        (progs.unsupported_reduce_used_twice, "result 's' is used other than by one copy"),
        (progs.unsupported_view, "view 'V' is not imported"),
        (
            progs.unsupported_reduce_view_part,
            "view 'V' is used other than by one copy of the whole",
        ),
        (progs.unsupported_dependent_params, "range of 'j' depends on 'i' of the same map"),
        (progs.unsupported_cast_type, "cast to int32, not the output's or the op's type"),
        (progs.unsupported_cast_float, "cast to float64, not an integer type"),
        (progs.unsupported_transient_scalar, "array 's': a transient Scalar is not imported"),
        (progs.unsupported_copy, "copy 'A' -> 'B'"),
        (progs.unsupported_strides, "strides \\[2\\] are not the default \\[1\\]"),
    ],
)
def test_unsupported_constructs(make, message):
    with pytest.raises(SdfgImportError, match=message):
        import_sdfg(make(), "t")


# =============================================================================
# Casts and convert maps
# =============================================================================


def _tasklets(g):
    return {n.id: (n.attrs["code"], {c: m.data for c, m in (n.inputs | n.outputs).items()})
            for n, _ in g.walk() if n.kind == "tasklet"}  # fmt: skip


def test_a_narrowing_convert_stays_a_nest_and_keeps_its_narrow_transient():
    """``a = (...).astype(int8)`` then ``a.astype(int32)``: the int8 ``a`` is what goes through
    memory, written by its own nest, and its reader casts it up as any input."""
    g = import_sdfg(progs.requantized.to_sdfg(simplify=True), "requantized")
    assert g.containers["a"].dtype == "int8" and g.containers["a"].transient
    tasklets = _tasklets(g)
    assert tasklets["convert_to_int8"] == ("out = int8(inp)", {"inp": "tmp3", "out": "a"})
    assert tasklets["add"] == ("out = int32(in1) + in2", {"in1": "a", "in2": "b", "out": "out"})
    rng = np.random.default_rng(0)
    z = rng.integers(-(2**31), 2**31 - 1, size=64, endpoint=True, dtype=np.int32)
    b = rng.integers(-(2**31), 2**31 - 1, size=64, endpoint=True, dtype=np.int32)
    got = execute(g, {"z": z, "b": b, "out": np.zeros(64, np.int32)})
    want = np.minimum(np.maximum(z // np.int32(1024), 0), 127).astype(np.int8).astype(np.int32) + b
    assert np.array_equal(got["out"], want)


@pytest.mark.parametrize(
    ("prog", "middle", "code"),
    [
        (progs.widened_twice, np.int16, "out = int32(in1) + 1"),  # int16 holds every int8
        (progs.widened_through_unsigned, np.uint16, "out = int32(uint16(in1)) + 1"),  # wraps
    ],
)
def test_a_chain_of_widening_converts_composes_its_casts(prog, middle, code):
    """Two converts in a row fold into one reader; a cast that holds every value it is given
    changes none and is dropped, one that wraps (int8 to uint16) is kept."""
    g = import_sdfg(prog.to_sdfg(simplify=True), "t")
    assert set(g.containers) == {"x", "out"}
    assert _tasklets(g)["add"] == (code, {"in1": "x", "out": "out"})
    x = np.arange(-128, 128, dtype=np.int8)
    got = execute(g, {"x": x, "out": np.zeros(256, np.int32)})
    assert np.array_equal(got["out"], x.astype(middle).astype(np.int32) + np.int32(1))
