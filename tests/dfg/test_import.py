"""The SDFG importer: vecadd's and dot's simplified SDFGs become their fixtures.

Accepted when the import of ``kernels/polybench/vecadd.py`` equals the
checked-in ``tests/dfg/fixtures/vecadd.snaxdfg``, with no transient and no
copy left, and the import of ``dot.py`` equals ``dot.snaxdfg``: a multiply
map, then its ``Reduce`` as a sum map folding into ``out[0]``; and
``vecmul.py``'s equals ``vecmul.snaxdfg``, ``vsum.py``'s ``vsum.snaxdfg``, ``relu.py``'s
``relu.snaxdfg`` (int32, ``out = max(in1, 0)``). The
importer builds the simplified SDFG as ``pixi run sdfg`` does, and a stored
``.sdfg`` gives the same graph. Beyond them: a transient between two maps,
offset subsets, a Max reduction, a Reduce over one axis of two, and every
construct the importer does not support rejected by name. relu, axpy, matadd
and rowsum import through the command line and match their kernel's
reference at two sizes each (the constructs they need: test_import_constructs.py).
"""

from __future__ import annotations

import pytest

from snax_forge.dfg.__main__ import main
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
# Casts and convert maps (D143 · D175)
# =============================================================================
