"""The SDFG importer: vecadd's and dot's simplified SDFGs become their fixtures.

Accepted when the import of ``kernels/polybench/vecadd.py`` equals the
checked-in ``tests/dfg/fixtures/vecadd.snaxdfg``, with no transient and no
copy left, and the import of ``dot.py`` equals ``dot.snaxdfg``: a multiply
map, then its ``Reduce`` as a sum map folding into ``out[0]``; and
``vecmul.py``'s equals ``vecmul.snaxdfg``, ``vsum.py``'s ``vsum.snaxdfg``. The
importer builds the simplified SDFG as ``pixi run sdfg`` does, and a stored
``.sdfg`` gives the same graph. Beyond them: a transient between two maps,
offset subsets, a Max reduction, a Reduce over one axis of two, and every
construct the importer does not support rejected by name. relu, axpy, matadd
and rowsum import through the command line and match their kernel's
reference at two sizes each (the constructs they need: test_import_constructs.py).
"""

from __future__ import annotations

import dace
import numpy as np
import pytest

from snax_forge.dfg import Graph
from snax_forge.dfg.__main__ import main
from snax_forge.dfg.import_sdfg import (
    SdfgImportError,
    clean,
    fresh,
    import_kernel,
    import_sdfg,
)
from snax_forge.sdfg.build import build
from snax_forge.sdfg.loader import load

from . import sdfg_programs as progs
from .helpers import fixture


@pytest.fixture(scope="module")
def vecadd_sdfg() -> dace.SDFG:
    return build(load("vecadd"), simplify=True)


# =============================================================================
# vecadd (the acceptance)
# =============================================================================


def test_no_transient_and_no_copy_left(vecadd_sdfg):
    """simplify already folded C[:] = A + B into a direct write."""
    g = import_sdfg(vecadd_sdfg, "vecadd")
    assert list(g.containers) == ["A", "B", "C"]
    assert not any(c.transient for c in g.containers.values())
    assert [n.kind for n, _ in g.walk()] == ["map", "tasklet"]
    assert g.node("add").outputs["out"].data == "C"


def test_the_kernel_is_int64():
    """vecadd imports as int64: every container's dtype."""
    g = import_kernel("vecadd")
    assert {c.dtype for c in g.containers.values()} == {"int64"}


def test_a_stored_sdfg_gives_the_same_graph(vecadd_sdfg, tmp_path):
    """What `pixi run sdfg vecadd` writes imports to the same graph."""
    path = tmp_path / "vecadd.simplified.sdfg"
    vecadd_sdfg.save(path)
    assert import_sdfg(dace.SDFG.from_file(path), "vecadd") == Graph.load(fixture("vecadd"))


@pytest.mark.parametrize("kernel", ["vecadd", "dot", "vecmul", "vsum"])
def test_cli_writes_the_fixture(tmp_path, capsys, kernel):
    """vecmul is vecadd with ``*`` (the map and tasklet named after Mult); vsum is dot's sum
    map on its own, the Reduce folding A into out[0] with add."""
    assert main(["import", kernel, "--out", str(tmp_path)]) == 0
    assert (tmp_path / f"{kernel}.snaxdfg").read_text() == fixture(kernel).read_text()
    assert capsys.readouterr().out.strip() == str(tmp_path / f"{kernel}.snaxdfg")


@pytest.mark.parametrize(
    ("kernel", "sizes"),
    [
        ("relu", ["1024", "7"]),
        ("axpy", ["1024", "7"]),
        ("matadd", ["256 x 256", "256 x 8"]),  # --n sets N only: one square, one not
        ("rowsum", ["256 x 256", "256 x 8"]),
    ],
)
def test_cli_imports_and_checks_the_b0_kernels(tmp_path, capsys, kernel, sizes):
    """Each imports, then runs in the reference executor equal to the kernel's reference."""
    assert main(["import", kernel, "--out", str(tmp_path)]) == 0
    path = str(tmp_path / f"{kernel}.snaxdfg")
    capsys.readouterr()
    for n in (None, "7" if sizes[1] == "7" else "8"):
        assert main(["check", path, "--kernel", kernel, *(["--n", n] if n else [])]) == 0
    out = capsys.readouterr().out
    assert out.count("OK ") == 2
    for size in sizes:
        dims = size.split(" x ")
        want = f"N = {dims[0]})" if len(dims) == 1 else f"M = {dims[0]}, N = {dims[1]})"
        assert want in out


def test_cli_reports_an_unsupported_kernel(tmp_path, capsys):
    assert main(["import", "jacobi1d", "--out", str(tmp_path)]) == 1
    assert "control flow is open item 36" in capsys.readouterr().err
    assert not (tmp_path / "jacobi1d.snaxdfg").exists()


# =============================================================================
# dot: a reduction
# =============================================================================


def test_dot_is_a_multiply_map_then_a_sum_map():
    """The Reduce's scalar and its copy are folded: the sum map writes out[0]."""
    g = import_kernel("dot")
    assert list(g.containers) == ["A", "B", "out", "tmp0"]
    assert [c for c, cont in g.containers.items() if cont.transient] == ["tmp0"]
    assert {c.dtype for c in g.containers.values()} == {"int64"}
    assert [(n.id, n.attrs.get("var")) for n in g.body] == [("mult_map", "i"), ("sum_map", "i")]
    s = g.node("sum")
    assert s.attrs["code"] == "out = in1"
    assert s.inputs["in1"].to_dict() == {"data": "tmp0", "subset": ["i"]}
    assert s.outputs["out"].to_dict() == {
        "data": "out",
        "subset": [0],
        "reduce": {"op": "add", "identity": 0},
    }


def test_max_reduction():
    """np.max: op max, identity the dtype's smallest value, named after the reduction."""
    g = import_sdfg(progs.amax.to_sdfg(simplify=True), "amax")
    assert list(g.containers) == ["A", "out"]
    assert [n.id for n, _ in g.walk()] == ["max_map", "max"]
    red = g.node("max").outputs["out"].reduce
    assert (red.op, red.identity) == ("max", int(np.iinfo(np.int64).min))


def test_reduce_over_one_axis():
    """A Reduce over axis 1 of A[4, 6]: a map per dimension, the output keeps i."""
    g = import_sdfg(progs.row_sum(), "row_sum")
    assert [(n.id, n.attrs.get("var")) for n, _ in g.walk()] == [
        ("sum_map", "i"),
        ("sum_map_1", "j"),
        ("sum", None),
    ]
    s = g.node("sum")
    assert s.inputs["in1"].subset == ["i", "j"]
    assert s.outputs["out"].subset == ["i"]


# =============================================================================
# Beyond vecadd
# =============================================================================


def test_transient_between_two_maps():
    """Two maps in one state, in order, sibling variables both ``i``, ``__tmp0`` -> ``tmp0``."""
    g = import_sdfg(progs.two.to_sdfg(simplify=True), "two")
    assert list(g.containers) == ["A", "B", "C", "tmp0"]
    assert g.containers["tmp0"].transient
    assert [(n.id, n.attrs.get("var")) for n in g.body] == [("add_map", "i"), ("mult_map", "i")]
    assert g.node("add").outputs["out"].data == "tmp0"
    assert g.node("mult").inputs["in1"].data == "tmp0"
    assert g.node("mult").attrs["code"] == "out = in1 * in2"


def test_offset_subsets_and_ranges():
    """B[1:-1] = A[:-2] + A[2:]: a range of N - 2 and indices i, i + 2, i + 1."""
    g = import_sdfg(progs.shift.to_sdfg(simplify=True), "shift")
    assert g.node("add_map").attrs["range"] == "0:N - 2"
    add = g.node("add")
    assert [m.subset for m in add.inputs.values()] == [["i"], ["i + 2"]]
    assert add.outputs["out"].subset == ["i + 1"]


def test_names():
    assert clean("_Add__map") == "add_map"
    assert clean("__tmp0") == "tmp0"
    assert clean("__in1") == "in1"
    assert clean("assign_14_4") == "assign_14_4"
    taken = {"add"}
    assert fresh("add", taken) == "add_1"
    assert fresh("add", taken) == "add_2"
    assert fresh("mul", taken) == "mul"


# =============================================================================
# Unsupported constructs are rejected by name
# =============================================================================


@pytest.mark.parametrize(
    ("kernel", "message"),
    [
        ("jacobi1d", "4 states .*control flow is open item 36"),
    ],
)
def test_unsupported_kernels(kernel, message):
    with pytest.raises(SdfgImportError, match=message):
        import_kernel(kernel)


def test_raw_vecadd_has_three_states():
    """The raw SDFG still has the transient copy in a state of its own."""
    with pytest.raises(SdfgImportError, match="3 states"):
        import_sdfg(build(load("vecadd"), simplify=False), "vecadd")


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
        (progs.unsupported_cast, "cast 'dace.int64\\(a\\)' .* a non-constant .* open item 37"),
        (progs.unsupported_cast_type, "cast to int32, not the output's type int64 .* open item 37"),
        (progs.unsupported_cast_float, "cast to float64, not an integer type .* open item 37"),
        (progs.unsupported_transient_scalar, "array 's': a transient Scalar is not imported"),
        (progs.unsupported_copy, "copy 'A' -> 'B'"),
        (progs.unsupported_strides, "strides \\[2\\] are not the default \\[1\\]"),
    ],
)
def test_unsupported_constructs(make, message):
    with pytest.raises(SdfgImportError, match=message):
        import_sdfg(make(), "t")
