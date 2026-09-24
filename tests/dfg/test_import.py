"""The SDFG importer (IMP1): vecadd's simplified SDFG becomes the vecadd fixture.

Accepted when the import of ``kernels/polybench/vecadd.py`` equals the
checked-in ``tests/dfg/fixtures/vecadd.snaxdfg``, with no transient and no
copy left. The importer builds the simplified SDFG as ``pixi run forge``
does, and a stored ``.sdfg`` gives the same graph. Beyond vecadd: a
transient between two maps, offset subsets, and every construct the
importer does not support rejected by name.
"""

from __future__ import annotations

import dace
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


def test_vecadd_import_is_the_fixture(vecadd_sdfg):
    g = import_sdfg(vecadd_sdfg, "vecadd")
    assert g.to_json() == fixture("vecadd").read_text()


def test_no_transient_and_no_copy_left(vecadd_sdfg):
    """simplify already folded C[:] = A + B into a direct write (D77)."""
    g = import_sdfg(vecadd_sdfg, "vecadd")
    assert list(g.containers) == ["A", "B", "C"]
    assert not any(c.transient for c in g.containers.values())
    assert [n.kind for n, _ in g.walk()] == ["map", "tasklet"]
    assert g.node("add").outputs["out"].data == "C"


def test_the_kernel_is_int64():
    """Open item 30, decided for vecadd: the kernel moves to the model's int64."""
    g = import_kernel("vecadd")
    assert {c.dtype for c in g.containers.values()} == {"int64"}


def test_a_stored_sdfg_gives_the_same_graph(vecadd_sdfg, tmp_path):
    """What `pixi run forge vecadd` writes imports to the same graph."""
    path = tmp_path / "vecadd.simplified.sdfg"
    vecadd_sdfg.save(path)
    assert import_sdfg(dace.SDFG.from_file(path), "vecadd") == Graph.load(fixture("vecadd"))


def test_cli_writes_the_fixture(tmp_path, capsys):
    assert main(["import", "vecadd", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "vecadd.snaxdfg").read_text() == fixture("vecadd").read_text()
    assert capsys.readouterr().out.strip() == str(tmp_path / "vecadd.snaxdfg")


def test_cli_reports_an_unsupported_kernel(tmp_path, capsys):
    assert main(["import", "dot", "--out", str(tmp_path)]) == 1
    assert "library node Reduce" in capsys.readouterr().err
    assert not (tmp_path / "dot.snaxdfg").exists()


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
        ("dot", "library node Reduce is not imported"),
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
        (progs.unsupported_wcr, "write-conflict resolution .* \\(DFG3\\)"),
        (progs.unsupported_two_params, "over 2 parameters"),
        (progs.unsupported_cast, "Call is not allowed .* open item 37"),
        (progs.unsupported_copy, "copy 'A' -> 'B'"),
        (progs.unsupported_strides, "strides \\[2\\] are not the default \\[1\\]"),
    ],
)
def test_unsupported_constructs(make, message):
    with pytest.raises(SdfgImportError, match=message):
        import_sdfg(make(), "t")
