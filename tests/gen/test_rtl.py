"""rtl.json (D130, C§17): the design point resolved for hw/chisel's ``Emit --spec`` and
checked against the block catalogue, and the fixtures hw/chisel's specs read.

Each refusal of the checks has a test; the GVSoC adapter's refusals, which rtl.py runs first, are
test_gvsoc.py's. The fixtures in ``hw/chisel/src/test/resources/rtl/<design>/`` (``rtl.json``,
the streamer hjson and ``vectors.json``) are regenerated from their recipes and must be equal
byte for byte.
"""

from __future__ import annotations

import copy
import json

import pytest

from snax_forge.gen import ELEMENTWISE_OP, GenError
from snax_forge.gen.rtl import (
    RtlSpec,
    catalogue,
    check_params,
    counter,
    resolve_csrs,
    rtl_spec,
    write_fixture,
    write_rtl,
)
from snax_forge.gen.schemas import validate
from tests.design.helpers import REPO

from .helpers import bound_vsum, designed, point

FIXTURES = REPO / "hw" / "chisel" / "src" / "test" / "resources" / "rtl"
# Each fixture: its recipe and --set
DESIGNS = {
    "vecadd": ("vecadd", []),
    "vecmul": ("vecmul", []),
    "vsum": ("vsum", []),
    "vsum_tree": ("vsum_tree", []),
    "vsum_tree_w8": ("vsum_tree", ["W=8"]),
    "vecadd_i32": ("vecadd_i32", []),
    "vecmul_i32": ("vecmul_i32", []),
    "vecadd_i8": ("vecadd_i8", []),
    "vsum_tree_i32": ("vsum_tree_i32", []),
    "vsum_tree_i8": ("vsum_tree_i8", []),
    "vecmul_i32_i64": ("vecmul_i32_i64", []),
}


@pytest.fixture
def vecadd(tmp_path):
    return point(tmp_path), tmp_path


def test_the_vecadd_spec(vecadd):
    path, root = vecadd
    spec = write_rtl(path, root / "rtl", root)
    d = json.loads((root / "rtl" / "rtl.json").read_text())
    assert d == spec.to_dict()
    assert d == {
        "design_point": "design/vecadd/design_point.json",
        "tag": "snax_forge",
        "units": [{
            "name": "acc",
            "block": "ElementwiseTiledSpatial",
            "params": {"inTypes": ["int64", "int64"], "opType": "int64", "outType": "int64",
                       "lanes": 4, "loopCountWidth": 32, "supportedOps": ["add"]},
            "ports": [{"brm": "a", "block": "a"}, {"brm": "b", "block": "b"},
                      {"brm": "out", "block": "out"}],
            "csrs": {"rw": [{"pin": "loopCount", "width": 32}], "start": "start",
                     "ro": ["busy", "counter"], "tie": {"opSel": 0}},
        }],
    }  # fmt: skip
    validate("rtl", d)
    assert RtlSpec.from_dict(d).to_dict() == d
    assert RtlSpec.load(root / "rtl" / "rtl.json") == spec


def test_the_accumulator_maps_its_port_and_has_two_fields(tmp_path):
    spec = rtl_spec(point(tmp_path, name="vsum", graph=bound_vsum()), tmp_path)
    (u,) = spec.units
    assert (u.block, u.params) == (
        "Accumulator",
        {"inType": "int64", "outType": "int64", "lanes": 1},
    )
    assert u.ports == [("a", "in"), ("out", "out")]
    assert (u.rw, u.start, u.tie) == ([("loopCount", 32), ("len", 32)], "start", {})


def test_the_catalogues_ops_are_elementwise_ops():
    """The catalogue's op encodings are blocks.py's (which tests read from the Chisel source)."""
    ops = catalogue()["ElementwiseTiledSpatial"]["params"]["supportedOps"]["values"]
    assert ops == ELEMENTWISE_OP


def _edit(f):
    """The catalogue with ElementwiseTiledSpatial's signature edited by ``f``."""

    def edited():
        cat = copy.deepcopy(catalogue())
        f(cat["ElementwiseTiledSpatial"])
        return cat

    return edited


REFUSALS = [
    (
        "gen.catalogue",
        lambda: {k: v for k, v in catalogue().items() if k != "ElementwiseTiledSpatial"},
    ),
    ("gen.params", _edit(lambda b: b["params"].update(depth={"type": "int", "min": 1}))),
    ("gen.param_type", _edit(lambda b: b["params"].update(opType={"type": "int", "min": 0}))),
    ("gen.param_limit", _edit(lambda b: b["params"]["outType"].update(values=["int32"]))),
    ("gen.param_limit", _edit(lambda b: b["params"]["lanes"].update(max=2))),
    ("gen.port_map", _edit(lambda b: b.update(ports={"a": "in", "c": "in", "out": "out"}))),
    ("gen.port_map", _edit(lambda b: b["ports"].update(b="out"))),
    ("gen.csr_pins", _edit(lambda b: b["csr"].update(rw=[{"pin": "len", "width": 32}]))),
    ("gen.csr_pins", _edit(lambda b: b["csr"]["rw"][0].update(width=64))),
]


@pytest.mark.parametrize(("code", "cat"), REFUSALS, ids=[f"{c}-{i}" for i, (c, _) in
                                                         enumerate(REFUSALS)])  # fmt: skip
def test_a_spec_the_catalogue_refuses(vecadd, code, cat):
    with pytest.raises(GenError) as e:
        rtl_spec(vecadd[0], vecadd[1], cat())
    assert e.value.code == code


def test_an_ops_param_must_be_op_names():
    sig = catalogue()["ElementwiseTiledSpatial"]["params"]
    good = {"inTypes": ["int64", "int64"], "opType": "int64", "outType": "int64", "lanes": 4,
            "loopCountWidth": 32, "supportedOps": ["add"]}  # fmt: skip
    check_params("E", good, sig)
    for ops, code in (("add", "gen.param_type"), ([], "gen.param_limit"),
                      (["add", "add"], "gen.param_limit"), (["pow"], "gen.param_limit")):  # fmt: skip
        with pytest.raises(GenError) as e:
            check_params("E", {**good, "supportedOps": ops}, sig)
        assert e.value.code == code


def test_a_tie_takes_one_op():
    csr = catalogue()["ElementwiseTiledSpatial"]["csr"]
    ops = {"supportedOps": ELEMENTWISE_OP}
    params = {"loopCountWidth": 32, "supportedOps": ["mul"]}
    assert resolve_csrs("E", params, ["loopCount"], csr, ops) == ([("loopCount", 32)], {"opSel": 2})
    with pytest.raises(GenError) as e:
        resolve_csrs("E", {**params, "supportedOps": ["add", "mul"]}, ["loopCount"], csr, ops)
    assert e.value.code == "gen.tie"


def test_the_counter_is_worked_from_the_shells_rules(tmp_path):
    from snax_forge.design.point import DesignPoint
    from snax_forge.gen.gvsoc import accelerator

    _, _, add = accelerator(DesignPoint.load(point(tmp_path)))
    _, _, acc = accelerator(DesignPoint.load(point(tmp_path, name="vsum", graph=bound_vsum())))
    assert counter(add, 16)[0] == 16
    assert counter(acc, 64)[0] == 65


@pytest.mark.parametrize("design", DESIGNS)
def test_the_fixture_is_reproduced(tmp_path, design):
    recipe, sets = DESIGNS[design]
    path = designed(tmp_path, recipe, design, sets)
    out = tmp_path / "fixture"
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        files = write_fixture(path, out, tmp_path)
    fixture = FIXTURES / design
    assert sorted(files) == sorted(f.name for f in fixture.iterdir())
    for f in files:
        assert (out / f).read_bytes() == (fixture / f).read_bytes(), f
    validate("rtl", json.loads((fixture / "rtl.json").read_text()))


def test_gen_rtl_writes_a_fixture(vecadd, capsys):
    """``gen-rtl --fixture`` writes write_fixture's files, the design point named from the root
    (the elaboration needs sbt, which the chisel job runs); a design point outside the root is
    refused."""
    from snax_forge.gen import gvsoc, rtl

    path, root = vecadd
    assert rtl.main([str(path), "--fixture", str(root / "fx")]) == 1
    assert "[gen.point]" in capsys.readouterr().err
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rtl, "ROOT", root)
        mp.setattr(gvsoc, "ROOT", root)
        mp.chdir(root)
        assert rtl.main([str(path), "--fixture", "fx"]) == 0
    assert sorted(f.name for f in (root / "fx").iterdir()) == [
        "rtl.json", "snax_forge_streamer.hjson", "vectors.json"
    ]  # fmt: skip
    assert "wrote fx/rtl.json" in capsys.readouterr().out


def test_a_type_param_must_be_type_names():
    sig = catalogue()["ElementwiseTiledSpatial"]["params"]
    good = {"inTypes": ["int8", "uint8"], "opType": "int16", "outType": "int16", "lanes": 8,
            "loopCountWidth": 32, "supportedOps": ["add"]}  # fmt: skip
    check_params("E", good, sig)
    for k, v, code in (("opType", 16, "gen.param_type"), ("inTypes", "int8", "gen.param_type"),
                       ("outType", "float32", "gen.param_limit"),
                       ("inTypes", ["int8", "int128"], "gen.param_limit")):  # fmt: skip
        with pytest.raises(GenError) as e:
            check_params("E", {**good, k: v}, sig)
        assert e.value.code == code


# =============================================================================
# Narrow and mixed designs (PACK0 R4): each port's type, the op at the inputs' result type
# =============================================================================


@pytest.mark.parametrize(("recipe", "dtype"), [("vecadd_i32", "int32"), ("vecadd_i8", "int8")])
def test_a_narrow_elementwise_design_is_written(tmp_path, recipe, dtype):
    spec = rtl_spec(tmp_path / designed(tmp_path, recipe, recipe), tmp_path)
    validate("rtl", spec.to_dict())
    (unit,) = spec.to_dict()["units"]
    params = {k: unit["params"][k] for k in ("inTypes", "opType", "outType")}
    assert params == {"inTypes": [dtype, dtype], "opType": dtype, "outType": dtype}


@pytest.mark.parametrize(
    ("recipe", "dtype"), [("vsum_tree_i32", "int32"), ("vsum_tree_i8", "int8")]
)
def test_a_narrow_fold_into_int64_is_written(tmp_path, recipe, dtype):
    """vsum_tree_i32 and vsum_tree_i8: the Accumulator casts each lane into its int64 sum."""
    spec = rtl_spec(tmp_path / designed(tmp_path, recipe, recipe), tmp_path)
    validate("rtl", spec.to_dict())
    (unit,) = spec.to_dict()["units"]
    assert unit["params"] == {"inType": dtype, "outType": "int64", "lanes": unit["params"]["lanes"]}


@pytest.mark.parametrize(
    ("a", "b", "out", "op_type"),
    [("int32", "int32", "int64", "int32"), ("int8", "uint8", "int16", "int16"),
     ("uint8", "int32", "int32", "int32"), ("uint32", "int32", "int64", "int64")],
)  # fmt: skip
def test_the_op_runs_at_the_inputs_result_type(a, b, out, op_type):
    from snax_forge.gen.blocks import elementwise_tiled_spatial
    from snax_forge.gen.rtl import RTL_PARAMS

    from .test_gvsoc import _dtypes, _instance

    inst = _instance(params={"W": 8, "op": "add"})  # 8 lanes: whole words of int8
    unit = elementwise_tiled_spatial(inst, _dtypes(a=a, b=b, out=out))
    params = RTL_PARAMS["ElementwiseTiledSpatial"](inst, unit)
    assert (params["inTypes"], params["opType"], params["outType"]) == ([a, b], op_type, out)
