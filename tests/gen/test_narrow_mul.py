"""``design.narrow_mul`` (D207): the design point keeps it only when true, and ``accelerator.json``
then puts ``narrowMul`` true on every Mac and every SimdLanes with a product, and nowhere else;
a default design's ``accelerator.json`` is as it was. The catalogue's ``narrowMul`` may be left
out of a block, and a value that is no bool is refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.design import DesignPoint
from snax_forge.design.check import load, run_checks
from snax_forge.gen import GenError
from snax_forge.gen.block import check_entry
from snax_forge.gen.blocks import DATAPATHS, NARROW_MUL, Fill
from snax_forge.gen.gvsoc import load_point
from tests.design.helpers import SNAX_FORGE

from .helpers import designed
from .test_block import MAC, entry

# A Mac, a SimdLanes mul, axpby and add, a TreeAccumulate and a bank
DESIGNS = ("dot_v1", "dot_v4", "axpy_v1", "vecadd_v1", "mvt_v6")  # vecadd: none marked


def _product(unit: dict) -> bool:
    return unit["datapath"] == "Mac" or (
        unit["datapath"] == "SimdLanes" and unit["params"]["op"] in ("mul", "axpby")
    )


def _units(desc: dict) -> list[dict]:
    return [u for a in desc["accelerators"] for g in a["groups"] for u in g["units"]]


@pytest.mark.parametrize("recipe", DESIGNS)
def test_narrow_mul_marks_every_product_and_nothing_else(tmp_path, recipe):
    wide_root, narrow_root = tmp_path / "wide", tmp_path / "narrow"  # one name, two roots
    wide_path = wide_root / designed(wide_root, recipe, recipe)
    narrow_path = narrow_root / designed(narrow_root, recipe, recipe, ["design.narrow_mul=true"])
    _, wide_point, wide = load_point(wide_path, wide_root)
    _, narrow_point, narrow = load_point(narrow_path, narrow_root)
    assert "narrow_mul" not in json.loads(wide_path.read_text())
    assert json.loads(narrow_path.read_text())["narrow_mul"] is True
    assert (wide_point.narrow_mul, narrow_point.narrow_mul) == (False, True)
    w, n = wide.to_dict(), narrow.to_dict()
    assert NARROW_MUL not in json.dumps(w)
    marked = [u["name"] for u in _units(n) if u["params"].get(NARROW_MUL) is True]
    assert marked == [u["name"] for u in _units(n) if _product(u)]
    for u in _units(n):
        u["params"].pop(NARROW_MUL, None)
    assert n == w


def test_a_simd_product_is_mul_axpby_or_an_axpbypcz_with_a_k():
    product = DATAPATHS["SimdLanes"].product
    assert product(Fill({}, "int32", {"op": "mul"}))
    assert product(Fill({}, "int32", {"op": "axpby"}, {"k1": "int32"}))
    assert product(Fill({}, "int32", {"op": "axpbypcz", "k2": 3, "d": 1}))
    assert product(Fill({}, "int32", {"op": "axpbypcz", "d": 1}, {"k3": "int32"}))
    for params in ({"op": "add"}, {"op": "axpbypcz", "d": 4}, {"op": "requant", "d": 2, "lo": 0,
                   "hi": 1}, {"op": "threshold", "k": 1, "hi": 1}):  # fmt: skip
        assert not product(Fill({}, "int32", params)), params
    assert DATAPATHS["TreeAccumulate"].product is None


def test_narrow_mul_that_is_no_bool_is_refused(tmp_path):
    from tests.gen.helpers import bound_vecmul

    graph = tmp_path / "vecmul.snaxdfg"
    bound_vecmul().save(graph)
    design = load(graph, SNAX_FORGE, design_sets=[("narrow_mul", "yes")])
    assert [p.code for p in run_checks(design)] == ["design.narrow_mul"]
    assert run_checks(load(graph, SNAX_FORGE, design_sets=[("narrow_mul", True)])) == []


def test_a_design_point_keeps_narrow_mul(tmp_path):
    path = tmp_path / designed(tmp_path, "dot_v4", "dot_v4_nm", ["design.narrow_mul=true"])
    p = DesignPoint.load(path)
    assert DesignPoint.from_dict(p.to_dict()) == p and p.narrow_mul is True


def test_a_block_may_leave_narrow_mul_out_and_set_it(tmp_path):
    check_entry(entry(tmp_path, "Mac", **MAC))
    check_entry(entry(tmp_path, "Mac", **MAC, narrowMul=True))
    with pytest.raises(GenError) as e:
        check_entry(entry(tmp_path, "Mac", **MAC, narrowMul=1))
    assert e.value.code == "gen.params" and "narrowMul: 1 is not a bool" in str(e.value)


def test_the_catalogue_gives_narrow_mul_a_default():
    cat = json.loads((Path(__file__).parents[2] / "hw" / "chisel" / "blocks.json").read_text())
    for name, sig in cat.items():
        has = NARROW_MUL in sig["params"]
        assert has == (DATAPATHS[name].product is not None if name in DATAPATHS else False), name
        if has:
            assert sig["params"][NARROW_MUL] == {"type": "bool", "default": False}
            assert sig["fill"][NARROW_MUL] == f"unit.params.{NARROW_MUL}"
