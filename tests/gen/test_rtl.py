"""gen-rtl (D130, C§17): accelerator.json, gen-gvsoc's bytes, checked against the datapath
catalogue for hw/chisel's ``Emit --spec``, and the fixtures hw/chisel's specs read.

Each refusal of the catalogue's checks has a test; accelerator.py's refusals, which rtl.py runs
first, are test_gvsoc.py's. The fixtures in ``hw/chisel/src/test/resources/rtl/<design>/``
(``accelerator.json``, the streamer hjson and ``vectors.json``) are regenerated from their
recipes and must be equal byte for byte.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from snax_forge.gen import ELEMENTWISE_OP, Accelerator, Csr, GenError, Group, Port, Unit
from snax_forge.gen.rtl import (
    accelerator_spec,
    catalogue,
    check_params,
    check_unit,
    counter,
    fill,
    live,
    write_accelerator,
    write_fixture,
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
    "dot_1": ("dot_1", []),
}


@pytest.fixture
def vecadd(tmp_path):
    return point(tmp_path), tmp_path


def test_gen_rtl_writes_gen_gvsocs_accelerator_json(vecadd):
    """The same bytes as gen-gvsoc's (D145), and they meet the schema."""
    from snax_forge.gen import generate

    path, root = vecadd
    acc = write_accelerator(path, root / "rtl", root)
    generate(path, root / "gvsoc", root)
    text = (root / "rtl" / "accelerator.json").read_text()
    assert text == (root / "gvsoc" / "accelerator.json").read_text() == acc.to_json()
    validate("accelerator", json.loads(text))


def test_the_fill_rule_gives_the_datapaths_params(vecadd, tmp_path):
    path, root = vecadd
    acc = accelerator_spec(path, root)
    (u,) = acc.groups[0].units
    sig = catalogue()["ElementwiseLanes"]
    assert fill(acc, u, sig) == {"inTypes": ["int64", "int64"], "opType": "int64",
                                 "outType": "int64", "lanes": 4, "op": "add"}  # fmt: skip
    vsum = accelerator_spec(
        point(tmp_path / "vsum", name="vsum", graph=bound_vsum()), tmp_path / "vsum"
    )
    (v,) = vsum.groups[0].units
    assert fill(vsum, v, catalogue()["TreeAccumulate"]) == {
        "inType": "int64", "outType": "int64", "lanes": 1, "op": "add"
    }  # fmt: skip


def test_the_catalogues_ops_are_elementwise_ops():
    """The catalogue's op names are blocks.py's (whose encodings tests read from the Chisel source)."""
    ops = catalogue()["ElementwiseLanes"]["params"]["op"]["values"]
    assert ops == list(ELEMENTWISE_OP)


def test_the_catalogue_has_the_datapaths_latency_span_and_pins():
    cat = catalogue()
    assert {k: (v["latency"], v["span"]) for k, v in cat.items()} == {
        "ElementwiseLanes": (0, 1), "TreeAccumulate": (1, 1), "Mac": ("max(accumulate, regOut)", 1),
        "SerialAccumulate": (1, "lanes"), "LinkBuffer": (1, 1),
    }  # fmt: skip
    pins = {k: live(v["pins"], {"accumulate": True, "regOut": False}) for k, v in cat.items()}
    assert {k: list(v) for k, v in pins.items()} == {
        "ElementwiseLanes": [], "TreeAccumulate": ["fire", "first"], "Mac": ["fire", "first"],
        "SerialAccumulate": ["read", "lane", "first"], "LinkBuffer": ["load", "wslot", "rslot"],
    }  # fmt: skip
    assert list(live(cat["Mac"]["pins"], {"accumulate": False, "regOut": True})) == ["en"]
    assert live(cat["Mac"]["pins"], {"accumulate": False, "regOut": False}) == {}
    for v in cat.values():
        assert set(v["fill"]) == set(v["params"])


def _edit(f):
    """The catalogue with ElementwiseLanes' signature edited by ``f``."""

    def edited():
        cat = copy.deepcopy(catalogue())
        f(cat["ElementwiseLanes"])
        return cat

    return edited


REFUSALS = [
    ("gen.catalogue", lambda: {k: v for k, v in catalogue().items() if k != "ElementwiseLanes"}),
    ("gen.timing", _edit(lambda d: d.update(latency=1))),
    ("gen.timing", _edit(lambda d: d.update(span=2))),
    ("gen.catalogue", _edit(lambda d: d["fill"].update(lanes="unit.width"))),
    ("gen.params", _edit(lambda d: d["params"]["outType"].update(values=["int32"]))),
    ("gen.params", _edit(lambda d: d["params"]["lanes"].update(min=8))),
    ("gen.params", _edit(lambda d: d["params"]["op"].update(values=["mul"]))),
    ("gen.port_map", _edit(lambda d: d.update(ports={"a": "in", "c": "in", "out": "out"}))),
    ("gen.port_map", _edit(lambda d: d["ports"].update(b="out"))),
]


@pytest.mark.parametrize(("code", "cat"), REFUSALS, ids=[f"{c}-{i}" for i, (c, _) in
                                                         enumerate(REFUSALS)])  # fmt: skip
def test_a_unit_the_catalogue_refuses(vecadd, code, cat):
    with pytest.raises(GenError) as e:
        accelerator_spec(vecadd[0], vecadd[1], cat())
    assert e.value.code == code


def test_a_type_param_must_be_type_names():
    sig = catalogue()["ElementwiseLanes"]["params"]
    good = {"inTypes": ["int8", "uint8"], "opType": "int16", "outType": "int16", "lanes": 8,
            "op": "add"}  # fmt: skip
    check_params("E", good, sig)
    for k, v in (("opType", 16), ("inTypes", "int8"), ("outType", "float32"),
                 ("inTypes", ["int8", "int128"]), ("op", "pow"), ("lanes", 0)):  # fmt: skip
        with pytest.raises(GenError) as e:
            check_params("E", {**good, k: v}, sig)
        assert e.value.code == "gen.params"


def test_the_counter_is_worked_from_the_controllers_rule(tmp_path):
    from snax_forge.design.point import DesignPoint
    from snax_forge.gen import describe

    (add,) = describe(DesignPoint.load(point(tmp_path)), "p").groups[0].units
    vsum = DesignPoint.load(point(tmp_path, name="vsum", graph=bound_vsum()))
    (acc,) = describe(vsum, "p").groups[0].units
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
    validate("accelerator", json.loads((fixture / "accelerator.json").read_text()))


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
        "accelerator.json", "snax_forge_streamer.hjson", "vectors.json"
    ]  # fmt: skip
    assert "wrote fx/accelerator.json" in capsys.readouterr().out


def test_a_fixtures_tasks_give_jobs_and_job_beats():
    """vectors.json's CSR values are the program's (lower/c.py's job_values)."""
    tasks = {d: json.loads((FIXTURES / d / "vectors.json").read_text())["tasks"][0]["csrs"]
             for d in ("vecadd", "vsum_tree_i32")}  # fmt: skip
    assert tasks == {"vecadd": {"JOBS": 16}, "vsum_tree_i32": {"JOBS": 1, "JOB_BEATS": 16}}


# =============================================================================
# Narrow and mixed designs (PACK0 R4): each port's type, the op at the inputs' result type
# =============================================================================


@pytest.mark.parametrize(("recipe", "dtype"), [("vecadd_i32", "int32"), ("vecadd_i8", "int8")])
def test_a_narrow_elementwise_design_is_filled(tmp_path, recipe, dtype):
    acc = accelerator_spec(tmp_path / designed(tmp_path, recipe, recipe), tmp_path)
    (u,) = acc.groups[0].units
    params = fill(acc, u, catalogue()["ElementwiseLanes"])
    assert {k: params[k] for k in ("inTypes", "opType", "outType")} == {
        "inTypes": [dtype, dtype], "opType": dtype, "outType": dtype
    }  # fmt: skip


@pytest.mark.parametrize(
    ("recipe", "dtype"), [("vsum_tree_i32", "int32"), ("vsum_tree_i8", "int8")]
)
def test_a_narrow_fold_into_int64_is_filled(tmp_path, recipe, dtype):
    """vsum_tree_i32 and vsum_tree_i8: TreeAccumulate casts each lane into its int64 sum."""
    acc = accelerator_spec(tmp_path / designed(tmp_path, recipe, recipe), tmp_path)
    (u,) = acc.groups[0].units
    params = fill(acc, u, catalogue()["TreeAccumulate"])
    assert params == {"inType": dtype, "outType": "int64", "lanes": params["lanes"],
                      "op": "add"}  # fmt: skip


# =============================================================================
# Mac and SerialAccumulate against the catalogue (B1x): ``when`` and expression latencies
# =============================================================================


def _alone(unit: Unit) -> Accelerator:
    """``unit`` as an accelerator of its own (one group, no link)."""
    ins = [f"{unit.name}.{k}" for k, p in unit.ports.items() if p.dir == "in"]
    outs = [f"{unit.name}.{k}" for k, p in unit.ports.items() if p.dir == "out"]
    n = len(unit.csrs)
    own = [Csr("START", "rw", 0, 32), Csr("JOBS", "rw", 1, 32), Csr("STATUS", "ro", 2 + n),
           Csr("COUNTER", "ro", 3 + n)]  # fmt: skip
    return Accelerator(unit.name, "p", "snax_forge", ins, outs, [Group(unit.name, own, [unit])])


def _mac(accumulate: bool, reg_out: bool = False, c: bool | None = None, latency: int | None = None,
         span: int = 1) -> Unit:  # fmt: skip
    """A Mac unit of 4 int64 lanes: ``c`` (by default the plain form's) and the latency
    ``max(accumulate, regOut)`` unless given."""
    ports = {k: Port(k, "in", "int64", 4) for k in ("a", "b")}
    if (not accumulate) if c is None else c:
        ports["c"] = Port("c", "in", "int64", 4)
    ports["out"] = Port("out", "out", "int64", 4)
    lat = max(accumulate, reg_out) if latency is None else latency
    csrs = [Csr("JOB_BEATS", "rw", 2, 32)] if accumulate else []
    return Unit("mac", "mac", "Mac", ports, {"accumulate": accumulate, "regOut": reg_out}, lat, 1,
                csrs, "int64", span)  # fmt: skip


@pytest.mark.parametrize(("accumulate", "reg_out"), [(True, False), (False, False), (False, True)])
def test_a_mac_unit_in_each_form_is_accepted(accumulate, reg_out):
    unit = _mac(accumulate, reg_out)
    acc = _alone(unit)
    check_unit(acc, unit, catalogue())
    assert fill(acc, unit, catalogue()["Mac"])["inTypes"] == ["int64"] * (2 if accumulate else 3)


def _behind_mac(unit: Unit) -> Accelerator:
    """dot 5's group: a Mac wired into ``unit`` (``sum``), which reads from the link."""
    from tests.gen.test_accelerator import dot4

    acc = dot4()
    (g,) = acc.groups
    return replace(acc, groups=[replace(g, units=[g.units[0], unit])])


def test_a_serial_accumulate_unit_spans_its_lanes():
    """Behind a link (dot 5): span 4 is the catalogue's; span 1 is gen.timing. Fed by the
    streamer it is gen.span, SNAX-GVSoC's limit until B2."""
    unit = Unit("sum", "accumulate", "SerialAccumulate",
                {"a": Port("in", "in", "int64", 4), "out": Port("out", "out", "int64", 1)},
                {"T": 1}, 1, 1, [], None, 4)  # fmt: skip
    check_unit(_behind_mac(unit), unit, catalogue())
    with pytest.raises(GenError) as e:
        bad = replace(unit, span=1)
        check_unit(_behind_mac(bad), bad, catalogue())
    assert e.value.code == "gen.timing"
    with pytest.raises(GenError) as e:
        _alone(unit)
    assert e.value.code == "gen.span"


@pytest.mark.parametrize(
    ("unit", "code"),
    [
        (_mac(True, c=True), "gen.port_map"),  # the accumulate form has no c
        (_mac(False, c=False), "gen.port_map"),  # the plain form needs c
        (_mac(True, span=2), "gen.span"),  # span > 1 on the streamer's inputs: SNAX-GVSoC's limit
        (_mac(True, latency=0), "gen.timing"),  # max(accumulate, regOut) = 1
        (_mac(False, True, latency=0), "gen.timing"),  # 1 with regOut
        (_mac(False, latency=1), "gen.timing"),  # 0 plain
    ],
    ids=["accumulate-c", "plain-no-c", "span", "latency-accumulate", "latency-regout",
         "latency-plain"],
)  # fmt: skip
def test_a_mac_unit_the_catalogue_refuses(unit, code):
    with pytest.raises(GenError) as e:
        check_unit(_alone(unit), unit, catalogue())
    assert e.value.code == code
