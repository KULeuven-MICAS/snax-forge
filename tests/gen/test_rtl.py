"""gen-rtl: accelerator.json, gen-gvsoc's bytes, checked against the datapath
catalogue for hw/chisel's ``Emit --spec``, and the fixtures hw/chisel's specs read.

Each refusal of the catalogue's checks has a test; accelerator.py's refusals, which rtl.py runs
first, are test_gvsoc.py's. The fixtures in ``hw/chisel/src/test/resources/rtl/<design>/``
(``vectors.json``; ``accelerator.json`` and the streamer hjsons, one per set, too for a design
that snax-gvsoc's ``forge/`` does not hold) are regenerated from their recipes and must be equal
byte for byte.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from snax_forge.gen import (
    Accelerator,
    Csr,
    Description,
    GenError,
    Group,
    Port,
    Unit,
)
from snax_forge.gen.accelerator import ACCELERATOR_JSON, Link
from snax_forge.gen.gvsoc import generate
from snax_forge.gen.rtl import (
    accelerator_spec,
    buffer_params,
    catalogue,
    check_params,
    check_unit,
    fill,
    flops,
    group_counter,
    halo_params,
    live,
    write_accelerator,
    write_fixture,
)
from tests.design.helpers import REPO
from tests.helpers import FORGE_FIXTURES, SUB, requires_submodule

from .helpers import designed, point, validate

RTL_FIXTURES = REPO / "hw" / "chisel" / "src" / "test" / "resources" / "rtl"
FORGE = SUB / "forge"
# forge/'s designs that the specs driving every fixture have no model for: axpby's CSR scalar
# (ShellScalarSpec checks it through ShellCasesSpec) and padded rows
NOT_RTL = {"axpy_v1", "rowsum_v2_padded"}
# Each fixture: its recipe and --set values; forge/'s designs at forge/'s sizes, then the ones a
# Chisel spec needs that forge/ does not hold (their accelerator.json and streamer hjsons beside)
DESIGNS = {
    **{d: (r, list(sets)) for d, (r, sets) in FORGE_FIXTURES.items() if d not in NOT_RTL},
    "vsum_v1": ("vsum_v1", []),  # ShellCasesSpec's fold wrapper: one lane, one input
    "vsum_v1_1_1": ("vsum_v1_1_1", []),  # ShellPackedSpec's sum, its output packed by hand
    "dot_v1_2": ("dot_v1_2", []),  # ShellEndToEndSpec's port mux run through two tasks
}

# SimdLanes' slot params of a unit with none (axpby's and axpbypcz's k1, k2 and k3, d, requant's lo,
# hi; each k from its CSR or not)
NO_SLOTS = {"k1": None, "k2": None, "kCsr": False, "k1Csr": False, "k2Csr": False, "k3": None,
            "d": None, "lo": None, "hi": None, "k3Csr": False}  # fmt: skip


@pytest.fixture
def vecadd(tmp_path):
    return point(tmp_path), tmp_path


def _edit(f):
    """The catalogue with SimdLanes' signature edited by ``f``."""

    def edited():
        cat = copy.deepcopy(catalogue())
        f(cat["SimdLanes"])
        return cat

    return edited


REFUSALS = [
    ("gen.catalogue", lambda: {k: v for k, v in catalogue().items() if k != "SimdLanes"}),
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
    sig = catalogue()["SimdLanes"]["params"]
    good = {"inTypes": ["int8", "uint8"], "opType": "int16", "outType": "int16", "lanes": 8,
            "op": "add", "streams": 2, "k": None} | NO_SLOTS  # fmt: skip
    check_params("E", good, sig)
    check_params("E", {**good, "inTypes": ["int8"], "streams": 1, "k": -(2**53)}, sig)
    for k, v in (("opType", 16), ("inTypes", "int8"), ("outType", "float32"),
                 ("inTypes", ["int8", "int128"]), ("op", "pow"), ("lanes", 0), ("streams", 0),
                 ("k", "0"), ("k", 1.5), ("k", 2**53 + 1), ("k1Csr", 1)):  # fmt: skip
        with pytest.raises(GenError) as e:
            check_params("E", {**good, k: v}, sig)
        assert e.value.code == "gen.params"


def test_a_halo_link_fills_from_its_link_and_holds_its_taps_minus_one_elements():
    """HaloLink's params from a halo link's taps and its ends' lanes and dtype (link.taps),
    checked against the catalogue; its flops H * W * width, H = ceil((T - 1) / W): 1 beat at W >= 2
    with T 3, 2 at W 1. LinkBuffer's fill is unchanged; a rule a link has not is refused."""
    cat = catalogue()
    sig = cat["HaloLink"]
    params = halo_params(3, 8, "uint8", sig)
    assert params == {"taps": 3, "lanes": 8, "dtype": "uint8"}
    check_params("HaloLink", params, sig["params"])
    assert list(live(sig["ports"], params)) == ["in", "taps"]
    assert list(live(sig["pins"], params)) == ["load"]
    for (taps, lanes, dtype), want in (((3, 8, "uint8"), 64), ((3, 1, "uint8"), 16),
                                       ((3, 32, "uint8"), 256), ((5, 8, "uint16"), 128),
                                       ((5, 2, "uint8"), 32), ((2, 4, "int32"), 128)):  # fmt: skip
        assert flops("HaloLink", halo_params(taps, lanes, dtype, sig), cat) == want
    assert buffer_params(2, 4, "int64", cat["LinkBuffer"]) == {"depth": 2, "lanes": 4,
                                                               "dtype": "int64"}  # fmt: skip
    with pytest.raises(GenError) as e:
        halo_params(3, 8, "uint8", {**sig, "fill": {**sig["fill"], "taps": "link.depth"}})
    assert e.value.code == "gen.catalogue"
    with pytest.raises(GenError) as e:
        check_params("HaloLink", {**params, "taps": 1}, sig["params"])
    assert e.value.code == "gen.params"


@requires_submodule
@pytest.mark.parametrize("design", DESIGNS)
def test_the_fixture_is_reproduced(tmp_path, design):
    """write_fixture's vectors.json is the fixture's. Its accelerator.json and streamer hjsons are
    snax-gvsoc's forge/<design>/ (test_fixture.py reproduces those); a fixture with no such folder
    keeps them beside its vectors.json, and they are reproduced here. The description meets its
    schema either way (the one written here, so not the submodule's copy)."""
    recipe, sets = DESIGNS[design]
    path = designed(tmp_path, recipe, design, sets)
    out = tmp_path / "fixture"
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        files = write_fixture(path, out, tmp_path)
        acc = write_accelerator(path, tmp_path / "spec", tmp_path)
        twin = FORGE / design
        if not twin.is_dir():
            write_accelerator(path, out, tmp_path)
            generate(path, tmp_path / "gen", tmp_path)
            streamers = [f"{acc.tag}_streamer_{a.name}.hjson" for a in acc.accelerators]
            for f in streamers:
                (out / f).write_bytes((tmp_path / "gen" / f).read_bytes())
            files += [ACCELERATOR_JSON, *streamers]
    fixture = RTL_FIXTURES / design
    assert sorted(files) == sorted(f.name for f in fixture.glob("*") if f.suffix != ".md")
    for f in files:
        assert (out / f).read_bytes() == (fixture / f).read_bytes(), f
    validate("accelerator", acc.to_dict())


def test_gen_rtl_writes_a_fixture(vecadd, capsys):
    """``gen-rtl --fixture`` writes write_fixture's vectors.json, the design point named from the
    root (no elaboration); a design point outside the root is refused."""
    from snax_forge.gen import gvsoc, rtl

    path, root = vecadd
    assert rtl.main([str(path), "--fixture", str(root / "fx")]) == 1
    assert "[gen.point]" in capsys.readouterr().err
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rtl, "ROOT", root)
        mp.setattr(gvsoc, "ROOT", root)
        mp.chdir(root)
        assert rtl.main([str(path), "--fixture", "fx"]) == 0
    assert sorted(f.name for f in (root / "fx").iterdir()) == ["vectors.json"]
    assert "wrote fx/vectors.json" in capsys.readouterr().out


@requires_submodule
def test_a_linked_fixtures_counter_is_the_group_rules():
    """A linked group's counter is snax-gvsoc's group rule stepped with an always-ready streamer:
    dot 4 at J 1 and T 2 is JT + 2 = 4; at J 16 and T 128 C4 = JT + 2; with a buffer of depth 2
    for its wire, which no formula states, a cycle more: JT + 3."""
    got = json.loads((RTL_FIXTURES / "dot_v4" / "vectors.json").read_text())["tasks"][0]["counter"]
    assert got == 4
    from tests.gen.test_accelerator import dot4

    (group,) = dot4().groups
    assert group_counter(group, 16, {"mac": 128})[0] == 2050
    deep = replace(group, links=[replace(group.links[0], depth=2)])
    assert group_counter(deep, 1, {"mac": 4})[0] == 7


@requires_submodule
def test_a_three_unit_chain_gets_its_counter():
    """A shape the table never had: mul (elementwise, L 0) -> a buffer of depth 1 -> acc (a
    fold, L 1, T 4 from JOB_BEATS) -> a wire -> sum (a fold, L 1, fixed T 2), J 1. mul fires in
    busy cycles 1-8, acc a cycle after each (2-9) and gives on its 4th and 8th (5, 9, pushed in
    6 and 10), sum fires on each push and gives on its 2nd, pushed in 11: E - s = 11."""

    def unit(name, datapath, ins, latency, params, csrs=()):
        ports = {p: Port(p, "in", "int64", 4) for p in ins} | {
            "out": Port("out", "out", "int64", 4)
        }
        return Unit(name, name, datapath, ports, params, latency, 1, list(csrs))

    mul = unit("mul", "SimdLanes", ("a", "b"), 0, {"op": "mul"})
    acc = unit("acc", "TreeAccumulate", ("a",), 1, {"op": "add"}, [Csr("JOB_BEATS", "rw", 2, 32)])
    total = unit("sum", "TreeAccumulate", ("a",), 1, {"op": "add", "T": 2})
    links = [Link("q", "mul.out", "acc.a", 1), Link("r", "acc.out", "sum.a", 0)]
    group = Group("chain", [], [mul, acc, total], links)
    value, working = group_counter(group, 1, {"acc": 4})
    assert value == 11
    assert "snax_group_rule.py" in working and "T mul 1, acc 4, sum 2: E - s = 11" in working
    assert group_counter(group, 0, {"acc": 4})[0] == 0  # the empty run


def test_a_fixtures_tasks_give_jobs_and_job_beats():
    """vectors.json's CSR values are the program's (lower/c.py's job_values)."""
    tasks = {d: json.loads((RTL_FIXTURES / d / "vectors.json").read_text())["tasks"][0]["csrs"]
             for d in ("vecadd_v1", "vsum_v1_1_1")}  # fmt: skip
    assert tasks == {"vecadd_v1": {"JOBS": 16}, "vsum_v1_1_1": {"JOBS": 1, "JOB_BEATS": 16}}


# =============================================================================
# Narrow and mixed designs: each port's type, the op at the inputs' result type
# =============================================================================


@pytest.mark.parametrize(("recipe", "dtype"), [("vecadd_v1_1", "int32"), ("vecadd_v1_2", "int8")])
def test_a_narrow_elementwise_design_is_filled(tmp_path, recipe, dtype):
    acc = accelerator_spec(tmp_path / designed(tmp_path, recipe, recipe), tmp_path)
    (u,) = acc.groups[0].units
    params = fill(acc, u, catalogue()["SimdLanes"])
    assert {k: params[k] for k in ("inTypes", "opType", "outType")} == {
        "inTypes": [dtype, dtype], "opType": dtype, "outType": dtype
    }  # fmt: skip


@pytest.mark.parametrize(("recipe", "dtype"), [("vsum_v1_1_1", "int32"), ("vsum_v1_1_2", "int8")])
def test_a_narrow_fold_into_int64_is_filled(tmp_path, recipe, dtype):
    """vsum_v1_1_1 and vsum_v1_1_2: TreeAccumulate casts each lane into its int64 sum."""
    acc = accelerator_spec(tmp_path / designed(tmp_path, recipe, recipe), tmp_path)
    (u,) = acc.groups[0].units
    params = fill(acc, u, catalogue()["TreeAccumulate"])
    assert params == {"inType": dtype, "outType": "int64", "lanes": params["lanes"],
                      "op": "add", "seed": False}  # fmt: skip


# =============================================================================
# Mac and SerialAccumulate against the catalogue: ``when`` and expression latencies
# =============================================================================


def _alone(unit: Unit) -> Description:
    """``unit`` as an accelerator of its own (one group, no link), the design's one."""
    ins = [f"{unit.name}.{k}" for k, p in unit.ports.items() if p.dir == "in"]
    outs = [f"{unit.name}.{k}" for k, p in unit.ports.items() if p.dir == "out"]
    n = len(unit.csrs)
    own = [Csr("START", "rw", 0, 1), Csr("JOBS", "rw", 1, 32), Csr("STATUS", "ro", 2 + n, 2),
           Csr("BUSY_COUNTER", "ro", 3 + n, 32)]  # fmt: skip
    acc = Accelerator(unit.name, ins, outs, [Group(unit.name, own, [unit])])
    return Description("p", "snax_forge_p", [acc])


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


# =============================================================================
# flops: the register bits of a unit's and a link's datapath
# =============================================================================


def test_axpys_unit_fills_k1_from_its_csr_and_k1_is_a_live_pin(tmp_path):
    """axpy's SimdLanes at axpby: k1 from its CSR (``scalars``), so k1Csr and the pin k1; baked,
    k1 the const and no pin. A scalar that is no live pin is refused (gen.port_map)."""
    from dataclasses import replace

    from .test_axpy import axpy

    sig = catalogue()["SimdLanes"]
    for name, graph, slots in (("axpy", axpy(), {"k1Csr": True}),
                               ("axpy_v1_1", axpy(baked=3), {"k1": 3})):  # fmt: skip
        acc = accelerator_spec(point(tmp_path / name, name=name, graph=graph), tmp_path / name)
        (u,) = acc.groups[0].units
        params = fill(acc, u, sig)
        assert (
            params
            == {
                "inTypes": ["int32", "int32"],
                "opType": "int32",
                "outType": "int32",
                "lanes": 2,
                "op": "axpby",
                "streams": 2,
                "k": None,
            }
            | NO_SLOTS
            | slots
        )
        assert list(live(sig["pins"], params)) == (["k1"] if "k1Csr" in slots else [])
        check_unit(acc, u, catalogue())
    with pytest.raises(GenError, match=r"scalar lo is no pin of SimdLanes here \(\[\]\)"):
        check_unit(acc, replace(u, scalars={"lo": "int32"}), catalogue())


def test_a_three_stream_unit_fills_axpbypcz_with_the_port_c_and_the_pin_k3(tmp_path):
    """SimdLanes at axpbypcz: three inputs (streams 3, the ports a, b and c live), k2 and d baked,
    k3 from its CSR (k3Csr, the pin k3); requant and threshold at one stream fill d, lo and hi, or
    k and hi, with no b or c. A c with no b is refused (gen.port_map): two streams are a and b."""
    from dataclasses import replace

    from .test_axpy import axpy

    sig = catalogue()["SimdLanes"]
    acc = accelerator_spec(point(tmp_path / "axpy", name="axpy", graph=axpy()), tmp_path / "axpy")
    (u,) = acc.groups[0].units
    a = next(p for p in u.ports.values() if p.port == "a")
    out = next(p for p in u.ports.values() if p.port == "out")
    three = replace(
        u,
        ports={"x": a, "y": replace(a, port="b"), "z": replace(a, port="c"), "out": out},
        params={"op": "axpbypcz", "k2": 2, "d": 4},
        scalars={"k3": "int32"},
    )
    params = fill(acc, three, sig)
    assert params == {"inTypes": ["int32"] * 3, "opType": "int32", "outType": "int32", "lanes": 2,
                      "op": "axpbypcz", "streams": 3, "k": None} | NO_SLOTS | {
                          "k2": 2, "d": 4, "k3Csr": True}  # fmt: skip
    assert list(live(sig["ports"], params)) == ["a", "b", "c", "out"]
    assert list(live(sig["pins"], params)) == ["k3"]
    check_unit(acc, three, catalogue())
    one = {"x": a, "out": out}
    for params_, want in (({"op": "requant", "d": 1024, "lo": 0, "hi": 127}, []),
                          ({"op": "threshold", "hi": 255}, ["k"])):  # fmt: skip
        unit = replace(u, ports=one, params=params_, scalars={"k": "int32"} if want else {})
        filled = fill(acc, unit, sig)
        assert {k: filled[k] for k in params_} == params_
        assert list(live(sig["ports"], filled)) == ["a", "out"]
        assert list(live(sig["pins"], filled)) == want
        check_unit(acc, unit, catalogue())
    with pytest.raises(GenError) as e:
        check_unit(acc, replace(u, ports={"x": a, "z": replace(a, port="c"), "out": out}),
                   catalogue())  # fmt: skip
    assert e.value.code == "gen.port_map"
    with pytest.raises(GenError) as e:
        check_params("E", {**params, "d": 4.0}, sig["params"])
    assert e.value.code == "gen.params"
