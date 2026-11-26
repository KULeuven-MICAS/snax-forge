"""The generator: vecadd on snax_forge as its description files (the cluster, one
streamer per set, accelerator.json), vecmul, vsum, and every refusal."""

from __future__ import annotations

import copy
import re
from dataclasses import replace
from pathlib import Path

import pytest

from snax_forge.brm import CONTROLLERS, Brm, Instance, load_brm
from snax_forge.common import generated_header
from snax_forge.design import Platform
from snax_forge.gen import DATAPATHS, ELEMENTWISE_OP, GenError, generate
from snax_forge.gen import __main__ as cli
from snax_forge.gen import csrs as csrs_mod
from snax_forge.gen import gvsoc as gen_mod
from snax_forge.gen.blocks import simd_lanes, tree_accumulate
from snax_forge.lower import c as c_mod
from tests.brm.helpers import IMPL
from tests.design.helpers import REPO, SNAX_FORGE
from tests.helpers import requires_submodule

from .helpers import (
    SOURCE,
    SUB,
    designed,
    one_unit,
    parsed,
    point,
)


def streamer_file(design: str, acc: str = "acc") -> str:
    """Set ``acc``'s streamer file, ``<tag>_streamer_<set>.hjson``, the tag snax_forge_<design>."""
    return f"snax_forge_{design}_streamer_{acc}.hjson"


FILES = ["snax_forge.hjson", streamer_file("vecadd"), "accelerator.json"]
# The slot of snax_forge's source, its only one, which the cluster file renames and sets
SLOT = ("snax_forge_core_template", "snax_acc_cfg", 0)

# vecadd at W = 4 on snax_forge: the template the design point's streamers give
TEMPLATE_W4 = {
    "data_reader_params": {
        "spatial_bounds": [[4], [4]], "temporal_dim": [1, 1], "num_channel": [4, 4],
        "fifo_depth": [8, 8],
    },
    "data_writer_params": {
        "spatial_bounds": [[4]], "temporal_dim": [1], "num_channel": [4], "fifo_depth": [8],
    },
}  # fmt: skip
# vecadd's accelerator at W = 4: one group of one SimdLanes unit,
# each port int64
GROUP_CSRS = [
    {"name": "START", "access": "rw", "offset": 0, "width": 1},
    {"name": "JOBS", "access": "rw", "offset": 1, "width": 32},
    {"name": "STATUS", "access": "ro", "offset": 2, "width": 2},
    {"name": "BUSY_COUNTER", "access": "ro", "offset": 3, "width": 32},
]  # fmt: skip
ADD_W4 = {
    "design_point": "design/vecadd/design_point.json", "tag": "snax_forge_vecadd",
    "accelerators": [{"name": "acc", "inputs": ["acc.a", "acc.b"], "outputs": ["acc.out"],
    "groups": [{"name": "acc", "csrs": GROUP_CSRS, "units": [{
        "name": "acc", "brm": "simd", "datapath": "SimdLanes",
        "ports": {p: {"port": p, "dir": "out" if p == "out" else "in", "dtype": "int64", "lanes": 4}
                  for p in ("a", "b", "out")},
        "opType": "int64", "params": {"op": "add"}, "timing": {"latency": 0, "ii": 1, "span": 1},
        "csrs": [],
    }], "links": []}]}],
}  # fmt: skip
# vsum on snax_forge: TreeAccumulate, a fold of one lane, one sum per T inputs; its JOB_BEATS
# between the group's JOBS and STATUS
VSUM = {
    "design_point": "design/vsum/design_point.json", "tag": "snax_forge_vsum",
    "accelerators": [{"name": "acc", "inputs": ["acc.a"], "outputs": ["acc.out"],
    "groups": [{"name": "acc", "csrs": [
        *GROUP_CSRS[:2], {**GROUP_CSRS[2], "offset": 3}, {**GROUP_CSRS[3], "offset": 4},
    ], "units": [{
        "name": "acc", "brm": "accumulate", "datapath": "TreeAccumulate",
        "ports": {"a": {"port": "in", "dir": "in", "dtype": "int64", "lanes": 1},
                  "out": {"port": "out", "dir": "out", "dtype": "int64", "lanes": 1}},
        "params": {"op": "add"}, "timing": {"latency": 1, "ii": 1, "span": 1},
        "csrs": [{"name": "JOB_BEATS", "access": "rw", "offset": 2, "width": 32}],
    }], "links": []}]}],
}  # fmt: skip


def slot(cluster: dict) -> dict:
    t, k, i = SLOT
    return cluster[t][k][i]


@pytest.fixture(scope="module")
def w4(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w4")
    out = tmp / "gvsoc" / "vecadd"
    return out, generate(point(tmp), out, root=tmp)


# =============================================================================
# vecadd at W = 4 on snax_forge
# =============================================================================


def test_the_cluster_file_is_the_source_renamed_with_the_slot_set(w4):
    out, _ = w4
    cluster, source = parsed(out / "snax_forge.hjson"), parsed(SOURCE)
    expected = {
        "snax_acc_name": "snax_forge_vecadd", "snax_tcdm_ports": 12,
        "sparse_interconnect_config": [[12, 1]],
        "snax_disable_csr_manager": True, "snax_forge_rtl": "rtl/files.f",
        "snax_streamer_sets": [{"name": "acc",
                                "snax_streamer_cfg": {"$ref": streamer_file("vecadd")},
                                "snax_num_rw_csr": 2, "snax_num_ro_csr": 2}],
        "snax_datapath_cfg": {"$ref": "accelerator.json"},
    }  # fmt: skip
    assert slot(cluster) == expected
    assert list(slot(cluster)) == list(expected)  # the slot's key order
    assert cluster["cluster"]["name"] == "snax_forge_cluster"
    assert cluster["cluster"]["bender_target"] == [
        "snitch_cluster", "snax_forge_cluster", "sparse_interconnect"
    ]  # fmt: skip
    # Outside the slot and the cluster's names, the source: its template, which only the slot
    # referred to, gone
    assert "snax_forge_streamer_template" not in cluster
    del source["snax_forge_streamer_template"]
    for d in (cluster, source):
        slot(d).clear()
        del d["cluster"]["name"], d["cluster"]["bender_target"]
    assert cluster == source
    assert list(cluster) == list(source)  # the source's key order


def test_every_generated_file_starts_with_the_one_header(w4):
    """hjson, headers, data.h and main.c: the generator, the design point, the command;
    accelerator.json and obs_marks.json are plain JSON; a set's streamer header is snax_cluster's
    text, its
    timestamp line naming the design point (headers.py)."""
    out, m = w4
    head = generated_header("design/vecadd/design_point.json", "")
    assert "`pixi run gen-gvsoc design/vecadd/design_point.json`" in head.replace("\n// ", " ")
    for f in m.files:
        if f in ("manifest.json", "accelerator.json", "obs_marks.json"):
            continue
        if f.startswith("include/streamer_"):
            assert "design point design/vecadd/design_point.json" in (out / f).read_text()
            continue
        text = (out / f).read_text()
        assert text.startswith(head), f
        assert all(len(line) <= 80 for line in text.splitlines() if line.startswith("//")), f


def test_elementwise_op_is_chisels():
    src = REPO / "hw/chisel/src/main/scala/snax/forge/elementwise/ElementwiseAlu.scala"
    body = src.read_text().split("object ElementwiseOp")[1].split("val width")[0]
    table = {k.lower(): int(v) for k, v in re.findall(r"val (\w+):\s*Int = (\d+)", body)}
    assert table == ELEMENTWISE_OP


# =============================================================================
# Refusals
# =============================================================================


def _refused(tmp_path, code, platform=None, source=None):
    """generate on snax_forge (or ``platform``), the root's source replaced by ``source``;
    refused with ``code`` before anything is written."""
    path = point(tmp_path, platform=platform)
    if source is not None:
        (tmp_path / "platforms" / SOURCE.name).write_text(source)
    with pytest.raises(GenError) as e:
        generate(path, tmp_path / "out", root=tmp_path)
    assert e.value.code == code
    assert not (tmp_path / "out").exists()
    return str(e.value)


@pytest.mark.parametrize(
    ("edit", "code", "text"),
    [
        (lambda s: s.replace("        snax_acc_cfg: [{\n",
                             '        snax_acc_cfg: [{snax_acc_name: "other"}, {\n'),
         "gen.slot", "['other', 'snax_forge']"),  # two slots
        (lambda s: re.sub(r"snax_acc_cfg: \[\{.*?\}\],", "snax_acc_cfg: [],", s, flags=re.DOTALL),
         "gen.slot", "has the slots []"),  # no slot
        (lambda s: s.replace('"snax_forge_cluster", "sparse_interconnect"', '"sparse_interconnect"'),
         "gen.source", "does not hold cluster.name 'snax_forge_cluster'"),  # bender target
        (lambda s: s.replace("        tcdm: {\n", '        tcdm: {$ref: "#/tcdm"},\n'
                             "        old_tcdm: {\n"),
         "gen.source", "cluster.tcdm is not an object written out"),  # nowhere to write
    ],
    ids=["two slots", "no slot", "bender target without the name", "tcdm not written out"],
)  # fmt: skip
def test_a_source_the_generator_refuses(tmp_path, edit, code, text):
    assert text in _refused(tmp_path, code, source=edit(SOURCE.read_text()))


@pytest.mark.parametrize(
    ("sets", "text"),
    [
        ([("l1.rows", 501)], "not whole KiB"),
    ],
)
def test_a_platform_value_the_hjson_cannot_express(tmp_path, sets, text):
    p = Platform.load(SNAX_FORGE).with_changes(sets)
    assert text in _refused(tmp_path, "gen.platform", p)


def test_the_design_checks_limits_are_not_the_cluster_files(tmp_path):
    """l1.max_xbar_inputs and l1.warn_xbar_inputs hold the design check only: raised by --set,
    the cluster file is written as on the default platform."""
    p = Platform.load(SNAX_FORGE).with_changes(
        [("l1.max_xbar_inputs", 256), ("l1.warn_xbar_inputs", 128)]
    )
    generate(
        point(tmp_path / "raised", platform=p),
        tmp_path / "raised" / "out",
        root=tmp_path / "raised",
    )
    generate(point(tmp_path / "plain"), tmp_path / "plain" / "out", root=tmp_path / "plain")
    written = sorted(
        f.relative_to(tmp_path / "raised" / "out")
        for f in (tmp_path / "raised" / "out").rglob("*.hjson")
    )
    assert written
    for f in written:
        assert (tmp_path / "raised" / "out" / f).read_text() == (
            tmp_path / "plain" / "out" / f
        ).read_text()


def _instance(edit=None, params=None, name="simd", impl=IMPL) -> Instance:
    """An instance (simd's by default), its BRM dict edited (no resolve: the
    datapath adapter's checks only)."""
    d = copy.deepcopy(load_brm(name).to_dict())
    if edit:
        edit(d)
    brm = Brm.from_dict(d)
    return Instance(brm, impl, params or {"W": 4, "op": "add", "streams": 2})


def _binding(impl=IMPL, **kw):
    return lambda d: d["implementations"][impl]["binding"].update(kw)


def _dtypes(ports=("a", "b", "out"), **kw):
    """The node's port dtypes (its containers', brm.port_dtypes): int64 unless given."""
    return {p: kw.get(p, "int64") for p in ports}


def _ports(**kw):
    def edit(d):
        for p in d["interface"]["ports"]:
            p.update(kw.get(p["name"], {}))

    return edit


ADAPTER_CASES = [
    ("op", None, {"W": 4, "op": "nand", "streams": 2}, "gen.op"),
    ("no k", None, {"W": 4, "op": "max", "streams": 1}, "gen.args"),
    ("k at two", None, {"W": 4, "op": "max", "streams": 2, "k": 0}, "gen.args"),
    ("k too wide", None, {"W": 8, "op": "add", "streams": 1, "k": 128}, "gen.k",
     _dtypes(("a", "out"), a="int8", out="int8")),
    ("k below", None, {"W": 4, "op": "add", "streams": 1, "k": -1}, "gen.k",
     _dtypes(("a", "out"), a="uint16", out="uint16")),
    ("args", _binding(args={"lanes": "W"}), {}, "gen.args"),
    ("float", None, {}, "gen.dtype", _dtypes(a="float64", b="float64", out="float64")),
    # 3 lanes of int8 neither fill whole words nor divide one (W = 4 would be sliced)
    ("lanes x N", None, {"W": 3, "op": "add", "streams": 2}, "gen.lanes",
     _dtypes(a="int8", b="int8", out="int8")),
    ("lanes vs port", _binding(args={"lanes": "W * 2", "op": "op"}), {}, "gen.lanes"),
]  # fmt: skip


@pytest.mark.parametrize(
    ("case", "edit", "params", "code", "dtypes"),
    [(*c, None)[:5] for c in ADAPTER_CASES], ids=[c[0] for c in ADAPTER_CASES],
)  # fmt: skip
def test_the_adapter_refuses(case, edit, params, code, dtypes):
    with pytest.raises(GenError) as e:
        simd_lanes(_instance(edit, params or None), dtypes)
    assert e.value.code == code


ACC = "chisel_adder_tree"


def _acc(edit=None, params=None) -> Instance:
    return _instance(edit, params or {"W": 1, "op": "add"}, name="accumulate", impl=ACC)


def _second_input(d):
    d["interface"]["ports"].insert(1, dict(d["interface"]["ports"][0], name="b"))
    d["dataflow"]["ports"]["b"] = d["dataflow"]["ports"]["a"]
    for impl in d["implementations"].values():  # the binding's ports map every BRM port
        impl["binding"]["ports"]["b"] = "b"


ACCUMULATOR_CASES = [
    ("no args", _binding(ACC, args={}), {}, "gen.args"),
    ("args", _binding(ACC, args={"lanes": "W", "accWidth": 64}), {}, "gen.args"),
    ("op", None, {"W": 1, "op": "max"}, "gen.op"),
    ("two inputs", _second_input, {}, "gen.ports"),
    ("rate in", _ports(a={"rate": "T"}), {}, "gen.rate"),
    ("lanes", _binding(ACC, args={"lanes": 1}), {"W": 2, "op": "add"}, "gen.lanes"),
    ("lanes out", _ports(out={"lanes": "W"}), {"W": 2, "op": "add"}, "gen.lanes"),
    # 3 lanes of int16 in: neither whole words nor a slice of one (1 lane of int32 is sliced)
    ("narrow", None, {"W": 3, "op": "add"}, "gen.lanes", _dtypes(("a", "out"), a="int16", out="int16")),
]  # fmt: skip


@pytest.mark.parametrize(
    ("case", "edit", "params", "code", "dtypes"),
    [(*c, None)[:5] for c in ACCUMULATOR_CASES], ids=[c[0] for c in ACCUMULATOR_CASES],
)  # fmt: skip
def test_the_accumulator_adapter_refuses(case, edit, params, code, dtypes):
    with pytest.raises(GenError) as e:
        tree_accumulate(_acc(edit, params or None), dtypes)
    assert e.value.code == code


def test_three_ports_are_refused():
    def three_inputs(d):
        d["interface"]["ports"].insert(2, dict(d["interface"]["ports"][1], name="c"))
        d["dataflow"]["ports"]["c"] = d["dataflow"]["ports"]["b"]
        d["function"]["code"] = "out = a + b + c"

    with pytest.raises(GenError) as e:
        simd_lanes(_instance(three_inputs))
    assert e.value.code == "gen.ports"


def _edited(monkeypatch, binding=None, timing=None):
    """lower/c.py's ``instances`` with the one instance's binding replaced by ``binding(b)``
    and its timing by ``timing``'s fields (no resolve: what describe refuses)."""
    real = c_mod.instances

    def edited(graph):
        (name, inst), = real(graph).items()  # fmt: skip
        b = inst.impl.binding if binding is None else binding(inst.impl.binding)
        t = inst.impl.timing if timing is None else replace(inst.impl.timing, **timing)
        brm = copy.copy(inst.brm)
        brm.implementations = {IMPL: replace(inst.impl, binding=b, timing=t)}
        return {name: Instance(brm, IMPL, inst.params)}

    monkeypatch.setattr(c_mod, "instances", edited)


@pytest.mark.parametrize(
    ("binding", "timing", "code"),
    [
        (lambda b: None, None, "gen.binding"),
        (lambda b: {**b, "datapath": "Gemm"}, None, "gen.datapath"),
        (lambda b: {**b, "controller": "other"}, None, "gen.controller"),
        (None, {"drain": 1}, "gen.drain"),
        (None, {"latency": 1}, "gen.timing"),
        (None, {"initiation_interval": 2}, "gen.timing"),
    ],
    ids=["null binding", "datapath", "controller", "drain", "latency", "ii"],
)  # fmt: skip
def test_what_describe_refuses(tmp_path, monkeypatch, binding, timing, code):
    """A null binding, an unregistered datapath, a controller other than counted (one
    registered later is not generated), a drain (the nest controller has none), a latency
    other than the datapath's, an II other than 1."""
    monkeypatch.setitem(CONTROLLERS, "other", "a test's")
    path = point(tmp_path)
    _edited(monkeypatch, binding, timing)
    with pytest.raises(GenError) as e:
        generate(path, tmp_path / "out", root=tmp_path)
    assert e.value.code == code
    assert not (tmp_path / "out").exists()
    assert "SimdLanes" in DATAPATHS


def test_the_command(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(gen_mod, "ROOT", tmp_path)
    path = point(tmp_path)
    out = tmp_path / "out"
    assert cli.main([str(path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "cluster snax_forge.hjson" in text and "task add_acc: 16 beats" in text
    assert "TARGET=snax_hjson" in text and f"--param hjson={out / 'snax_forge.hjson'}" in text
    assert (out / "manifest.json").is_file()
    (tmp_path / "platforms" / SOURCE.name).unlink()
    assert cli.main([str(path), "--out", str(out)]) == 1
    assert "[gen.source]" in capsys.readouterr().err


@requires_submodule
def test_the_constants_are_snax_accelerators():
    text = (SUB / "snax" / "snax_accelerator.py").read_text()
    assert "SNAX_CSR_FIRST = 0x3C0" in text and "SNAX_ACC_CSR_LAST = 0x5FD" in text
    assert (csrs_mod.SNAX_CSR_FIRST, csrs_mod.SNAX_ACC_CSR_LAST) == (0x3C0, 0x5FD)


def test_the_accelerators_counter_may_land_on_0x5fd_not_past():
    """vecadd's streamer (18 CSRs) and a unit of 552 R/W CSRs: the counter on 0x5FD, the last
    CSR an accelerator may use; one more puts it on 0x5FE, a task queue of snax_cluster's."""
    k = 0x5FD - 0x3C0 - 18 - 4 + 1
    layout = csrs_mod.csr_layout({"acc": TEMPLATE_W4}, one_unit(k))
    assert layout.accelerator["ACC_RO_BUSY_COUNTER"] == 0x5FD
    with pytest.raises(GenError) as e:
        csrs_mod.csr_layout({"acc": TEMPLATE_W4}, one_unit(k + 1))
    assert e.value.code == "gen.csrs"
    assert "the entry's last CSR 1534 is past 1533" in str(e.value)


# =============================================================================
# Narrow designs: the streamer in words, each port's type
# =============================================================================


def _narrow(tmp_path, recipe: str, sets: tuple[str, ...] = ()) -> Path:
    """``recipes/<folder>/<recipe>.json`` (``--set`` each of ``sets``) designed and generated under
    ``tmp_path``; the out folder."""
    point = designed(tmp_path, recipe, recipe, sets)
    out = tmp_path / "gvsoc" / recipe
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        generate(point, out, root=tmp_path)
    return out


def test_the_output_folder_is_the_design_point_s_or_its_flow_s(tmp_path):
    """gen-gvsoc, gen-rtl and bundle default to under/<folder>: the design step's folder, or for
    a flow's design point (``out/flow/<name>/design/``) the flow's, with its ``--set`` suffix."""
    from snax_forge.gen.gvsoc import out_dir

    under = tmp_path / "out" / "rtl"
    flow = tmp_path / "out" / "flow" / "dot_1_dry_W_vmul8_W_vsum4" / "design" / "design_point.json"
    assert out_dir(None, under, flow) == under / "dot_1_dry_W_vmul8_W_vsum4"
    design = tmp_path / "out" / "design" / "vecadd_W8" / "design_point.json"
    assert out_dir(None, under, design) == under / "vecadd_W8"
    assert out_dir(tmp_path / "x", under, flow) == tmp_path / "x"
