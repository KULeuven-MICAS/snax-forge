"""The generator (D134): vecadd on snax_forge as three hjson files, vecmul,
vsum, and every refusal."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from snax_forge.brm import CONTROLS, Brm, Instance, load_brm
from snax_forge.common import generated_header
from snax_forge.design import DesignError, DesignPoint, Platform
from snax_forge.dfg import Graph
from snax_forge.gen import BLOCKS, ELEMENTWISE_OP, GenError, Manifest, generate
from snax_forge.gen import __main__ as cli
from snax_forge.gen import gvsoc as gen_mod
from snax_forge.gen.blocks import accumulator, elementwise_tiled_spatial
from snax_forge.gen.data import array_text, arrays
from snax_forge.lower import c as c_mod
from tests.brm.helpers import IMPL
from tests.design.helpers import REPO, SNAX_FORGE, chained, graph_dict

from .helpers import (
    SOURCE,
    SUB,
    bound_vecmul,
    bound_vsum,
    parsed,
    point,
    snax_cfg,
    snax_streamer,
)

needs_submodule = pytest.mark.skipif(
    not (SUB / "snax" / "snax_cfg.py").exists(),
    reason="the snax-gvsoc submodule is not checked out (only CI's gvsoc job fetches it)",
)
FILES = ["snax_forge.hjson", "snax_forge_streamer.hjson", "snax_forge_acc.hjson"]
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
# vecadd's datapath block at W = 4: ElementwiseTiledSpatial as a SnaxUnit and its
# SnaxShell's field; int64 gives signed True (only min and max read it)
TILED_ADD = {
    "unit": "SnaxUnit", "inputs": ["a", "b"], "outputs": ["out"], "lanes": 4, "data_width": 64,
    "op": 0, "signed": True, "control": "counted", "wait": "lockstep", "latency": 0, "ii": 1,
    "fields": ["loopCount"],
}  # fmt: skip
# vsum on snax_forge: the Accumulator, a fold of one lane, one sum per T inputs
ACCUMULATOR = {
    "unit": "SnaxUnit", "inputs": ["a"], "outputs": ["out"], "lanes": 1, "data_width": 64,
    "op": 0, "signed": True, "control": "counted", "config": ["count", "rate"],
    "wait": "pipelined", "latency": 1, "ii": 1, "fields": ["loopCount", "len"],
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


def test_a_seed_reaches_data_h_header_and_arrays(tmp_path):
    """A seed given to generate is the one data.h's header names and its inputs are made with."""
    out, path = tmp_path / "gvsoc" / "vecadd", point(tmp_path)
    generate(path, out, root=tmp_path, seed=3)
    text = (out / "data.h").read_text()
    assert "the inputs from make_inputs (seed 3)" in text
    p = DesignPoint.load(path)
    assert arrays(p, 3) != arrays(p, 0)
    assert text.endswith("\n".join(array_text(a) for a in arrays(p, 3)))


def test_vecadd_w4_files_and_manifest(w4):
    out, m = w4
    assert m.design == "vecadd" and m.cluster == "snax_forge.hjson"
    assert m.files == [
        *FILES, "include/streamer_csr_addr_map.h", "include/acc_csr.h",
        "include/snitch_cluster_cfg.h", "include/snitch_cluster_addrmap.h", "data.h", "main.c",
        "manifest.json",
    ]  # fmt: skip
    assert m.program == "main.c"
    assert sorted(str(f.relative_to(out)) for f in out.rglob("*") if f.is_file()) == sorted(m.files)
    assert m.trace == "snax_acc0/unit"
    assert m.design_point == "design/vecadd/design_point.json"
    assert m.symbols == {"N": 64}
    d = json.loads((out / "manifest.json").read_text())
    assert list(d) == ["design", "design_point", "symbols", "cluster", "files", "program",
                       "trace", "csrs", "tasks"]  # fmt: skip
    assert (out.parent.parent / m.design_point).is_file()
    assert m.csrs.to_dict() == {
        "streamer": {"first": 960, "last": 977}, "fields": {"loopCount": 978},
        "start": 979, "busy": 980, "counter": 981,
    }  # fmt: skip
    assert [t.to_dict() for t in m.tasks] == [
        {"name": "add_acc", "beats": 16, "csrs": {"loopCount": 16}}
    ]
    assert Manifest.load(out / "manifest.json") == m
    assert Manifest.from_dict(m.to_dict()) == m


def test_the_streamer_file_is_the_design_points_streamers(w4):
    out, _ = w4
    assert parsed(out / "snax_forge_streamer.hjson") == TEMPLATE_W4


def test_the_streamer_file_is_the_hjsons_template(w4):
    """Lowered per design point, vecadd at W = 4 gives the source's own template back."""
    out, _ = w4
    hand = parsed(SOURCE)["snax_forge_streamer_template"]
    for group in ("data_reader_params", "data_writer_params"):
        assert parsed(out / "snax_forge_streamer.hjson")[group] == hand[group]


def test_the_accelerator_file_is_vecadds_datapath_block(w4):
    out, _ = w4
    assert parsed(out / "snax_forge_acc.hjson") == TILED_ADD
    assert "config" not in parsed(out / "snax_forge_acc.hjson")  # the default, left out


def test_the_cluster_file_is_the_source_renamed_with_the_slot_set(w4):
    out, _ = w4
    cluster, source = parsed(out / "snax_forge.hjson"), parsed(SOURCE)
    expected = {
        "snax_acc_name": "snax_forge", "snax_tcdm_ports": 12,
        "sparse_interconnect_config": [[12, 1]], "snax_num_rw_csr": 2, "snax_num_ro_csr": 2,
        "snax_disable_csr_manager": True, "snax_forge_rtl": "rtl/files.f",
        "snax_streamer_cfg": {"$ref": "snax_forge_streamer.hjson"},
        "snax_datapath_cfg": {"$ref": "snax_forge_acc.hjson"},
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


@needs_submodule
def test_snax_gvsocs_reader_takes_the_folder(w4):
    """snax-gvsoc's snax_cfg.load: the whole-file refs resolve and the slot's counts agree."""
    out, _ = w4
    cfg = snax_cfg().load(out / "snax_forge.hjson")
    (acc,) = cfg.cores[0].accelerators
    assert (acc.name, acc.tcdm_ports, acc.streamer) == ("snax_forge", 12, TEMPLATE_W4)
    assert acc.datapath.fields == ["loopCount"] and acc.datapath.config == ["count"]
    assert (cfg.tcdm.size, cfg.tcdm.banks, cfg.tcdm.sparse_interconnect) == (131072, 32, True)


@needs_submodule
def test_streamer_csrs_are_snax_streamers(w4):
    """The generator's copy of snax_streamer.py's nb_csrs gives what the real one gives."""
    out, _ = w4
    template = parsed(out / "snax_forge_streamer.hjson")
    assert gen_mod.streamer_csrs(template) == snax_streamer().nb_csrs(template) == 18
    assert gen_mod.channels(template) == snax_streamer().nb_tcdm_ports(template) == 12


def test_every_generated_file_starts_with_the_one_header(w4):
    """hjson, headers, data.h and main.c: the generator, the design point, the command."""
    out, m = w4
    head = generated_header("design/vecadd/design_point.json", "")
    assert "`pixi run gen-gvsoc design/vecadd/design_point.json`" in head.replace("\n// ", " ")
    for f in m.files:
        if f == "manifest.json":
            continue
        text = (out / f).read_text()
        assert text.startswith(head), f
        assert all(len(line) <= 80 for line in text.splitlines() if line.startswith("//")), f


def test_vecadd_w8_gives_eight_channels_per_mover(tmp_path):
    out = tmp_path / "out"
    m = generate(point(tmp_path, w=8, name="vecadd_w8"), out, root=tmp_path)
    assert parsed(out / "snax_forge_streamer.hjson")["data_reader_params"]["num_channel"] == [8, 8]
    s = slot(parsed(out / "snax_forge.hjson"))
    assert (s["snax_tcdm_ports"], s["sparse_interconnect_config"]) == (24, [[24, 1]])
    assert m.csrs.start == 979 and m.tasks[0].beats == 8


def test_vecmul_is_vecadd_with_op_mul(tmp_path, w4):
    """elementwise_mul's binding gives vecadd's accelerator with op 2."""
    out = tmp_path / "gvsoc" / "vecmul"
    m = generate(point(tmp_path, name="vecmul", graph=bound_vecmul()), out, root=tmp_path)
    assert parsed(out / "snax_forge_acc.hjson") == {**TILED_ADD, "op": 2}
    assert parsed(out / "snax_forge_streamer.hjson") == TEMPLATE_W4
    assert m.symbols == {"N": 64} and m.design_point == "design/vecmul/design_point.json"
    assert m.files == w4[1].files and m.csrs == w4[1].csrs and m.trace == "snax_acc0/unit"
    assert [t.to_dict() for t in m.tasks] == [
        {"name": "mult_acc", "beats": 16, "csrs": {"loopCount": 16}}
    ]
    assert parsed(out / "snax_forge.hjson") == parsed(w4[0] / "snax_forge.hjson")
    for h in ("streamer_csr_addr_map.h", "acc_csr.h"):
        mine, add = (out / "include" / h).read_text(), (w4[0] / "include" / h).read_text()
        assert mine.replace("vecmul", "vecadd") == add, h


def test_vsum_is_the_accumulator(tmp_path):
    """accumulate's Accumulator, one reader and one writer of one lane, the shell's fields
    loopCount and len, one task of 64 beats summing all 64 (T = n)."""
    out = tmp_path / "gvsoc" / "vsum"
    m = generate(point(tmp_path, name="vsum", graph=bound_vsum()), out, root=tmp_path)
    assert parsed(out / "snax_forge_acc.hjson") == ACCUMULATOR
    one = {"spatial_bounds": [[1]], "temporal_dim": [1], "num_channel": [1], "fifo_depth": [8]}
    assert parsed(out / "snax_forge_streamer.hjson") == {
        "data_reader_params": one, "data_writer_params": one
    }  # fmt: skip
    s = slot(parsed(out / "snax_forge.hjson"))
    assert (s["snax_tcdm_ports"], s["sparse_interconnect_config"]) == (2, [[2, 1]])
    assert (s["snax_num_rw_csr"], s["snax_num_ro_csr"]) == (3, 2)
    assert m.symbols == {"N": 64}
    assert m.csrs.to_dict() == {
        "streamer": {"first": 960, "last": 972}, "fields": {"loopCount": 973, "len": 974},
        "start": 975, "busy": 976, "counter": 977,
    }  # fmt: skip
    assert [t.to_dict() for t in m.tasks] == [
        {"name": "sum_acc", "beats": 64, "csrs": {"loopCount": 64, "len": 64}}
    ]
    assert "#define ACC_RW_LEN 974\n" in (out / "include" / "acc_csr.h").read_text()


def test_vsum_at_w4_is_the_adder_tree(tmp_path):
    """accumulate's adder tree at W = 4 is the Accumulator with lanes 4, one task of 16 beats
    of 4 lanes summing all 64 (T = n)."""
    out = tmp_path / "gvsoc" / "vsum_tree"
    graph = bound_vsum(4, "chisel_adder_tree")
    m = generate(point(tmp_path, name="vsum_tree", graph=graph), out, root=tmp_path)
    assert parsed(out / "snax_forge_acc.hjson") == {**ACCUMULATOR, "lanes": 4}
    assert [t.to_dict() for t in m.tasks] == [
        {"name": "sum_acc", "beats": 16, "csrs": {"loopCount": 16, "len": 16}}
    ]


def test_the_platforms_values_are_written_back(tmp_path):
    """The reverse of hjson_platform.py's mapping: a platform with --set changes gives a cluster
    file that reads back as that platform."""
    sets = [("l1.n_banks", 16), ("l1.rows", 2048), ("l1.base_addr", 0x2000_0000),
            ("l1.wide_bits", 256), ("l2.beat_bits", 256), ("l2.base_addr", 0x9000_0000),
            ("l2.size_bytes", 1 << 30)]  # fmt: skip
    platform = Platform.load(SNAX_FORGE).with_changes(sets)
    out = tmp_path / "out"
    generate(point(tmp_path, platform=platform), out, root=tmp_path)
    c = parsed(out / "snax_forge.hjson")
    assert (c["cluster"]["tcdm"]["banks"], c["cluster"]["tcdm"]["size"]) == (16, 256)
    assert (c["cluster"]["cluster_base_addr"], c["cluster"]["dma_data_width"]) == (0x2000_0000, 256)
    assert c["dram"] == {"address": 0x9000_0000, "length": 1 << 30}
    assert c["cluster"]["data_width"] == 64


def test_two_out_folders_give_byte_identical_files(tmp_path, monkeypatch):
    """Nothing written depends on the out folder or the working directory: the fixture diff."""
    path = point(tmp_path)
    a, b = tmp_path / "a", tmp_path / "b" / "deeper" / "vecadd"
    ma = generate(path, a, root=tmp_path)
    monkeypatch.chdir(path.parent.parent)  # and the design point given relative to it
    mb = generate(Path("vecadd/design_point.json"), b, root=tmp_path)
    assert ma == mb
    for f in ma.files:
        text = (a / f).read_bytes()
        assert text == (b / f).read_bytes(), f
        assert str(tmp_path).encode() not in text, f
    for f in ma.files[:-1]:
        head = generated_header("design/vecadd/design_point.json", "")
        assert (a / f).read_text().startswith(head), f


def test_elementwise_op_is_chisels():
    src = REPO / "hw/chisel/src/main/scala/snax/forge/elementwise/ElementwiseAlu.scala"
    body = src.read_text().split("object ElementwiseOp")[1].split("val width")[0]
    table = {k.lower(): int(v) for k, v in re.findall(r"val (\w+):\s*Int = (\d+)", body)}
    assert table == ELEMENTWISE_OP


# =============================================================================
# Refusals
# =============================================================================


def test_a_design_point_outside_the_root(tmp_path):
    with pytest.raises(GenError) as e:
        generate(point(tmp_path / "elsewhere"), tmp_path / "out", root=tmp_path / "root")
    assert e.value.code == "gen.point" and not (tmp_path / "out").exists()


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


def test_a_platform_without_a_source_hjson(tmp_path):
    p = Platform.load(SNAX_FORGE).with_changes([("hjson", None)])
    assert "pixi run design from-hjson" in _refused(tmp_path, "gen.source", p)


def test_the_sources_only_slot_takes_the_accelerator_slot_whatever_its_name(tmp_path):
    p = Platform.load(SNAX_FORGE).with_changes([("accelerator_slot", "snax_forge_x")])
    path = point(tmp_path, platform=p)
    src = tmp_path / "platforms" / SOURCE.name
    src.write_text(src.read_text().replace('snax_acc_name: "snax_forge"', 'snax_acc_name: "any"'))
    out = tmp_path / "out"
    m = generate(path, out, root=tmp_path)
    assert m.files[:3] == [
        "snax_forge.hjson",
        "snax_forge_x_streamer.hjson",
        "snax_forge_x_acc.hjson",
    ]
    s = slot(parsed(out / "snax_forge.hjson"))
    assert s["snax_acc_name"] == "snax_forge_x"
    assert s["snax_streamer_cfg"] == {"$ref": "snax_forge_x_streamer.hjson"}


def test_the_cluster_takes_snax_forges_name_whatever_the_sources(tmp_path):
    """A source named otherwise (snax_cluster's own name): cluster.name and its entry in
    cluster.bender_target become snax_forge_cluster, the other entries kept."""
    path = point(tmp_path)
    src = tmp_path / "platforms" / SOURCE.name
    src.write_text(src.read_text().replace('"snax_forge_cluster"', '"snax_alu_cluster"'))
    assert src.read_text().count('"snax_alu_cluster"') == 2  # the name and the bender entry
    generate(path, tmp_path / "out", root=tmp_path)
    cluster = parsed(tmp_path / "out" / "snax_forge.hjson")["cluster"]
    assert cluster["name"] == "snax_forge_cluster"
    assert cluster["bender_target"] == [
        "snitch_cluster", "snax_forge_cluster", "sparse_interconnect"
    ]  # fmt: skip


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


def _instance(edit=None, params=None, name="elementwise_add", impl=IMPL) -> Instance:
    """An instance (elementwise_add's by default), its BRM dict edited (no resolve: the
    adapter's checks only)."""
    d = copy.deepcopy(load_brm(name).to_dict())
    if edit:
        edit(d)
    brm = Brm.from_dict(d)
    return Instance(brm, impl, params or {"W": 4, "op": "add"})


def _binding(impl=IMPL, **kw):
    return lambda d: d["implementations"][impl]["binding"].update(kw)


def _ports(**kw):
    def edit(d):
        for p in d["interface"]["ports"]:
            p.update(kw.get(p["name"], {}))

    return edit


ADAPTER_CASES = [
    ("wait", _binding(wait="pipelined"), {}, "gen.wait"),
    ("latency", lambda d: d["implementations"][IMPL]["timing"].update(latency=1), {}, "gen.timing"),
    ("ii", lambda d: d["implementations"][IMPL]["timing"].update(initiation_interval=2), {},
     "gen.timing"),
    ("op", None, {"W": 4, "op": "nand"}, "gen.op"),
    ("args", _binding(args={"lanes": "W"}), {}, "gen.args"),
    ("mixed dtype", _ports(b={"dtype": "int32"}), {}, "gen.dtype"),
    ("float", _ports(a={"dtype": "float64"}, b={"dtype": "float64"}, out={"dtype": "float64"}),
     {}, "gen.dtype"),
    ("lanes x N", _ports(a={"dtype": "int8"}, b={"dtype": "int8"}, out={"dtype": "int8"}),
     {}, "gen.lanes"),
    ("lanes vs port", _binding(args={"lanes": "W * 2", "supportedOps": "op"}), {}, "gen.lanes"),
]  # fmt: skip


@pytest.mark.parametrize(
    ("case", "edit", "params", "code"), ADAPTER_CASES, ids=[c[0] for c in ADAPTER_CASES]
)
def test_the_adapter_refuses(case, edit, params, code):
    with pytest.raises(GenError) as e:
        elementwise_tiled_spatial(_instance(edit, params or None))
    assert e.value.code == code


ACC = "chisel_accumulator"


def _acc(edit=None, params=None) -> Instance:
    return _instance(edit, params or {"W": 1, "op": "add"}, name="accumulate", impl=ACC)


def _timing(**kw):
    return lambda d: d["implementations"][ACC]["timing"].update(kw)


def _csrs(*pairs):
    return _binding(ACC, csrs=[{"field": f, "value": v} for f, v in pairs])


def _second_input(d):
    d["interface"]["ports"].insert(1, dict(d["interface"]["ports"][0], name="b"))
    d["dataflow"]["ports"]["b"] = d["dataflow"]["ports"]["a"]
    for impl in d["implementations"].values():  # the binding's ports map every BRM port
        impl["binding"]["ports"]["b"] = "b"


ACCUMULATOR_CASES = [
    ("wait", _binding(ACC, wait="lockstep"), {}, "gen.wait"),
    ("latency", _timing(latency=0), {}, "gen.timing"),
    ("drain", _timing(drain=1), {}, "gen.timing"),
    ("no args", _binding(ACC, args={}), {}, "gen.args"),
    ("args", _binding(ACC, args={"lanes": "W", "accWidth": 64}), {}, "gen.args"),
    ("op", None, {"W": 1, "op": "max"}, "gen.op"),
    ("two inputs", _second_input, {}, "gen.ports"),
    ("rate in", _ports(a={"rate": "T"}), {}, "gen.rate"),
    ("mixed dtype", _ports(a={"dtype": "int32"}), {}, "gen.dtype"),
    ("lanes", _binding(ACC, args={"lanes": 1}), {"W": 2, "op": "add"}, "gen.lanes"),
    ("lanes out", _ports(out={"lanes": "W"}), {"W": 2, "op": "add"}, "gen.lanes"),
    ("narrow", _ports(a={"dtype": "int32"}, out={"dtype": "int32"}), {}, "gen.lanes"),
    ("fields order", _csrs(("len", "T"), ("loopCount", "n")), {}, "gen.fields"),
    ("no len", _csrs(("loopCount", "n")), {}, "gen.fields"),
]  # fmt: skip


@pytest.mark.parametrize(
    ("case", "edit", "params", "code"), ACCUMULATOR_CASES, ids=[c[0] for c in ACCUMULATOR_CASES]
)
def test_the_accumulator_adapter_refuses(case, edit, params, code):
    with pytest.raises(GenError) as e:
        accumulator(_acc(edit, params or None))
    assert e.value.code == code


def test_the_accumulator_adapter_gives_a_fold():
    unit = accumulator(_acc())
    assert (unit.inputs, unit.outputs, unit.lanes, unit.fields, unit.config) == (
        ["a"], ["out"], 1, ["loopCount", "len"], ["count", "rate"]
    )  # fmt: skip
    assert BLOCKS["Accumulator"] is accumulator


def test_the_adder_tree_is_the_accumulator_at_w_lanes():
    tree = _instance(None, {"W": 4, "op": "add"}, name="accumulate", impl="chisel_adder_tree")
    assert accumulator(tree) == replace(accumulator(_acc()), lanes=4)


@pytest.mark.parametrize(("adapter", "inst"), [
    (elementwise_tiled_spatial, lambda: _instance(_binding(control="other"))),
    (accumulator, lambda: _acc(_binding(ACC, control="other"))),
])  # fmt: skip
def test_a_control_other_than_counted_is_refused(monkeypatch, adapter, inst):
    """counted is the only control (D129); one registered later is not generated."""
    monkeypatch.setitem(CONTROLS, "other", "a test's")
    with pytest.raises(GenError) as e:
        adapter(inst())
    assert e.value.code == "gen.control"


def test_the_adapter_reads_unsigned_and_narrow_dtypes():
    u16 = {"dtype": "uint16"}
    unit = elementwise_tiled_spatial(_instance(_ports(a=u16, b=u16, out=u16)))
    assert (unit.data_width, unit.signed, unit.lanes) == (16, False, 4)


def test_three_ports_are_refused():
    def three_inputs(d):
        d["interface"]["ports"].insert(2, dict(d["interface"]["ports"][1], name="c"))
        d["dataflow"]["ports"]["c"] = d["dataflow"]["ports"]["b"]
        d["function"]["code"] = "out = a + b + c"

    with pytest.raises(GenError) as e:
        elementwise_tiled_spatial(_instance(three_inputs))
    assert e.value.code == "gen.ports"


def test_a_null_symbol_stops_at_the_load(tmp_path):
    """No rule of the generator's own: DesignPoint.load refuses it (graph.symbols)."""
    path = point(tmp_path)
    d = json.loads(path.read_text())
    d["graph"]["symbols"]["N"] = None
    path.write_text(json.dumps(d))
    with pytest.raises(DesignError, match=r"graph\.symbols"):
        generate(path, tmp_path / "out", root=tmp_path)
    assert not (tmp_path / "out").exists()


def test_two_instances_are_refused(tmp_path):
    gd = graph_dict()
    chained(gd)
    with pytest.raises(GenError) as e:
        generate(point(tmp_path, graph=Graph.from_dict(gd)), tmp_path / "out", root=tmp_path)
    assert e.value.code == "gen.instances" and "['acc', 'acc2']" in str(e.value)


@pytest.mark.parametrize(("binding", "code"), [(None, "gen.binding"), ("Gemm", "gen.block")])
def test_a_null_binding_and_an_unregistered_block(tmp_path, monkeypatch, binding, code):
    real = c_mod.instances

    def edited(graph):
        (name, inst), = real(graph).items()  # fmt: skip
        b = None if binding is None else {**inst.impl.binding, "block": binding}
        impl = replace(inst.impl, binding=b)
        brm = copy.copy(inst.brm)
        brm.implementations = {IMPL: impl}
        return {name: Instance(brm, IMPL, inst.params)}

    path = point(tmp_path)
    monkeypatch.setattr(c_mod, "instances", edited)
    with pytest.raises(GenError) as e:
        generate(path, tmp_path / "out", root=tmp_path)
    assert e.value.code == code
    assert "ElementwiseTiledSpatial" in BLOCKS


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


@pytest.mark.skipif(
    not (SUB / "snax").is_dir(), reason="the snax-gvsoc submodule is not checked out"
)
def test_the_constants_are_snax_accs():
    text = (SUB / "snax" / "snax_acc.py").read_text()
    assert "SNAX_CSR_FIRST = 0x3C0" in text and "SNAX_ACC_CSR_LAST = 0x5FD" in text
    assert (gen_mod.SNAX_CSR_FIRST, gen_mod.SNAX_ACC_CSR_LAST) == (0x3C0, 0x5FD)


def test_the_shells_counter_may_land_on_0x5fd_not_past():
    """vecadd's streamer (18 CSRs) and 553 fields: the counter on 0x5FD, the last CSR an
    accelerator may use; one more field puts it on 0x5FE, a task queue of snax_cluster's."""
    fields = [f"f{i}" for i in range(0x5FD - 0x3C0 - 18 - 2)]
    assert gen_mod.csr_layout(TEMPLATE_W4, fields).counter == 0x5FD
    with pytest.raises(GenError) as e:
        gen_mod.csr_layout(TEMPLATE_W4, [*fields, "one_more"])
    assert e.value.code == "gen.csrs" and "the shell's last CSR 1534 is past 1533" in str(e.value)
