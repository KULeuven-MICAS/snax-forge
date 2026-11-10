"""The generator (D134): vecadd on snax_forge as its description files (the cluster, one
streamer per set, accelerator.json; D157), vecmul, vsum, and every refusal."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from snax_forge.brm import CONTROLLERS, Brm, Instance, load_brm
from snax_forge.common import generated_header
from snax_forge.design import DesignError, DesignPoint, Platform
from snax_forge.gen import DATAPATHS, ELEMENTWISE_OP, Description, GenError, Manifest, generate
from snax_forge.gen import __main__ as cli
from snax_forge.gen import csrs as csrs_mod
from snax_forge.gen import gvsoc as gen_mod
from snax_forge.gen.blocks import Port, simd_lanes, tree_accumulate
from snax_forge.gen.data import array_text, arrays
from snax_forge.lower import c as c_mod
from tests.brm.helpers import IMPL
from tests.design.helpers import REPO, SNAX_FORGE
from tests.helpers import bound_vsum, requires_submodule

from .helpers import (
    SOURCE,
    SUB,
    bound_vecmul,
    designed,
    one_unit,
    parsed,
    point,
    snax_cfg,
    snax_streamer,
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
# vecadd's accelerator at W = 4 (C§17's example): one group of one SimdLanes unit,
# each port int64 (D143)
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
        *FILES, "include/streamer_acc_csr_addr_map.h", "include/acc_csr.h",
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
        "streamers": [{"name": "acc", "first": 960, "last": 977, "trace": "snax_acc0/streamer_acc"}],
        "accelerator": {"ACC_RW_START": 978, "ACC_RW_JOBS": 979, "ACC_RO_STATUS": 980,
                        "ACC_RO_BUSY_COUNTER": 981},
    }  # fmt: skip
    assert [t.to_dict() for t in m.tasks] == [
        {"name": "add_acc", "beats": 16, "csrs": {"JOBS": 16}}
    ]
    assert Manifest.load(out / "manifest.json") == m
    assert Manifest.from_dict(m.to_dict()) == m


def test_the_streamer_file_is_the_design_points_streamers(w4):
    out, _ = w4
    assert parsed(out / streamer_file("vecadd")) == TEMPLATE_W4


def test_the_streamer_file_is_the_hjsons_template(w4):
    """Lowered per design point, vecadd at W = 4 gives the source's own template back."""
    out, _ = w4
    hand = parsed(SOURCE)["snax_forge_streamer_template"]
    for group in ("data_reader_params", "data_writer_params"):
        assert parsed(out / streamer_file("vecadd"))[group] == hand[group]


def test_the_accelerator_file_is_vecadds_description(w4):
    """Plain JSON, no header comment: its design_point names the source (C§17)."""
    out, _ = w4
    text = (out / "accelerator.json").read_text()
    assert json.loads(text) == ADD_W4
    assert Description.load(out / "accelerator.json").to_json() == text


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


@requires_submodule
def test_snax_gvsocs_reader_takes_the_folder(w4):
    """snax-gvsoc's snax_cfg.load (G1): the whole-file refs resolve, the slot is the tag with one
    streamer set per accelerator (slot.sets), each set's counts agree, and the nested
    accelerator.json reads as written (its CSRs in CSR order)."""
    out, _ = w4
    cfg = snax_cfg().load(out / "snax_forge.hjson")
    (acc,) = cfg.cores[0].accelerators
    assert (acc.name, acc.tcdm_ports) == ("snax_forge_vecadd", 12)
    ((s,),) = (acc.sets,)
    assert (s.name, s.streamer, s.num_rw_csr, s.num_ro_csr) == ("acc", TEMPLATE_W4, 2, 2)
    assert (acc.num_rw_csr, acc.num_ro_csr) == (2, 2)
    d = acc.description
    assert (d.tag, d.design_point) == ("snax_forge_vecadd", "design/vecadd/design_point.json")
    (a,) = d.accelerators
    assert (a.name, a.inputs, a.outputs) == ("acc", ["acc.a", "acc.b"], ["acc.out"])
    assert [(c.name, c.offset) for c in a.csrs] == [
        ("START", 0),
        ("JOBS", 1),
        ("STATUS", 2),
        ("BUSY_COUNTER", 3),
    ]
    (unit,) = a.groups[0].units
    assert (unit.datapath, unit.op_type, unit.params) == (
        "SimdLanes",
        "int64",
        {"op": "add"},
    )
    assert (cfg.tcdm.size, cfg.tcdm.banks, cfg.tcdm.sparse_interconnect) == (131072, 32, True)


@requires_submodule
def test_streamer_csrs_are_snax_streamers(w4):
    """The generator's copies of snax_streamer.py's nb_csrs and snax_cfg.py's
    streamer_channels give what the real ones give."""
    out, _ = w4
    template = parsed(out / streamer_file("vecadd"))
    assert csrs_mod.streamer_csrs(template) == snax_streamer().nb_csrs(template) == 18
    assert csrs_mod.channels(template) == snax_cfg().streamer_channels(template) == 12


def test_every_generated_file_starts_with_the_one_header(w4):
    """hjson, headers, data.h and main.c: the generator, the design point, the command;
    accelerator.json is plain JSON; a set's streamer header is snax_cluster's text, its
    timestamp line naming the design point (headers.py)."""
    out, m = w4
    head = generated_header("design/vecadd/design_point.json", "")
    assert "`pixi run gen-gvsoc design/vecadd/design_point.json`" in head.replace("\n// ", " ")
    for f in m.files:
        if f in ("manifest.json", "accelerator.json"):
            continue
        if f.startswith("include/streamer_"):
            assert "design point design/vecadd/design_point.json" in (out / f).read_text()
            continue
        text = (out / f).read_text()
        assert text.startswith(head), f
        assert all(len(line) <= 80 for line in text.splitlines() if line.startswith("//")), f


def test_vecadd_w8_gives_eight_channels_per_mover(tmp_path):
    out = tmp_path / "out"
    m = generate(point(tmp_path, w=8, name="vecadd_w8"), out, root=tmp_path)
    reader = parsed(out / streamer_file("vecadd_w8"))["data_reader_params"]
    assert reader["num_channel"] == [8, 8]
    s = slot(parsed(out / "snax_forge.hjson"))
    assert (s["snax_tcdm_ports"], s["sparse_interconnect_config"]) == (24, [[24, 1]])
    assert m.csrs.accelerator["ACC_RW_START"] == 978 and m.tasks[0].beats == 8
    assert m.tasks[0].csrs == {"JOBS": 8}


def test_vecmul_is_vecadd_with_op_mul(tmp_path, w4):
    """simd's binding gives vecadd's accelerator with op mul."""
    out = tmp_path / "gvsoc" / "vecmul"
    m = generate(point(tmp_path, name="vecmul", graph=bound_vecmul()), out, root=tmp_path)
    want = copy.deepcopy(ADD_W4)
    want["design_point"], want["tag"] = "design/vecmul/design_point.json", "snax_forge_vecmul"
    want["accelerators"][0]["groups"][0]["units"][0].update(brm="simd",
                                                            params={"op": "mul"})  # fmt: skip
    assert json.loads((out / "accelerator.json").read_text()) == want
    assert parsed(out / streamer_file("vecmul")) == TEMPLATE_W4
    assert m.symbols == {"N": 64} and m.design_point == "design/vecmul/design_point.json"
    assert [f.replace("vecmul", "vecadd") for f in m.files] == w4[1].files
    assert m.csrs == w4[1].csrs and m.trace == "snax_acc0/unit"
    assert [t.to_dict() for t in m.tasks] == [
        {"name": "mult_acc", "beats": 16, "csrs": {"JOBS": 16}}
    ]
    cluster = (out / "snax_forge.hjson").read_text().replace("vecmul", "vecadd")
    assert cluster.split("\n{", 1)[1] == (w4[0] / "snax_forge.hjson").read_text().split("\n{", 1)[1]
    for h in ("streamer_acc_csr_addr_map.h", "acc_csr.h"):
        mine, add = (out / "include" / h).read_text(), (w4[0] / "include" / h).read_text()
        assert mine.replace("vecmul", "vecadd") == add, h


def test_vsum_is_the_tree_accumulate(tmp_path):
    """accumulate's TreeAccumulate, one reader and one writer of one lane, the group's JOBS and
    the unit's JOB_BEATS, one task of 64 beats summing all 64 (T = n: one job of 64)."""
    out = tmp_path / "gvsoc" / "vsum"
    m = generate(point(tmp_path, name="vsum", graph=bound_vsum()), out, root=tmp_path)
    assert json.loads((out / "accelerator.json").read_text()) == VSUM
    one = {"spatial_bounds": [[1]], "temporal_dim": [1], "num_channel": [1], "fifo_depth": [8]}
    assert parsed(out / streamer_file("vsum")) == {
        "data_reader_params": one, "data_writer_params": one
    }  # fmt: skip
    s = slot(parsed(out / "snax_forge.hjson"))
    assert (s["snax_tcdm_ports"], s["sparse_interconnect_config"]) == (2, [[2, 1]])
    ((the_set),) = s["snax_streamer_sets"]
    assert (the_set["snax_num_rw_csr"], the_set["snax_num_ro_csr"]) == (3, 2)
    assert m.symbols == {"N": 64}
    assert m.csrs.to_dict() == {
        "streamers": [{"name": "acc", "first": 960, "last": 972, "trace": "snax_acc0/streamer_acc"}],
        "accelerator": {"ACC_RW_START": 973, "ACC_RW_JOBS": 974, "ACC_RW_JOB_BEATS": 975,
                        "ACC_RO_STATUS": 976, "ACC_RO_BUSY_COUNTER": 977},
    }  # fmt: skip
    assert [t.to_dict() for t in m.tasks] == [
        {"name": "sum_acc", "beats": 64, "csrs": {"JOBS": 1, "JOB_BEATS": 64}}
    ]
    assert "#define ACC_RW_JOB_BEATS 975\n" in (out / "include" / "acc_csr.h").read_text()


def test_vsum_at_w4_is_the_adder_tree(tmp_path):
    """accumulate's adder tree at W = 4 is TreeAccumulate with 4 input lanes, one task of 16
    beats of 4 lanes summing all 64 (T = n)."""
    out = tmp_path / "gvsoc" / "vsum_tree"
    graph = bound_vsum(4, "chisel_adder_tree")
    m = generate(point(tmp_path, name="vsum_tree", graph=graph), out, root=tmp_path)
    (unit,) = json.loads((out / "accelerator.json").read_text())["accelerators"][0]["groups"][0][
        "units"
    ]
    assert unit["ports"]["a"]["lanes"] == 4 and unit["ports"]["out"]["lanes"] == 1
    assert [t.to_dict() for t in m.tasks] == [
        {"name": "sum_acc", "beats": 16, "csrs": {"JOBS": 1, "JOB_BEATS": 16}}
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
        if f != "accelerator.json" and not f.startswith("include/streamer_"):  # C§17, headers.py
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
        "snax_forge_x_vecadd_streamer_acc.hjson",
        "accelerator.json",
    ]
    s = slot(parsed(out / "snax_forge.hjson"))
    assert s["snax_acc_name"] == "snax_forge_x_vecadd"
    assert s["snax_streamer_sets"][0]["snax_streamer_cfg"] == {
        "$ref": "snax_forge_x_vecadd_streamer_acc.hjson"
    }


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
    ("lanes x N", None, {}, "gen.lanes", _dtypes(a="int8", b="int8", out="int8")),
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
    ("narrow", None, {}, "gen.lanes", _dtypes(("a", "out"), a="int32", out="int32")),
]  # fmt: skip


@pytest.mark.parametrize(
    ("case", "edit", "params", "code", "dtypes"),
    [(*c, None)[:5] for c in ACCUMULATOR_CASES], ids=[c[0] for c in ACCUMULATOR_CASES],
)  # fmt: skip
def test_the_accumulator_adapter_refuses(case, edit, params, code, dtypes):
    with pytest.raises(GenError) as e:
        tree_accumulate(_acc(edit, params or None), dtypes)
    assert e.value.code == code


def test_the_accumulator_adapter_gives_a_fold():
    """Its ports through the binding's map, no opType, params op add; latency 1."""
    fill = tree_accumulate(_acc())
    assert fill.ports == {"a": Port("in", "in", None, 1), "out": Port("out", "out", None, 1)}
    assert (fill.op_type, fill.params) == (None, {"op": "add"})
    assert DATAPATHS["TreeAccumulate"].fill is tree_accumulate
    assert DATAPATHS["TreeAccumulate"].latency == 1 and DATAPATHS["SimdLanes"].latency == 0


def test_the_adder_tree_is_tree_accumulate_at_w_lanes():
    tree = _instance(None, {"W": 4, "op": "add"}, name="accumulate", impl="chisel_adder_tree")
    assert tree_accumulate(tree).ports["a"].lanes == 4
    assert tree_accumulate(tree).ports["out"] == tree_accumulate(_acc()).ports["out"]


def test_the_adapter_reads_unsigned_and_narrow_dtypes():
    """Each port its container's dtype, opType the inputs' NumPy result type."""
    u16 = simd_lanes(_instance(), _dtypes(a="uint16", b="uint16", out="uint16"))
    assert [p.dtype for p in u16.ports.values()] == ["uint16"] * 3 and u16.op_type == "uint16"
    mixed = simd_lanes(_instance(), _dtypes(a="int32", b="int32", out="int64"))
    assert [p.dtype for p in mixed.ports.values()] == ["int32", "int32", "int64"]
    assert mixed.op_type == "int32"  # vmul int32 into int64: the op at int32, then extended
    signs = simd_lanes(_instance(), _dtypes(a="int16", b="uint16", out="int32"))
    assert signs.op_type == "int32"


def test_one_stream_is_a_and_out_with_k():
    """At one stream the unit is a and out, opType a's dtype, params op and k; at two, as
    before, op only (an unset k never read)."""
    one = simd_lanes(_instance(None, {"W": 4, "op": "max", "streams": 1, "k": -3}),
                     _dtypes(("a", "out"), a="int32", out="int32"))  # fmt: skip
    assert list(one.ports) == ["a", "out"] and one.op_type == "int32"
    assert one.params == {"op": "max", "k": -3}
    two = simd_lanes(_instance(), _dtypes())
    assert list(two.ports) == ["a", "b", "out"] and two.params == {"op": "add"}
    edge = {"W": 8, "op": "add", "streams": 1, "k": -128}
    assert simd_lanes(_instance(None, edge), _dtypes(("a", "out"), a="int8", out="int8")).params[
        "k"] == -128  # fmt: skip


def test_relu_is_one_unit_of_one_input_and_k(tmp_path):
    """relu's recipe gives one SimdLanes unit: a and out at int32, op max, k 0; one reader and
    one writer of two channels each."""
    from .helpers import bound_recipe

    out = tmp_path / "gvsoc" / "relu"
    m = generate(point(tmp_path, name="relu", graph=bound_recipe("relu", "relu")), out,
                 root=tmp_path)  # fmt: skip
    (unit,) = json.loads((out / "accelerator.json").read_text())["accelerators"][0]["groups"][0][
        "units"]  # fmt: skip
    assert unit["ports"] == {"a": {"port": "a", "dir": "in", "dtype": "int32", "lanes": 4},
                             "out": {"port": "out", "dir": "out", "dtype": "int32", "lanes": 4}}  # fmt: skip
    assert (unit["opType"], unit["params"]) == ("int32", {"op": "max", "k": 0})
    s = parsed(out / streamer_file("relu"))
    assert s["data_reader_params"]["num_channel"] == [2]
    assert s["data_writer_params"]["num_channel"] == [2]
    assert [t.to_dict() for t in m.tasks] == [
        {"name": "numpy_maximum_acc", "beats": 16, "csrs": {"JOBS": 16}}]  # fmt: skip


def test_three_ports_are_refused():
    def three_inputs(d):
        d["interface"]["ports"].insert(2, dict(d["interface"]["ports"][1], name="c"))
        d["dataflow"]["ports"]["c"] = d["dataflow"]["ports"]["b"]
        d["function"]["code"] = "out = a + b + c"

    with pytest.raises(GenError) as e:
        simd_lanes(_instance(three_inputs))
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
    """A null binding, an unregistered datapath, a controller other than counted (D129: one
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


def _two_sets(k: int) -> Description:
    """Two vecadd-shaped accelerators a and b, b's unit with ``k`` R/W CSRs of its own, the
    offsets running on from a's (D157)."""
    from dataclasses import replace as dc_replace

    from snax_forge.gen import Csr

    (a,) = one_unit(0, "a").accelerators
    (b,) = one_unit(k, "b").accelerators
    g = b.groups[0]
    g = dc_replace(g, csrs=[dc_replace(c, offset=c.offset + 4) for c in g.csrs],
                   units=[dc_replace(g.units[0], csrs=[Csr(c.name, c.access, c.offset + 4, c.width)
                                                       for c in g.units[0].csrs])])  # fmt: skip
    return Description("p", "snax_forge_two", [a, dc_replace(b, groups=[g])])


def test_the_whole_entrys_layout_packs_the_sets_from_960():
    """b_0 = 960, b_{k+1} = b_k + s_k + c_k (C§16): set a's streamer 960-977 and own 978-981,
    set b's streamer 982-999 and own from 1000; gen.csrs when the last set's last CSR passes
    0x5FD, the whole design counted."""
    templates = {"a": TEMPLATE_W4, "b": TEMPLATE_W4}
    layout = csrs_mod.csr_layout(templates, _two_sets(0))
    assert [s.to_dict() for s in layout.streamers] == [
        {"name": "a", "first": 960, "last": 977, "trace": "snax_acc0/streamer_a"},
        {"name": "b", "first": 982, "last": 999, "trace": "snax_acc0/streamer_b"},
    ]  # fmt: skip
    assert layout.accelerator == {"A_RW_START": 978, "A_RW_JOBS": 979, "A_RO_STATUS": 980,
                                  "A_RO_BUSY_COUNTER": 981, "B_RW_START": 1000, "B_RW_JOBS": 1001,
                                  "B_RO_STATUS": 1002, "B_RO_BUSY_COUNTER": 1003}  # fmt: skip
    k = 0x5FD - 1003
    assert csrs_mod.csr_layout(templates, _two_sets(k)).accelerator["B_RO_BUSY_COUNTER"] == 0x5FD
    with pytest.raises(GenError) as e:
        csrs_mod.csr_layout(templates, _two_sets(k + 1))
    assert e.value.code == "gen.csrs" and "the entry's last CSR 1534" in str(e.value)


# =============================================================================
# Narrow designs (D142): the streamer in words, each port's type
# =============================================================================


def _narrow(tmp_path, recipe: str) -> Path:
    """``recipes/<folder>/<recipe>.json`` designed and generated under ``tmp_path``; the out folder."""
    point = designed(tmp_path, recipe, recipe)
    out = tmp_path / "gvsoc" / recipe
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        generate(point, out, root=tmp_path)
    return out


def test_vecadd_i32_two_channels_per_port(tmp_path):
    out = _narrow(tmp_path, "vecadd_i32")
    s = parsed(out / streamer_file("vecadd_i32"))
    assert (s["data_reader_params"]["spatial_bounds"], s["data_reader_params"]["num_channel"]) == (
        [[2], [2]], [2, 2]
    )  # fmt: skip
    assert (s["data_writer_params"]["spatial_bounds"], s["data_writer_params"]["num_channel"]) == (
        [[2]], [2]
    )  # fmt: skip
    (unit,) = json.loads((out / "accelerator.json").read_text())["accelerators"][0]["groups"][0][
        "units"
    ]
    assert unit["ports"] == {
        p: {"port": p, "dir": "out" if p == "out" else "in", "dtype": "int32", "lanes": 4}
        for p in ("a", "b", "out")
    }
    assert unit["opType"] == "int32"
    assert slot(parsed(out / "snax_forge.hjson"))["snax_tcdm_ports"] == 6


def test_vsum_tree_i32_int32_in_int64_out(tmp_path):
    out = _narrow(tmp_path, "vsum_tree_i32")
    s = parsed(out / streamer_file("vsum_tree_i32"))
    assert s["data_reader_params"]["spatial_bounds"] == [[2]]
    assert s["data_reader_params"]["num_channel"] == [2]
    assert (s["data_writer_params"]["spatial_bounds"], s["data_writer_params"]["num_channel"]) == (
        [[1]], [1]
    )  # fmt: skip
    (unit,) = json.loads((out / "accelerator.json").read_text())["accelerators"][0]["groups"][0][
        "units"
    ]
    assert unit["ports"] == {"a": {"port": "in", "dir": "in", "dtype": "int32", "lanes": 4},
                             "out": {"port": "out", "dir": "out", "dtype": "int64", "lanes": 1}}  # fmt: skip
    assert "opType" not in unit


def test_vecadd_i8_one_channel_per_port(tmp_path):
    s = parsed(_narrow(tmp_path, "vecadd_i8") / streamer_file("vecadd_i8"))
    for group in ("data_reader_params", "data_writer_params"):
        assert all(b == [1] for b in s[group]["spatial_bounds"])
        assert all(c == 1 for c in s[group]["num_channel"])


def test_matadd_int32_bounds_in_words(tmp_path):
    """matadd (int32 again, its recipe's 1 x 4 tiles of W = 4): 4 lanes of int32, 2 words."""
    s = parsed(_narrow(tmp_path, "matadd") / streamer_file("matadd"))
    assert s["data_reader_params"]["spatial_bounds"] == [[2], [2]]
    assert s["data_reader_params"]["num_channel"] == [2, 2]
    assert s["data_writer_params"]["num_channel"] == [2]


def test_generating_again_changes_no_file(tmp_path):
    """The same design point again: every file keeps its mtime (written only when its text
    changes), and a file the earlier manifest listed that this generation does not write goes."""
    out = tmp_path / "gvsoc" / "vecadd"
    path = point(tmp_path)
    manifest = generate(path, out, root=tmp_path)
    stamps = {f: (out / f).stat().st_mtime_ns for f in manifest.files}
    old = json.loads((out / "manifest.json").read_text())
    old["files"].append("include/gone.h")
    (out / "manifest.json").write_text(json.dumps(old))
    (out / "include" / "gone.h").write_text("// an earlier generation's")
    stamps["manifest.json"] = (out / "manifest.json").stat().st_mtime_ns
    generate(path, out, root=tmp_path)
    assert not (out / "include" / "gone.h").exists()
    changed = [f for f in manifest.files if (out / f).stat().st_mtime_ns != stamps[f]]
    assert changed == ["manifest.json"]  # the one file changed by hand, written back


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
