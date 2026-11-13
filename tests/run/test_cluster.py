"""run/cluster.py: the cluster as a run is named, from the description files (the slot's sets
form) and the design point."""

import json

import hjson
import pytest

from snax_forge.gen.csrs import csr_layout
from snax_forge.run import Cluster, RunError, cluster_from, load_cluster
from snax_forge.run.cluster import csrs_of

from .helpers import STREAMER_FILE, files


def test_names_and_cores(cluster):
    c = cluster
    a = c.accelerator
    assert (c.design, c.nb_cores, c.dma_core, a.name, a.core, a.slot) == (
        "vecadd",
        2,
        1,
        "acc",
        0,
        "snax_forge_vecadd",
    )
    assert c.cores == ["core0", "core1"]
    assert c.sources == ["acc", "acc_a", "acc_b", "acc_out", "dma", "core0", "core1"]


def test_movers_readers_first_named_by_port(cluster):
    a = cluster.accelerator
    assert [(m.name, m.gvsoc, m.port, m.write, m.lanes, m.fifo_depth, m.set)
            for m in a.movers] == [
        ("acc_a", "reader0", "a", False, 4, 8, "acc"),
        ("acc_b", "reader1", "b", False, 4, 8, "acc"),
        ("acc_out", "writer0", "out", True, 4, 8, "acc"),
    ]  # fmt: skip
    assert a.mover("writer0").ports == [f"acc_out.lane{i}" for i in range(4)]
    assert a.mover("reader1").fifo == "acc_b.fifo"
    with pytest.raises(RunError, match="reader2"):
        a.mover("reader2")


def test_csrs_as_the_generated_header_has_them(cluster):
    """vecadd's streamer_csr_addr_map.h and acc_csr.h: start 0x3cf, busy 0x3d0, the group's
    START 0x3d2, JOBS 0x3d3, STATUS 0x3d4 (busy, bit 0), BUSY_COUNTER 0x3d5."""
    a = cluster.accelerator
    c = a.csrs
    (s,) = c.sets
    assert (s.first, s.start, s.busy, s.last) == (
        0x3C0,
        0x3CF,
        0x3D0,
        0x3D1,
    )
    assert (c.fields, c.start, c.busy, c.counter) == ({"ACC_RW_JOBS": 0x3D3}, 0x3D2, 0x3D4, 0x3D5)
    assert [a.block_of(x) for x in (0x3C0, 0x3D0, 0x3D2, 0x3D5, 0x3D6)] == [
        "acc_streamer",
        "acc_streamer",
        "acc",
        "acc",
        None,
    ]


@pytest.mark.parametrize("k", [0, 1, 3])
def test_csrs_agree_with_the_generator(k):
    """accelerator.json's CSRs as the generator numbers them: START, STATUS and BUSY_COUNTER as
    start, busy and counter, the other R/W ones as fields."""
    from tests.gen.helpers import one_unit

    _, templates, _, _, _ = files()
    templates["acc"]["data_reader_params"]["temporal_dim"] = [2, 1]
    acc = one_unit(k)
    c = csrs_of(templates, acc.to_dict(), "acc")
    g = csr_layout(templates, acc)
    n = g.accelerator
    ((r,),) = (g.streamers,)
    assert (c.sets[0].first, c.sets[0].last) == (r.first, r.last)
    assert (c.start, c.busy, c.counter) == (n["ACC_RW_START"], n["ACC_RO_STATUS"],
                                            n["ACC_RO_BUSY_COUNTER"])  # fmt: skip
    assert c.fields == {k: v for k, v in n.items() if "_RW_" in k and k != "ACC_RW_START"}


def test_l1_and_banks(cluster):
    c = cluster
    assert (c.l1_base, c.l1_size, c.banks, c.bank_bytes) == (0x10000000, 128 * 1024, 32, 8)
    assert c.is_l1(0x10000000) and c.is_l1(0x1001FFFF)
    assert not c.is_l1(0x10020000) and not c.is_l1(0x80000000)


def test_the_word_interleaved_map_and_the_dma_beat(cluster):
    c = cluster
    assert c.dma_beat_bytes == 64
    # bank = (a / w) mod B, row = a / (w B): 8-byte words over 32 banks
    assert c.bank_row(0x10000000) == (0, 0)
    assert c.bank_row(0x10000008) == (1, 0)
    assert c.bank_row(0x10000100) == (0, 1)
    assert c.bank_row(0x10000208) == (1, 2)
    assert c.bank_row(0x100005F8) == (31, 5)


@pytest.mark.parametrize(
    ("edit", "code"),
    [
        (lambda c, s, u, p, m: c["acc_core"].update(xdma=True), "cluster.cores"),  # 2 DMA cores
        (lambda c, s, u, p, m: p["streamers"].pop("acc_b"), "cluster.movers"),  # not the point's
        # the manifest's streamers: another range, no trace path, none at all
        (lambda c, s, u, p, m: m["csrs"]["streamers"][0].update(last=978), "cluster.manifest"),
        (lambda c, s, u, p, m: m["csrs"]["streamers"][0].pop("trace"), "cluster.manifest"),
        (lambda c, s, u, p, m: m.pop("csrs"), "cluster.manifest"),
    ],
)
def test_a_cluster_that_does_not_fit_is_refused(edit, code):
    cluster, streamer, unit, point, manifest = files()
    edit(cluster, streamer, unit, point, manifest)
    with pytest.raises(RunError) as e:
        cluster_from(cluster, streamer, unit, point, manifest)
    assert e.value.code == code


def test_each_set_has_its_trace_path(cluster):
    """The manifest's trace path, its crossbar inputs' prefix from it."""
    (s,) = cluster.accelerator.csrs.sets
    assert (s.name, s.trace, s.xbar) == ("acc", "snax_acc0/streamer_acc", "snax_acc0.streamer_acc")
    assert cluster.accelerator.set_at("/chip/soc/cluster_0/snax_acc0/streamer_acc") == s
    assert cluster.accelerator.set_at("/chip/soc/cluster_0/snax_acc0/streamer") is None


def test_load_from_files(tmp_path):
    cluster, streamer, unit, point, manifest = files()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "snax_forge.hjson").write_text(hjson.dumps(cluster))
    (tmp_path / STREAMER_FILE).write_text(hjson.dumps(streamer["acc"]))
    (tmp_path / "accelerator.json").write_text(json.dumps(unit))
    (tmp_path / "design_point.json").write_text(json.dumps(point))
    c = load_cluster(tmp_path / "snax_forge.hjson", tmp_path / "design_point.json")
    assert c.accelerator == cluster_from(*files()).accelerator
    assert c.meta == {"cluster": "snax_forge.hjson"}
    (tmp_path / "accelerator.json").unlink()
    with pytest.raises(RunError) as e:
        load_cluster(tmp_path / "snax_forge.hjson", tmp_path / "design_point.json")
    assert e.value.code == "cluster.ref"
    (tmp_path / "accelerator.json").write_text(json.dumps(unit))
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(RunError) as e:
        load_cluster(tmp_path / "snax_forge.hjson", tmp_path / "design_point.json")
    assert e.value.code == "cluster.manifest"


def test_the_old_one_streamer_slot_is_refused(tmp_path):
    """The slot in the form before the sets (snax_streamer_cfg, no snax_streamer_sets) does not
    read: cluster.sets."""
    cluster, templates, unit, point, _ = files()
    (slot,) = cluster["acc_core"]["snax_acc_cfg"]
    del slot["snax_streamer_sets"]
    slot["snax_streamer_cfg"] = {"$ref": STREAMER_FILE}
    (tmp_path / "snax_forge.hjson").write_text(hjson.dumps(cluster))
    (tmp_path / STREAMER_FILE).write_text(hjson.dumps(templates["acc"]))
    (tmp_path / "accelerator.json").write_text(json.dumps(unit))
    (tmp_path / "design_point.json").write_text(json.dumps(point))
    with pytest.raises(RunError) as e:
        load_cluster(tmp_path / "snax_forge.hjson", tmp_path / "design_point.json")
    assert e.value.code == "cluster.sets"


def test_two_sets_read_set_by_set(tmp_path):
    """dot_serial as the generator writes it: two streamer sets of the one entry, each its movers
    counted from reader0 and writer0 within the set, its streamer range (mul 960-977, sum
    982-994), each mover's CSRs from its own set's first (beats.py's mover_csrs)."""
    from snax_forge.gen import generate
    from snax_forge.run.beats import mover_csrs
    from tests.gen.helpers import bound_recipe
    from tests.gen.helpers import point as gen_point

    path = gen_point(tmp_path, name="dot_serial", graph=bound_recipe("dot_serial", "dot"))
    generate(path, tmp_path / "gen", tmp_path)
    c = load_cluster(tmp_path / "gen" / "snax_forge.hjson", path)
    a = c.accelerator
    assert (a.name, a.slot) == ("snax_forge_dot_serial", "snax_forge_dot_serial")
    assert [(m.name, m.gvsoc, m.set) for m in a.movers] == [
        ("mul_a", "reader0", "mul"), ("mul_b", "reader1", "mul"), ("mul_out", "writer0", "mul"),
        ("sum_a", "reader0", "sum"), ("sum_out", "writer0", "sum"),
    ]  # fmt: skip
    assert [(s.name, s.first, s.start, s.busy, s.counter) for s in a.csrs.sets] == [
        ("mul", 960, 975, 976, 977), ("sum", 982, 992, 993, 994),
    ]  # fmt: skip
    assert [(u.name, u.path, u.start, u.counter) for u in a.units] == [
        ("mul", "unit_mul", 978, 981), ("sum", "unit_sum", 995, 999),
    ]  # fmt: skip
    assert [a.block_of(x) for x in (960, 978, 982, 994, 997)] == [
        "mul_streamer", "mul", "sum_streamer", "sum_streamer", "sum"]  # fmt: skip
    assert mover_csrs(c) == {"mul_a": 960, "mul_b": 965, "mul_out": 970, "sum_a": 982,
                             "sum_out": 987}  # fmt: skip
    # each set's streamer at its own path, its movers by their names within it
    assert [s.trace for s in a.csrs.sets] == ["snax_acc0/streamer_mul", "snax_acc0/streamer_sum"]
    assert [a.mover("reader0", x).name for x in ("mul", "sum")] == ["mul_a", "sum_a"]
    assert Cluster.from_dict(json.loads(json.dumps(c.to_dict()))) == c
