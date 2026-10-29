"""run/cluster.py: the cluster as a run is named, from the hjson files and the design point."""

import json

import hjson
import pytest

from snax_forge.gen.csrs import csr_layout
from snax_forge.run import Cluster, RunError, cluster_from, load_cluster
from snax_forge.run.cluster import csrs_of

from .helpers import files


def test_names_and_cores(cluster):
    c = cluster
    a = c.accelerator
    assert (c.design, c.nb_cores, c.dma_core, a.name, a.core, a.slot) == (
        "vecadd",
        2,
        1,
        "acc",
        0,
        "snax_forge",
    )
    assert c.cores == ["core0", "core1"]
    assert c.sources == ["acc", "acc_a", "acc_b", "acc_out", "dma", "core0", "core1"]


def test_movers_readers_first_named_by_port(cluster):
    a = cluster.accelerator
    assert [(m.name, m.gvsoc, m.port, m.write, m.lanes, m.fifo_depth) for m in a.movers] == [
        ("acc_a", "reader0", "a", False, 4, 8),
        ("acc_b", "reader1", "b", False, 4, 8),
        ("acc_out", "writer0", "out", True, 4, 8),
    ]
    assert a.mover("writer0").ports == [f"acc_out.lane{i}" for i in range(4)]
    assert a.mover("reader1").fifo == "acc_b.fifo"
    with pytest.raises(RunError, match="reader2"):
        a.mover("reader2")


def test_csrs_as_the_generated_header_has_them(cluster):
    """vecadd's streamer_csr_addr_map.h and acc_csr.h: start 0x3cf, busy 0x3d0, the group's
    START 0x3d2, JOBS 0x3d3, STATUS 0x3d4 (busy, bit 0), COUNTER 0x3d5."""
    a = cluster.accelerator
    c = a.csrs
    assert (c.streamer_first, c.streamer_start, c.streamer_busy, c.streamer_last) == (
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
    """accelerator.json's CSRs as the generator numbers them: START, STATUS and COUNTER as
    start, busy and counter, the other R/W ones as fields."""
    from tests.gen.helpers import one_unit

    _, streamer, _, _ = files()
    streamer["data_reader_params"]["temporal_dim"] = [2, 1]
    acc = one_unit(k)
    c = csrs_of(streamer, acc.to_dict(), "acc")
    g = csr_layout(streamer, acc)
    n = g.accelerator
    assert (c.streamer_first, c.streamer_last) == (g.streamer_first, g.streamer_last)
    assert (c.start, c.busy, c.counter) == (n["ACC_RW_START"], n["ACC_RO_STATUS"],
                                            n["ACC_RO_COUNTER"])  # fmt: skip
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


def test_round_trip(cluster):
    c = cluster
    assert Cluster.from_dict(json.loads(json.dumps(c.to_dict()))) == c


@pytest.mark.parametrize(
    ("edit", "code"),
    [
        (lambda c, s, u, p: c["acc_core"].update(xdma=True), "cluster.cores"),  # two DMA cores
        (lambda c, s, u, p: p["streamers"].pop("acc_b"), "cluster.movers"),  # not the point's
    ],
)
def test_a_cluster_that_does_not_fit_is_refused(edit, code):
    cluster, streamer, unit, point = files()
    edit(cluster, streamer, unit, point)
    with pytest.raises(RunError) as e:
        cluster_from(cluster, streamer, unit, point)
    assert e.value.code == code


def test_load_from_files(tmp_path):
    cluster, streamer, unit, point = files()
    (tmp_path / "snax_forge.hjson").write_text(hjson.dumps(cluster))
    (tmp_path / "snax_forge_streamer.hjson").write_text(hjson.dumps(streamer))
    (tmp_path / "accelerator.json").write_text(json.dumps(unit))
    (tmp_path / "design_point.json").write_text(json.dumps(point))
    c = load_cluster(tmp_path / "snax_forge.hjson", tmp_path / "design_point.json")
    assert c.accelerator == cluster_from(*files()).accelerator
    assert c.meta == {"cluster": "snax_forge.hjson"}
    (tmp_path / "accelerator.json").unlink()
    with pytest.raises(RunError) as e:
        load_cluster(tmp_path / "snax_forge.hjson", tmp_path / "design_point.json")
    assert e.value.code == "cluster.ref"
