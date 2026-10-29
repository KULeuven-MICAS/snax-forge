"""A platform read from snax_cluster's cluster file (D131), and its source."""

from __future__ import annotations

import tracemalloc
from dataclasses import replace

import pytest

from snax_forge.design import Design, Platform, run_checks
from snax_forge.design import __main__ as cli
from snax_forge.design.hjson_platform import platform_from_hjson

from .helpers import REPO, SNAX_FORGE, bound_vecadd, platform_dict

# snax_forge's source: snax-forge's base cluster
SOURCE = REPO / "platforms" / "snax_forge_cluster.hjson"

# A cluster file with what the reader needs, in the hjson's own form.
CLUSTER = """{
  cluster: {
    cluster_base_addr: 268435456, data_width: 64, dma_data_width: 512,
    tcdm: {size: 128, banks: 32},
    hives: [{cores: [{$ref: "#/acc_core"}, {isa: "rv32ima"}]}],
  },
  dram: {address: 2147483648, length: 2147483648},
  acc_core: {snax_acc_cfg: [ACCS]},
  // a streamer template as snax_alu_cluster's
  template: {
    data_reader_params: {temporal_dim: [1, 1], fifo_depth: [8, 8]},
    data_writer_params: {temporal_dim: [1], fifo_depth: [8]},
  },
}"""
ACC = '{snax_acc_name: "NAME", snax_streamer_cfg: {$ref: "#/template"}}'


def cluster(tmp_path, text: str = CLUSTER, accs: str = ACC.replace("NAME", "a"), name="c.hjson"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.replace("ACCS", accs))
    return path


def read(path, name: str = "c"):
    """The platform of a test's cluster file, its source named from the file's folder."""
    return platform_from_hjson(path, name, path.parent)


def test_snax_forge_round_trips_byte_for_byte():
    p = Platform.load(SNAX_FORGE)
    assert p.to_json() == SNAX_FORGE.read_text()
    assert Platform.from_dict(p.to_dict()) == p


def test_the_source_gives_the_checked_in_platform():
    """platforms/snax_forge.json is `design from-hjson platforms/snax_forge_cluster.hjson
    snax_forge`; its accelerator_slot is snax_forge, the source's slot's name too."""
    p = platform_from_hjson(SOURCE, "snax_forge")
    assert p.to_json() == SNAX_FORGE.read_text()
    assert (p.hjson, p.accelerator_slot) == ("platforms/snax_forge_cluster.hjson", "snax_forge")


def test_a_source_outside_the_root_is_refused(tmp_path):
    with pytest.raises(Exception) as e:
        platform_from_hjson(cluster(tmp_path), "c")
    assert e.value.codes == ["hjson.path"] and "copy the file into platforms/" in str(e.value)


def test_what_is_read_and_what_is_small16s(tmp_path):
    p = read(cluster(tmp_path))
    assert p == replace(Platform.load(SNAX_FORGE), name="c", hjson="c.hjson")
    l1, l2 = p.l1, p.l2
    assert (l1.n_banks, l1.width_bits, l1.rows, l1.base_addr, l1.wide_bits) == (
        32, 64, 512, 0x1000_0000, 512
    )  # fmt: skip
    assert (l2.base_addr, l2.size_bytes, l2.beat_bits, l2.width_bits) == (2**31, 2**31, 512, 64)
    assert p.default.fifo_depth == 8  # temporal_dim is the graph's (D84), not read
    assert p.wait_mode == platform_dict()["wait_mode"]
    assert (p.base, p.changes, p.entries) == (None, {}, {})


def test_two_accelerators_are_refused(tmp_path):
    accs = ACC.replace("NAME", "a") + ", " + ACC.replace("NAME", "b")
    with pytest.raises(Exception) as e:
        read(cluster(tmp_path, accs=accs))
    assert e.value.codes == ["hjson.accelerators"]
    assert "2 accelerators ['a', 'b']; a platform has one accelerator slot" in str(e.value)


def test_movers_that_differ_name_open_item_78(tmp_path):
    text = CLUSTER.replace("fifo_depth: [8]", "fifo_depth: [4]")
    with pytest.raises(Exception) as e:
        read(cluster(tmp_path, text))
    assert e.value.codes == ["hjson.movers"]
    msg = str(e.value)
    assert "snax_acc_cfg[0].snax_streamer_cfg.fifo_depth" in msg
    assert "data_reader_params [8, 8], data_writer_params [4]" in msg and "open item 78" in msg


@pytest.mark.parametrize(
    ("old", "new", "where"),
    [
        ("banks: 32", "", "cluster.tcdm.banks"),
        ("dram: {address: 2147483648, length: 2147483648}", "dram: {address: 0}", "dram.length"),
        ('{$ref: "#/acc_core"}', '{$ref: "#/no_core"}', "#.cluster.hives[0].cores[0]"),
    ],
)
def test_a_missing_key_is_named(tmp_path, old, new, where):
    with pytest.raises(Exception) as e:
        read(cluster(tmp_path, CLUSTER.replace(old, new)))
    assert e.value.problems[0].where == where


def test_tcdm_that_is_not_whole_rows_and_a_bad_name(tmp_path):
    with pytest.raises(Exception) as e:
        read(cluster(tmp_path, CLUSTER.replace("banks: 32", "banks: 3")))
    assert e.value.codes == ["hjson.value"]
    with pytest.raises(Exception) as e:
        read(cluster(tmp_path), "snax-alu")
    assert "is not an identifier" in str(e.value)


def test_from_hjson_writes_the_platform_and_keeps_an_existing_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PLATFORMS", tmp_path / "platforms")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    acc = ACC.replace("NAME", "snax_alu")
    src = cluster(tmp_path, accs=acc, name="platforms/snax_forge_cluster.hjson")
    assert cli.main(["from-hjson", str(src), "snax_forge"]) == 0
    assert (tmp_path / "platforms" / "snax_forge.json").read_text() == SNAX_FORGE.read_text()
    assert cli.main(["from-hjson", str(src), "snax_forge"]) == 1
    assert "pass --force" in capsys.readouterr().err
    assert cli.main(["from-hjson", str(src), "snax_forge", "--force"]) == 0
    bad = cluster(tmp_path, accs="")
    assert cli.main(["from-hjson", str(bad), "other"]) == 1
    assert "[hjson.accelerators]" in capsys.readouterr().err
    outside = cluster(tmp_path.parent / f"{tmp_path.name}_outside", accs=acc)
    assert cli.main(["from-hjson", str(outside), "other"]) == 1
    assert "[hjson.path]" in capsys.readouterr().err


def test_a_slot_name_that_is_no_identifier(tmp_path):
    with pytest.raises(Exception) as e:
        read(cluster(tmp_path, accs=ACC.replace("NAME", "snax-alu")))
    assert e.value.codes == ["hjson.value"] and "snax_acc_name" in str(e.value)


def test_vecadd_w4_designs_on_snax_forge_inside_the_tcdm_without_allocating_it():
    """The first nonzero L1 base end to end in the design step."""
    tracemalloc.start()
    d = Design(Platform.load(SNAX_FORGE), bound_vecadd(4))
    assert run_checks(d) == []
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 64 << 20  # nothing the size of either memory
    lo, hi = 0x1000_0000, 0x1000_0000 + 128 * 1024
    for c, mems in d.memory.layouts.items():
        a, b = mems["l1"].span()
        assert lo <= a and b + 8 <= hi, c
    assert [mems["l1"].base for mems in d.memory.layouts.values()] == [lo, lo + 512, lo + 1024]
