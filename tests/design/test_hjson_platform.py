"""A platform read from snax_cluster's cluster file (D131), and its source."""

from __future__ import annotations

import tracemalloc
from dataclasses import replace

import pytest

from snax_forge.design import Design, Platform, run_checks
from snax_forge.design import __main__ as cli
from snax_forge.design.hjson_platform import platform_from_hjson

from .helpers import CFG, REPO, SNAX_ALU, bound_vecadd, platform_dict

NEW, OLD = CFG / "snax_alu_cluster.hjson", CFG / "snax_alu_cluster_old.hjson"
# snax_alu's source: a copy of snax-gvsoc's sw/cfg/snax_alu_cluster.hjson
SOURCE = REPO / "platforms" / "snax_alu.hjson"
needs_submodule = pytest.mark.skipif(
    not NEW.exists(), reason="the snax-gvsoc submodule is not checked out (CI never fetches it)"
)

# A cluster file with what the reader needs, in the hjson's own form.
CLUSTER = """{
  cluster: {
    cluster_base_addr: 268435456, data_width: 64, dma_data_width: 512,
    tcdm: {size: 128, banks: 32},
    hives: [{cores: [{$ref: "#/acc_core"}, {isa: "rv32ima"}]}],
  },
  dram: {address: 2147483648, length: 2147483648},
  acc_core: {snax_acc_cfg: [ACCS]},
  // a streamer template as snax_alu's
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


def test_snax_alu_round_trips_byte_for_byte():
    p = Platform.load(SNAX_ALU)
    assert p.to_json() == SNAX_ALU.read_text()
    assert Platform.from_dict(p.to_dict()) == p


def test_the_source_gives_the_checked_in_platform():
    """platforms/snax_alu.json is `design from-hjson platforms/snax_alu.hjson snax_alu`."""
    p = platform_from_hjson(SOURCE, "snax_alu")
    assert p.to_json() == SNAX_ALU.read_text()
    assert (p.hjson, p.accelerator_slot) == ("platforms/snax_alu.hjson", "snax_alu")


@needs_submodule
def test_the_source_is_the_submodules_hjson_verbatim():
    """The copy is refreshed by hand; its diff is the check (as D126)."""
    assert SOURCE.read_bytes() == NEW.read_bytes()


@needs_submodule
def test_the_old_hjson_gives_the_same_platform_but_for_the_name():
    old = platform_from_hjson(OLD, "snax_alu_old")
    assert (old.name, old.hjson) == (
        "snax_alu_old",
        "gvsoc/snax-gvsoc/sw/cfg/snax_alu_cluster_old.hjson",
    )
    assert replace(
        old, name="snax_alu", hjson=SOURCE.relative_to(REPO).as_posix()
    ) == Platform.load(SNAX_ALU)


def test_a_source_outside_the_root_is_refused(tmp_path):
    with pytest.raises(Exception) as e:
        platform_from_hjson(cluster(tmp_path), "c")
    assert e.value.codes == ["hjson.path"] and "copy the file into platforms/" in str(e.value)


def test_what_is_read_and_what_is_small16s(tmp_path):
    p = read(cluster(tmp_path))
    assert p == replace(Platform.load(SNAX_ALU), name="c", hjson="c.hjson", accelerator_slot="a")
    l1, l2 = p.l1, p.l2
    assert (l1.n_banks, l1.width_bits, l1.rows, l1.base_addr, l1.wide_bits) == (
        32, 64, 512, 0x1000_0000, 512
    )  # fmt: skip
    assert (l2.base_addr, l2.size_bytes, l2.beat_bits, l2.width_bits) == (2**31, 2**31, 512, 64)
    assert (p.default.temporal_dims, p.default.fifo_depth) == (1, 8)
    small = platform_dict()
    assert (p.default.addr_depth, p.default.prio) == (8, 0)
    for k in ("xbar", "dma", "controller", "register_window", "wait_mode"):
        assert p.to_dict()[k] == small[k]
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
    src = cluster(tmp_path, accs=acc, name="platforms/snax_alu.hjson")
    assert cli.main(["from-hjson", str(src), "snax_alu"]) == 0
    assert (tmp_path / "platforms" / "snax_alu.json").read_text() == SNAX_ALU.read_text()
    assert cli.main(["from-hjson", str(src), "snax_alu"]) == 1
    assert "pass --force" in capsys.readouterr().err
    assert cli.main(["from-hjson", str(src), "snax_alu", "--force"]) == 0
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


def test_vecadd_w4_designs_on_snax_alu_inside_the_tcdm_without_allocating_it():
    """The first nonzero L1 base end to end in the design step, beside a 2 GiB L2."""
    tracemalloc.start()
    d = Design(Platform.load(SNAX_ALU), bound_vecadd(4))
    assert run_checks(d) == []
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 64 << 20  # nothing the size of either memory
    lo, hi = 0x1000_0000, 0x1000_0000 + 128 * 1024
    for c, mems in d.memory.layouts.items():
        a, b = mems["l1"].span()
        assert lo <= a and b + 8 <= hi, c
        assert mems["l2"].base >= 2**31
    assert [mems["l1"].base for mems in d.memory.layouts.values()] == [lo, lo + 512, lo + 1024]
