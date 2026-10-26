"""The generated C headers: the accelerator's and the runtime's view of the cluster."""

from __future__ import annotations

import re

import pytest

from snax_forge.common import generated_header
from snax_forge.design import Platform
from snax_forge.gen import GenError, generate
from snax_forge.gen.headers import cluster_headers, define_name, streamer_header

from .helpers import SNAX_FORGE, SOURCE, SUB, parsed, point

CHISEL = SUB / "sw" / "generated" / "snax_forge" / "streamer_csr_addr_map.h"
SHELL = SUB / "sw" / "apps" / "snax-shell" / "src" / "snax-shell.c"
DEFINE = re.compile(r"^#define (\w+) (\d+)$", re.MULTILINE)
SYNCED = SUB / "sw" / "generated" / "snax_forge"
CLUSTER_HEADERS = ["snitch_cluster_cfg.h", "snitch_cluster_addrmap.h"]

# vecadd at W = 4 on snax_forge: three movers of one spatial and one temporal dimension
STREAMER_W4 = [
    ("BASE_PTR_READER_0_LOW", 960), ("BASE_PTR_READER_0_HIGH", 961),
    ("S_STRIDE_BASE_READER_0", 962), ("S_STRIDE_NUM_READER_0", 1), ("S_STRIDE_READER_0_0", 962),
    ("T_BOUND_BASE_READER_0", 963), ("T_BOUND_NUM_READER_0", 1), ("T_BOUND_READER_0_0", 963),
    ("T_STRIDE_BASE_READER_0", 964), ("T_STRIDE_NUM_READER_0", 1), ("T_STRIDE_READER_0_0", 964),
    ("BASE_PTR_READER_1_LOW", 965), ("BASE_PTR_READER_1_HIGH", 966),
    ("S_STRIDE_BASE_READER_1", 967), ("S_STRIDE_NUM_READER_1", 1), ("S_STRIDE_READER_1_0", 967),
    ("T_BOUND_BASE_READER_1", 968), ("T_BOUND_NUM_READER_1", 1), ("T_BOUND_READER_1_0", 968),
    ("T_STRIDE_BASE_READER_1", 969), ("T_STRIDE_NUM_READER_1", 1), ("T_STRIDE_READER_1_0", 969),
    ("BASE_PTR_WRITER_0_LOW", 970), ("BASE_PTR_WRITER_0_HIGH", 971),
    ("S_STRIDE_BASE_WRITER_0", 972), ("S_STRIDE_NUM_WRITER_0", 1), ("S_STRIDE_WRITER_0_0", 972),
    ("T_BOUND_BASE_WRITER_0", 973), ("T_BOUND_NUM_WRITER_0", 1), ("T_BOUND_WRITER_0_0", 973),
    ("T_STRIDE_BASE_WRITER_0", 974), ("T_STRIDE_NUM_WRITER_0", 1), ("T_STRIDE_WRITER_0_0", 974),
    ("STREAMER_START_CSR", 975), ("STREAMER_BUSY_CSR", 976),
    ("STREAMER_PERFORMANCE_COUNTER_CSR", 977),
]  # fmt: skip
ACC_W4 = [("ACC_RW_LOOP_COUNT", 978), ("ACC_RW_START", 979), ("ACC_RO_BUSY", 980),
          ("ACC_RO_PERF_COUNT", 981)]  # fmt: skip


def defines(text: str) -> list[tuple[str, int]]:
    return [(k, int(v)) for k, v in DEFINE.findall(text)]


@pytest.fixture(scope="module")
def include(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w4")
    m = generate(point(tmp), tmp / "out", root=tmp)
    assert [f for f in m.files if f.startswith("include/")] == [
        "include/streamer_csr_addr_map.h", "include/acc_csr.h", "include/snitch_cluster_cfg.h",
        "include/snitch_cluster_addrmap.h",
    ]  # fmt: skip
    return tmp / "out" / "include"


def test_the_streamer_header_numbers(include):
    text = (include / "streamer_csr_addr_map.h").read_text()
    assert defines(text) == STREAMER_W4
    assert text.startswith(generated_header("design/vecadd/design_point.json", ""))


@pytest.mark.skipif(not CHISEL.exists(), reason="the snax-gvsoc submodule is not checked out")
def test_the_streamer_header_is_the_chisel_ones(include):
    assert defines((include / "streamer_csr_addr_map.h").read_text()) == defines(CHISEL.read_text())


def test_the_accelerator_header_numbers(include):
    assert defines((include / "acc_csr.h").read_text()) == ACC_W4


@pytest.mark.skipif(not SHELL.exists(), reason="the snax-gvsoc submodule is not checked out")
def test_snax_shell_drives_the_accelerator_header(include):
    """snax-gvsoc's snax-shell includes vecadd's acc_csr.h: every CSR name it uses is ours."""
    ours = dict(defines((include / "acc_csr.h").read_text()))
    used = set(re.findall(r"\bACC_\w+", SHELL.read_text()))
    assert used and used <= set(ours), used - set(ours)


def test_more_dimensions_follow_streamer_scalas_loops():
    """Per mover: the pointer, then each index of the spatial strides, temporal bounds, strides."""
    movers = [{"name": "READER_0", "spatial_bounds": [2, 4], "temporal_dim": 3}]
    text, last = streamer_header(movers, 960, "p")
    assert defines(text) == [
        ("BASE_PTR_READER_0_LOW", 960), ("BASE_PTR_READER_0_HIGH", 961),
        ("S_STRIDE_BASE_READER_0", 962), ("S_STRIDE_NUM_READER_0", 2),
        ("S_STRIDE_READER_0_0", 962), ("S_STRIDE_READER_0_1", 963),
        ("T_BOUND_BASE_READER_0", 964), ("T_BOUND_NUM_READER_0", 3),
        ("T_BOUND_READER_0_0", 964), ("T_BOUND_READER_0_1", 965), ("T_BOUND_READER_0_2", 966),
        ("T_STRIDE_BASE_READER_0", 967), ("T_STRIDE_NUM_READER_0", 3),
        ("T_STRIDE_READER_0_0", 967), ("T_STRIDE_READER_0_1", 968), ("T_STRIDE_READER_0_2", 969),
        ("STREAMER_START_CSR", 970), ("STREAMER_BUSY_CSR", 971),
        ("STREAMER_PERFORMANCE_COUNTER_CSR", 972),
    ]  # fmt: skip
    assert last == 972


NAMES = [
    ("loopCount", "LOOP_COUNT"),
    ("opSel", "OP_SEL"),
    ("len", "LEN"),
    ("csr2Mode", "CSR2_MODE"),
]


@pytest.mark.parametrize(("field", "name"), NAMES)
def test_define_names(field, name):
    assert define_name(field) == name


# =============================================================================
# The runtime's cluster headers
# =============================================================================


def below_comments(text: str) -> list[str]:
    """A header's lines but its comments and blank lines (the generated header comment, the
    template's copyright)."""
    return [line for line in text.splitlines() if line and not line.startswith("//")]


def values(text: str) -> dict[str, int]:
    """Every define of an address map, its expression evaluated over the earlier ones."""
    out: dict[str, int] = {}
    for name, value in re.findall(r"^#define (\w+) (.+)$", text, re.MULTILINE):
        out[name] = eval(re.sub(r"[A-Z_]\w*", lambda m: str(out[m[0]]), value))
    return out


@pytest.mark.skipif(not SYNCED.exists(), reason="the snax-gvsoc submodule is not checked out")
@pytest.mark.parametrize("name", CLUSTER_HEADERS)
def test_the_base_clusters_headers_are_the_synced_ones(include, name):
    """snax_cluster's templates on the same cluster, as snax-gvsoc's sw/ holds them."""
    ours = below_comments((include / name).read_text())
    assert ours == below_comments((SYNCED / name).read_text())


def test_a_64_kib_l1_moves_the_peripherals_and_the_zero_memory(tmp_path):
    p = Platform.load(SNAX_FORGE).with_changes([("l1.rows", 256)])
    generate(point(tmp_path, platform=p), tmp_path / "out", root=tmp_path)
    text = (tmp_path / "out" / "include" / "snitch_cluster_addrmap.h").read_text()
    v = values(text)
    assert "#define CLUSTER_TCDM_SIZE 0x10000" in text
    assert "#define CLUSTER_PERIPH_BASE_ADDR (CLUSTER_TCDM_BASE_ADDR + CLUSTER_TCDM_SIZE)" in text
    assert (v["CLUSTER_TCDM_BASE_ADDR"], v["CLUSTER_PERIPH_BASE_ADDR"]) == (
        0x1000_0000,
        0x1001_0000,
    )
    assert (v["CLUSTER_ZERO_MEM_START_ADDR"], v["CLUSTER_ZERO_MEM_END_ADDR"]) == (
        0x1002_0000, 0x1003_0000
    )  # fmt: skip


def test_the_base_clusters_address_map_values(include):
    v = values((include / "snitch_cluster_addrmap.h").read_text())
    assert (v["CLUSTER_TCDM_SIZE"], v["CLUSTER_PERIPH_BASE_ADDR"]) == (0x20000, 0x1002_0000)
    assert (v["CLUSTER_ZERO_MEM_START_ADDR"], v["CLUSTER_ZERO_MEM_END_ADDR"]) == (
        0x1003_0000, 0x1004_0000
    )  # fmt: skip
    assert (v["CLUSTER_SIZE"], v["CLUSTER_ADDRWIDTH"], v["CLINT_BASE_ADDR"]) == (
        0x40_0000, 22, 0xFFFF_0000
    )  # fmt: skip


def test_three_cores_are_counted_over_the_hives(include):
    """As clustergen counts them: every core of every hive, ``$ref``s resolved."""
    cluster = parsed(include.parent / "snax_forge.hjson")
    cluster["cluster"]["hives"][0]["cores"].append({"$ref": "#/dma_core_template"})
    cfg = cluster_headers(cluster, "p")["snitch_cluster_cfg.h"]
    assert below_comments(cfg) == [
        "#define CFG_CLUSTER_NR_CORES 3",
        "#define CFG_CLUSTER_BASE_HARTID 0",
    ]


def test_a_cluster_size_not_a_power_of_two_is_refused(tmp_path):
    path = point(tmp_path)
    src = tmp_path / "platforms" / SOURCE.name
    text = src.read_text()
    assert text.count("cluster_base_offset: 4194304,") == 1
    src.write_text(text.replace("cluster_base_offset: 4194304,", "cluster_base_offset: 3145728,"))
    with pytest.raises(GenError) as e:
        generate(path, tmp_path / "out", root=tmp_path)
    assert e.value.code == "gen.platform" and "3145728 (0x300000)" in str(e.value)
    assert not (tmp_path / "out").exists()
