"""The generated accelerator's C headers."""

from __future__ import annotations

import re

import pytest

from snax_forge.common import generated_header
from snax_forge.gen import generate
from snax_forge.gen.headers import define_name, streamer_header

from .helpers import SUB, point

CHISEL = SUB / "sw" / "generated" / "snax_alu_cluster_old" / "streamer_csr_addr_map.h"
SHELL = SUB / "sw" / "apps" / "snax-shell" / "src" / "snax-shell.c"
DEFINE = re.compile(r"^#define (\w+) (\d+)$", re.MULTILINE)

# vecadd at W = 4 on snax_alu: three movers of one spatial and one temporal dimension
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
        "include/streamer_csr_addr_map.h", "include/acc_csr.h"
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
