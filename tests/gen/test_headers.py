"""The generated C headers: per streamer set its streamer's (byte for byte snax_cluster's
StreamerSwHeaderGen's), per accelerator its own CSRs', and the runtime's view of the
cluster."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from snax_forge.gen import GenError, generate
from tests.helpers import requires_submodule

from .helpers import SOURCE, SUB, point

CHISEL = SUB / "sw" / "generated" / "snax_forge" / "streamer_csr_addr_map.h"
SHELL = SUB / "sw" / "apps" / "snax-shell" / "src" / "snax-shell.c"
DEFINE = re.compile(r"^#define (\w+) (\d+)$", re.MULTILINE)
SYNCED = SUB / "sw" / "generated" / "snax_forge"
CLUSTER_HEADERS = ["snitch_cluster_cfg.h", "snitch_cluster_addrmap.h"]
# snax_cluster ry/slot1's StreamerSwHeaderGen (Streamer.scala's writeCsrAddrMap at 7e97280e) on
# its snaxgen test's two sets (util/snaxgen/test_streamer_sets.hjson): vmul, three movers, with
# --csr-base 960 --macro-prefix VMUL_ --header-name streamer_vmul_csr_addr_map.h; vsum, two,
# with --csr-base 982 --macro-prefix VSUM_; dot_v7's mul and sum have their shapes
SNAX_CLUSTER = Path(__file__).parent / "fixtures" / "snax_cluster"

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
# with its set's prefix (vecadd's one set, acc)
PREFIXED_W4 = [(f"ACC_{k}", v) for k, v in STREAMER_W4]
ACC_W4 = [("ACC_RW_START", 978), ("ACC_RW_JOBS", 979), ("ACC_RO_STATUS", 980),
          ("ACC_RO_BUSY_COUNTER", 981)]  # fmt: skip


def defines(text: str) -> list[tuple[str, int]]:
    return [(k, int(v)) for k, v in DEFINE.findall(text)]


@pytest.fixture(scope="module")
def include(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w4")
    m = generate(point(tmp), tmp / "out", root=tmp)
    assert [f for f in m.files if f.startswith("include/")] == [
        "include/streamer_acc_csr_addr_map.h", "include/acc_csr.h",
        "include/snitch_cluster_cfg.h", "include/snitch_cluster_addrmap.h",
    ]  # fmt: skip
    return tmp / "out" / "include"


@requires_submodule
def test_snax_shell_drives_the_accelerator_header(include):
    """snax-gvsoc's snax-shell includes vecadd's acc_csr.h and its set's streamer header (G1:
    streamer_acc_csr_addr_map.h, its macros prefixed ACC_): every CSR name it uses is ours."""
    ours = dict(defines((include / "acc_csr.h").read_text()))
    ours |= dict(defines((include / "streamer_acc_csr_addr_map.h").read_text()))
    used = set(re.findall(r"\bACC_\w+", SHELL.read_text()))
    assert used and used <= set(ours), used - set(ours)


# =============================================================================
# The runtime's cluster headers
# =============================================================================


def values(text: str) -> dict[str, int]:
    """Every define of an address map, its expression evaluated over the earlier ones."""
    out: dict[str, int] = {}
    for name, value in re.findall(r"^#define (\w+) (.+)$", text, re.MULTILINE):
        out[name] = eval(re.sub(r"[A-Z_]\w*", lambda m: str(out[m[0]]), value))
    return out


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
