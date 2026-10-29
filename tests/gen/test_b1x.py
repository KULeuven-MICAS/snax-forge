"""B1x's designs on the generator (D112, D147, C§17): one accelerator of linked units, streamers
on the outer ports only, one task per group, and the register writes of dot 6 and matvec 6."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.gen import generate

from .helpers import designed


def _generated(tmp_path: Path, recipe: str) -> Path:
    path = designed(tmp_path, recipe, recipe)
    out = tmp_path / "gvsoc"
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        generate(path, out, tmp_path)
    return out


def _core0(out: Path) -> list[str]:
    """The accelerator core's lines up to its start, without the HIGH halves and spatial strides."""
    text = (out / "main.c").read_text()
    body = text[text.index("int forge_core_0(void) {") : text.index("STREAMER_START_CSR")]
    return [ln.strip() for ln in body.splitlines()[1:-1]  # the last: the start's own line
            if "csrw_ss" in ln and "_HIGH" not in ln and "S_STRIDE" not in ln]  # fmt: skip


def test_dot6_writes_its_movers_jobs_and_the_macs_beats(tmp_path):
    """dot 6 at N = 512, W = 4: A, B and out on the contiguous plan; the writer's one beat at
    temporal stride 0 (its spatial stride is the word, 8)."""
    out = _generated(tmp_path, "dot_6")
    assert _core0(out) == [
        "csrw_ss(BASE_PTR_READER_0_LOW, 0x10000000);", "csrw_ss(T_BOUND_READER_0_0, 128);",
        "csrw_ss(T_STRIDE_READER_0_0, 32);",
        "csrw_ss(BASE_PTR_READER_1_LOW, 0x10001000);", "csrw_ss(T_BOUND_READER_1_0, 128);",
        "csrw_ss(T_STRIDE_READER_1_0, 32);",
        "csrw_ss(BASE_PTR_WRITER_0_LOW, 0x10002000);", "csrw_ss(T_BOUND_WRITER_0_0, 1);",
        "csrw_ss(T_STRIDE_WRITER_0_0, 0);",
        "csrw_ss(DOT_RW_JOBS, 1);", "csrw_ss(DOT_RW_MAC_JOB_BEATS, 128);",
    ]  # fmt: skip
    main = (out / "main.c").read_text()
    assert "csrw_ss(DOT_RW_START, 1);" in main and "csrr_ss(DOT_RO_COUNTER)" in main
    acc = json.loads((out / "accelerator.json").read_text())
    (group,) = acc["groups"]
    assert (acc["name"], acc["inputs"], acc["outputs"]) == ("dot", ["mac.a", "mac.b"], ["sum.out"])
    assert group["links"] == [{"name": "out_partial", "from": "mac.out", "to": "sum.a", "depth": 1}]
    sum_unit = group["units"][1]
    assert (sum_unit["datapath"], sum_unit["params"], sum_unit["timing"]) == (
        "SerialAccumulate", {"T": 1}, {"latency": 1, "ii": 1, "span": 4}
    )  # fmt: skip
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["tasks"] == [{"name": "mult_dot", "beats": 128,
                                  "csrs": {"JOBS": 1, "MAC_JOB_BEATS": 128}}]  # fmt: skip


def test_matvec6_rereads_p_for_every_row(tmp_path):
    """matvec 6, 16 x 512, W = 4: A's reader a second loop over the rows (4096 B), p's the
    same loop at stride 0, the writer 16 words at stride 8; JOBS 16."""
    out = _generated(tmp_path, "matvec_6")
    assert _core0(out) == [
        "csrw_ss(BASE_PTR_READER_0_LOW, 0x10000000);", "csrw_ss(T_BOUND_READER_0_0, 128);",
        "csrw_ss(T_BOUND_READER_0_1, 16);", "csrw_ss(T_STRIDE_READER_0_0, 32);",
        "csrw_ss(T_STRIDE_READER_0_1, 4096);",
        "csrw_ss(BASE_PTR_READER_1_LOW, 0x10010000);", "csrw_ss(T_BOUND_READER_1_0, 128);",
        "csrw_ss(T_BOUND_READER_1_1, 16);", "csrw_ss(T_STRIDE_READER_1_0, 32);",
        "csrw_ss(T_STRIDE_READER_1_1, 0);",
        "csrw_ss(BASE_PTR_WRITER_0_LOW, 0x10011000);", "csrw_ss(T_BOUND_WRITER_0_0, 16);",
        "csrw_ss(T_STRIDE_WRITER_0_0, 8);",
        "csrw_ss(MATVEC_RW_JOBS, 16);", "csrw_ss(MATVEC_RW_MAC_JOB_BEATS, 128);",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("recipe", "fold"), [("dot_2", "SUM"), ("dot_3", "SUM"), ("dot_4", "MAC"), ("dot_5", "MAC")]
)
def test_one_group_names_one_fold_csr(tmp_path, recipe, fold):
    """C§17's table: START 0, JOBS 1, the fold's JOB_BEATS 2, STATUS 3, COUNTER 4; the tree or
    serial sum after a MAC has a fixed T of 1 and no JOB_BEATS."""
    out = _generated(tmp_path, recipe)
    header = (out / "include" / "dot_csr.h").read_text()
    names = [ln.split()[1] for ln in header.splitlines() if ln.startswith("#define DOT_")]
    assert names == ["DOT_RW_START", "DOT_RW_JOBS", f"DOT_RW_{fold}_JOB_BEATS", "DOT_RO_STATUS",
                     "DOT_RO_COUNTER"]  # fmt: skip
    hjson = (out / "snax_forge_streamer.hjson").read_text()
    assert hjson.count("spatial_bounds") == 2  # readers and writers: the outer ports only
