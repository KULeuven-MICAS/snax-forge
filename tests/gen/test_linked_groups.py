"""Linked groups on the generator: one accelerator of linked units, streamers
on the outer ports only, one task per group, and the register writes of dot 6 and matvec 6."""

from __future__ import annotations

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
    body = text[text.index("int forge_core_0(void) {") : text.index("_STREAMER_START_CSR")]
    return [ln.strip() for ln in body.splitlines()[1:-1]  # the last: the start's own line
            if "csrw_ss" in ln and "_HIGH" not in ln and "S_STRIDE" not in ln]  # fmt: skip


@pytest.mark.slow  # 16 x 512 through the sandbox and the generator
def test_matvec6_rereads_p_for_every_row(tmp_path):
    """matvec 6, 16 x 512, W = 4: A's reader a second loop over the rows (4096 B), p's the
    same loop at stride 0, the writer 16 words at stride 8; JOBS 16."""
    out = _generated(tmp_path, "matvec_v6")
    assert _core0(out) == [
        "csrw_ss(MATVEC_BASE_PTR_READER_0_LOW, 0x10000000);", "csrw_ss(MATVEC_T_BOUND_READER_0_0, 128);",
        "csrw_ss(MATVEC_T_BOUND_READER_0_1, 16);", "csrw_ss(MATVEC_T_STRIDE_READER_0_0, 32);",
        "csrw_ss(MATVEC_T_STRIDE_READER_0_1, 4096);",
        "csrw_ss(MATVEC_BASE_PTR_READER_1_LOW, 0x10010000);", "csrw_ss(MATVEC_T_BOUND_READER_1_0, 128);",
        "csrw_ss(MATVEC_T_BOUND_READER_1_1, 16);", "csrw_ss(MATVEC_T_STRIDE_READER_1_0, 32);",
        "csrw_ss(MATVEC_T_STRIDE_READER_1_1, 0);",
        "csrw_ss(MATVEC_BASE_PTR_WRITER_0_LOW, 0x10011000);", "csrw_ss(MATVEC_T_BOUND_WRITER_0_0, 16);",
        "csrw_ss(MATVEC_T_STRIDE_WRITER_0_0, 8);",
        "csrw_ss(MATVEC_RW_JOBS, 16);", "csrw_ss(MATVEC_RW_MAC_JOB_BEATS, 128);",
    ]  # fmt: skip
