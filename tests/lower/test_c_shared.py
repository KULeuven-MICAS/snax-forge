"""The program of a shared set (D169, PORT1): each shared mover written once per task (its
group's port, else idle), the narrow user's mask, PORT_MUX_SEL after the wait and before
STREAMER_START, and ``c.port_mux``."""

from __future__ import annotations

import re

import pytest

from snax_forge.lower import CError
from snax_forge.lower import c as c_mod
from tests.helpers import steps

from .test_c import text


def mvt(tmp_path, **kw) -> str:
    return text(tmp_path, graph=steps("mvt_1_dot2_shared", W=8)[-1], name="mvt_1_dot2_shared", **kw)


def test_the_select_follows_the_tasks_after_their_waits(tmp_path):
    """mvt_1_dot2_shared at W = 8: dot for x1 (0), vadd for x1 (1), dot for x2 (0), vadd for x2
    (1); each select after the wait on the streamer, then STREAMER_START, then the group's START."""
    lines = [x.strip() for x in mvt(tmp_path).splitlines()]
    sel = [i for i, x in enumerate(lines) if "MVT_VADD_RW_PORT_MUX_SEL" in x]
    assert [lines[i] for i in sel] == [f"csrw_ss(MVT_VADD_RW_PORT_MUX_SEL, {k});"
                                       for k in (0, 1, 0, 1)]  # fmt: skip
    for k, i in enumerate(sel):
        assert lines[i + 1] == "csrw_ss(MVT_VADD_STREAMER_START_CSR, 1);"
        assert lines[i + 2] == f"csrw_ss(MVT_VADD_RW_{('MVT', 'VADD')[k % 2]}_START, 1);"
        if k:  # the last wait on the streamer before it, no start between
            wait = max(j for j in range(i) if "MVT_VADD_STREAMER_BUSY_CSR" in lines[j])
            assert not any("START" in x for x in lines[wait:i])


def test_each_mover_once_a_task_with_its_users_mask(tmp_path):
    """Per task each of the three movers is written once (12 tasks' worth of BASE_PTR_LOW: 3 per
    task, 4 tasks); the writer's mask is vsum_x1.out's one channel (0x1, spatial stride 0, so a
    mask that failed writes over its word) or vadd.out's eight (0xff)."""
    c = mvt(tmp_path)
    assert len(re.findall(r"MVT_VADD_BASE_PTR_\w+_LOW", c)) == 12
    masks = re.findall(r"csrw_ss\(MVT_VADD_ENABLED_CHANNEL_WRITER_0, (0x[0-9a-f]+)\);", c)
    assert masks == ["0x00000001", "0x000000ff", "0x00000001", "0x000000ff"]
    strides = re.findall(r"csrw_ss\(MVT_VADD_S_STRIDE_WRITER_0_0, (\d+)\);", c)
    assert strides == ["0", "8", "0", "8"]


def test_a_select_without_a_wait_is_refused(tmp_path, monkeypatch):
    """c.port_mux: the waits taken out, the second task's select would change under a busy
    streamer."""
    real = c_mod.placed_waits
    monkeypatch.setattr(c_mod, "placed_waits", lambda tasks, point: [
        w for w in real(tasks, point) if w.component == "dma"])  # fmt: skip
    with pytest.raises(CError, match=r"\[c.port_mux\] .*PORT_MUX_SEL of mvt_vadd to vadd \(1\) "
                       r"while its previous task \w+ has no wait"):  # fmt: skip
        mvt(tmp_path)
