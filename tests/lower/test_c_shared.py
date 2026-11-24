"""The program of a shared set: each shared mover written once per task (its
group's port, else idle), the narrow user's mask, PORT_MUX_SEL after the wait and before
STREAMER_START, and ``c.port_mux``."""

from __future__ import annotations

import pytest

from snax_forge.lower import CError
from snax_forge.lower import c as c_mod
from tests.helpers import steps

from .test_c import text


def mvt(tmp_path, **kw) -> str:
    return text(tmp_path, graph=steps("mvt_v1_2_2", W=8)[-1], name="mvt_v1_2_2", **kw)


def test_a_select_without_a_wait_is_refused(tmp_path, monkeypatch):
    """c.port_mux: the waits taken out, the second task's select would change under a busy
    streamer."""
    real = c_mod.placed_waits
    monkeypatch.setattr(c_mod, "placed_waits", lambda tasks, point: [
        w for w in real(tasks, point) if w.component == "dma"])  # fmt: skip
    with pytest.raises(CError, match=r"\[c.port_mux\] .*PORT_MUX_SEL of mvt_vadd to vadd \(1\) "
                       r"while its previous task \w+ has no wait"):  # fmt: skip
        mvt(tmp_path)
