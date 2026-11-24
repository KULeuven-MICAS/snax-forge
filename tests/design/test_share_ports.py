"""Shared movers: the crossbar's inputs by hand, the movers and their masks, the
fit rule, and the crossbar's warning and limit."""

from __future__ import annotations

import pytest

from snax_forge.design import Design
from snax_forge.design.check import NOTES, run_checks
from snax_forge.design.platform import Platform
from tests.helpers import steps

from .helpers import SNAX_FORGE


@pytest.mark.slow
def test_above_64_inputs_a_note_above_128_a_refusal():
    """Mvt_2 at W = 8 (84 inputs) and vecmul at W = 32 (98) pass with the note
    design.xbar_inputs; vecmul at W = 64 (194) is refused (connect.xbar_inputs); shared, mvt_v2
    takes 50 and gets no note."""
    pf = Platform.load(SNAX_FORGE)
    for name, params, n in (("mvt_v2", {"W_x1": 8, "W_x2": 8}, 84),
                            ("vecmul_v1", {"W": 32, "N": 2048}, 98)):  # fmt: skip
        d = Design(pf, steps(name, **params)[-1])
        assert run_checks(d) == []
        (note,) = NOTES["design.xbar_inputs"](d)
        assert note.message.startswith(f"{n} TCDM crossbar inputs, above 64 (l1.warn_xbar_inputs)")
    d = Design(pf, steps("mvt_v2_2", W_x1=8, W_x2=8)[-1])
    assert list(NOTES["design.xbar_inputs"](d)) == []
    wide = Design(pf, steps("vecmul_v1", W=64, N=2048)[-1])
    (p,) = [p for p in run_checks(wide) if p.code == "connect.xbar_inputs"]
    assert p.message.startswith("194 TCDM crossbar inputs, the platform takes at most 128")
    assert p.fix == "fewer lanes, or share_ports on a container between two accelerators"
