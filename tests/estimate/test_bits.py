"""The register bits of the estimate: a control unit's registers from its group
description (estimate/bits.py), and the RTL fixtures' ``ctrl_registers.json`` (hw/chisel's
CtrlRegistersSpec, which compares them with the emitted control units) current."""

from __future__ import annotations

import importlib.util
import json

import pytest

from snax_forge.estimate import ctrl_registers
from tests.helpers import REPO

_spec = importlib.util.spec_from_file_location("ctrl_regs", REPO / "util" / "ctrl_regs.py")
ctrl_regs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ctrl_regs)

JW = 32


def _unit(name: str, latency: int, span: int = 1, ii: int = 1) -> dict:
    return {"name": name, "L": latency, "ii": ii, "span": span, "inputs": 1}


def _link(frm: str, to: str, depth: int) -> dict:
    return {"from": frm, "to": to, "to_input": 0, "depth": depth}


def test_a_fixed_t_counts_in_its_own_bits_and_the_ii_gap_and_job_width():
    """A fixed T of 5 is a 3-bit count; II 3 a 2-bit gap; a buffer of 4 a 3-bit occupancy and
    2-bit slots; the jobs digits follow job_width, and JOBS has no copy."""
    units, links = [_unit("a", 1, ii=3), _unit("b", 2)], [_link("a", "b", 4)]
    regs = dict(ctrl_registers(units, links, "b", {"a": 5, "b": 6}, 16))
    assert (regs["a_job_beats_counter"], regs["a_b_job_beats_counter"]) == (3, 3)
    assert (regs["a_gap"], regs["b_v"], regs["a_jobs_counter"]) == (2, 2, 16)
    assert (regs["a_b_occupancy"], regs["a_b_wslot"], regs["a_b_rslot"]) == (3, 2, 2)
    assert "a_job_beats" not in regs and "b_jobs" not in regs


def test_the_rtl_fixtures_ctrl_registers_json_is_current():
    """hw/chisel's CtrlRegistersSpec compares every fixture's emitted control units with
    ``ctrl_registers.json``, which must be ``ctrl_registers`` on each fixture's accelerator.json
    as it is now (``pixi run python util/ctrl_regs.py --json`` writes it again)."""
    if not ctrl_regs.FORGE.is_dir():
        pytest.skip("gvsoc/snax-gvsoc is not checked out: the fixtures' accelerator.json")
    assert ctrl_regs.REGISTERS_JSON.read_text() == ctrl_regs.estimated()
    units = json.loads(ctrl_regs.REGISTERS_JSON.read_text())
    assert units["mvt_v5_sequential"]["snax_forge_mvt_v5_mvt_ctrl"][-2:] == [
        ["vmul_x1_vsum_x1_occupancy", 1],
        ["done", 1],
    ]


# =============================================================================
# Against SYN2: every datapath's count, and each control unit's within 10 %
# =============================================================================
