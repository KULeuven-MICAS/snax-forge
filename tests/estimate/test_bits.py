"""The register bits of the estimate (D154, C§19): a control unit's registers from its group
description (estimate/bits.py), and every datapath's and control unit's bits against what DC
kept in SYN2's 48 points (``fixtures/syn2_flops.json``, util/syn2_flops.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.estimate import ctrl_bits, ctrl_registers

from .helpers import estimated

SYN2 = Path(__file__).parent / "fixtures" / "syn2_flops.json"
JW = 32


def _unit(name: str, latency: int, span: int = 1, ii: int = 1) -> dict:
    return {"name": name, "L": latency, "ii": ii, "span": span, "inputs": 1}


def _link(frm: str, to: str, depth: int) -> dict:
    return {"from": frm, "to": to, "to_input": 0, "depth": depth}


def test_a_fold_alone_counts_its_t_from_the_start():
    """vsum: one TreeAccumulate, T from JOB_BEATS: its latch, counter, jobs digit and v."""
    regs = ctrl_registers([_unit("acc", 1)], [], "acc", {"acc": None}, JW)
    assert regs == [("busy", 1), ("acc_jobs", 32), ("acc_job_beats", 32),
                    ("acc_job_beats_counter", 32), ("acc_jobs_counter", 32), ("acc_v", 1),
                    ("done", 1)]  # fmt: skip
    assert ctrl_bits([_unit("acc", 1)], [], "acc", {"acc": None}, JW) == 131


def test_a_unit_without_a_fold_has_no_t_and_a_v_at_l_0():
    """vecmul: ElementwiseLanes, L 0, T 1: its jobs digit; v is declared as one bit (max(L, 1))
    but never written, so firtool drops it."""
    regs = ctrl_registers([_unit("acc", 0)], [], "acc", {"acc": 1}, JW)
    assert regs == [("busy", 1), ("acc_jobs", 32), ("acc_jobs_counter", 32), ("done", 1)]
    declared = ctrl_registers([_unit("acc", 0)], [], "acc", {"acc": 1}, JW, declared=True)
    assert declared == [*regs[:3], ("acc_v", 1), ("done", 1)]


def test_dot_5_counts_the_lane_and_a_wires_slots():
    """dot 5 at W = 8: the MAC's T from the start, the serial sum's fixed T 1 (no digit for
    it), its lane of 3 bits; the wire's two 1-bit slots are declared but dropped."""
    units, links = [_unit("mac", 1), _unit("sum", 1, span=8)], [_link("mac", "sum", 0)]
    regs = ctrl_registers(units, links, "sum", {"mac": None, "sum": 1}, JW)
    assert regs == [("busy", 1), ("sum_jobs", 32), ("mac_job_beats", 32),
                    ("mac_job_beats_counter", 32), ("mac_jobs_counter", 32), ("mac_v", 1),
                    ("sum_jobs_counter", 32), ("sum_v", 1), ("sum_lane", 3), ("done", 1)]  # fmt: skip
    declared = dict(ctrl_registers(units, links, "sum", {"mac": None, "sum": 1}, JW, True))
    assert (declared["mac_sum_wslot"], declared["mac_sum_rslot"]) == (1, 1)


def test_a_unit_at_l_0_on_a_wire_borrows_and_one_on_a_buffer_counts():
    """dot 3: the elementwise mul at L 0 on a wire borrows the sum's test (no digit, no jobs
    counter); dot 2: on a buffer it counts in mixed radix, a digit for the sum's T."""
    units = [_unit("mul", 0), _unit("sum", 1)]
    fixed = {"mul": 1, "sum": None}
    wire = dict(ctrl_registers(units, [_link("mul", "sum", 0)], "sum", fixed, JW))
    buf = dict(ctrl_registers(units, [_link("mul", "sum", 1)], "sum", fixed, JW))
    assert "mul_jobs_counter" not in wire and "mul_sum_job_beats_counter" not in wire
    assert (buf["mul_jobs_counter"], buf["mul_sum_job_beats_counter"]) == (32, 32)
    assert buf["mul_sum_occupancy"] == 1 and "mul_sum_wslot" not in buf  # one slot: dropped


def test_a_fixed_t_counts_in_its_own_bits_and_the_ii_gap_and_job_width():
    """A fixed T of 5 is a 3-bit count; II 3 a 2-bit gap; a buffer of 4 a 3-bit occupancy and
    2-bit slots; JOBS and the jobs digits follow job_width."""
    units, links = [_unit("a", 1, ii=3), _unit("b", 2)], [_link("a", "b", 4)]
    regs = dict(ctrl_registers(units, links, "b", {"a": 5, "b": 6}, 16))
    assert (regs["a_job_beats_counter"], regs["a_b_job_beats_counter"]) == (3, 3)
    assert (regs["a_gap"], regs["b_v"], regs["b_jobs"], regs["a_jobs_counter"]) == (2, 2, 16, 16)
    assert (regs["a_b_occupancy"], regs["a_b_wslot"], regs["a_b_rslot"]) == (3, 2, 2)
    assert "a_job_beats" not in regs


# =============================================================================
# Against SYN2: every datapath and every control unit exactly
# =============================================================================


@pytest.fixture(scope="module")
def syn2():
    """Per SYN2 point (recipe, W) the fixture's row and the estimate's one group."""
    rows = json.loads(SYN2.read_text())["points"]
    out = []
    for r in rows:
        (g,) = estimated(r["recipe"], W=r["W"]).groups
        out.append((r, g))
    return out


@pytest.mark.slow
def test_syn2_has_the_48_design_points():
    rows = json.loads(SYN2.read_text())["points"]
    want = {(r, w) for r in ("vecadd", "vecmul", "vsum_tree", *(f"dot_{k}" for k in range(2, 7)))
            for w in (1, 2, 4, 8, 16, 32)}  # fmt: skip
    assert {(r["recipe"], r["W"]) for r in rows} == want


@pytest.mark.slow
def test_every_datapath_has_the_flops_dc_kept(syn2):
    """Each SYN2 instance, unit or buffer, has the catalogue's flops on its params; the estimate
    lists nothing else but wires, which hold none."""
    for r, g in syn2:
        want = {k: v["flops"] for k, v in r["instances"].items()}
        assert {k: g.bits[k] for k in want} == want, r["point"]
        assert all(v == 0 for k, v in g.bits.items() if k not in want), (r["point"], g.bits)


@pytest.mark.slow
def test_ctrl_bits_equal_syn2s(syn2, capsys):
    """The control unit's counted bits equal DC's at every point (firtool drops a v at L 0 and
    a wire's or depth-1 buffer's slots, and ctrl_bits leaves them out); the table is printed."""
    lines = [
        "| point | recipe | W | counted | synthesized | difference |",
        "|---|---|---|---|---|---|",
    ]
    for r, g in syn2:
        lines.append(f"| {r['point']} | {r['recipe']} | {r['W']} | {g.ctrl_bits} | {r['ctrl']} | "
                     f"{g.ctrl_bits - r['ctrl']:+d} |")  # fmt: skip
    with capsys.disabled():
        print("\nctrl bits, counted (estimate/bits.py) against SYN2's (DC):\n" + "\n".join(lines))
    assert all(g.ctrl_bits == r["ctrl"] for r, g in syn2), [
        r["point"] for r, g in syn2 if g.ctrl_bits != r["ctrl"]
    ]
