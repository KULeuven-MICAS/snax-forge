"""The cost of the estimate (estimate/cost.py, C§20): CHAR1's model on a technology's table,
its counting, and the kernel energy against snax-forge-experiments' model of SYN2's four designs
(``fixtures/char_model.json``, util/char_model.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.char import Table, load
from snax_forge.estimate import Estimate, estimate
from snax_forge.estimate.cost import Block, block_cost, group_cost
from tests.helpers import steps

from .helpers import SNAX_FORGE, estimated

MODEL = Path(__file__).parent / "fixtures" / "char_model.json"
TOL = 0.005
TABLE = Table("t", 10.0, {"snax_forge": "x"}, 2.0, 0.003, 0.001, 1.5, {
    "TreeAccumulate": [{"params": {"inType": "int64", "outType": "int64", "op": "add"},
                        "lanes_rule": "interpolate",
                        "points": [{"lanes": 2, "comb_um2": 100.0, "e_comb_pj": 4.0}]}]})  # fmt: skip
TREE = {"inType": "int64", "outType": "int64", "op": "add", "lanes": 2}


def test_a_block_costs_its_comb_per_fire_its_loads_and_its_holds():
    """e_fire = e_comb + n_bits (e_load − e_hold), hold = n_bits e_hold, task = n_fire e_fire +
    C_acc hold; area = A_comb + n_bits A_flop."""
    c = block_cost(TABLE, Block("TreeAccumulate", TREE, 10), 64, 100)
    assert (c.found, c.comb_um2, c.flops_um2, c.total_um2) == ("measured", 100.0, 128.0, 228.0)
    assert c.e_fire_pj == pytest.approx(4.0 + 64 * 0.002) and c.e_hold_pj == pytest.approx(0.064)
    assert c.task_nj == pytest.approx((10 * 4.128 + 100 * 0.064) / 1000)


def test_a_block_not_characterized_counts_its_comb_as_0_and_its_flops():
    for b in (Block("TreeAccumulate", {**TREE, "lanes": 64}, 10), Block(None, {}, 10)):
        c = block_cost(TABLE, b, 64, 100)
        assert (c.found, c.comb_um2, c.flops_um2) == (None, 0.0, 128.0)
        assert c.task_nj == pytest.approx((10 * 0.128 + 100 * 0.064) / 1000)


def test_a_groups_ctrl_is_its_bits_held_times_the_counter_factor():
    g = group_cost(TABLE, {"acc": Block("TreeAccumulate", TREE, 10)}, {}, {"acc": 64}, 131, 100)
    assert (g.ctrl_bits, g.ctrl_um2) == (131, 262.0)
    assert g.ctrl_nj == pytest.approx(1.5 * 131 * 100 * 0.001 / 1000)
    assert g.area == pytest.approx(228.0 + 262.0)
    assert g.energy == pytest.approx(g.units["acc"].task_nj + g.ctrl_nj) and g.missing == []


def test_a_unit_fires_its_firings_times_its_span_a_buffer_its_producers_firings():
    """dot 5 at W = 8: the Mac 256 fires, the serial sum one firing of 8 beats; dot 2 at W = 8:
    the depth-1 buffer loads once per beat of its producer, the mul's 256."""
    (g,) = estimated("dot_5", W=8, N=2048).groups
    assert (g.cost.units["mac"].fires, g.cost.units["sum"].fires) == (256, 8)
    (g,) = estimated("dot_2", W=8, N=2048).groups
    assert g.cost.links["tmp0"].fires == g.cost.units["mul"].fires == 256
    assert (
        g.cost.links["tmp0"].found == "measured" and g.cost.links["tmp0"].datapath == "LinkBuffer"
    )


def test_the_estimate_lists_what_is_not_characterized():
    e = estimated("vecadd_i32")
    assert e.missing == ["acc"] and e.groups[0].cost.units["acc"].found is None
    assert e.to_dict()["cost"]["missing"] == ["acc"]


def test_no_technology_skips_the_cost_and_both_round_trip():
    g = steps("dot_5", W=8, N=64)[-1]
    plain = estimate(g, None, None)
    assert plain.to_dict()["cost"] is None and plain.groups[0].cost is None
    assert plain.area is None and plain.energy is None
    costed = estimate(g, None, "nangate15")
    assert costed.to_dict()["cost"]["tech"] == "nangate15"
    assert costed.char_source == load("nangate15").source
    for e in (plain, costed):
        d = e.to_dict()
        assert Estimate.from_dict(json.loads(json.dumps(d))).to_dict() == d


def test_an_older_estimate_without_cost_still_reads():
    d = estimated("vecmul", W=8).to_dict()
    del d["cost"]
    for g in d["groups"]:
        del g["cost"], g["bits"], g["ctrl_bits"]
    e = Estimate.from_dict(d)
    assert e.tech is None and e.groups[0].cost is None and e.groups[0].bits == {}


@pytest.mark.slow
def test_the_kernel_energy_is_chars_model_of_syn2(capsys):
    """vecmul, vsum_tree, dot 4 and dot 5 at W = 8 and 32, N = 2048, on both technologies: the
    estimate's kernel energy (units, links and ctrl) against snax-forge-experiments' model, the
    same formula on the same inputs (its C_acc SYN2's window, its ctrl bits DC's); within 0.5 %."""
    rows = json.loads(MODEL.read_text())["rows"]
    lines = ["| tech | recipe | W | C_acc | ctrl bits | estimate nJ | model nJ | ratio |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    bad = []
    for r in rows:
        e = estimate(steps(r["recipe"], W=r["W"], N=2048)[-1], _platform(), r["tech"])
        (g,) = e.groups
        ratio = e.energy / r["kernel_nj"]
        lines.append(f"| {r['tech']} | {r['recipe']} | {r['W']} | {e.accelerator_cycles} | "
                     f"{g.ctrl_bits} | {e.energy:.3f} | {r['kernel_nj']:.3f} | {ratio:.4f} |")  # fmt: skip
        assert (e.accelerator_cycles, g.ctrl_bits) == (r["c_acc"], r["ctrl_bits"]), r
        if abs(ratio - 1) > TOL:
            bad.append((r["tech"], r["recipe"], r["W"], ratio))
    with capsys.disabled():
        print("\nkernel energy, the estimate against CHAR1's model:\n" + "\n".join(lines))
    assert len(rows) == 16 and not bad, bad


def _platform():
    from snax_forge.design import Platform

    return Platform.load(SNAX_FORGE)
