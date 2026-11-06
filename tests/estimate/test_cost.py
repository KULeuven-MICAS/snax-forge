"""The cost of the estimate (estimate/cost.py, C§20): CHAR1's model on a technology's table,
its counting, and the kernel energy against snax-forge-experiments' model of SYN2's four designs
(``fixtures/char_model.json``, util/char_model.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.char import Table, load
from snax_forge.estimate import Estimate, estimate
from snax_forge.estimate.cost import CAVEAT, Block, block_cost, dec3, group_cost, heading, sig3
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
    """Its energy f_ctrl n_bits e_hold per cycle; its area its flops' plus an equal comb area (a
    placeholder, D154)."""
    g = group_cost(TABLE, {"acc": Block("TreeAccumulate", TREE, 10)}, {}, {"acc": 64}, 131, 100)
    assert (g.ctrl_bits, g.ctrl_flops_um2, g.ctrl_comb_um2, g.ctrl_um2) == (
        131,
        262.0,
        262.0,
        524.0,
    )
    assert g.to_dict()["ctrl"]["area"] == {"flops": 262.0, "comb": 262.0, "total": 524.0}
    assert g.ctrl_nj == pytest.approx(1.5 * 131 * 100 * 0.001 / 1000)
    assert g.area == pytest.approx(228.0 + 524.0)
    assert g.energy == pytest.approx(g.units["acc"].task_nj + g.ctrl_nj) and g.missing == []


def test_a_unit_fires_its_firings_times_its_span_a_buffer_its_producers_outputs():
    """dot 5 at W = 8: the Mac 256 fires, the serial sum one firing of 8 beats; dot 2 at W = 8:
    the depth-1 buffer loads once per output of its producer, the mul's 256; dot 6 at W = 8:
    once, the Mac giving once per job of 256 beats (DRY1-3)."""
    (g,) = estimated("dot_5", W=8, N=2048).groups
    assert (g.cost.units["mac"].fires, g.cost.units["sum"].fires) == (256, 8)
    (g,) = estimated("dot_2", W=8, N=2048).groups
    assert g.cost.links["tmp0"].fires == g.cost.units["mul"].fires == 256
    assert (
        g.cost.links["tmp0"].found == "measured" and g.cost.links["tmp0"].datapath == "LinkBuffer"
    )
    (g,) = estimated("dot_6", W=8, N=2048).groups
    assert (g.cost.units["mac"].fires, g.cost.links["out_partial"].fires) == (256, 1)


def test_l1_accesses_cost_4_e_load_per_bit_per_accelerator():
    """D154's placeholder: the 64-bit words an accelerator's streamers read and write in L1, at 4
    × e_load per bit, added to its energy (not to a group's): vecmul at W = 8 and N = 2048 6144
    words, 27.005 nJ on nangate45 (4.40 pJ a word), 1.41 pJ a word on nangate15; a packed dtype
    counts words, not elements (vecadd_i32: 3 × 32 words for 64 int32; vsum_tree_i8: 8 + 1 for 64
    int8); a link's container is not in L1 (dot 2: tmp0's 2 × 2048 words gone from dot 1's)."""
    e = estimated("vecmul", W=8, N=2048)
    (x,) = e.l1.values()
    assert (list(e.l1), x.words, x.bits, dec3(x.energy_nj)) == (["acc"], 6144, 393216, "27.005")
    assert f"{1000 * x.energy_nj / x.words:.2f}" == "4.40"
    assert e.energy == pytest.approx(e.kernel_energy + x.energy_nj)
    assert e.kernel_energy == pytest.approx(sum(g.cost.energy for g in e.groups))
    (a,) = e.to_dict()["accelerators"]  # cost.l1 under its accelerator (C§19)
    assert (a["name"], a["cost"]) == ("acc", {"l1": {"words": 6144, "bits": 393216,
                                                     "energy": x.energy_nj}})  # fmt: skip
    assert "l1" not in e.to_dict()["cost"]
    fifteen = estimate(steps("vecmul", W=8, N=2048)[-1], None, "nangate15").l1["acc"]
    assert f"{1000 * fifteen.energy_nj / fifteen.words:.2f}" == "1.41"
    assert estimated("vecadd_i32").l1_words == {"acc": 96}
    assert estimated("vsum_tree_i8").l1_words == {"acc": 9}
    one, enc, two = (estimated(r, W=8, N=2048).l1_words for r in ("dot_1", "dot_1_enc", "dot_2"))
    # per accelerator: dot 1's two, dot_1_enc's one (the same words), dot 2's link keeps tmp0 out
    assert (one, enc, two) == ({"vmul": 6144, "vsum": 2049}, {"dot": 8193}, {"dot": 4097})


def test_the_estimate_lists_what_is_not_characterized():
    e = estimated("vecadd_i32")
    assert e.missing == ["acc"] and e.groups[0].cost.units["acc"].found is None
    assert e.to_dict()["cost"]["missing"] == ["acc"]


def test_no_technology_skips_the_cost_and_both_round_trip():
    g = steps("dot_5", W=8, N=64)[-1]
    plain = estimate(g, None, None)
    assert plain.to_dict()["cost"] is None and plain.groups[0].cost is None
    assert plain.area is None and plain.energy is None and plain.l1 == {}
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


def test_a_cost_is_shown_to_3_significant_figures():
    """An area as a reader sees it (design.md, the viewer): 3 significant figures, thousands
    split."""
    assert [sig3(x) for x in (26.33877, 63186.17, 97.2592, 0.1353, 999.6, 0.0, 2723.84)] == [
        "26.3", "63 200", "97.3", "0.135", "1 000", "0", "2 720"]  # fmt: skip
    assert sig3(None) is None


def test_an_energy_is_shown_to_3_decimals():
    """Display precision only, so two close designs can be told apart (dot 4 and dot 5 at W =
    8: 26.285 and 26.339 nJ); the caveat beside it is unchanged."""
    assert [dec3(x) for x in (26.28497, 26.33877, 97.2592, 3.6596, 0.1353, 0.0, 1234.5)] == [
        "26.285", "26.339", "97.259", "3.660", "0.135", "0.000", "1 234.500"]  # fmt: skip
    assert dec3(None) is None


def test_dot_4_and_dot_5_read_apart():
    four, five = (estimated(r, W=8, N=2048).kernel_energy for r in ("dot_4", "dot_5"))
    assert (dec3(four), dec3(five)) == ("26.285", "26.339") and sig3(four) == sig3(five)
    assert heading("nangate15") == "Cost estimate (first-order) on nangate15"
    assert "about 20 %" in CAVEAT and "csr" in CAVEAT and "4 × e_load per bit" in CAVEAT


@pytest.mark.slow
def test_the_kernel_energy_is_the_experiments_formula(capsys):
    """A consistency check, not one of accuracy: vecmul, vsum_tree, dot 4 and dot 5 at W = 8
    and 32, N = 2048, on both technologies, the estimate's kernel energy (units, links and
    ctrl) against snax-forge-experiments' model, the same formula on the same inputs (its C_acc
    SYN2's window, its ctrl bits DC's); within 0.5 %, the table's rounding."""
    rows = json.loads(MODEL.read_text())["rows"]
    lines = ["| tech | recipe | W | C_acc | ctrl bits | estimate nJ | model nJ | ratio |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    bad = []
    for r in rows:
        e = estimate(steps(r["recipe"], W=r["W"], N=2048)[-1], _platform(), r["tech"])
        (g,) = e.groups
        ratio = e.kernel_energy / r["kernel_nj"]  # the L1 accesses are not the kernel's
        lines.append(f"| {r['tech']} | {r['recipe']} | {r['W']} | {e.accelerator_cycles} | "
                     f"{g.ctrl_bits} | {e.kernel_energy:.3f} | {r['kernel_nj']:.3f} | {ratio:.4f} |")  # fmt: skip
        assert (e.accelerator_cycles, g.ctrl_bits) == (r["c_acc"], r["ctrl_bits"]), r
        if abs(ratio - 1) > TOL:
            bad.append((r["tech"], r["recipe"], r["W"], ratio))
    with capsys.disabled():
        print(
            "\nkernel energy, the estimate against the experiments' formula (consistency):\n"
            + "\n".join(lines)
        )
    assert len(rows) == 16 and not bad, bad


def _platform():
    from snax_forge.design import Platform

    return Platform.load(SNAX_FORGE)


def test_a_one_stream_simd_is_not_the_two_input_point():
    """relu at int64 binds a one-stream SimdLanes (op max, k 0): the table's SimdLanes points
    are two-input ones, so it is not characterized (``char.missing``), its flops still counted;
    vecadd's two-stream add at int64 is."""
    from snax_forge.design import Platform
    from snax_forge.dfg import Graph
    from snax_forge.sandbox import Recipe, apply_recipe
    from tests.helpers import graph_dict, recipe_path

    d = graph_dict("relu")
    for c in d["containers"].values():
        c["dtype"] = "int64"
    g = apply_recipe(Recipe.load(recipe_path("relu")), Graph.from_dict(d))[-1].graph
    e = estimate(g, Platform.load(SNAX_FORGE))
    assert e.missing == ["acc"] and e.groups[0].cost.units["acc"].found is None
    assert estimated("vecadd").missing == []
