"""The cost of the estimate (estimate/cost.py, C§20): CHAR1's model on a technology's table,
its counting, and the kernel energy against snax-forge-experiments' model of SYN2's four designs
(``fixtures/char_model.json``, util/char_model.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.char import TECHS, CharError, Table, load
from snax_forge.design.platform import Platform
from snax_forge.estimate import Estimate, estimate
from snax_forge.estimate.cost import CAVEAT, Block, block_cost, dec3, group_cost, heading, sig3
from tests.helpers import steps

from .helpers import SNAX_FORGE, copies, estimated

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


# Rows of snax_forge/char/nangate45/ HWCOST's hand-worked designs read (D167)
READER_8_1_8 = (38525.045548, 39.65753)  # depth, loops, channels -> area µm², pJ per beat
READER_8_1_4 = (19635.587779, 20.02126)
READER_8_1_1 = (5469.225952, 5.315563)
READER_8_2_4 = (20155.085786, 20.29734)
BANK = (13522.7, 3.53819, 6.4677)  # 512 × 64: area µm², pJ per read, pJ per write


def test_vecmul_at_w8_by_part():
    """The hand-worked design of the HWCOST notes: vecmul at W = 8, N = 2048, three ports of 8
    channels and one loop, 256 beats each, 6144 words; the crossbar at 24 + 2 + 1 = 27 inputs
    on 32 banks, its table area linear in inputs (t = 3/8 between 24 and 32), its 16 readers'
    channels read-only and 8 writer's write-only (X71: × (1 − 0.396496 · 16/27 − 0.41077 · 8/27)),
    every access at a full crossbar's price (32 inputs on 32 banks, a grid point); 32 banks,
    4096 reads and 2048 writes."""
    e = estimated("vecmul", W=8, N=2048)
    assert [(s.counts.name, s.counts.write, s.counts.depth, s.counts.loops, s.counts.channels,
             s.counts.words, s.counts.beats) for s in e.streamer_costs] == [
        ("acc_a", False, 8, 1, 8, 2048, 256), ("acc_b", False, 8, 1, 8, 2048, 256),
        ("acc_out", True, 8, 1, 8, 2048, 256)]  # fmt: skip
    p = e.parts
    assert p["streamers"].area_um2 == pytest.approx(3 * READER_8_1_8[0])
    assert p["streamers"].energy_nj == pytest.approx(3 * 256 * READER_8_1_8[1] / 1000)
    x = e.xbar
    assert (x.inputs, x.banks, x.accesses, x.extrapolated) == (27, 32, 6144, ())
    assert (x.read_only, x.write_only, x.form) == (16, 8, "factor")
    table = 124078.629252 + 3 / 8 * (164650.280089 - 124078.629252)
    assert x.area_um2 == pytest.approx(table * (1 - 0.396496 * 16 / 27 - 0.41077 * 8 / 27))
    assert x.e_access_pj == 2.739443  # xbar.csv's 32 inputs on 32 banks
    m = e.memory
    assert (m.n, m.depth, m.width, m.reads, m.writes) == (32, 512, 64, 4096, 2048)
    assert m.area_um2 == pytest.approx(32 * BANK[0])
    assert m.energy_nj == pytest.approx((4096 * BANK[1] + 2048 * BANK[2]) / 1000)
    shown = {k: (sig3(v.area_um2), dec3(v.energy_nj)) for k, v in p.items()}
    # control 362 µm², 0.066 nJ: 34 bits, CTRL1's 66 less its 32-bit copy of JOBS (D168)
    assert shown == {"datapath": ("55 300", "23.137"), "control": ("362", "0.066"),
                     "streamers": ("116 000", "30.457"), "crossbar": ("89 600", "16.831"),
                     "memory": ("433 000", "27.738")}  # fmt: skip
    assert (sig3(e.area), sig3(e.area_design), dec3(e.energy)) == ("694 000", "261 000", "98.229")
    assert e.energy == pytest.approx(sum(v.energy_nj for v in p.values()))
    assert e.area_design == pytest.approx(e.area - m.area_um2)
    assert e.kernel_energy == pytest.approx(p["datapath"].energy_nj + p["control"].energy_nj)
    (acc,) = e.l1.values()  # its words through the crossbar and the banks
    assert (list(e.l1), acc.words, acc.reads, acc.writes) == (["acc"], 6144, 4096, 2048)
    assert acc.energy_nj == pytest.approx(x.energy_nj + m.energy_nj)


def test_dot_4_by_part():
    """dot 4 at W = 4, N = 512: two readers of 4 channels (128 beats), a one-lane writer of one
    word; the crossbar at 4 + 4 + 1 + 3 = 12 inputs, its table area cost_tables.md's example 3
    (61,450.9 µm²) with 8 read-only and 1 write-only (X71), its accesses at a full crossbar's
    price."""
    e = estimated("dot_4")
    s = {c.counts.name: c for c in e.streamer_costs}
    assert [(k, c.counts.channels, c.counts.words, c.counts.beats) for k, c in s.items()] == [
        ("mac_a", 4, 512, 128), ("mac_b", 4, 512, 128), ("sum_out", 1, 1, 1)]  # fmt: skip
    assert e.parts["streamers"].area_um2 == pytest.approx(2 * READER_8_1_4[0] + READER_8_1_1[0])
    assert e.parts["streamers"].energy_nj == pytest.approx(
        (2 * 128 * READER_8_1_4[1] + READER_8_1_1[1]) / 1000)  # fmt: skip
    assert (e.xbar.read_only, e.xbar.write_only) == (8, 1)
    assert e.xbar.area_um2 == pytest.approx(
        61450.9 * (1 - 0.396496 * 8 / 12 - 0.41077 / 12), abs=0.1
    )
    assert e.xbar.e_access_pj == 2.739443
    assert (e.memory.reads, e.memory.writes) == (1024, 1)


def test_mvt_1_dot2_counts_a_shared_set_once_and_its_words_twice():
    """mvt 1 with dot 2 (D165): one vmul → link → vsum group run for both halves, so its
    streamers are priced once and move both halves' words (A twice, 2 × 40 × 40 words over 4
    channels and 2 loops); the vadd's set is its own accelerator; 24 inputs, a grid value, 16
    of them read-only (both sets' readers) and 5 write-only (vsum's one, vadd's four)."""
    e = estimated("mvt_1_dot2")
    s = {c.counts.name: c for c in e.streamer_costs}
    assert list(s) == ["vmul_x1_a", "vmul_x1_b", "vsum_x1_out", "vadd_a", "vadd_b", "vadd_out"]
    a = s["vmul_x1_a"]
    assert (a.counts.loops, a.counts.channels, a.counts.words, a.counts.beats) == (2, 4, 3200, 800)
    assert (a.area_um2, a.e_beat_pj) == READER_8_2_4
    assert (e.xbar.inputs, e.xbar.read_only, e.xbar.write_only) == (24, 16, 5)
    assert e.xbar.area_um2 == pytest.approx(
        124078.629252 * (1 - 0.396496 * 16 / 24 - 0.41077 * 5 / 24)
    )
    assert e.xbar.e_access_pj == 2.739443
    assert sum(x.energy_nj for x in e.l1.values()) == pytest.approx(
        e.xbar.energy_nj + e.memory.energy_nj)  # fmt: skip


def test_dot_1_shared_prices_each_mover_once():
    """dot 1 with share_ports (D169): three movers of 4 channels, each priced once at its own
    depth, loops and channels on its first user; vsum.out, one channel of writer0, priced per beat
    at its one channel and no area; 15 inputs where dot 1 has 20."""
    e = estimated("dot_1_shared")
    s = {c.counts.name: c for c in e.streamer_costs}
    assert [
        (k, c.counts.mover, c.counts.channels, c.counts.mover_channels, c.counts.area)
        for k, c in s.items()
    ] == [
        ("vmul_a", "reader0", 4, 4, True),
        ("vmul_b", "reader1", 4, 4, True),
        ("vmul_out", "writer0", 4, 4, True),
        ("vsum_a", "reader0", 4, 4, False),
        ("vsum_out", "writer0", 1, 4, False),
    ]
    assert e.parts["streamers"].area_um2 == pytest.approx(3 * READER_8_1_4[0])
    assert (s["vsum_out"].area_um2, s["vsum_out"].e_beat_pj) == (0.0, READER_8_1_1[1])
    assert (e.xbar.inputs, estimated("dot_1").xbar.inputs) == (15, 20)
    # its movers on the crossbar: reader0 and reader1 (8 read-only), writer0 (4 write-only)
    assert (e.xbar.read_only, e.xbar.write_only) == (8, 4)
    assert [x.to_dict() for x in e.streamer_counts] == [
        x.counts.to_dict() for x in e.streamer_costs]  # fmt: skip


def test_words_are_counted_packed_and_a_link_keeps_its_container_out():
    """A packed dtype counts words, not elements (vecadd_i32: 3 × 32 words for 64 int32;
    vsum_tree_i8: 8 + 1 for 64 int8); a link's container is not in L1 (dot 2: tmp0's 2 × 2048
    words gone from dot 1's)."""
    assert estimated("vecadd_i32").l1_words == {"acc": 96}
    assert estimated("vsum_tree_i8").l1_words == {"acc": 9}
    one, enc, two = (estimated(r, W=8, N=2048).l1_words for r in ("dot_1", "dot_1_enc", "dot_2"))
    assert (one, enc, two) == ({"vmul": 6144, "vsum": 2049}, {"dot": 8193}, {"dot": 4097})


def test_not_costed_without_a_platform_or_the_tables():
    """No platform: no streamer, crossbar or bank counts; the datapath and control parts stay,
    the others are None and named. (nangate15's case, no HWCOST tables, dropped: the
    technology is no longer offered, D167.)"""
    bare = estimate(steps("vecmul", W=8, N=2048)[-1], None, "nangate45")
    assert bare.not_costed == ["streamers, crossbar and memory: no platform"]
    assert bare.parts["memory"] == bare.parts["crossbar"] == bare.parts["streamers"]
    assert bare.parts["memory"].area_um2 is None and bare.l1 == {}
    assert bare.energy == pytest.approx(bare.kernel_energy)
    with pytest.raises(CharError, match="no longer offered"):
        estimate(steps("vecmul", W=8, N=2048)[-1], Platform.load(SNAX_FORGE), "nangate15")


def test_a_point_outside_the_grid_is_extrapolated_and_named():
    """vecmul at W = 32: 32 channels per port, past the streamer table's 16, and 96 + 3
    crossbar inputs, past its 64 (the check refuses the design; the estimate still prices it)."""
    e = estimated("vecmul", W=32, N=2048)
    assert e.extrapolated == ["acc_a", "acc_b", "acc_out", "crossbar"]
    assert all(s.extrapolated == ("channels",) for s in e.streamer_costs)
    assert (e.xbar.inputs, e.xbar.extrapolated) == (99, ("inputs",))
    assert e.to_dict()["cost"]["extrapolated"] == e.extrapolated


def test_an_access_costs_the_same_whatever_the_inputs():
    """One price per access for a bank count (D167): the crossbar's energy follows its accesses
    alone, so dot 5 linked (20 inputs) costs less than unlinked (36 inputs), by its 16 fewer
    accesses, where pricing at each design's own inputs had it the other way."""
    s = steps("dot_5", W=8, N=2048)
    two, one = (estimate(s[k], Platform.load(SNAX_FORGE)) for k in (5, 6))
    assert (two.xbar.inputs, one.xbar.inputs) == (36, 20)
    assert two.xbar.e_access_pj == one.xbar.e_access_pj == 2.739443
    assert (two.xbar.accesses, one.xbar.accesses) == (4113, 4097)
    assert (dec3(two.energy), dec3(one.energy)) == ("72.455", "72.243")  # ctrl without copies, D168


def test_the_cost_round_trips():
    for e in (estimated("vecmul", W=8, N=2048), estimated("mvt_1_dot2")):
        d = e.to_dict()
        back = Estimate.from_dict(json.loads(json.dumps(d)))
        assert back.to_dict() == d and back.parts == e.parts


def test_the_estimate_lists_what_is_not_characterized():
    e = estimated("vecadd_i32")
    assert e.missing == ["acc"] and e.groups[0].cost.units["acc"].found is None
    assert e.to_dict()["cost"]["missing"] == ["acc"]


def test_no_technology_skips_the_cost_and_both_round_trip():
    g = steps("dot_5", W=8, N=64)[-1]
    plain = estimate(g, None, None)
    assert plain.to_dict()["cost"] is None and plain.groups[0].cost is None
    assert plain.area is None and plain.energy is None and plain.l1 == {}
    costed = estimate(g, None, "nangate45")
    assert costed.to_dict()["cost"]["tech"] == "nangate45"
    assert costed.char_source == load("nangate45").source
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
    8: 26.160 and 26.210 nJ); the caveat beside it is unchanged."""
    assert [dec3(x) for x in (26.28497, 26.33877, 97.2592, 3.6596, 0.1353, 0.0, 1234.5)] == [
        "26.285", "26.339", "97.259", "3.660", "0.135", "0.000", "1 234.500"]  # fmt: skip
    assert dec3(None) is None


def test_dot_4_and_dot_5_read_apart():
    four, five = (estimated(r, W=8, N=2048).kernel_energy for r in ("dot_4", "dot_5"))
    # 26.160 and 26.210: their control units without CTRL1's 64 bits of copies (D168)
    assert (dec3(four), dec3(five)) == ("26.160", "26.210") and sig3(four) == sig3(five)
    assert heading("nangate45") == "Cost estimate (first-order) on nangate45"
    assert "about 30 %" in CAVEAT and "csr" in CAVEAT and "4 × e_load" not in CAVEAT


@pytest.mark.slow
def test_the_kernel_energy_is_the_experiments_formula(capsys):
    """A consistency check, not one of accuracy: vecmul, vsum_tree, dot 4 and dot 5 at W = 8
    and 32, N = 2048, on nangate45 (its nangate15 rows skipped: the technology is no longer
    offered, D167), the estimate's kernel energy (units, links and ctrl) against
    snax-forge-experiments' model, the same formula on the same inputs (its C_acc SYN2's
    window, its ctrl bits DC's); within 0.5 %, the table's rounding. DC's ctrl bits are CTRL1's,
    with the copies of JOBS and each JOB_BEATS CTRL2 removed (D168): the estimate's bits plus
    those (``copies``), and its kernel energy plus their control energy, are compared, until
    CTRL2's R5 refreshes the fixture."""
    rows = [r for r in json.loads(MODEL.read_text())["rows"] if r["tech"] in TECHS]
    lines = ["| tech | recipe | W | C_acc | ctrl bits | estimate nJ | model nJ | ratio |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    bad = []
    for r in rows:
        e = estimate(steps(r["recipe"], W=r["W"], N=2048)[-1], _platform(), r["tech"])
        (g,) = e.groups
        extra = copies(r["recipe"])
        was = e.kernel_energy + g.cost.ctrl_nj * extra / g.ctrl_bits  # with CTRL1's copies
        ratio = was / r["kernel_nj"]  # the L1 accesses are not the kernel's
        lines.append(f"| {r['tech']} | {r['recipe']} | {r['W']} | {e.accelerator_cycles} | "
                     f"{g.ctrl_bits} + {extra} | {was:.3f} | {r['kernel_nj']:.3f} | {ratio:.4f} |")  # fmt: skip
        assert (e.accelerator_cycles, g.ctrl_bits + extra) == (r["c_acc"], r["ctrl_bits"]), r
        if abs(ratio - 1) > TOL:
            bad.append((r["tech"], r["recipe"], r["W"], ratio))
    with capsys.disabled():
        print(
            "\nkernel energy, the estimate against the experiments' formula (consistency):\n"
            + "\n".join(lines)
        )
    assert len(rows) == 8 and not bad, bad


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


def test_the_clusters_dot_3_crossbar_in_both_forms():
    """dot 3 at W = 16, N = 2048, the design CSYN1 synthesized in the whole cluster: two readers of
    16 channels and a one-channel writer, 36 inputs on 32 banks (X71's check). The factor form
    gives 119,308 µm² (the cluster's sparse interconnect is 107,685), the crosspoint option
    115,763; the energy per access is the same."""
    e = estimated("dot_3", W=16, N=2048)
    f, c = e.xbar, e.xbar_in("crosspoint")
    assert (f.inputs, f.read_only, f.write_only, f.form, c.form) == (
        36,
        32,
        1,
        "factor",
        "crosspoint",
    )
    assert (round(f.area_um2), round(c.area_um2)) == (119308, 115763)
    assert f.e_access_pj == c.e_access_pj and f.energy_nj == c.energy_nj
    assert e.parts["crossbar"].area_um2 == f.area_um2
