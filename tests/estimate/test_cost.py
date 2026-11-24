"""The cost of the estimate (estimate/cost.py): the block cost model on a technology's table,
its counting, and dot 3's five parts against snax-forge-experiments' own implementation on the
clock-gated tables (``fixtures/dot3_gated.json``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.char import Table, load
from snax_forge.estimate import Estimate, estimate
from snax_forge.estimate.cost import (
    Block,
    group_cost,
    sig3,
)
from tests.helpers import steps

from .helpers import estimated

GATED = Path(__file__).parent / "fixtures" / "dot3_gated.json"
TABLE = Table("t", 10.0, {"snax_forge": "x"}, 2.0, 0.003, 0.0015, {
    "TreeAccumulate": [{"params": {"inType": "int64", "outType": "int64", "op": "add"},
                        "lanes_rule": "interpolate",
                        "points": [{"lanes": 2, "comb_um2": 100.0, "e_comb_pj": 4.0}]}]})  # fmt: skip
TREE = {"inType": "int64", "outType": "int64", "op": "add", "lanes": 2}


def test_a_groups_ctrl_is_its_bits_times_e_ctrl_every_cycle():
    """Its energy e_ctrl n_bits per cycle; its area its flops' plus an equal comb area (a
    placeholder)."""
    g = group_cost(TABLE, {"acc": Block("TreeAccumulate", TREE, 10)}, {}, {"acc": 64}, 131, 100)
    assert (g.ctrl_bits, g.ctrl_flops_um2, g.ctrl_comb_um2, g.ctrl_um2) == (
        131,
        262.0,
        262.0,
        524.0,
    )
    assert g.to_dict()["ctrl"]["area"] == {"flops": 262.0, "comb": 262.0, "total": 524.0}
    assert g.ctrl_nj == pytest.approx(0.0015 * 131 * 100 / 1000)
    assert g.area == pytest.approx(228.0 + 524.0)
    assert g.energy == pytest.approx(g.units["acc"].task_nj + g.ctrl_nj) and g.missing == []


# Rows of snax_forge/char/nangate45/ HWCOST's hand-worked designs read
READER_8_1_8 = (31739.119316, 22.49923)  # depth, loops, channels -> area µm², pJ per beat
READER_8_1_4 = (16195.409662, 11.44502)
READER_8_1_1 = (4556.313922, 3.229712)
READER_8_2_4 = (16691.765669, 11.54413)
BANK = (13522.7, 3.53819, 6.4677)  # 512 × 64: area µm², pJ per read, pJ per write


def test_dot_1_shared_prices_each_mover_once():
    """dot 1 with share_ports: three movers of 4 channels, each priced once at its own
    depth, loops and channels on its first user; vsum.out, one channel of writer0, priced per beat
    at its one channel and no area; 14 inputs where dot 1 has 19."""
    e = estimated("dot_v1_2")
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
    assert (e.xbar.inputs, estimated("dot_v1").xbar.inputs) == (14, 19)
    # its movers on the crossbar: reader0 and reader1 (8 read-only), writer0 (4 write-only)
    assert (e.xbar.read_only, e.xbar.write_only) == (8, 4)
    assert [x.to_dict() for x in e.streamer_counts] == [
        x.counts.to_dict() for x in e.streamer_costs]  # fmt: skip


def test_the_cost_round_trips():
    for e in (estimated("vecmul_v1", W=8, N=2048), estimated("mvt_v1_2")):
        d = e.to_dict()
        back = Estimate.from_dict(json.loads(json.dumps(d)))
        assert back.to_dict() == d and back.parts == e.parts


def test_no_technology_skips_the_cost_and_both_round_trip():
    g = steps("dot_v5", W=8, N=64)[-1]
    plain = estimate(g, None, None)
    assert plain.to_dict()["cost"] is None and plain.groups[0].cost is None
    assert plain.area is None and plain.energy is None and plain.l1 == {}
    costed = estimate(g, None, "nangate45")
    assert costed.to_dict()["cost"]["tech"] == "nangate45"
    assert costed.char_source == load("nangate45").source
    for e in (plain, costed):
        d = e.to_dict()
        assert Estimate.from_dict(json.loads(json.dumps(d))).to_dict() == d


def test_a_cost_is_shown_to_3_significant_figures():
    """An area as a reader sees it (design.md, the viewer): 3 significant figures, thousands
    split."""
    assert [sig3(x) for x in (26.33877, 63186.17, 97.2592, 0.1353, 999.6, 0.0, 2723.84)] == [
        "26.3", "63 200", "97.3", "0.135", "1 000", "0", "2 720"]  # fmt: skip
    assert sig3(None) is None
