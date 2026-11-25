"""The per-technology tables (snax_forge/char/): every entry against the catalogue
(``hw/chisel/blocks.json``), and the lookup rules on small tables."""

from __future__ import annotations

import pytest

from snax_forge.char import TECHS, Lookup, Table, load, lookup
from snax_forge.gen.block import STREAM_BLOCKS
from snax_forge.gen.rtl import catalogue, check_params

I64 = {"inType": "int64", "outType": "int64"}


@pytest.mark.parametrize("tech", TECHS)
def test_every_entry_is_a_catalogue_datapath_with_complete_params(tech):
    """Each entry's datapath is the catalogue's (or a stream block, FifoPacker or FifoSlicer), its params with
    each point's lanes (and depth) a valid and complete param set there, its rule known, its
    points unique, a narrow fold's slope given. A block measured before it took a seed (a Mac)
    or a bank has no seed or rows key in any entry, its unseeded one-row form."""
    cat, t = catalogue(), load(tech)
    assert t.tech == tech and t.source["snax_forge"] and t.source["snax_forge_experiments"]
    assert t.flop_um2 > 0 and t.e_load_pj > t.e_ctrl_pj > 0
    for dp, entries in t.blocks.items():
        sig = STREAM_BLOCKS[dp] if dp in STREAM_BLOCKS else cat[dp]["params"]
        for key in ("seed", "rows"):
            if key in sig and not any(key in e["params"] for e in entries):
                sig = {k: v for k, v in sig.items() if k != key}
        for e in entries:
            rule = e["lanes_rule"]
            assert rule in ("copy", "interpolate", "slope", "none"), (dp, rule)
            assert (rule == "slope") == ("slope" in e), (dp, e["params"])
            keys = [(p.get("lanes"), p.get("depth")) for p in e["points"]]
            assert len(keys) == len(set(keys)), (dp, keys)
            for p in e["points"]:
                full = {**e["params"], **{k: p[k] for k in ("lanes", "depth") if k in p}}
                assert set(full) == set(sig), (dp, sorted(full), sorted(sig))
                check_params(dp, full, sig)
                # >= 0: a LinkBuffer of depth 1 is its flops alone, no comb energy when gated
                assert p["comb_um2"] > 0 and p["e_comb_pj"] >= 0


@pytest.mark.parametrize("tech", TECHS)
def test_the_entries_params_are_unique_per_datapath(tech):
    for dp, entries in load(tech).blocks.items():
        ps = [e["params"] for e in entries]
        assert all(ps.count(p) == 1 for p in ps), dp


def _table(blocks) -> Table:
    return Table("t", 10.0, {}, 1.0, 0.002, 0.001, blocks)


def _pt(lanes, comb, e, depth=None):
    return {"lanes": lanes, **({} if depth is None else {"depth": depth}), "comb_um2": comb,
            "e_comb_pj": e}  # fmt: skip


SMALL = _table({
    "Mac": [{"params": {"inTypes": ["int64", "int64"], "accumulate": True}, "lanes_rule": "copy",
             "points": [_pt(1, 10.0, 1.0), _pt(8, 70.0, 7.5)]}],
    "TreeAccumulate": [{"params": {**I64, "op": "add"}, "lanes_rule": "interpolate",
                        "points": [_pt(2, 20.0, 2.0), _pt(4, 40.0, 3.0), _pt(8, 60.0, 5.0)]}],
    "LinkBuffer": [{"params": {"dtype": "int64"}, "lanes_rule": "copy",
                    "points": [_pt(1, 1.0, 0.1, 1), _pt(8, 9.0, 0.9, 1), _pt(1, 3.0, 0.3, 2),
                               _pt(1, 7.0, 0.7, 4)]}],
    "SerialAccumulate": [{"params": {"inType": "int8", "outType": "int32"}, "lanes_rule": "slope",
                          "slope": {"comb_um2": 2.0, "e_comb_pj": 0.25},
                          "points": [_pt(8, 50.0, 4.0)]}],
    "SimdLanes": [{"params": {"inTypes": ["int8", "int8"], "opType": "int16", "outType": "int16",
                              "op": "mul", "k": None}, "lanes_rule": "copy",
                   "points": [_pt(1, 5.0, 0.5)]}],
    "FifoPacker": [{"params": {"inLanes": 1, "laneType": "int32", "words": 1}, "lanes_rule": "none",
                    "points": [{"comb_um2": 100.0, "e_comb_pj": 0.5, "flops": 33, "load_bits": 17.0}]}],
})  # fmt: skip
MAC = {"inTypes": ["int64", "int64"], "accumulate": True}


def test_outside_the_measured_range_is_none():
    tree = {**I64, "op": "add"}
    assert lookup(SMALL, "TreeAccumulate", tree, 1) is None  # below the first lanes
    assert lookup(SMALL, "TreeAccumulate", tree, 16) is None  # above the last
    assert lookup(SMALL, "LinkBuffer", {"dtype": "int64"}, 1, 8) is None  # deeper than measured
    assert lookup(SMALL, "Mac", {**MAC, "accumulate": False}, 8) is None  # no entry
    assert lookup(SMALL, "SerialAccumulate", I64, 8) is None  # no entry at this pair
    assert lookup(SMALL, "LaneAccumulate", I64, 8) is None  # no datapath


def test_a_seeded_fold_is_priced_as_its_unseeded_entry():
    tree = {**I64, "op": "add"}
    table = _table({"TreeAccumulate": [{**SMALL.blocks["TreeAccumulate"][0],
                                        "params": {**tree, "seed": False}}]})  # fmt: skip
    seeded = lookup(table, "TreeAccumulate", {**tree, "seed": True}, 4)
    assert seeded == Lookup(40.0, 3.0, "unseeded")
    got = lookup(table, "TreeAccumulate", {**tree, "seed": True}, 6)
    assert (got.comb_um2, got.found) == (50.0, "unseeded")
    seeded_only = _table({"TreeAccumulate": [{**SMALL.blocks["TreeAccumulate"][0],
                                              "params": {**tree, "seed": True, "op": "max"}}]})  # fmt: skip
    assert lookup(seeded_only, "TreeAccumulate", {**tree, "seed": True}, 4) is None  # no unseeded


def test_a_packer_by_its_lane_type_and_d_out():
    got = lookup(SMALL, "FifoPacker", {"d_out": 2, "lanes": 1, "dtype": "int32"}, 1)
    assert got == Lookup(100.0, 0.5, "measured", 17.0)
    assert lookup(SMALL, "FifoPacker", {"d_out": 2, "lanes": 2, "dtype": "int16"}, 2) is None


def test_a_mixed_pair_at_its_own_entry_in_either_order_else_its_width_pair():
    mac = {"opType": "int32", "outType": "int32", "accumulate": True}
    table = _table({"Mac": [
        {"params": {**mac, "inTypes": ["int32", "int32"]}, "lanes_rule": "copy",
         "points": [_pt(8, 80.0, 8.0)]},
        {"params": {**mac, "inTypes": ["int8", "int32"]}, "lanes_rule": "copy",
         "points": [_pt(8, 40.0, 4.0)]}]})  # fmt: skip
    for ins in (["int8", "int32"], ["int32", "int8"]):
        assert lookup(table, "Mac", {**mac, "inTypes": ins}, 8) == Lookup(40.0, 4.0, "measured")
    got = lookup(table, "Mac", {**mac, "inTypes": ["int16", "int32"]}, 8)  # not measured
    assert got == Lookup(80.0, 8.0, "measured")  # its width pair, int32 x int32
