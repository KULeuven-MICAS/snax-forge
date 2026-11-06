"""The per-technology tables (snax_forge/char/, C§20): every entry against the catalogue
(``hw/chisel/blocks.json``), and the lookup rules on small tables."""

from __future__ import annotations

import pytest

from snax_forge.char import DATAPATH_RENAMES, FOUND, TECHS, CharError, Lookup, Table, load, lookup
from snax_forge.gen.rtl import catalogue, check_params

I64 = {"inType": "int64", "outType": "int64"}


@pytest.mark.parametrize("tech", TECHS)
def test_every_entry_is_a_catalogue_datapath_with_complete_params(tech):
    """Each entry's datapath is the catalogue's (through ``DATAPATH_RENAMES``), its params with each point's lanes (and
    depth) a valid and complete param set there, its rule known, its points unique."""
    cat, t = catalogue(), load(tech)
    assert t.tech == tech and t.source["snax_forge"] and t.source["snax_forge_experiments"]
    assert t.flop_um2 > 0 and t.e_load_pj > t.e_hold_pj > 0 and t.counter_factor >= 1
    renamed = {table: cat_name for cat_name, table in DATAPATH_RENAMES.items()}
    assert set(renamed) <= set(t.blocks) and set(DATAPATH_RENAMES) <= set(cat)
    for table_dp, entries in t.blocks.items():
        dp = renamed.get(table_dp, table_dp)  # the table's old name, the catalogue's new one
        assert dp in cat, dp
        sig = cat[dp]["params"]
        for e in entries:
            assert e["lanes_rule"] in ("copy", "interpolate"), (dp, e["lanes_rule"])
            keys = [(p["lanes"], p.get("depth")) for p in e["points"]]
            assert len(keys) == len(set(keys)), (dp, keys)
            for p in e["points"]:
                full = {**e["params"], "lanes": p["lanes"]}
                if "depth" in p:
                    full["depth"] = p["depth"]
                assert set(full) == set(sig), (dp, sorted(full), sorted(sig))
                check_params(dp, full, sig)
                assert p["comb_um2"] > 0 and p["e_comb_pj"] > 0


@pytest.mark.parametrize("tech", TECHS)
def test_the_entries_params_are_unique_per_datapath(tech):
    for dp, entries in load(tech).blocks.items():
        ps = [e["params"] for e in entries]
        assert all(ps.count(p) == 1 for p in ps), dp


def test_an_unknown_technology_is_refused():
    with pytest.raises(CharError, match=r"\[char.tech\]"):
        load("tsmc7")


def _table(blocks) -> Table:
    return Table("t", 10.0, {}, 1.0, 0.002, 0.001, 1.0, blocks)


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
})  # fmt: skip
MAC = {"inTypes": ["int64", "int64"], "accumulate": True}


def test_a_measured_point_is_used_as_is():
    assert lookup(SMALL, "Mac", MAC, 8) == Lookup(70.0, 7.5, "measured")
    assert lookup(SMALL, "TreeAccumulate", {**I64, "op": "add"}, 4) == Lookup(40.0, 3.0, "measured")
    assert lookup(SMALL, "LinkBuffer", {"dtype": "int64"}, 8, 1) == Lookup(9.0, 0.9, "measured")


def test_the_params_lanes_and_depth_are_ignored():
    """The generator's params hold lanes and depth; the arguments say which point."""
    got = lookup(SMALL, "Mac", {**MAC, "lanes": 99}, 8)
    assert got == Lookup(70.0, 7.5, "measured")


def test_copy_is_lanes_times_the_lanes_1_point():
    assert lookup(SMALL, "Mac", MAC, 4) == Lookup(40.0, 4.0, "copy")
    assert lookup(SMALL, "Mac", MAC, 64) == Lookup(640.0, 64.0, "copy")
    assert lookup(SMALL, "LinkBuffer", {"dtype": "int64"}, 4, 2) == Lookup(12.0, 1.2, "copy")


def test_interpolate_is_linear_between_the_measured_lanes_around_it():
    got = lookup(SMALL, "TreeAccumulate", {**I64, "op": "add"}, 6)
    assert (got.comb_um2, got.e_comb_pj, got.found) == (50.0, 4.0, "interpolate")
    got = lookup(SMALL, "TreeAccumulate", {**I64, "op": "add"}, 3)
    assert (got.comb_um2, got.e_comb_pj) == (30.0, 2.5)


def test_a_linkbuffers_depth_is_measured_or_interpolated_then_copied():
    got = lookup(SMALL, "LinkBuffer", {"dtype": "int64"}, 1, 3)
    assert (got.comb_um2, got.e_comb_pj, got.found) == (5.0, 0.5, "interpolate")
    got = lookup(SMALL, "LinkBuffer", {"dtype": "int64"}, 2, 3)
    assert (got.comb_um2, got.e_comb_pj, got.found) == (10.0, 1.0, "interpolate")


def test_outside_the_measured_range_is_none():
    tree = {**I64, "op": "add"}
    assert lookup(SMALL, "TreeAccumulate", tree, 1) is None  # below the first lanes
    assert lookup(SMALL, "TreeAccumulate", tree, 16) is None  # above the last
    assert lookup(SMALL, "LinkBuffer", {"dtype": "int64"}, 1, 8) is None  # deeper than measured
    assert lookup(SMALL, "Mac", {**MAC, "accumulate": False}, 8) is None  # no entry
    assert lookup(SMALL, "LinkBuffer", {"dtype": "int32"}, 1, 1) is None
    assert lookup(SMALL, "SerialAccumulate", I64, 8) is None  # no datapath


@pytest.mark.parametrize("tech", TECHS)
def test_the_real_tables_on_chars_named_cases(tech):
    """The cases C§20 names: TreeAccumulate at lanes 64 and SerialAccumulate at lanes 1 are
    outside, int32 has no entry; a Mac at W = 4 is copied, a SerialAccumulate at 12
    interpolated, a LinkBuffer of depth 3 interpolated."""
    t = load(tech)
    i64 = {"inTypes": ["int64", "int64"], "opType": "int64", "outType": "int64"}
    mac = {**i64, "accumulate": True, "regOut": False}
    assert lookup(t, "TreeAccumulate", {**I64, "op": "add"}, 64) is None
    assert lookup(t, "SerialAccumulate", I64, 1) is None
    assert lookup(t, "SerialAccumulate", {"inType": "int32", "outType": "int32"}, 8) is None
    assert lookup(t, "Mac", mac, 32).found == "measured"
    assert lookup(t, "Mac", mac, 4).found == "copy"
    assert lookup(t, "SerialAccumulate", I64, 12).found == "interpolate"
    assert lookup(t, "LinkBuffer", {"dtype": "int64"}, 1, 3).found == "interpolate"
    assert lookup(t, "LinkBuffer", {"dtype": "int64"}, 1, 32) is None
    assert {lookup(t, "SimdLanes", {**i64, "op": op}, 8).found
            for op in ("add", "mul")} == {"measured"}  # fmt: skip
    assert set(FOUND) == {"measured", "copy", "interpolate"}
