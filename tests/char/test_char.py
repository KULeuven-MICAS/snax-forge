"""The per-technology tables (snax_forge/char/): every entry against the catalogue
(``hw/chisel/blocks.json``), and the lookup rules on small tables."""

from __future__ import annotations

import pytest

from snax_forge.char import FOUND, TECHS, CharError, Lookup, Table, load, lookup, pair_params
from snax_forge.gen.block import STREAM_BLOCKS
from snax_forge.gen.rtl import catalogue, check_params

I64 = {"inType": "int64", "outType": "int64"}


@pytest.mark.parametrize("tech", TECHS)
def test_every_entry_is_a_catalogue_datapath_with_complete_params(tech):
    """Each entry's datapath is the catalogue's (or a stream block, FifoPacker or FifoSlicer), its params with
    each point's lanes (and depth) a valid and complete param set there, its rule known, its
    points unique, a narrow fold's slope given. A block measured before it took a seed (a Mac,
    D187) has no seed key in any entry, its unseeded form."""
    cat, t = catalogue(), load(tech)
    assert t.tech == tech and t.source["snax_forge"] and t.source["snax_forge_experiments"]
    assert t.flop_um2 > 0 and t.e_load_pj > t.e_ctrl_pj > 0
    for dp, entries in t.blocks.items():
        sig = STREAM_BLOCKS[dp] if dp in STREAM_BLOCKS else cat[dp]["params"]
        if "seed" in sig and not any("seed" in e["params"] for e in entries):
            sig = {k: v for k, v in sig.items() if k != "seed"}
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


def test_an_unknown_technology_is_refused():
    with pytest.raises(CharError, match=r"\[char.tech\]"):
        load("tsmc7")


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
    assert lookup(SMALL, "SerialAccumulate", I64, 8) is None  # no entry at this pair
    assert lookup(SMALL, "LaneAccumulate", I64, 8) is None  # no datapath


def test_copy_starts_from_the_pairs_smallest_lanes():
    """A pair measured at W0 = 8 only copies from there (lanes / W0 of its point)."""
    t = _table({"Mac": [{"params": MAC, "lanes_rule": "copy", "points": [_pt(8, 80.0, 8.0)]}]})
    assert lookup(t, "Mac", MAC, 2) == Lookup(20.0, 2.0, "copy")
    assert lookup(t, "Mac", MAC, 32) == Lookup(320.0, 32.0, "copy")


def test_a_narrow_fold_adds_its_slope_per_lane_from_the_nearest_point():
    tree = {"inType": "int8", "outType": "int32"}
    assert lookup(SMALL, "SerialAccumulate", tree, 8) == Lookup(50.0, 4.0, "measured")
    assert lookup(SMALL, "SerialAccumulate", tree, 32) == Lookup(98.0, 10.0, "slope")
    assert lookup(SMALL, "SerialAccumulate", tree, 2) == Lookup(38.0, 2.5, "slope")
    assert lookup(SMALL, "SerialAccumulate", {**tree, "outType": "uint32"}, 8).found == "measured"


def test_a_narrow_linkbuffer_is_the_int64_one_by_its_width():
    assert lookup(SMALL, "LinkBuffer", {"dtype": "int32"}, 8, 1) == Lookup(4.5, 0.45, "width")
    got = lookup(SMALL, "LinkBuffer", {"dtype": "uint16"}, 1, 3)
    assert (got.comb_um2, got.e_comb_pj, got.found) == (1.25, 0.125, "width")


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


def test_a_seeded_mac_is_priced_as_a_table_measured_before_the_seed():
    """D187: no entry of the block has a seed key (measured before it took one), so a seeded
    unit is priced at the entry without it, and an unseeded one (the catalogue's seed false) is
    that entry, measured."""
    assert lookup(SMALL, "Mac", {**MAC, "seed": True}, 8) == Lookup(70.0, 7.5, "unseeded")
    assert lookup(SMALL, "Mac", {**MAC, "seed": False}, 8) == Lookup(70.0, 7.5, "measured")
    assert lookup(SMALL, "Mac", MAC, 8) == Lookup(70.0, 7.5, "measured")
    tree = {**I64, "op": "add"}
    assert lookup(SMALL, "TreeAccumulate", {**tree, "seed": True}, 4) == Lookup(40.0, 3.0,
                                                                              "unseeded")  # fmt: skip


def test_simdlanes_is_priced_at_its_widest_input_and_its_op_or_output():
    """D176: inputs at the widest, the output at most the op's width, unsigned as signed, the
    constants left None ignored."""
    base = {"op": "mul", "k": None}
    for ins, op, out in (
        (["int8", "int8"], "int16", "int16"),
        (["int8", "uint8"], "int16", "int64"),
        (["int8", "int8"], "int32", "int16"),
    ):
        got = lookup(SMALL, "SimdLanes", {**base, "inTypes": ins, "opType": op, "outType": out}, 4)
        assert got == Lookup(20.0, 2.0, "copy"), (ins, op, out)
    assert pair_params(
        "SimdLanes",
        {"inTypes": ["int8", "int32"], "opType": "int32", "outType": "int64", "lanes": 4},
    )["inTypes"] == ["int32", "int32"]


def test_a_packer_by_its_lane_type_and_d_out():
    got = lookup(SMALL, "FifoPacker", {"d_out": 2, "lanes": 1, "dtype": "int32"}, 1)
    assert got == Lookup(100.0, 0.5, "measured", 17.0)
    assert lookup(SMALL, "FifoPacker", {"d_out": 2, "lanes": 2, "dtype": "int16"}, 2) is None


@pytest.mark.parametrize("tech", TECHS)
def test_a_slicer_by_its_lane_type_lanes_and_d_in(tech):
    """A sliced input's FifoSlicer, one word in: one measured point per lane type and lanes (an
    unsigned type as the signed one), with the bits that load a pop, 64 / d_in + log2 d_in; at one
    d_in the same block whatever its lanes. Two words in are not in the table."""
    t = load(tech)
    got = {}
    for dtype, lanes, d_in, bits in (("int32", 1, 2, 33), ("int16", 1, 4, 18), ("int16", 2, 2, 33),
                                     ("int8", 1, 8, 11), ("uint8", 2, 4, 18), ("int8", 4, 2, 33)):  # fmt: skip
        got[dtype, lanes] = lookup(t, "FifoSlicer", {"d_in": d_in, "lanes": lanes, "dtype": dtype},
                                   lanes)  # fmt: skip
        assert (got[dtype, lanes].found, got[dtype, lanes].load_bits) == ("measured", bits)
    assert got["int32", 1] == got["int16", 2] == got["int8", 4]
    assert got["int16", 1] == got["uint8", 2]
    assert lookup(t, "FifoSlicer", {"d_in": 4, "lanes": 1, "dtype": "int32"}, 1) is None


@pytest.mark.parametrize("tech", TECHS)
def test_the_real_tables_on_chars_named_cases(tech):
    """The cases the cost model names: the int64 TreeAccumulate at lanes 64 and SerialAccumulate at
    lanes 1 are outside; a Mac at W = 4 is copied, a SerialAccumulate at 12 interpolated, a
    LinkBuffer of depth 3 interpolated; every narrow pair is there (D176)."""
    t = load(tech)
    i64 = {"inTypes": ["int64", "int64"], "opType": "int64", "outType": "int64"}
    mac = {**i64, "accumulate": True, "regOut": False}
    assert lookup(t, "TreeAccumulate", {**I64, "op": "add", "seed": False}, 64) is None
    assert lookup(t, "SerialAccumulate", {**I64, "seed": False}, 1) is None
    assert (
        lookup(
            t, "SerialAccumulate", {"inType": "int32", "outType": "int32", "seed": False}, 8
        ).found
        == "measured"
    )
    assert (
        lookup(
            t, "SerialAccumulate", {"inType": "int8", "outType": "int32", "seed": False}, 12
        ).found
        == "slope"
    )
    for i in (8, 16, 32, 64):
        for o in (8, 16, 32, 64):
            if o < i:
                continue
            ty = {"inTypes": [f"int{i}"] * 2, "opType": f"int{o}", "outType": f"int{o}"}
            assert lookup(t, "Mac", {**ty, "accumulate": True, "regOut": False}, 8) is not None
            assert lookup(t, "TreeAccumulate", {"inType": f"int{i}", "outType": f"int{o}", "op": "add",
                                                "seed": False}, 8) is not None  # fmt: skip
            for op in ("add", "mul"):
                assert lookup(t, "SimdLanes", {**ty, "op": op, "streams": 2, "kCsr": False,
                                               "k1Csr": False, "k2Csr": False}, 1) is not None  # fmt: skip
    assert lookup(t, "LinkBuffer", {"dtype": "int8"}, 1, 4).found == "width"
    assert (
        lookup(t, "FifoPacker", {"d_out": 8, "lanes": 1, "dtype": "int8"}, 1).load_bits is not None
    )
    assert lookup(t, "Mac", mac, 32).found == "measured"
    assert lookup(t, "Mac", mac, 4).found == "copy"
    assert lookup(t, "SerialAccumulate", {**I64, "seed": False}, 12).found == "interpolate"
    assert lookup(t, "LinkBuffer", {"dtype": "int64"}, 1, 3).found == "interpolate"
    assert lookup(t, "LinkBuffer", {"dtype": "int64"}, 1, 32) is None
    simd = {**i64, "streams": 2, "kCsr": False, "k1Csr": False, "k2Csr": False}
    assert {lookup(t, "SimdLanes", {**simd, "op": op}, 8).found
            for op in ("add", "mul")} == {"measured"}  # fmt: skip
    tree = {**I64, "op": "add"}
    seeded = lookup(t, "TreeAccumulate", {**tree, "seed": True}, 8)
    plain = lookup(t, "TreeAccumulate", {**tree, "seed": False}, 8)
    assert (seeded.comb_um2, seeded.e_comb_pj, seeded.found) == (
        plain.comb_um2, plain.e_comb_pj, "unseeded"
    )  # fmt: skip
    assert set(FOUND) == {"measured", "copy", "interpolate", "slope", "width", "unseeded"}
