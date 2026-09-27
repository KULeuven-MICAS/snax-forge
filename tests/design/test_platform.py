"""The platform file (DP1a, D84): format, round trip, --set changes, working copies, save."""

from __future__ import annotations

import json

import pytest

from snax_forge.design import DesignError, Platform, StreamerOptions, parse_value
from snax_forge.design.platform import parse

from .helpers import ALU4, SMALL16, platform_dict

# =============================================================================
# 1. small16 and the format
# =============================================================================


def test_small16_round_trips_byte_for_byte():
    p = Platform.load(SMALL16)
    assert p.to_json() == SMALL16.read_text()
    assert Platform.from_dict(json.loads(p.to_json())) == p


def test_small16_is_alu4_without_the_accelerator():
    """Every platform value is the one alu4.json has; the streamer default is alu4's streamers
    without what the port decides (write, n_ports)."""
    p, a = Platform.load(SMALL16), json.loads(ALU4.read_text())
    comps = {c["name"]: c for c in a["components"]}
    assert p.l1.to_dict() == a["l1"] and p.l2.to_dict() == a["l2"]
    assert p.xbar == comps["xbar"]["config"] and p.dma.to_dict() == comps["dma"]["config"]
    assert p.controller.to_dict() == comps["ctl"]["config"]
    assert p.register_window == a["register_map"]["window"]
    for s in ("acc_a", "acc_b", "acc_out"):
        cfg = dict(comps[s]["config"])
        del cfg["write"], cfg["n_ports"]
        assert p.default.to_dict() == cfg
    assert (p.name, p.base, p.changes, p.entries, p.wait_mode) == ("small16", None, {}, {}, "poll")


def test_missing_keys_take_defaults_and_every_field_is_written():
    p = Platform.from_dict({"name": "bare"})
    assert p.l2 is None and p.default == StreamerOptions() and p.register_window == 32
    d = p.to_dict()
    assert list(d) == ["name", "base", "changes", "l1", "l2", "xbar", "dma", "controller",
                       "register_window", "wait_mode", "streamers"]  # fmt: skip
    assert d["l2"] is None and d["streamers"] == {"default": StreamerOptions().to_dict()}


def test_streamer_entries_round_trip_as_written():
    d = platform_dict()
    d["streamers"]["acc_out"] = {"fifo_depth": 4, "spatial_bounds": [2, 2]}
    p = Platform.from_dict(d)
    assert p.to_dict()["streamers"]["acc_out"] == {"fifo_depth": 4, "spatial_bounds": [2, 2]}
    assert p.options("acc_out").fifo_depth == 4 and p.options("acc_a").fifo_depth == 2
    assert p.spatial_bounds("acc_out") == [2, 2] and p.spatial_bounds("acc_a") is None


# =============================================================================
# 2. Loading reports every problem
# =============================================================================


def test_every_problem_is_reported_at_once():
    d = platform_dict()
    d["colour"] = "blue"  # unknown key
    d["l1"]["n_banks"] = "16"  # wrong type
    d["dma"]["dims"] = 0  # rejected by DmaConfig
    d["wait_mode"] = "sleep"
    d["streamers"]["default"]["fifo_depth"] = 0
    _, found = parse(d)
    got = {(p.code, p.where) for p in found}
    assert got == {
        ("platform.keys", "platform"),
        ("platform.keys", "l1.n_banks"),
        ("platform.values", "dma"),
        ("platform.values", "wait_mode"),
        ("platform.streamer", "streamers.default"),
    }
    with pytest.raises(DesignError) as e:
        Platform.from_dict(d)
    assert len(e.value.problems) == 5 and "5 problems" in str(e.value)


@pytest.mark.parametrize(
    "entry, code, text",
    [
        ({"fifo_dpeth": 4}, "platform.keys", "unknown keys ['fifo_dpeth']"),
        ({"fifo_depth": 0}, "platform.streamer", ">= 1"),
        ({"fifo_depth": True}, "platform.keys", "must be an int"),
        ({"spatial_bounds": []}, "platform.streamer", "non-empty list"),
        ({"spatial_bounds": [4, 0]}, "platform.streamer", "non-empty list"),
        ([4], "platform.keys", "must be an object"),
    ],
)
def test_a_bad_streamer_entry_is_named(entry, code, text):
    d = platform_dict()
    d["streamers"]["acc_a"] = entry
    _, found = parse(d)
    assert [p.code for p in found] == [code] and text in found[0].message


def test_write_and_n_ports_in_an_entry_load_for_the_connect_check():
    """Only the graph knows the port, so connect.derived reports them (test_check)."""
    d = platform_dict()
    d["streamers"]["acc_a"] = {"write": True, "n_ports": 8}
    assert Platform.from_dict(d).entries["acc_a"] == {"write": True, "n_ports": 8}


def test_a_platform_without_name_or_as_a_list_is_rejected():
    assert [p.where for p in parse({})[1]] == ["name"]
    assert [p.code for p in parse([1])[1]] == ["platform.keys"]


# =============================================================================
# 3. --set, working copies and save
# =============================================================================


def test_with_changes_gives_a_working_copy():
    base = Platform.load(SMALL16)
    w = base.with_changes([("l1.n_banks", 32), ("streamers.acc_out.fifo_depth", 4)])
    assert (w.name, w.base) == ("small16", "small16")
    assert w.changes == {"l1.n_banks": 32, "streamers.acc_out.fifo_depth": 4}
    assert w.l1.n_banks == 32 and w.entries == {"acc_out": {"fifo_depth": 4}}
    assert base.l1.n_banks == 16 and base.changes == {}  # the input is not changed
    assert base.with_changes() == base.with_changes([]) and base.with_changes().base == "small16"


def test_continuing_adds_to_the_changes_and_keeps_the_base():
    w = Platform.load(SMALL16).with_changes([("l1.n_banks", 32), ("l1.rows", 128)])
    again = Platform.from_dict(json.loads(w.to_json())).with_changes(
        [("l1.n_banks", 64), ("streamers.default.temporal_dims", 2)]
    )
    assert again.base == "small16"
    assert list(again.changes.items()) == [
        ("l1.rows", 128), ("l1.n_banks", 64), ("streamers.default.temporal_dims", 2)
    ]  # fmt: skip
    assert (again.l1.n_banks, again.l1.rows, again.default.temporal_dims) == (64, 128, 2)


def test_saved_as_renames_and_keeps_the_record():
    w = Platform.load(SMALL16).with_changes([("streamers.default.temporal_dims", 2)])
    s = w.saved_as("small16_t2")
    assert (s.name, s.base, s.changes) == ("small16_t2", "small16", w.changes)
    assert Platform.from_dict(s.to_dict()) == s
    # Continuing from a saved platform keeps the root base and the full difference.
    c = s.with_changes([("l1.rows", 128)])
    assert c.base == "small16" and list(c.changes) == ["streamers.default.temporal_dims", "l1.rows"]
    with pytest.raises(DesignError, match="not an identifier"):
        w.saved_as("small-16")


@pytest.mark.parametrize(
    "path, value, text",
    [
        ("name", "x", "written by the tools"),
        ("changes", {}, "written by the tools"),
        ("l1.nbanks", 32, "no such field"),
        ("l1.n_banks", "32", "must be an int"),
        ("l1.n_banks.x", 1, "is a value, not a section"),
        ("xbar.check_hold", 1, "must be a bool"),
        ("streamers.acc_a", 4, "streamers.<streamer>.<option>"),
        ("streamers.acc_a.write", True, "comes from the accelerator port"),
        ("streamers.acc_a.n_ports", 8, "comes from the accelerator port"),
        ("streamers.default.spatial_bounds", [4], "unknown option"),
        ("streamers.acc_a.fifo_dpeth", 4, "unknown option"),
        ("streamers.acc_a.spatial_bounds", 4, "must be a list"),
        ("streamers.acc-a.fifo_depth", 4, "not a streamer name"),
        ("l1..rows", 4, "not a path"),
    ],
)
def test_a_bad_set_is_named(path, value, text):
    with pytest.raises(DesignError) as e:
        Platform.load(SMALL16).with_changes([(path, value)])
    (p,) = e.value.problems
    assert p.code == "platform.keys" and p.where == f"platform.{path}" and text in p.message


def test_a_set_on_a_missing_l2_and_a_close_miss_say_what_to_do():
    no_l2 = Platform.from_dict({"name": "l1only"})
    with pytest.raises(DesignError) as e:
        no_l2.with_changes([("l2.size_bytes", 1 << 16)])
    assert "has no l2" in e.value.problems[0].message and e.value.problems[0].fix
    with pytest.raises(DesignError) as e:
        Platform.load(SMALL16).with_changes([("l1.nbanks", 32)])
    assert e.value.problems[0].fix == "did you mean l1.n_banks?"


def test_a_set_the_config_rejects_is_a_value_problem():
    with pytest.raises(DesignError) as e:
        Platform.load(SMALL16).with_changes([("l1.n_banks", 0)])
    assert e.value.codes == ["platform.values"]


def test_parse_value_takes_json_or_text():
    assert parse_value("32") == 32 and parse_value("true") is True
    assert parse_value("[2, 2]") == [2, 2] and parse_value('"int32"') == "int32"
    assert parse_value("int32") == "int32"
