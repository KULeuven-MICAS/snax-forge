"""The cluster view's queries (viz/cluster.py) on hand-written events with obvious answers.

1. at a cycle: a grant's element and value, on its lane and its bank; a read's data at its
   ``back``; a FIFO count held between two ``fifo`` lines; a mover's class inside a class
   run; the DMA's open copy, its beat and its read data back from L1 and L2; only that
   cycle's lines; what a level lacks
2. stepping: it skips empty cycles, goes both ways, stops at the window's ends
3. recorded runs: an accelerator of two groups (dot 1), a shared mover's port mux (dot 1
   shared), a linked group at task (dot 4), a group of two outputs (bicg_v3)
"""

import pytest

from snax_forge.viz import cluster as cv

from .helpers import L1, L2, Events


def run():
    """add_acc_a from 0 to 20: A[1] (value 2) granted on acc_a.lane0 at 10, its data back at
    12; the lane's queue 2 from 5, 0 from 9; acc_a busy 1-4, stall_xbar 5-9, idle 10-19; a
    copy of A issued at 2, its first beat at 3, done at 6; acc fires at 13."""
    ev = Events()
    ev.task(0, "acc_a", "add_acc_a", 20, {"busy": 4, "stall_xbar": 5, "stall_fifo": 0, "idle": 10})
    ev.task(0, "acc", "add_acc", 20)
    for t, mode in ((1, "busy"), (5, "stall_xbar"), (10, "idle")):
        ev.add({"t": t, "k": "class", "src": "acc_a", "task": "add_acc_a", "mode": mode})
    ev.add({"t": 1, "k": "class", "src": "acc", "task": "add_acc", "mode": "busy"})
    ev.grant(10, "acc_a.lane0", L1 + 8, b=1, task="add_acc_a", back=12)
    ev.stall(10, "acc_b.lane1", L1 + 8, "acc_a.lane0", L1 + 8, b=1, task="add_acc_b")
    for t, n in ((5, 2), (9, 0)):
        ev.add({"t": t, "k": "fifo", "src": "acc_a", "task": "add_acc_a", "lane": 0, "count": n})
    ev.copy(2, 2, "load_A", 6)
    ev.dma(3, 2, 0, "w", L1, L2, "load_A")
    ev.dma(4, 2, 1, "w", L1 + 64, L2 + 64, "load_A")
    ev.fire(13, 0, {"acc_a": 0, "acc_b": 0}, {"acc_a": [1], "acc_b": [100]})
    return ev


# =============================================================================
# 1. At a cycle
# =============================================================================


def test_a_grants_element_and_value_at_its_cycle():
    at = cv.cluster_at(run().index("stall", total=30), 10)
    g = at["movers"]["acc_a.lane0"]["grant"]
    assert (g["region"], g["elems"], g["vals"], g["bank"]) == ("A", [1], [2], 1)
    (access,) = at["banks"][1]["access"]
    assert (access["port"], access["op"], access["elems"], access["vals"], access["value"]) == (
        "acc_a.lane0", "r", [1], [2], 2)  # fmt: skip
    assert at["banks"][1]["superbank"] == 0 and at["banks"][8]["superbank"] == 1
    assert at["movers"]["acc_a.lane1"]["grant"] is None


def test_a_reads_data_at_its_back():
    idx = run().index("stall", total=30)
    for t, want in ((10, []), (11, []), (12, [10])):
        at = cv.cluster_at(idx, t)
        assert [e["t"] for e in at["movers"]["acc_a.lane0"]["back"]] == want
        assert [e["t"] for e in at["banks"][1]["back"]] == want


def test_a_fifo_count_holds_between_its_lines():
    idx = run().index("stall", total=30)
    got = [cv.cluster_at(idx, t)["movers"]["acc_a.lane0"]["fifo"] for t in range(3, 11)]
    assert got == [0, 0, 2, 2, 2, 2, 0, 0]
    assert cv.cluster_at(idx, 7)["movers"]["acc_a.lane1"]["fifo"] == 0


def test_a_movers_class_inside_a_class_run():
    idx = run().index("stall", total=30)
    assert [cv.class_at(idx, "acc_a", t) for t in (0, 1, 4, 5, 9, 10, 19, 20)] == [
        None, "busy", "busy", "stall_xbar", "stall_xbar", "idle", "idle", None]  # fmt: skip
    at = cv.cluster_at(idx, 7)
    assert at["movers"]["acc_a.lane3"]["class"] == "stall_xbar"
    assert at["movers"]["acc_a.lane3"]["task"] == "add_acc_a"
    assert at["movers"]["acc_out.lane0"]["task"] is None


def test_the_dmas_open_copy_and_its_beat():
    idx = run().index("stall", total=30)
    (cp,) = cv.cluster_at(idx, 3)["dma"]["copies"]
    assert (cp["task"], cp["dir"], cp["bytes"]) == ("load_A", "l2_to_l1", 512)
    (b,) = cp["beat"]
    assert (b["i"], b["n"], b["elems"]) == (0, 2, list(range(8)))
    banks = cv.cluster_at(idx, 3)["banks"]
    assert [(banks[j]["access"][0]["elems"], banks[j]["access"][0]["vals"]) for j in (0, 7)] == [
        ([0], [1]), ([7], [8])]  # fmt: skip
    assert cv.cluster_at(idx, 7)["dma"]["copies"] == []


def test_a_copys_read_data_comes_back_at_its_back():
    """A store's L2 burst back at 3 (the helper's back is its t) and its L1 beat at 4."""
    ev = Events()
    ev.copy(2, 7, "store_C", 6)
    ev.burst(3, 7, "r", L2, 64, "store_C")
    ev.dma(4, 7, 0, "r", L1, L2, "store_C")
    idx = ev.index("beat", total=30)
    for t, want in ((2, []), (3, [("burst", 3)]), (4, [("dma", 4)]), (5, [])):
        (cp,) = cv.cluster_at(idx, t)["dma"]["copies"]
        assert [(e["k"], e["t"]) for e in cp["back"]] == want
    assert "back" not in cv.cluster_at(ev.index("task", total=30), 3)["dma"]["copies"][0]


def test_a_cycle_lists_only_its_lines_and_its_conflicts():
    idx = run().index("stall", total=30)
    at = cv.cluster_at(idx, 10)
    assert {(e["t"], e["k"]) for e in at["events"]} == {(10, "grant"), (10, "stall"),
                                                        (10, "class")}  # fmt: skip
    assert at["interconnect"] == {"traffic": True, "conflicts": [
        {"bank": 1, "served": "acc_a.lane0", "refused": ["acc_b.lane1"]}]}  # fmt: skip
    assert at["banks"][1]["stalled"][0]["served"] == "acc_a.lane0"
    assert cv.cluster_at(idx, 11)["events"] == []
    assert cv.cluster_at(idx, 13)["acc"]["fire"]["vals"] == {"acc_a": [1], "acc_b": [100]}


def test_below_a_level_its_fields_are_absent_and_named():
    at = cv.cluster_at(run().index("beat", total=30), 10)
    assert "fifo" not in at["movers"]["acc_a.lane0"] and "stalled" not in at["banks"][1]
    assert at["needs"] == {"stall": "stall", "fifo": "stall", "stalled": "stall",
                           "conflicts": "stall"}  # fmt: skip
    at = cv.cluster_at(run().index("task", total=30), 10)
    assert "grant" not in at["movers"]["acc_a.lane0"] and at["needs"]["grant"] == "beat"
    assert at["movers"]["acc_a.lane0"]["task"] == "add_acc_a"
    assert at["filtered"] is None  # no beat lines to cover


# =============================================================================
# 2. Stepping
# =============================================================================


def test_step_skips_empty_cycles_both_ways_and_stops_at_the_ends():
    idx = run().index("stall", total=30)
    assert idx.event_cycles == [1, 3, 4, 5, 9, 10, 13]
    fwd, t = [], 0
    while (n := cv.step(idx, t, 1, "event")) != t:
        fwd.append(n)
        t = n
    assert fwd == idx.event_cycles
    back, t = [], 29
    while (n := cv.step(idx, t, -1, "event")) != t:
        back.append(n)
        t = n
    assert back == idx.event_cycles[::-1]
    assert cv.step(idx, 13, 1, "event") == 13 and cv.step(idx, 1, -1, "event") == 1


def test_step_by_boundary_visits_starts_dones_and_copies():
    idx = run().index("stall", total=30)
    assert idx.boundary_cycles == [0, 2, 6, 20]
    assert [cv.step(idx, t, 1, "boundary") for t in (0, 2, 6, 20)] == [2, 6, 20, 20]
    assert cv.step(idx, 7, -1, "boundary") == 6


def test_a_bad_step_is_refused():
    with pytest.raises(ValueError):
        cv.step(run().index("stall", total=30), 0, 2, "event")


# =============================================================================
# The recorded runs: two groups on shared movers, two sets, a group of two outputs
# =============================================================================


def _recorded(tmp_path_factory, fixture, level):
    from snax_forge.viz import api
    from tests.run.helpers import RECORDED

    from .helpers import recorded_run

    name = f"{fixture}_{level}"
    out = recorded_run(tmp_path_factory.mktemp(fixture) / name, level, RECORDED / fixture)
    return api.load_run(out, name)


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    """mvt_v1_2_2 at stall: one accelerator of two groups, mvt and vadd, on shared movers."""
    return _recorded(tmp_path_factory, "mvt_v1_2_2", "stall")


@pytest.fixture(scope="module")
def two_sets(tmp_path_factory):
    """mvt_v5_sequential at task: two accelerators, mvt and mac_x2, each its own set."""
    return _recorded(tmp_path_factory, "mvt_v5_sequential", "task")


def test_the_cluster_view_names_the_unit_that_fires(shared):
    """At an mvt firing ``acc`` is mvt, its task and fire its own, vadd's entry quiet; at a vadd
    firing the other way round."""
    idx = shared.index
    for unit, other, task in (
        ("mvt", "vadd", "tmp0_mult_mvt"),
        ("vadd", "mvt", "augassign_18_4_vadd"),
    ):
        t = next(e["t"] for e in idx.events if e["k"] == "fire" and e["src"] == unit)
        a = cv.cluster_at(idx, t)["acc"]
        assert (a["name"], a["accelerator"], a["fire"]["src"]) == (unit, "mvt_vadd", unit)
        units = {u["name"]: u for u in a["units"]}
        assert units[other]["task"] is None and units[other]["fire"] is None
        assert units[unit]["task"] == task


def test_the_schedule_splits_the_detail_by_unit(shared):
    from snax_forge.viz import schedule

    idx = shared.index
    bars = [b for lane in schedule.lanes(idx)["lanes"] for b in lane.get("bars", []) if "idle" in b]
    assert bars and not any(b["idle"] for b in bars)  # every mover is used by both groups
    d = schedule.detail(idx, 0, idx.total_cycles)
    assert {u: {e["src"] for e in es} for u, es in d["units"].items()} == {
        "mvt": {"mvt"}, "vadd": {"vadd"}}  # fmt: skip


def test_the_cluster_view_has_a_streamer_per_set(two_sets, shared):
    """Two streamers inside the cluster, mvt's busy once mvt starts and mac_x2's quiet; the
    shared set one, holding both groups."""
    idx = two_sets.index
    t = next(e["t"] for e in idx.events if e["k"] == "start" and e["src"] == "mvt")
    a = cv.cluster_at(idx, t + 1)
    assert [(s["name"], s["groups"], s["first"], s["last"], s["movers"]) for s in a["streamers"]] == [
        ("mvt", ["mvt"], 960, 986, ["vmul_x1_a", "vmul_x1_b", "vsum_x1_seed", "vsum_x1_out"]),
        ("mac_x2", ["mac_x2"], 992, 1018, ["mac_x2_a", "mac_x2_b", "mac_x2_seed", "mac_x2_out"]),
    ]  # fmt: skip
    assert [(s["busy"], s["task"]) for s in a["streamers"]] == [(True, "tmp1_mult_vmul_x1_a"),
                                                                (False, None)]  # fmt: skip
    assert {u["name"]: u["set"] for u in a["acc"]["units"]} == {"mvt": "mvt", "mac_x2": "mac_x2"}
    (one,) = cv.cluster_at(shared.index, 300)["streamers"]
    assert (one["name"], one["groups"]) == ("mvt_vadd", ["mvt", "vadd"])


def test_the_schedule_groups_lanes_accelerator_then_group(two_sets, shared):
    """As the editor's Bindings: per accelerator, per group its movers then its unit."""
    from snax_forge.viz import schedule

    def order(run):
        return [(x["accelerator"], x["group"], x["name"]) for x in schedule.lanes(run.index)["lanes"]
                if x["kind"] in ("mover", "accelerator")]  # fmt: skip

    assert order(two_sets) == [
        *(("mvt", "mvt", m) for m in ("vmul_x1_a", "vmul_x1_b", "vsum_x1_seed", "vsum_x1_out", "mvt")),
        *(("mac_x2", "mac_x2", m) for m in ("mac_x2_a", "mac_x2_b", "mac_x2_seed", "mac_x2_out",
                                            "mac_x2")),
    ]  # fmt: skip
    assert [x[:2] for x in order(shared)] == [("mvt_vadd", "mvt")] * 4 + [("mvt_vadd", "vadd")] * 4


def test_a_shared_movers_mux_serves_the_user_of_the_firing_unit(shared):
    """At an mvt firing reader0, reader1 and writer0 serve vmul_x1_a, vmul_x1_b and vsum_x1_out,
    on every lane; at a vadd firing writer0 serves vadd_out; an unshared run has no user."""
    idx = shared.index
    t = next(e["t"] for e in idx.events if e["k"] == "fire" and e["src"] == "mvt")
    at = cv.cluster_at(idx, t)
    got = tuple(
        at["movers"][f"mvt_vadd_{m}.lane0"]["user"] for m in ("reader0", "reader1", "writer0")
    )
    assert got == ("vmul_x1_a", "vmul_x1_b", "vsum_x1_out")
    assert at["movers"]["mvt_vadd_reader0.lane7"]["user"] == "vmul_x1_a"
    assert "mvt_vadd_reader0" in at["acc"]["fire"]["b"]
    t = next(e["t"] for e in idx.events if e["k"] == "fire" and e["src"] == "vadd")
    assert cv.cluster_at(idx, t)["movers"]["mvt_vadd_writer0.lane0"]["user"] == "vadd_out"
    assert "user" not in cv.cluster_at(run().index("stall", total=30), 10)["movers"]["acc_a.lane0"]


def test_a_linked_group_at_task_says_what_needs_beat(two_sets):
    """mvt (vmul_x1 → vsum_x1 linked): at task the grants need beat, with no ``filtered``."""
    idx = two_sets.index
    t = next(e["t"] for e in idx.events if e["k"] == "start" and e["src"] == "mvt")
    at = cv.cluster_at(idx, t + 1)
    assert at["filtered"] is None and at["needs"]["grant"] == "beat" and "linked" not in at
    assert at["acc"]["task"] == "tmp1_mult_mvt"


def test_a_group_of_two_outputs_shows_each_push_of_a_cycle(tmp_path_factory):
    """bicg_v4_1 at t = 400: q's push 38 and s's first push in one cycle, each with its sink."""
    idx = _recorded(tmp_path_factory, "bicg_v4_1", "beat").index
    (u,) = cv.cluster_at(idx, 400)["acc"]["units"]
    assert [(p["unit"], p["n"]) for p in u["pushes"]] == [("sum", 38), ("mac", 0)]
    assert u["push"] is u["pushes"][0]
