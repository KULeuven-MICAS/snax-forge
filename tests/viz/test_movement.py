"""Data movement (movement.py, api.py): journeys, residency, patterns and conflicts as queries
on the run index.

1. the journey of one element: its accesses, its firings with their operands' values, the push
   and where the results went; a push that folds many firings; a repeated read; two runs of one
   component, crossed through the task pairing
2. residency: each element's arrival, uses and departure, and the windows of them
3. patterns: the configured nest gives every lane's grant addresses; held cycles are the
   profile's stalls
4. conflicts placed on the layout, in a window, with counts per word, bank and port
5. what a run below a level, or a filtered one, gets
6. a group of two outputs on a forked input (bicg_v3): each firing's push per output, a packed
   word's results

Most cases run on the recorded vecadd run (tests/run/fixtures/vecadd, converted by
helpers.recorded_run), the rest on hand-written events (helpers.Events).
"""

from __future__ import annotations

import pytest

from snax_forge.viz import api, movement

from .helpers import L1, Events, recorded_run, run_dir

B_PIN = [("B.l1.base", L1 + 576)]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("moves")
    out = {lv: api.load_run(recorded_run(base / f"rec_{lv}", lv), f"rec_{lv}")
           for lv in ("task", "beat", "stall")}  # fmt: skip
    out["B_pinned"] = api.load_run(run_dir(base / "pinned", "vecadd_v1", "beat", memory_sets=B_PIN))
    return out


def events_of(rv, k, **match):
    return [e for e in rv.events if e["k"] == k and all(e.get(f) == v for f, v in match.items())]


# =============================================================================
# 1. Journeys
# =============================================================================


def test_a_held_back_read_names_what_its_bank_served(runs):
    j = api.journey_view(runs["stall"], "B", 0)
    (held,) = [h for h in j["hops"] if h["act"] == "held back"]
    (stall,) = events_of(runs["stall"], "stall", src="acc_b.lane0")
    assert (held["t"], held["by"], held["bank"], held["row"]) == (stall["t"], "acc_b.lane0", 0, 2)
    assert held["served"] == {"port": "acc_a.lane0", "addr": L1, "row": 0, "element": "A[0]"}


def test_a_read_handed_out_twice_feeds_two_firings():
    """acc_a's beat 0 is fired twice (a repeat), with acc_b's beats 0 and 1."""
    ev = Events()
    ev.grant(1, "acc_a.lane0", L1, b=0, task="add_acc_a")
    ev.grant(1, "acc_b.lane0", L1 + 512, b=0, task="add_acc_b")
    ev.grant(2, "acc_b.lane0", L1 + 520, b=1, task="add_acc_b")
    ev.fire(3, 0, {"acc_a": 0, "acc_b": 0}, {"acc_a": [1], "acc_b": [100]})
    ev.fire(4, 1, {"acc_a": 0, "acc_b": 1}, {"acc_a": [1], "acc_b": [101]})
    j = movement.journey(ev.index(), "A", 0)
    assert [f["n"] for f in j["firings"]] == [0, 1]
    assert [f["inputs"]["acc_b"] for f in j["firings"]] == [["B[0]"], ["B[1]"]]
    assert [f["operands"]["acc_b"]["vals"] for f in j["firings"]] == [[100], [101]]
    assert all(f["push"] is None for f in j["firings"])  # no push traced


def test_an_index_outside_the_region_is_refused(runs):
    with pytest.raises(ValueError):
        api.journey_view(runs["stall"], "A", 64)
    with pytest.raises(ValueError):
        api.journey_view(runs["stall"], "Q", 0)
    assert api.journey_view(runs["stall"], "A", [5])["element"] == "A[5]"


# =============================================================================
# 2. Residency
# =============================================================================


# =============================================================================
# 3. Patterns
# =============================================================================


def expand(base, nest):
    out = [0]
    for bound, stride in nest:
        out = [o + j * stride for j in range(bound) for o in out]
    return [base + o for o in out]


@pytest.mark.parametrize("level", ["beat", "stall"])
def test_every_pattern_is_its_lanes_grants(runs, level):
    rv = runs[level]
    pats = movement.patterns(rv.index)
    lanes = [p for p in pats if p["port"] != "dma"]
    assert len(lanes) == sum(m.lanes for m in rv.cluster.accelerator.movers)
    for p in lanes:
        got = [e["addr"] for e in events_of(rv, "grant", src=p["port"], task=p["task"])]
        assert expand(p["addr"]["base"], p["addr"]["nest"]) == got, p["port"]
        assert p["beats"] == p["addr"]["addresses"] == len(got)
        assert p["ideal_last"] == p["first"] + p["beats"] - 1
    dmas = [p for p in pats if p["port"] == "dma"]
    assert [p["task"] for p in dmas] == ["load_A", "load_B", "store_C"]
    assert all(p["beats"] == p["size"] // rv.cluster.dma_beat_bytes for p in dmas)


# =============================================================================
# 4. Conflicts
# =============================================================================


# =============================================================================
# 5. Below a level, and filtered
# =============================================================================


# =============================================================================
# 6. A group of two outputs (tests/run/fixtures/bicg_v4_1)
# =============================================================================


@pytest.fixture(scope="module")
def fork(tmp_path_factory):
    from tests.run.helpers import RECORDED

    base = tmp_path_factory.mktemp("fork")
    return api.load_run(recorded_run(base / "rec", "beat", RECORDED / "bicg_v4_1"), "rec")


def test_a_packed_words_result_comes_from_its_own_firings(fork):
    """q[3] is slice 1 of sum_out's word 1 (two int32 a word, D174), carried as the seed of the
    second strip (D192): produced by firings 12 to 15 of each strip only; s[1] by the 40 firings
    of its bank row."""
    j = api.journey_view(fork, "q", 3)
    assert [h["by"] for h in j["hops"] if h["k"] == "grant"] == [
        "sum_seed.lane0",
        "sum_out.lane0",
    ] * 2
    assert [(f["role"], f["n"]) for f in j["firings"]] == [
        ("produced", n) for n in (12, 13, 14, 15, 172, 173, 174, 175)]  # fmt: skip
    assert {x for f in j["firings"] for x in f["outputs"]["sum_out"]} == {"q[3]"}
    assert [f["n"] for f in api.journey_view(fork, "s", 1)["firings"]] == list(range(40))
