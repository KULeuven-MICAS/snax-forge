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

Most cases run on the recorded vecadd run (tests/run/fixtures/vecadd, converted by
helpers.recorded_run), the rest on hand-written events (helpers.Events). Not ported from the
SNAX-MODEL tests (git ``cad6619``): ``test_fit_nest`` and ``test_fmul_fits_its_two_loops``
(the nest is the configured one, nothing is fitted; no fmul run), ``vecadd_conflict`` and
``vecadd_tiled`` (SNAX-MODEL scenarios), ``test_the_b_pinned_run_has_no_conflicts`` (the
synthetic events stall on a reader's own bank, so a stall-level B-pinned run does not convert;
the B-pinned journey is checked at beat), the memory tab's marks (test_memory.py now).
"""

from __future__ import annotations

import pytest

from snax_forge.report.run import build_movement
from snax_forge.viz import api, movement

from .helpers import L1, Events, recorded_run, run_dir, tasks, two_runs

B_PIN = [("B.l1.base", L1 + 576)]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("moves")
    out = {lv: api.load_run(recorded_run(base / f"rec_{lv}", lv), f"rec_{lv}")
           for lv in ("task", "beat", "stall")}  # fmt: skip
    out["B_pinned"] = api.load_run(run_dir(base / "pinned", "vecadd", "beat", memory_sets=B_PIN))
    return out


def events_of(rv, k, **match):
    return [e for e in rv.events if e["k"] == k and all(e.get(f) == v for f, v in match.items())]


# =============================================================================
# 1. Journeys
# =============================================================================


def test_the_journey_of_a5(runs):
    """A[5]: from L2 by the load's burst, into L1 by its DMA beat, read by acc_a.lane1, fired
    with B[5], pushed as C[5] = A[5] + B[5], written by acc_out.lane1, stored by a DMA beat and
    its burst."""
    rv = runs["stall"]
    j = api.journey_view(rv, "A", 5)
    assert j["available"] and j["element"] == "A[5]" and j["memories"] == ["l2", "l1"]
    assert [(h["k"], h["mem"], h["act"], h["by"]) for h in j["hops"]] == [
        ("burst", "l2", "read", "dma"), ("dma", "l1", "write", "dma"),
        ("grant", "l1", "read", "acc_a.lane1")]  # fmt: skip
    burst, dma, grant = j["hops"]
    assert burst["id"] == dma["id"] and dma["i"] == 0 and dma["l2"] == burst["addr"]
    assert (grant["b"], grant["lane"], grant["owner"]) == (1, 1, "acc_a")
    assert dma["value"] == grant["value"]  # the value moves with the element
    (f,) = j["firings"]
    assert (f["n"], f["role"], f["via"], f["lane"]) == (1, "consumed", "acc_a.lane1", 1)
    assert f["inputs"] == {"acc_a": ["A[4]", "A[5]", "A[6]", "A[7]"],
                           "acc_b": ["B[4]", "B[5]", "B[6]", "B[7]"]}  # fmt: skip
    a5, b5 = f["operands"]["acc_a"]["vals"][1], f["operands"]["acc_b"]["vals"][1]
    assert a5 == grant["value"]
    assert f["push"]["n"] == 1 and f["push"]["first"] == f["push"]["last"] == 1
    assert f["outputs"] == {"acc_out": ["C[4]", "C[5]", "C[6]", "C[7]"]}
    assert f["push"]["vals"][1] == a5 + b5
    c5 = f["results"]["C[5]"]
    assert [(h["k"], h["mem"], h["act"]) for h in c5] == [
        ("grant", "l1", "write"), ("dma", "l1", "read"), ("burst", "l2", "write")]  # fmt: skip
    assert c5[0]["value"] == c5[1]["value"] == a5 + b5 and c5[0]["b"] == 1
    assert (
        c5[1]["id"] == c5[2]["id"] and c5[2]["addr"] <= c5[1]["l2"] < c5[2]["addr"] + c5[2]["size"]
    )
    assert c5[0]["t"] > f["t"] and c5[1]["t"] >= c5[0]["t"]


def test_a_written_element_names_the_firings_it_came_from(runs):
    j = api.journey_view(runs["stall"], "C", 5)
    assert j["memories"] == ["l2", "l1"]
    assert [h["k"] for h in j["hops"]] == ["grant", "dma", "burst"]
    (f,) = j["firings"]
    assert (f["n"], f["role"], f["via"]) == (1, "produced", "acc_out.lane1")


def test_a_held_back_read_names_what_its_bank_served(runs):
    j = api.journey_view(runs["stall"], "B", 0)
    (held,) = [h for h in j["hops"] if h["act"] == "held back"]
    (stall,) = events_of(runs["stall"], "stall", src="acc_b.lane0")
    assert (held["t"], held["by"], held["bank"], held["row"]) == (stall["t"], "acc_b.lane0", 0, 2)
    assert held["served"] == {"port": "acc_a.lane0", "addr": L1, "row": 0, "element": "A[0]"}


def test_b5_on_the_b_pinned_run(runs):
    """B at L1 + 576: B[5] is in bank 13 of row 2, written there by a DMA beat and read there."""
    got = [(h["k"], h["act"], h["bank"], h["row"]) for h in api.journey_view(runs["B_pinned"], "B", 5)["hops"]
           if h["mem"] == "l1"]  # fmt: skip
    assert got == [
        ("dma", "write", 8, 2),
        ("grant", "read", 13, 2),
    ]  # the beat's first word: bank 8


def test_a_push_that_folds_many_firings():
    """An accumulation: four firings of acc_a's beats 0-3 fold into push 0 (first 0, last 3),
    written by acc_out as C[0]; every input's journey reaches it."""
    ev = Events()
    for n in range(4):
        ev.grant(n, "acc_a.lane0", L1 + 8 * n, b=n, task="add_acc_a")
        ev.fire(n + 2, n, {"acc_a": n}, {"acc_a": [n + 1]})
    ev.push(6, 0, 0, 3, [10])
    ev.grant(7, "acc_out.lane0", L1 + 1024, op="w", b=0, task="add_acc_out")
    idx = ev.index()
    j = movement.journey(idx, "C", 0)
    assert [f["n"] for f in j["firings"]] == [0, 1, 2, 3]
    assert all(f["role"] == "produced" and f["push"]["n"] == 0 for f in j["firings"])
    for n in range(4):
        (f,) = movement.journey(idx, "A", n)["firings"]
        assert f["n"] == n and (f["push"]["first"], f["push"]["last"]) == (0, 3)
        assert list(f["results"]) == ["C[0]"] and f["results"]["C[0]"][0]["value"] == 101


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


def test_the_journey_in_a_second_run_stays_in_it():
    """b and n restart at 0 in the second run: A[1]'s journey reaches the second run's firing,
    push and writer's grant (C[1]), A[0]'s the first's (C[0])."""
    idx = two_runs().index(task_list=tasks("add", "add2"))
    for flat, (p, t) in enumerate((("add", 0), ("add2", 100))):
        (f,) = movement.journey(idx, "A", flat)["firings"]
        assert (f["task"], f["n"], f["t"], f["role"]) == (f"{p}_acc", 0, t + 3, "consumed")
        assert f["inputs"] == {"acc_a": [f"A[{flat}]"], "acc_b": [f"B[{flat}]"]}
        assert f["operands"]["acc_b"]["grants"][0]["task"] == f"{p}_acc_b"
        assert (f["push"]["t"], f["push"]["vals"]) == (t + 3, [101 + 2 * flat])
        assert f["outputs"] == {"acc_out": [f"C[{flat}]"]}
        (g,) = [x for x in f["push"]["writer"]["grants"] if x is not None]
        assert (g["t"], g["task"]) == (t + 4, f"{p}_acc_out")
        (back,) = movement.journey(idx, "C", flat)["firings"]  # and back, from the writer
        assert (back["task"], back["t"], back["role"]) == (f"{p}_acc", t + 3, "produced")


def test_an_index_outside_the_region_is_refused(runs):
    with pytest.raises(ValueError):
        api.journey_view(runs["stall"], "A", 64)
    with pytest.raises(ValueError):
        api.journey_view(runs["stall"], "Q", 0)
    assert api.journey_view(runs["stall"], "A", [5])["element"] == "A[5]"


# =============================================================================
# 2. Residency
# =============================================================================


@pytest.mark.parametrize("level", ["beat", "stall"])
def test_residency_is_the_journeys(runs, level):
    """Every container in each memory it lives in; per element the times its journey shows."""
    rv = runs[level]
    res = movement.residency(rv.index)
    assert [(r["region"], r["mem"]) for r in res] == [(r.container, r.memory) for r in rv.regions]
    for r in res:
        for x in r["per_element"]:
            hs = [
                h
                for h in api.journey_view(rv, r["region"], x["flat"])["hops"]
                if h["mem"] == r["mem"]
            ]
            arrival = next((h["t"] for h in hs if h["act"] == "write"), None)
            after = [h for h in hs if arrival is None or h["t"] > arrival]
            uses = [h["t"] for h in after if h["act"] == "read" and h["by"] != "dma"]
            dep = next((h["t"] for h in after if h["act"] == "read" and h["by"] == "dma"), None)
            assert (x["arrival"], x["uses"], x["departure"]) == (arrival, uses, dep)
        window = lambda xs: [min(xs), max(xs)] if xs else None
        el = r["per_element"]
        assert r["arrival"] == window([x["arrival"] for x in el if x["arrival"] is not None])
        assert r["use"] == window([t for x in el for t in x["uses"]])
        assert r["departure"] == window([x["departure"] for x in el if x["departure"] is not None])
        assert r["present_at_start"] == sum(x["arrival"] is None for x in el)


def test_residency_of_the_recorded_run(runs):
    rv = runs["stall"]
    got = {(r["region"], r["mem"]): r for r in movement.residency(rv.index)}
    load_a = [e["t"] for e in events_of(rv, "dma", task="load_A")]
    (burst_a,) = [e["t"] for e in events_of(rv, "burst", task="load_A")]
    assert got["A", "l2"]["present_at_start"] == 64 and got["A", "l2"]["departure"] == [burst_a] * 2
    assert got["A", "l1"]["arrival"] == [min(load_a), max(load_a)]
    reads = [e["t"] for e in rv.events if e["k"] == "grant" and e["src"].startswith("acc_a.")]
    assert got["A", "l1"]["use"] == [min(reads), max(reads)] and got["A", "l1"]["departure"] is None
    store = [e["t"] for e in events_of(rv, "dma", task="store_C")]
    assert got["C", "l1"]["departure"] == [min(store), max(store)] and got["C", "l1"]["use"] is None
    bursts = [e["t"] for e in events_of(rv, "burst", task="store_C")]
    assert got["C", "l2"]["arrival"] == [min(bursts), max(bursts)]
    a0 = got["A", "l1"]["per_element"][0]
    assert a0["wait"] == a0["uses"][0] - a0["arrival"]


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


def test_held_cycles_are_the_profiles_stalls(runs):
    rv = runs["stall"]
    pats = [p for p in movement.patterns(rv.index) if p["port"] != "dma"]
    prof = rv.outputs.profile
    for p in pats:
        assert len(p["held"]) == prof.ports[p["port"]].stalls, p["port"]
    for name, s in prof.streamers.items():
        cycles = {h["t"] for p in pats if p["owner"] == name for h in p["held"]}
        assert len(cycles) == s.cycles["stall_xbar"], name
    b0 = next(p for p in pats if p["port"] == "acc_b.lane0")
    (h,) = b0["held"]
    assert (h["element"], h["by"], h["by_element"]) == ("B[0]", "acc_a.lane0", "A[0]")


def test_held_cycles_need_the_stall_level(runs):
    for p in (p for p in movement.patterns(runs["beat"].index) if p["port"] != "dma"):
        assert p["held"] is None and "needs --trace stall" in p["held_reason"]


# =============================================================================
# 4. Conflicts
# =============================================================================


def test_the_conflicts_of_the_recorded_run(runs):
    rv = runs["stall"]
    c = api.conflicts_view(rv)
    snax = sum(p.stalls for name, p in rv.outputs.profile.ports.items()
               if rv.index.mover_of(name) is not None)  # fmt: skip
    assert c["available"] and c["count"] == len(events_of(rv, "stall")) == snax == 4
    assert c["banks"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert c["ports"] == {f"acc_b.lane{i}": 1 for i in range(4)}
    for x in c["conflicts"]:  # B waits, A is served, in the same bank
        assert x["waiting"]["element"].startswith("B[") and x["served"]["element"].startswith("A[")
        assert x["waiting"]["owner"] == "acc_b" and x["served"]["owner"] == "acc_a"
        assert rv.cluster.bank_row(x["waiting"]["addr"])[0] == x["bank"]
        assert rv.cluster.bank_row(x["served"]["addr"])[0] == x["bank"]
    w = {x["element"]: x for x in c["words"]}
    t = c["conflicts"][0]["t"]
    assert w["B[0]"] == {"addr": L1 + 512, "bank": 0, "row": 2, "held": [t], "won": [],
                         "element": "B[0]"}  # fmt: skip
    assert w["A[0]"]["won"] == [t]
    assert sum(len(x["held"]) for x in c["words"]) == sum(len(x["won"]) for x in c["words"]) == 4
    assert build_movement(rv).conflicts == c["count"]  # the report's section counts the same


def test_a_window_of_conflicts(runs):
    t = api.conflicts_view(runs["stall"])["conflicts"][0]["t"]
    c = api.conflicts_view(runs["stall"], t, t + 1)
    assert c["count"] == 4 and {x["t"] for x in c["conflicts"]} == {t}
    assert api.conflicts_view(runs["stall"], t + 1)["count"] == 0


# =============================================================================
# 5. Below a level, and filtered
# =============================================================================


def test_a_task_level_run_gets_the_reason(runs):
    rv = runs["task"]
    for got in (api.journey_view(rv, "A", 0), api.conflicts_view(rv)):
        assert got["available"] is False and "needs --trace " in got["reason"]
        assert "traced at task" in got["reason"]
    assert "traced at task" in build_movement(rv).reason


def test_a_beat_level_run_lacks_only_what_needs_stall(runs):
    rv = runs["beat"]
    m = build_movement(rv)
    assert m.reason is None and m.residency and m.patterns
    assert m.conflicts is None and "--trace stall" in m.conflicts_reason
    c = api.conflicts_view(rv)
    assert c["available"] is False and "traced at beat" in c["reason"]
    j = api.journey_view(rv, "B", 0)
    assert j["available"] and not [h for h in j["hops"] if h["act"] == "held back"]


def test_a_filtered_trace_says_so(tmp_path):
    rv = api.load_run(run_dir(tmp_path / "f", "vecadd", "beat", trace_sources=["acc", "acc_a"]))
    m = build_movement(rv)
    assert m.reason is None and m.sources == ["acc", "acc_a"]
    j = api.journey_view(rv, "A", 0)
    assert j["filtered"] == {"sources": ["acc", "acc_a"]}
    assert [h["k"] for h in j["hops"]] == ["grant"]  # no DMA lines: the beat's arrival is missing
    assert build_movement(api.load_run(run_dir(tmp_path / "w", "vecadd", "beat"))).sources is None
