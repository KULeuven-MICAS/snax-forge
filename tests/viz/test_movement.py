"""Data movement (VIS4b, D97): journeys, residency, patterns and conflicts from the beat trace.

1. the journey of one element, and a reduction's many firings
2. residency windows equal the extremes of the elements' journeys
3. every pattern regenerates its lane's addresses; held-back cycles equal the profile
4. conflicts placed on the layout, and the memory tab's marks
5. firings with a repeated read (D69), matched on a small made-up trace
6. what a run without a beat trace, or with a filtered one, gets
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from snax_forge.snax_model.scenario import ClusterConfig, Region, Scenario, run, write_outputs
from snax_forge.viz import api, memory, movement
from tests.flow.helpers import B_PIN, run_flows

from .helpers import SCEN, run_dir

FLOWS = {
    "default": {},
    "B_pinned": {"memory_sets": B_PIN},
    "W8": {"recipe_sets": {"W": 8}},
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The flow runs at beat level, vecadd_conflict, vecadd_tiled, fmul, and reduce with regions."""
    base = tmp_path_factory.mktemp("moves")
    flows = run_flows(base, FLOWS, "beat")
    out = {name: api.load_run(f.out / "run", name) for name, f in flows.items()}
    for name in ("vecadd_conflict", "vecadd_tiled", "fmul"):
        out[name] = api.load_run(run_dir(base / name, name, "beat"), name)
    sc = Scenario.load(SCEN / "reduce" / "scenario.json")
    sc.regions = [Region("IN", "l1", 0, (64,), (8,)), Region("OUT", "l1", 1024, (4,), (8,))]
    out["reduce"] = api.load_run(
        write_outputs(run(sc, trace_level="beat"), base / "reduce"), "reduce"
    )
    return out


def hops(j):
    return [(h["t"], h["mem"], h["act"], h["by"], h.get("bank"), h.get("row")) for h in j["hops"]]


# =============================================================================
# 1. Journeys
# =============================================================================


def test_the_journey_of_a5(runs):
    """A[5]: from L2, into L1, read by acc_a.1, fired with B[5], and its result C[5] stored."""
    j = api.journey_view(runs["default"], "A", 5)
    assert j["available"] and j["element"] == "A[5]" and j["memories"] == ["l2", "l1"]
    assert hops(j) == [
        (13, "l2", "read", "dma", None, None),
        (14, "l2", "data back", "dma", None, None),
        (15, "l1", "write", "dma.wide", 5, 0),
        (42, "l1", "read", "acc_a.1", 5, 0),
        (43, "l1", "data back", "acc_a.1", 5, 0),
    ]
    (f,) = j["firings"]
    assert (f["acc"], f["task"], f["n"], f["t"], f["lane"], f["role"]) == (
        "acc",
        0,
        1,
        45,
        1,
        "consumed",
    )
    assert f["inputs"] == {"a": ["A[5]"], "b": ["B[5]"]} and f["outputs"] == {"out": ["C[5]"]}
    c5 = [
        (h["t"], h["mem"], h["act"], h["by"], h.get("bank"), h.get("row"))
        for h in f["results"]["C[5]"]
    ]
    assert c5 == [
        (46, "l1", "write", "acc_out.1", 5, 8),
        (73, "l1", "read", "dma.wide", 5, 8),
        (74, "l1", "data back", "dma.wide", 5, 8),
        (75, "l2", "write", "dma", None, None),
    ]


def test_a_held_back_read_names_what_its_bank_served(runs):
    j = api.journey_view(runs["default"], "B", 8)
    (held,) = [h for h in j["hops"] if h["act"] == "held back"]
    assert (held["t"], held["by"], held["bank"], held["row"]) == (44, "acc_b.0", 8, 4)
    assert held["served"] == {"port": "acc_a.0", "w": False, "row": 0, "element": "A[8]"}


def test_b5_on_the_b_pinned_run(runs):
    """B at 576: B[5] lands with the beat at 576 in cycle 28 and is read from bank 13 at 43."""
    got = hops(api.journey_view(runs["B_pinned"], "B", 5))
    assert (28, "l1", "write", "dma.wide", 13, 4) in got
    assert (43, "l1", "read", "acc_b.1", 13, 4) in got
    assert not [h for h in got if h[2] == "held back"]


def test_a_reduced_element_lists_every_firing_it_came_from(runs):
    """reduce: T = 4 firings of four lanes each make one output, OUT[1] from IN[16..31]."""
    j = api.journey_view(runs["reduce"], "OUT", 1)
    fs = [f for f in j["firings"] if f["role"] == "produced"]
    assert [f["n"] for f in fs] == [4, 5, 6, 7]
    assert [x for f in fs for x in f["inputs"]["in"]] == [f"IN[{i}]" for i in range(16, 32)]
    assert all(f["outputs"] == {"out": ["OUT[1]"]} for f in fs)
    j = api.journey_view(runs["reduce"], "IN", 17)
    (f,) = j["firings"]
    assert (f["n"], f["inputs"], list(f["results"])) == (4, {"in": ["IN[17]"]}, ["OUT[1]"])


def test_an_index_outside_the_region_is_refused(runs):
    with pytest.raises(ValueError):
        api.journey_view(runs["default"], "A", 64)
    with pytest.raises(ValueError):
        api.journey_view(runs["default"], "Q", 0)
    assert api.journey_view(runs["default"], "A", [5])["element"] == "A[5]"


# =============================================================================
# 2. Residency
# =============================================================================


@pytest.mark.parametrize("name", ["default", "B_pinned"])
def test_residency_windows_are_the_extremes_of_the_journeys(runs, name):
    rv = runs[name]
    for res in api.movement_view(rv)["residency"]:
        region = next(
            r for r in rv.outputs.regions if r.name == res["region"] and r.mem == res["mem"]
        )
        arr, uses, dep = [], [], []
        for flat in range(region.size):
            hs = [
                h for h in api.journey_view(rv, region.name, flat)["hops"] if h["mem"] == region.mem
            ]
            w = [h["t"] for h in hs if h["act"] == "write"][:1]
            arr += w
            uses += [h["t"] for h in hs if h["act"] == "read" and h["owner"] != "dma"]
            dep += [
                h["t"]
                for h in hs
                if h["act"] == "read" and h["owner"] == "dma" and (not w or h["t"] > w[0])
            ][:1]
        window = lambda xs: [min(xs), max(xs)] if xs else None
        assert (res["arrival"], res["use"], res["departure"]) == (
            window(arr),
            window(uses),
            window(dep),
        )
        assert res["present_at_start"] == region.size - len(arr)


def test_residency_of_the_default_run(runs):
    got = {(r["region"], r["mem"]): r for r in api.movement_view(runs["default"])["residency"]}
    assert got["A", "l2"]["present_at_start"] == 64 and got["A", "l2"]["departure"] == [13, 20]
    assert got["A", "l1"]["arrival"] == [15, 22] and got["A", "l1"]["use"] == [41, 63]
    assert got["C", "l1"]["arrival"] == [45, 67] and got["C", "l1"]["departure"] == [73, 80]
    assert got["C", "l2"]["arrival"] == [75, 82] and got["C", "l2"]["departure"] is None


# =============================================================================
# 3. Patterns
# =============================================================================


@pytest.mark.parametrize(
    "name", ["default", "B_pinned", "W8", "vecadd_conflict", "vecadd_tiled", "fmul", "reduce"]
)
def test_every_pattern_regenerates_its_addresses(runs, name):
    rv = runs[name]
    idx = api.moves(rv)
    l1 = idx.layouts["l1"]
    pats = api.movement_view(rv)["patterns"]
    assert pats and all(p["addr"] is not None for p in pats)
    for p in pats:
        span = rv.spans[p["owner"]][p["task"]]
        if "side" in p:  # a DMA side: its beats
            got = [
                a for t, _, a, m in idx.dma_hops[p["owner"], p["side"]] if span[0] <= t <= span[1]
            ]
        else:
            beats = [
                h
                for h in idx.lane_hops[p["port"]]
                if span[0] <= h.t <= span[1] and h.act in ("read", "write")
            ]
            if len({h.t for h in beats}) < len(beats):  # a wide port: one address per beat
                beats = [h for i, h in enumerate(beats) if i == 0 or h.t != beats[i - 1].t]
            got = [l1.base_addr + h.word * l1.word_bytes for h in beats]
        assert movement.expand_nest(p["addr"]["base"], p["addr"]["nest"]) == got, p["port"]


def test_the_patterns_of_the_default_run(runs):
    pats = {(p["port"], p["task"]): p for p in api.movement_view(runs["default"])["patterns"]}
    b1 = pats["acc_b.1", 0]
    assert b1["addr"] == {"base": 520, "nest": [[16, 32]]}  # B[4j + 1]: base 1, 4 per beat
    assert b1["element"] == {"region": "B", "base": 1, "nest": [[16, 4]]}
    assert (b1["beats"], b1["first"], b1["last"], b1["ideal_last"], b1["held_cycles"]) == (
        16,
        42,
        64,
        57,
        7,
    )
    assert pats["acc_a.1", 0]["idle"] == 7  # waits for acc_b: no request, not held back
    assert pats["dma.src", 0]["mem"] == "l2" and pats["dma.dst", 0]["mem"] == "l1"


def test_fmul_fits_its_two_loops(runs):
    pats = {(p["port"], p["task"]): p for p in api.movement_view(runs["fmul"])["patterns"]}
    assert pats["acc_a.0", 0]["addr"] == {"base": 0, "nest": [[8, 8], [2, 256]]}
    assert pats["acc_a.0", 1]["addr"]["base"] == 128


@pytest.mark.parametrize("name", ["default", "B_pinned", "vecadd_conflict", "vecadd_tiled", "fmul"])
def test_held_back_cycles_are_the_profiles_stalls(runs, name):
    rv = runs[name]
    pats = api.movement_view(rv)["patterns"]
    prof = rv.outputs.profile.to_dict()
    for s, v in prof["streamers"].items():
        cycles = {h["t"] for p in pats if p["owner"] == s for h in p.get("held", [])}
        assert len(cycles) == v["cycles"]["stall_xbar"], s
    for port, v in prof["ports"].items():
        assert sum(len(p.get("held", [])) for p in pats if p["port"] == port) == v["stalls"], port
    if name == "default":
        assert prof["streamers"]["acc_b"]["cycles"]["stall_xbar"] == 7
    if name == "B_pinned":
        assert all(v["cycles"]["stall_xbar"] == 0 for v in prof["streamers"].values())


def test_fit_nest():
    assert movement.fit_nest([5]) == (5, [])
    assert movement.fit_nest([0, 8, 16, 256, 264, 272]) == (0, [[3, 8], [2, 256]])
    assert movement.fit_nest([3, 3, 3]) == (3, [[3, 0]])
    assert movement.fit_nest([0, 1, 3]) is None
    assert movement.expand_nest(0, [[3, 8], [2, 256]]) == [0, 8, 16, 256, 264, 272]


# =============================================================================
# 4. Conflicts
# =============================================================================


def test_the_conflicts_of_vecadd_conflict(runs):
    rv = runs["vecadd_conflict"]
    c = api.conflicts_view(rv)
    stalls = [e for e in rv.events if e["k"] == "stall" and e["mem"] == "l1"]
    assert c["count"] == len(stalls) == 28
    assert c["banks"] == {"0": 3, "1": 3, "2": 3, "3": 3, "8": 4, "9": 4, "10": 4, "11": 4}
    assert c["ports"] == {f"acc_b.{i}": 7 for i in range(4)}
    for x in c["conflicts"]:  # B always waits, A always wins, in the same bank
        assert x["waiting"]["element"].startswith("B[") and x["served"]["element"].startswith("A[")
        assert x["waiting"]["port"].startswith("acc_b.") and x["served"]["port"].startswith(
            "acc_a."
        )
    first = c["conflicts"][0]
    assert (first["t"], first["bank"], first["waiting"]["element"], first["served"]["element"]) == (
        44,
        8,
        "B[8]",
        "A[8]",
    )
    w = {x["element"]: x for x in c["words"]}
    assert w["B[8]"] == {"bank": 8, "row": 4, "element": "B[8]", "held": [44], "won": []}
    assert w["A[8]"]["won"] == [44]
    assert sum(len(x["held"]) for x in c["words"]) == sum(len(x["won"]) for x in c["words"]) == 28


def test_a_window_of_conflicts(runs):
    c = api.conflicts_view(runs["vecadd_conflict"], 44, 45)
    assert c["count"] == 4 and {x["t"] for x in c["conflicts"]} == {44}


def test_the_b_pinned_run_has_no_conflicts(runs):
    assert api.conflicts_view(runs["B_pinned"])["count"] == 0
    assert api.movement_view(runs["B_pinned"])["conflicts"] == {
        "count": 0,
        "banks": {},
        "ports": {},
    }


def test_conflict_marks_fold_the_memory_tab(runs):
    """Rows fold only with equal marks; a fold line says how many conflicts it hides."""
    rv = runs["vecadd_conflict"]
    l1 = api.memory_view(rv, "conflicts")["memories"][0]
    lines = l1["lines"]
    rows = {x["row"]: x for x in lines if x["kind"] == "row"}
    assert rows[0]["marks"] == [0] * 8 + [1] * 4 + [0] * 4  # A[8..11] won in cycle 44
    assert rows[4]["marks"] == [0] * 8 + [1] * 4 + [0] * 4  # B[8..11] waited
    folds = [x for x in lines if x["kind"] == "fold"]
    for f in folds:  # the hidden rows' marks add up to what the fold line says
        got = api.memory_rows(rv, "l1", f["from"], f["to"] + 1, "conflicts")["rows"]
        assert sum(sum(r["marks"]) for r in got) == f["marked"]
    total = sum(sum(x["marks"]) for x in lines if x["kind"] == "row") + sum(
        f["marked"] for f in folds
    )
    assert total == 2 * 28  # each conflict marks the word that waited and the word served
    assert "marks" not in api.memory_view(rv)["memories"][0]["lines"][0]  # none unless asked


# =============================================================================
# 5. A repeated read (D69) on a made-up trace
# =============================================================================


def test_a_read_handed_out_twice_feeds_two_firings():
    """acc_a reads two beats, acc_b four, and the accelerator fires four times: each of
    acc_a's reads feeds two firings in order (T = 2), each write carries one firing."""
    cl = ClusterConfig.load(SCEN / "clusters" / "alu4.json")
    regions = [Region(n, "l1", b, (16,), (8,)) for n, b in (("A", 0), ("B", 512), ("C", 1024))]
    layouts = {"l1": memory.layout_of("l1", cl, regions)}

    def grant(t, port, addr, w=False):
        word = addr // 8
        return {"t": t, "k": "grant", "src": "xbar", "port": port, "mem": "l1", "w": w, "addr": addr,
                "banks": [word % 16], "row": word // 16}  # fmt: skip

    events = [grant(1, "acc_a.0", 0), grant(2, "acc_a.0", 32)]
    events += [grant(t, "acc_b.0", 512 + 32 * i) for i, t in enumerate(range(1, 5))]
    events += [{"t": t, "k": "fire", "src": "acc", "n": n} for n, t in enumerate(range(5, 9))]
    events += [grant(t, "acc_out.0", 1024 + 32 * i, True) for i, t in enumerate(range(6, 10))]
    events.sort(key=lambda e: e["t"])
    ports = {f"{s}.{i}": {"owner": s} for s in ("acc_a", "acc_b", "acc_out") for i in range(4)}
    spans = {s: [(0, 20)] for s in ("acc", "acc_a", "acc_b", "acc_out")}
    trace = SimpleNamespace(level="beat", filter_sources=None, filter_window=None)
    idx = movement.build_index(events, cl, ports, layouts, spans, trace)
    assert [f.n for f in idx.fed["acc_a.0", 0]] == [0, 1] and [
        f.n for f in idx.fed["acc_a.0", 1]
    ] == [2, 3]
    assert [f.n for f in idx.fed["acc_b.0", 2]] == [2]
    assert [[f.n for f in idx.made["acc_out.0", k]] for k in range(4)] == [[0], [1], [2], [3]]
    j = movement.journey(idx, regions, "A", 4)  # A[4]: acc_a's second read
    assert [f["n"] for f in j["firings"]] == [2, 3]
    assert [f["inputs"]["b"] for f in j["firings"]] == [["B[8]"], ["B[12]"]]


# =============================================================================
# 6. Without a beat trace, and with a filtered one
# =============================================================================


def test_a_task_level_run_gets_the_reason(tmp_path):
    rv = api.load_run(run_dir(tmp_path / "t", "vecadd", "task"))
    for got in (api.movement_view(rv), api.journey_view(rv, "A", 0), api.conflicts_view(rv)):
        assert got["available"] is False and "needs a beat-level trace" in got["reason"]
        assert "level task" in got["reason"]
    with pytest.raises(api.MovementUnavailable):
        api.memory_view(rv, "conflicts")
    assert "marks" in api.memory_view(rv) and api.memory_view(rv)["marks"] is None


def test_a_filtered_trace_says_so(tmp_path):
    rv = api.load_run(run_dir(tmp_path / "f", "vecadd", "beat", trace_window=(40, 50)))
    m = api.movement_view(rv)
    assert m["available"] and m["filtered"] == {"sources": None, "window": [40, 50]}
    assert all(40 <= c["t"] < 50 for c in api.conflicts_view(rv)["conflicts"])


# =============================================================================
# 7. Time marks for the memory tab (arrival, first use, wait)
# =============================================================================


def texts_of(lines):
    out = []
    for x in lines:
        if x["kind"] == "row":
            out.append(("row", x["row"], x["marks"]))
        elif x["kind"] == "fold":
            out.append(("fold", x["from"], x["to"], x["mark_step"], x["mark_span"]))
        else:
            out.append(("empty", x["from"], x["to"]))
    return out


def test_arrival_folds_by_a_constant_step(runs):
    """The DMA writes one 8-word beat per cycle, so A and B arrive 2 cycles per row; C is
    written by acc_out at the pace acc_b's stalls allow, 6 cycles per row."""
    l1 = api.memory_view(runs["default"], "arrival")["memories"][0]
    assert l1["mark_range"] == [15, 67]
    got = texts_of(l1["lines"])
    assert got[0] == ("row", 0, [15] * 8 + [16] * 8)
    assert got[1] == ("fold", 1, 2, 2, [17, 20])
    assert got[4] == ("fold", 5, 6, 2, [30, 33])
    assert got[6] == ("row", 8, [45] * 4 + [46] * 4 + [48] * 4 + [49] * 4)
    assert got[7] == ("fold", 9, 10, 6, [51, 61])
    assert got[-1] == ("empty", 12, 63)


def test_use_and_wait_marks(runs):
    view = {m["mem"]: m for m in api.memory_view(runs["default"], "wait")["memories"]}
    rows = {x["row"]: x["marks"] for x in view["l1"]["lines"] if x["kind"] == "row"}
    assert rows[0][:4] == [26, 26, 26, 26]  # A[0..3]: in L1 at 15, first read at 41
    assert rows[4][:4] == [14, 14, 14, 14]  # B[0..3]: in L1 at 28, first read at 42
    assert rows[8] == [None] * 16  # C is written, never read by a streamer
    assert view["l2"]["mark_range"] is None  # nothing in L2 is read by a streamer
    use = api.memory_view(runs["default"], "use")["memories"][0]
    assert use["mark_range"] == [41, 64]


def test_time_rows_on_request_carry_their_marks(runs):
    got = api.memory_rows(runs["default"], "l1", 1, 3, "arrival")["rows"]
    assert [r["marks"] for r in got] == [[17] * 8 + [18] * 8, [19] * 8 + [20] * 8]
    with pytest.raises(memory.MemoryViewError):
        api.memory_view(runs["default"], "later")


def test_the_time_step_rule():
    ts = memory._time_step
    assert ts([1, 2, None], [3, 4, None]) == (True, 2)
    assert ts([1, 2, None], [3, 5, None]) == (False, None)
    assert ts([1, None], [3, 4]) == (False, None)  # a word with a time next to one without
    assert ts([None, None], [None, None]) == (True, None)
    marks = memory.TimeMarks({0: [1, 1], 1: [3, 3], 2: [5, 5], 3: [8, 8]})
    cl = ClusterConfig.load(SCEN / "clusters" / "alu4.json")
    lay = memory.layout_of("l1", cl, [Region("A", "l1", 0, (4, 16), (128, 8))])
    kinds = [(x["kind"], x.get("row", x.get("from"))) for x in memory.fold(lay, marks)]
    assert kinds == [
        ("row", 0),
        ("fold", 1),
        ("row", 2),
        ("row", 3),
        ("empty", 4),
    ]  # step 2, then 3
