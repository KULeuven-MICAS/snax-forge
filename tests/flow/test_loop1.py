"""The loop1 walk-through (D98, D171): every claim examples/loop1/README.md makes, checked on the
runs it describes, on SNAX-GVSoC (marker ``gvsoc``, D134: in the image, skipped where
out/snax-gvsoc/install is missing): what does not depend on timing exactly, each cycle number it
states within 10 % (``near``; beyond it, investigate before updating the page).

The README's two flow commands are run as written (from the repo root, into a
temporary folder named as the README names it), and the default once more at trace
``off``. Each quoted table is marked ``<!-- excerpt: FOLDER/report/FILE -->``; every line
of it must be a line of that report. The claims made in prose are checked one by one
below, in the README's order, from the profile, the trace and the viewer's queries.
"""

from __future__ import annotations

import shlex

import pytest

from snax_forge.flow import __main__ as cli
from snax_forge.viz import api
from tests.design.helpers import REPO

from .helpers import excerpts_hold, near, needs_gvsoc

pytestmark = [pytest.mark.gvsoc, needs_gvsoc]

README = REPO / "examples" / "loop1" / "README.md"
DEFAULT, WIDE = "vecadd_N512_stall", "vecadd_N512_W8_stall"
COMMANDS = {
    DEFAULT: "pixi run flow recipes/elementwise/vecadd.json --platform platforms/snax_forge.json --set N=512 "
    "--trace stall",
    WIDE: "pixi run flow recipes/elementwise/vecadd.json --platform platforms/snax_forge.json --set N=512 "
    "--set W=8 --trace stall",
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Both flows, run from the README's commands, and the default at trace off; folder ->
    RunView (the off run as ``"off"``)."""
    root = tmp_path_factory.mktemp("loop1")
    off = COMMANDS[DEFAULT].removesuffix(" --trace stall")
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(REPO)
        for folder, cmd in [*COMMANDS.items(), ("off", off)]:
            args = shlex.split(cmd)[3:]  # after "pixi run flow"
            assert cli.main([*args, "--out", str(root / folder)]) == 0
            out[folder] = api.load_run(root / folder, folder)
    return root, out


def text() -> str:
    return README.read_text()


def tasks(rv) -> dict[str, tuple[int, int]]:
    """Each task's (start, done), from the schedule's bars."""
    return {b["task"]: (b["from"], b["to"]) for lane in api.schedule_view(rv)["lanes"]
            for b in lane.get("bars", [])}  # fmt: skip


def folds(rv) -> list[dict]:
    return [s for s in api.schedule_view(rv)["segments"] if s["folded"]]


def lane(rv, name: str) -> dict:
    (x,) = [x for x in api.schedule_view(rv)["lanes"] if x["name"] == name]
    return x


def grants(rv, port: str) -> list[dict]:
    return [e for e in rv.events if e["k"] == "grant" and e["src"] == port]


# =============================================================================
# The commands, the folders and the excerpts
# =============================================================================


def test_the_readme_gives_both_commands_and_folders():
    t = text()
    for folder, cmd in COMMANDS.items():
        assert f"\n{cmd}\n" in t and f"`out/flow/{folder}/`" in t
    assert f"pixi run view out/flow/{DEFAULT}\n" in t
    assert f"pixi run view out/flow/{DEFAULT} out/flow/{WIDE}\n" in t


def test_the_flows_get_the_folder_names_the_readme_uses(runs):
    root, _ = runs
    for folder, w in ((DEFAULT, 4), (WIDE, 8)):
        log = (root / folder / "flow.log").read_text()
        assert log.startswith(f"flow {folder} (vecadd: W={w}, N=512; N=512) on snax_forge")


def test_every_excerpt_is_a_part_of_its_report(runs):
    root, _ = runs
    assert excerpts_hold(text(), root) == 13


# =============================================================================
# The claims in prose, in the README's order
# =============================================================================


def test_the_cycles_do_not_depend_on_the_trace_level(runs):
    _, rv = runs
    assert rv[DEFAULT].total_cycles == rv["off"].total_cycles and near(rv["off"].total_cycles, 471)
    assert rv["off"].trace_level == "off"
    stdout = (runs[0] / "off" / "run" / "stdout.txt").read_text()
    assert f"forge cycles: {rv['off'].total_cycles}\n" in stdout


def test_the_vectors_lie_16_rows_over_all_32_banks(runs):
    _, rv = runs
    d = rv[DEFAULT]
    words = " ".join(text().split())  # the README's lines joined, however they wrap
    assert d.cluster.banks == 32 and "16 bank rows long over all 32 banks" in words
    for r in d.regions:
        if r.memory == "l1":
            assert r.shape[0] * r.itemsize == 512 * 8 == 16 * 32 * d.cluster.bank_bytes


def test_the_accelerator_fires_every_cycle_after_its_streamers_head_start(runs):
    _, rv = runs
    d = rv[DEFAULT]
    acc = d.outputs.profile.accelerators["acc"]
    assert (acc.firings, acc.cycles["busy"]) == (128, 128)
    assert acc.cycles["stall_in"] == 0 and "never starved" in text()
    t = tasks(d)
    movers = min(t[f"add_acc_{m}"][0] for m in ("a", "b", "out"))
    start, done = t["add_acc"]
    assert near(start - movers, 10) and near(movers, 183) and near(start, 193)
    assert near(done - start, 129)
    # the counter: sound (D171), and below the estimate's by the head start
    counter = rv["off"].outputs.profile.functional_check["tasks"]["add_acc"]["counter"]
    assert 128 <= counter < 131 and near(counter, 131)  # its beats, below the estimate's
    assert "reads 128 against the estimate's 131" in " ".join(text().split())


def test_acc_b_is_held_at_its_first_request_then_one_beat_behind(runs):
    _, rv = runs
    d = rv[DEFAULT]
    assert near(d.outputs.profile.streamers["acc_b"].cycles["stall_xbar"], 2)
    cs = api.conflicts_view(d, 0, d.total_cycles)["conflicts"]
    first = cs[0]["t"]
    assert near(first, 185)
    assert [(c["t"], c["bank"], c["waiting"]["element"], c["served"]["element"]) for c in cs[:4]] == [
        (first, b, f"B[{b}]", f"A[{b}]") for b in range(4)]  # fmt: skip
    assert {c["served"]["port"].split(".")[0] for c in cs[:4]} == {"acc_a"}
    # once more when the accelerator starts, then acc_out's first write against acc_b
    rest = [(c["waiting"]["element"][0], c["served"]["element"][0]) for c in cs[4:]]
    assert rest == [("B", "A")] * 4 + [("C", "B")] * 4
    for lane_ in range(4):
        a, b = grants(d, f"acc_a.lane{lane_}"), grants(d, f"acc_b.lane{lane_}")
        assert a[0]["t"] == first and len(a) == len(b) == 128
        assert all(y["t"] - x["t"] == 1 for x, y in zip(a, b, strict=True))


def test_the_accelerator_is_busy_27_2_percent_of_the_window(runs):
    _, rv = runs
    p = rv[DEFAULT].outputs.profile
    assert near(100 * p.accelerators["acc"].cycles["busy"] / p.total_cycles, 27.2)
    assert "27.2% of the window" in text()


def test_the_schedule_folds_the_setup_the_gap_and_the_end(runs):
    _, rv = runs
    d = rv[DEFAULT]
    fs = folds(d)
    stated = [(0, 34), (325, 356), (428, 471)]
    assert len(fs) == 3 and all(
        near(f["from"], a) and near(f["to"], b) for f, (a, b) in zip(fs, stated, strict=True)
    ) and fs[0]["from"] == 0  # fmt: skip
    assert set(fs[0]["cores"]) == {"core0"}  # one core (D170)
    first = fs[0]["cores"]["core0"]
    assert first["funcs"] == ["main", "forge_core_0"] and first["csr_writes"] == 0
    assert d.outputs.profile.cores["core0"].csr_writes == 18
    assert "forge_check" in fs[-1]["cores"]["core0"]["funcs"]
    assert min(b["from"] for b in lane(d, "dma")["bars"]) == fs[0]["to"]


def test_the_run_is_serial_between_the_folds(runs):
    _, rv = runs
    d = rv[DEFAULT]
    t = tasks(d)
    assert near(t["load_A"][0], 34) and near(t["load_B"][1], 176)
    assert t["load_B"][1] < t["add_acc"][0] < t["add_acc"][1] < t["store_C"][0]
    assert near(t["store_C"][0], 356) and near(t["store_C"][1], 427)
    assert t["store_C"][1] - t["store_C"][0] == 71 and near(d.total_cycles - 428, 43)
    waits = [w.block for w in d.outputs.profile.cores["core0"].waits]
    assert waits == ["acc"] and d.outputs.profile.cores["core0"].cycles["barrier"] == 0
    assert near(d.outputs.profile.cores["core0"].cycles["poll"], 127)
    assert near(d.outputs.profile.dmas["dma"].busy, 213)


def test_read_data_comes_back_a_cycle_after_its_grant(runs):
    _, rv = runs
    reads = [e for e in rv[DEFAULT].events if e["k"] == "grant" and e["op"] == "r"
             and e["src"].startswith("acc_")]  # fmt: skip
    assert reads and {e["back"] - e["t"] for e in reads} == {1}


def test_the_stepping_keys_go_by_event_and_by_boundary(runs):
    _, rv = runs
    d = rv[DEFAULT]
    assert api.step_view(d, 300, 1, "event")["t"] == 301  # a grant every cycle in the task
    after = min(x for a, b in tasks(d).values() for x in (a, b) if x > 300)
    assert api.step_view(d, 300, 1, "boundary")["t"] == after  # the next task boundary


def test_the_prediction_holds(runs):
    """Section 7 on what the code layout does not move: the firings, the accelerator's busy
    cycles and each copy's cycles (the task table) with the DMA's busy cycles (the profile); the
    counter and the window shown, not predicted."""
    _, rv = runs
    d, w = rv[DEFAULT].outputs.profile, rv[WIDE].outputs.profile
    assert w.accelerators["acc"].firings == 512 // 8 == 64
    assert d.accelerators["acc"].firings == 128
    assert (d.accelerators["acc"].cycles["busy"], w.accelerators["acc"].cycles["busy"]) == (128, 64)
    td, tw = tasks(rv[DEFAULT]), tasks(rv[WIDE])
    for copy in ("load_A", "load_B", "store_C"):
        assert td[copy][1] - td[copy][0] == tw[copy][1] - tw[copy][0]
    assert all(near(tw[c][1] - tw[c][0], n) for c, n in (("load_A", 77), ("load_B", 128),
                                                          ("store_C", 71)))  # fmt: skip
    assert d.dmas["dma"].busy == w.dmas["dma"].busy and near(w.dmas["dma"].busy, 213)
    # shown: the counters, sound (D171) and stated; W=8's 3 starved cycles; the windows
    counter = {k: p.functional_check["tasks"]["add_acc"]["counter"] for k, p in
               ((DEFAULT, d), (WIDE, w))}  # fmt: skip
    assert near(counter[DEFAULT], 128) and near(counter[WIDE], 67)
    assert near(w.accelerators["acc"].cycles["stall_in"], 3)
    assert near(d.total_cycles, 471) and near(w.total_cycles, 391)
    assert d.total_cycles - w.total_cycles > 64
    t = text()
    assert "| busy cycles 64, half of 128 | 64 |" in t and "window went from 471 to 391" in t
    i, j = t.index("## 5. Predict"), t.index("## 6. Rerun")
    assert "- firings 64 (N/W, was 128);" in t[i:j]  # written before the rerun's section
    assert "64, half of W=4's 128" in t[i:j] and "load_A 77, load_B 128, store_C 71" in t[i:j]
