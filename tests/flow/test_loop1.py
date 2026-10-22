"""The loop1 walk-through (D98): every number examples/loop1/README.md states, checked on the
runs it describes, on SNAX-GVSoC (marker ``gvsoc``, D128: in the image, skipped where
out/snax-gvsoc/install is missing).

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

from .helpers import excerpts_hold, needs_gvsoc

pytestmark = [pytest.mark.gvsoc, needs_gvsoc]

README = REPO / "examples" / "loop1" / "README.md"
DEFAULT, WIDE = "vecadd_N512_stall", "vecadd_N512_W8_stall"
COMMANDS = {
    DEFAULT: "pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json --set N=512 "
    "--trace stall",
    WIDE: "pixi run flow recipes/vecadd.json --platform platforms/snax_forge.json --set N=512 "
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
    assert excerpts_hold(text(), root) == 12


# =============================================================================
# The claims in prose, in the README's order
# =============================================================================


def test_the_cycles_do_not_depend_on_the_trace_level(runs):
    _, rv = runs
    assert rv[DEFAULT].total_cycles == rv["off"].total_cycles == 486
    assert rv["off"].trace_level == "off"
    stdout = (runs[0] / "off" / "run" / "stdout.txt").read_text()
    assert "forge cycles: 486\n" in stdout


def test_the_vectors_lie_16_rows_over_all_32_banks(runs):
    _, rv = runs
    d = rv[DEFAULT]
    assert d.cluster.banks == 32 and "16 bank\nrows long over all 32 banks" in text()
    for r in d.regions:
        if r.memory == "l1":
            assert r.shape[0] * r.itemsize == 512 * 8 == 16 * 32 * d.cluster.bank_bytes


def test_the_accelerator_fires_every_cycle_after_3_starved_ones(runs):
    _, rv = runs
    d = rv[DEFAULT]
    acc = d.outputs.profile.accelerators["acc"]
    assert (acc.firings, acc.cycles["busy"], acc.cycles["stall_in"]) == (128, 128, 3)
    start, done = tasks(d)["add_acc"]
    assert done - start == 132
    assert d.index.runs["add_acc"][:2] == [(224, 227, "stall_in"), (227, 355, "busy")]
    assert "starved for the first 3 (cycles 224–226)" in text()


def test_acc_b_is_held_once_at_its_first_request_then_one_beat_behind(runs):
    _, rv = runs
    d = rv[DEFAULT]
    assert d.outputs.profile.streamers["acc_b"].cycles["stall_xbar"] == 1
    cs = api.conflicts_view(d, 0, d.total_cycles)["conflicts"]
    assert [(c["t"], c["bank"], c["waiting"]["element"], c["served"]["element"]) for c in cs] == [
        (224, b, f"B[{b}]", f"A[{b}]") for b in range(4)]  # fmt: skip
    assert {c["served"]["port"].split(".")[0] for c in cs} == {"acc_a"}
    for lane_ in range(4):
        a, b = grants(d, f"acc_a.lane{lane_}"), grants(d, f"acc_b.lane{lane_}")
        assert a[0]["t"] == 224 and len(a) == len(b) == 128
        assert all(y["t"] - x["t"] == 1 for x, y in zip(a, b, strict=True))
        # one beat behind: when acc_b reads beat j, acc_a reads beat j + 1, four banks on
        at = {x["t"]: x["bank"] for x in a}
        assert all((at[y["t"]] - y["bank"]) % 32 == 4 for y in b if y["t"] in at)


def test_the_accelerator_is_busy_26_3_percent_of_the_window(runs):
    _, rv = runs
    p = rv[DEFAULT].outputs.profile
    assert f"{100 * p.accelerators['acc'].cycles['busy'] / p.total_cycles:.1f}" == "26.3"


def test_the_schedule_folds_the_setup_and_the_end(runs):
    _, rv = runs
    d = rv[DEFAULT]
    fs = folds(d)
    assert [(f["from"], f["to"]) for f in fs] == [(0, 77), (439, 486)]
    first = fs[0]["cores"]["core0"]
    assert first["csr_writes"] == 12 and d.outputs.profile.cores["core0"].csr_writes == 18
    assert {e["block"] for e in lane(d, "core0")["csr"] if e["t"] < 77} == {"acc_streamer"}
    assert first["funcs"] == ["main", "forge_core_0"]
    assert fs[0]["cores"]["core1"]["funcs"] == ["main", "forge_core_1"]
    assert fs[1]["cores"]["core0"]["funcs"][-2:] == ["main", "forge_check"]
    assert min(b["from"] for b in lane(d, "dma")["bars"]) == 77


def test_the_run_is_serial_between_the_folds(runs):
    _, rv = runs
    d = rv[DEFAULT]
    t = tasks(d)
    assert (t["load_A"][0], t["load_B"][1]) == (77, 218)
    assert t["add_acc"] == (223, 355) and t["store_C"] == (367, 438)
    assert t["store_C"][1] - t["store_C"][0] == 71 and d.total_cycles - 439 == 47
    waits = [(w.block, w.first, w.last) for w in d.outputs.profile.cores["core0"].waits]
    assert ("barrier", 84, 221) in waits
    assert d.outputs.profile.cores["core0"].cycles["poll"] == 131
    assert d.outputs.profile.dmas["dma"].busy == 212


def test_read_data_comes_back_a_cycle_after_its_grant(runs):
    _, rv = runs
    reads = [e for e in rv[DEFAULT].events if e["k"] == "grant" and e["op"] == "r"
             and e["src"].startswith("acc_")]  # fmt: skip
    assert reads and {e["back"] - e["t"] for e in reads} == {1}


def test_the_stepping_keys_go_by_event_and_by_boundary(runs):
    _, rv = runs
    d = rv[DEFAULT]
    assert api.step_view(d, 300, 1, "event")["t"] == 301  # a grant every cycle in the task
    assert api.step_view(d, 300, 1, "boundary")["t"] == 352  # add_acc_a's done


def test_the_prediction_holds(runs):
    _, rv = runs
    d, w = rv[DEFAULT].outputs.profile, rv[WIDE].outputs.profile
    assert w.accelerators["acc"].firings == 512 // 8 == 64
    assert d.accelerators["acc"].firings == 128
    saved = d.accelerators["acc"].cycles["busy"] - w.accelerators["acc"].cycles["busy"]
    assert w.accelerators["acc"].cycles["busy"] < d.accelerators["acc"].cycles["busy"]
    assert 0 < d.total_cycles - w.total_cycles < saved
    assert (saved, d.total_cycles, w.total_cycles) == (64, 486, 432)
    t = text()
    i, j = t.index("## 5. Predict"), t.index("## 6. Rerun")
    assert "- firings 64 (N/W, was 128);" in t[i:j]  # written before the rerun's section


def test_where_the_54_cycles_went(runs):
    _, rv = runs
    d, w = tasks(rv[DEFAULT]), tasks(rv[WIDE])
    for copy in ("load_A", "load_B", "store_C"):  # the same cycles: 77, 126 and 71
        assert d[copy][1] - d[copy][0] == w[copy][1] - w[copy][0]
    assert [w[c][1] - w[c][0] for c in ("load_A", "load_B", "store_C")] == [77, 126, 71]
    acc = (d["add_acc"][1] - d["add_acc"][0]) - (w["add_acc"][1] - w["add_acc"][0])
    assert acc == 64 and 64 - (rv[DEFAULT].total_cycles - rv[WIDE].total_cycles) == 10
    assert d["load_A"][0] - w["load_A"][0] == 2  # the setup ends 2 cycles earlier
    assert (d["store_C"][0] - d["add_acc"][1], w["store_C"][0] - w["add_acc"][1]) == (12, 23)
    fs = [(f["from"], f["to"]) for f in folds(rv[WIDE])]
    assert (293, 312) in fs
    end = {k: v.total_cycles - folds(v)[-1]["from"] for k, v in rv.items() if k != "off"}
    assert end == {DEFAULT: 47, WIDE: 48}
