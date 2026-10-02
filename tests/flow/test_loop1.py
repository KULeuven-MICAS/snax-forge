"""LOOP1 (D98): every number examples/loop1/README.md states, checked on the runs it describes.

The README's two flow commands are run as written (from the repo root, into a
temporary folder named as the README names it). Each quoted table is marked
``<!-- excerpt: FOLDER/report/FILE -->``; every line of it must be a line of
that report. The claims made in prose are checked one by one below, in the
README's order.
"""

from __future__ import annotations

import shlex

import pytest

from snax_forge.flow import __main__ as cli
from snax_forge.flow import run_flow
from snax_forge.report import run as report_run
from snax_forge.viz import api
from tests.design.helpers import RECIPE, SMALL16

from .helpers import excerpts_hold

REPO = RECIPE.parents[1]
README = REPO / "examples" / "loop1" / "README.md"
DEFAULT, MOVED = "vecadd", "vecadd_B.l1.base576"
COMMANDS = {
    DEFAULT: "pixi run flow recipes/vecadd.json --platform platforms/small16.json --trace beat",
    MOVED: "pixi run flow recipes/vecadd.json --platform platforms/small16.json --trace beat "
    "--set memory.B.l1.base=576",
}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Both flows, run from the README's commands; folder -> (summary, RunView)."""
    root = tmp_path_factory.mktemp("loop1")
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(REPO)
        for folder, cmd in COMMANDS.items():
            args = shlex.split(cmd)[3:]  # after "pixi run flow"
            assert cli.main([*args, "--out", str(root / folder)]) == 0
            out[folder] = api.load_run(root / folder / "run", folder)
    return root, out


def text() -> str:
    return README.read_text()


def conflicts(rv) -> list[dict]:
    return api.conflicts_view(rv, 0, rv.outputs.run["total_cycles"])["conflicts"]


def read_at(rv, region: str, index: int, owner: str) -> int:
    """The cycle ``owner`` read element ``region[index]`` from L1."""
    (t,) = [
        h["t"]
        for h in api.journey_view(rv, region, index)["hops"]
        if h["mem"] == "l1" and h["act"] == "read" and h["owner"] == owner
    ]
    return t


def row_cells(rv, row: int) -> list:
    (l1,) = [m for m in api.memory_view(rv)["memories"] if m["mem"] == "l1"]
    (line,) = [x for x in l1["lines"] if x["kind"] == "row" and x["row"] == row]
    names = [r["name"] for r in l1["regions"]]
    return [[(names[i], e) for i, e in cell] for cell in line["cells"]]


# =============================================================================
# The commands, the folders and the excerpts
# =============================================================================


def test_the_readme_gives_both_commands_and_folders():
    t = text()
    for folder, cmd in COMMANDS.items():
        assert f"\n{cmd}\n" in t and f"`out/flow/{folder}/`" in t


def test_the_flows_get_the_folder_names_the_readme_uses(runs):
    root, _ = runs
    for folder in COMMANDS:
        log = (root / folder / "flow.log").read_text()
        assert log.startswith(f"flow {folder} (vecadd: W=4; N=64) on small16")


def test_every_excerpt_is_a_part_of_its_report(runs):
    root, _ = runs
    assert excerpts_hold(text(), root) == 16


# =============================================================================
# The claims in prose, in the README's order
# =============================================================================


def test_the_cycles_do_not_depend_on_the_trace_level(tmp_path):
    f = run_flow(RECIPE, SMALL16, out=tmp_path / "t", trace_level="task")
    assert f.result.total_cycles == 85


def test_the_held_back_cycles_listed_are_the_first_three():
    assert report_run.HELD_SHOWN == 3
    assert "(only the first three are listed per\nport)" in text()


def test_acc_a_pauses_and_acc_b_catches_up(runs):
    _, rv = runs
    d = rv[DEFAULT]
    # acc_a reads A[0..3] at 41 and A[4..7] at 42, then nothing at 43; A[8..11] at 44
    assert [read_at(d, "A", i, "acc_a") for i in (0, 4, 8)] == [41, 42, 44]
    # acc_b one cycle behind: B[0..3] at 42, B[4..7] at 43, then asks for B[8] at 44
    assert [read_at(d, "B", i, "acc_b") for i in (0, 4)] == [42, 43]
    assert conflicts(d)[0]["t"] == 44


def test_every_conflict_is_acc_b_waiting_for_the_same_element_acc_a_got(runs):
    _, rv = runs
    cs = conflicts(rv[DEFAULT])
    assert len(cs) == 28
    for c in cs:
        assert (c["waiting"]["owner"], c["served"]["owner"]) == ("acc_b", "acc_a")
        wi, si = c["waiting"]["element"], c["served"]["element"]
        assert wi.startswith("B[") and si.startswith("A[") and wi[1:] == si[1:]
    assert {c["bank"] for c in cs} == {0, 1, 2, 3, 8, 9, 10, 11}


def test_the_memory_tab_rows(runs):
    _, rv = runs
    assert row_cells(rv[DEFAULT], 4) == [[("B", i)] for i in range(16)]
    # A in rows 0-3 of the same banks
    assert row_cells(rv[DEFAULT], 0) == [[("A", i)] for i in range(16)]
    assert row_cells(rv[MOVED], 4) == [[]] * 8 + [[("B", i)] for i in range(8)]


def test_b_moves_8_banks_away_from_a(runs):
    _, rv = runs
    regions = {(r.name, r.mem): r for r in rv[MOVED].outputs.regions}
    a, b = regions["A", "l1"], regions["B", "l1"]
    for i in range(a.size):
        assert (b.address((i,)) // 8 - a.address((i,)) // 8) % 16 == 8


def test_the_prediction_holds(runs):
    _, rv = runs
    m = rv[MOVED]
    held = {
        k: v.outputs.profile.streamers["acc_b"].cycles.get("stall_xbar", 0) for k, v in rv.items()
    }
    assert held == {DEFAULT: 7, MOVED: 0}
    assert conflicts(m) == []
    assert m.outputs.run["total_cycles"] < 85


def test_acc_b_stays_within_one_beat_of_acc_a_after_the_move(runs):
    _, rv = runs
    m = rv[MOVED]
    for j in range(16):
        assert abs(read_at(m, "B", 4 * j, "acc_b") - read_at(m, "A", 4 * j, "acc_a")) <= 1


def test_where_the_cycles_went(runs):
    _, rv = runs
    total = {k: v.outputs.run["total_cycles"] for k, v in rv.items()}
    assert total[DEFAULT] - total[MOVED] == 8
    assert "where the 8 cycles went" in text() and "7 cycles shorter" in text()
