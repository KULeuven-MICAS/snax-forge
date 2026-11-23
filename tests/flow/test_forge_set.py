"""``pixi run forge-set`` (flow/forge_set.py): a set's forges, then ``out/forge-set/<set>.md``
with each bundle's expected lines and the one line that checks them all in snax_cluster.

Each forge is stubbed (``run_forge``): it writes the terminal output forge prints and a bundle
folder with its ``forge_expected.txt``, as forge does.
"""

from __future__ import annotations

import importlib.util

import pytest

from snax_forge.flow import forge_set as fs
from tests.design.helpers import REPO

_spec = importlib.util.spec_from_file_location("md_reflow", REPO / "util" / "md_reflow.py")
md_reflow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(md_reflow)


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    """forge_set with its folders under ``tmp_path`` and a fake forge; ``fail`` the designs whose
    forge exits 1."""
    monkeypatch.setattr(fs, "ROOT", tmp_path)
    monkeypatch.setattr(fs, "OUT", tmp_path / "out" / "forge-set")
    monkeypatch.setattr(fs, "warm_emit", lambda: None)
    monkeypatch.chdir(tmp_path)
    fail: set[str] = set()

    def run_forge(design, output):
        recipe = design.name
        b = tmp_path / "out" / "bundle" / recipe
        b.mkdir(parents=True, exist_ok=True)
        (b / "forge_expected.txt").write_text(
            f"forge counter {recipe}: 19\nforge data: 0 errors (64 words)\n"
        )
        output.write_text(
            f"forge {recipe}: {fs.find_recipe(design.recipe).relative_to(REPO)} on platforms/snax_forge.json\n"
            f"  bundle    out/bundle/{recipe}/: snax_forge_{recipe}_shell_wrapper, snax-forge 0123456789ab\n"
            "snax_cluster on Verilator, from snax_cluster in its pixi environment:\n"
            f"  B={b}/snax_forge.hjson\n"
            "  pixi run make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=$B\n"
        )
        return 1 if recipe in fail else 0

    monkeypatch.setattr(fs, "run_forge", run_forge)
    return fail


def test_the_quick_set_writes_its_table_and_the_check_line(tmp_path, stubbed, capsys):
    assert fs.main(["quick", "-j", "2"]) == 0
    text = (tmp_path / "out" / "forge-set" / "quick.md").read_text()
    names = [d.name for d in fs.QUICK]
    dirs = [tmp_path / "out" / "bundle" / d for d in names]
    for d, b in zip(names, dirs, strict=True):
        assert (
            f"| {d} | `{b}` | `forge counter {d}: 19`<br>`forge data: 0 errors (64 words)` |"
            in text
        )
        assert f"### {d}\n\n```\nsnax_cluster on Verilator" in text
    line = (
        f'make snax-forge-check BUNDLES="{" ".join(map(str, dirs))}" SIM=vsim '
        "SNAX_CLUSTER_IMAGE=<image>"
    )
    assert f"```\n{line}\n```" in text
    assert fs.IMAGE_HINT in text
    assert capsys.readouterr().out.endswith(f"  {line}\n  {fs.IMAGE_HINT}\n")
    assert md_reflow.reflow(text.splitlines())[1] == []  # one line per paragraph already


def test_a_failing_forge_stops_the_set(tmp_path, stubbed, capsys):
    stubbed.add("dot_v1")
    assert fs.main(["quick", "--sim", "vlt"]) == 1
    err = capsys.readouterr().err
    assert (
        "forge-set quick FAILED: dot_v1 (exit 1), its output in out/forge-set/quick/dot_v1.txt"
        in err
    )
    assert not (tmp_path / "out" / "forge-set" / "quick.md").exists()


def test_the_full_set_holds_the_quick_one():
    assert fs.SETS["full"][: len(fs.QUICK)] == fs.QUICK and len(fs.SETS["full"]) == 26
    assert fs.SETS["full"][-3] == fs.Design("relu_v1") and fs.Design("relu_v1") not in fs.QUICK


def test_the_pack_set_runs_every_narrow_recipe_once_at_i8():
    """The narrow types' set: each recipe at i8 and its smallest W, dot_v7 at i32 only, and
    dot_v3 and matvec_v3 again at W = 32, dot_v3 at i16 and i32, vecadd at W = 1 and matvec_v3 at
    W = 4 (sliced inputs, D180); the names forge's folders take."""
    pack = fs.SETS["pack"]
    assert len(pack) == 30 and len({d.name for d in pack}) == 30
    assert pack[-2:] == [fs.Design("vecadd_v1", ("types=i8", "W=1")),
                         fs.Design("matvec_v3", ("types=i8", "W=4"))]  # fmt: skip
    assert fs.Design("matvec_v3", ("types=i8", "W=8")) in fs.QUICK
    assert fs.Design("matvec_v3", ("types=i8", "W=8")).name == "matvec_v3_typesi8_W8"


def test_a_design_carries_its_own_sets():
    """rowsum_v2 runs at its defaults (contiguous) and with the padded_rows layout: forge gets
    the --set, and the design is named as forge names its folders (flow/run.py's
    default_name)."""
    flat, padded = fs.SETS["full"][-2:]
    assert flat == fs.Design("rowsum_v2") and flat.name == "rowsum_v2"
    assert padded == fs.Design("rowsum_v2", ("memory.passes.layout=padded_rows",))
    assert padded.name == "rowsum_v2_passes.layoutpadded_rows"
    assert fs.Design("relu_v1").name == "relu_v1"


def test_the_full_set_passes_each_designs_sets_to_forge(tmp_path, stubbed, monkeypatch, capsys):
    calls = []
    stub = fs.run_forge

    def recorded(design, output):
        calls.append(design)
        return stub(design, output)

    monkeypatch.setattr(fs, "run_forge", recorded)
    assert fs.main(["full", "-j", "4"]) == 0
    assert sorted(d.name for d in calls) == sorted(d.name for d in fs.SETS["full"])
    text = (tmp_path / "out" / "forge-set" / "full.md").read_text()
    assert "| rowsum_v2 | " in text and "| rowsum_v2_passes.layoutpadded_rows | " in text
    assert "  rowsum_v2_passes.layoutpadded_rows passed" in capsys.readouterr().out


def test_every_sets_recipe_is_found_in_its_folder():
    """forge_set takes its names through ``find_recipe``: each set's recipes are in a folder of
    ``recipes/``, and the sets keep their names."""
    assert [d.recipe for d in fs.QUICK] == ["vecadd_v1_1", "vsum_v1_1_2", "dot_v1", "dot_v1_1",
                                            "dot_v4", "axpy_v1", "vsum_v1_1", "matvec_v3",
                                            "vecadd_v1_1"]  # fmt: skip
    assert all(not d.sets for d in fs.QUICK[:6]) and all(d.sets == fs.I8 for d in fs.QUICK[6:8])
    assert fs.QUICK[8] == fs.Design("vecadd_v1_1", ("W=1",))  # sliced inputs (D180)
    for name in {d.recipe for names in fs.SETS.values() for d in names}:
        path = fs.find_recipe(name)
        assert path.parent.parent == REPO / "recipes" and path.stem == name


def test_the_mvt_set_is_its_own():
    """mvt at its defaults, 64 x 64: mvt_1, mvt_v1_2 and mvt_v1_2_2, then mvt_v2 and
    mvt_v2_2, each at W = 4 and 8, contiguous and padded_rows (mvt_v2 at 8 has 84 crossbar
    inputs, above 64: a note), mvt_v2_1 at its defaults (the twin check), mvt_v3 at its defaults (W = 8, contiguous), mvt_v4 at W = 8 and 4,
    mvt_v3_1 at W = 8, contiguous and padded_rows, under each schedule of D179, and mvt_v4,
    mvt_v4_1 and mvt_v4_1_1 (int8 in, int32 out) at W = 8, contiguous and
    padded_rows, sequential (D183), and mvt_v5 (half 2 a seeded Mac, D186 · D187) at its
    defaults, sequential contiguous and padded_rows and pipeline contiguous, and
    mvt_v5_2 (half 2 an unseeded Mac linked into a vadd) sequential contiguous; in
    neither quick nor full."""
    names = [d.name for d in fs.SETS["mvt"]]
    pad = "_passes.layoutpadded_rows"
    two = ["mvt_v2", "mvt_v2_2"]
    assert names == [
        *(f"{r}{w}{p}" for r in ("mvt_v1", "mvt_v1_2", "mvt_v1_2_2")
          for w in ("", "_W8") for p in ("", pad)),
        *(f"{r}{w}{p}" for r in two for w in ("", "_W_x1-8_W_x2-8") for p in ("", pad)),
        "mvt_v2_1",
        "mvt_v3",
        *(f"mvt_v4{w}{p}" for w in ("", "_W_x1-4") for p in ("", pad)),
        *(f"mvt_v3_1{p}_schedule{s}" for p in ("", pad)
          for s in ("serial", "pipeline", "pipeline_compute")),
        *(f"{r}{p}_schedulesequential"
          for r in ("mvt_v4", "mvt_v4_1", "mvt_v4_1_1") for p in ("", pad)),
        "mvt_v5_schedulesequential", f"mvt_v5{pad}_schedulesequential",
        "mvt_v5_schedulepipeline",
        "mvt_v5_2_schedulesequential",
    ]  # fmt: skip
    assert len(names) == 42
    assert not {d.recipe for d in fs.SETS["mvt"]} & {d.recipe for d in fs.SETS["full"]}


def test_the_bicg_set_is_its_own():
    """bicg at 64 x 64, W = 8: bicg_v1 contiguous and padded_rows, bicg_v2 (and padded_rows
    under pipeline), and bicg_v2_1 (int8 in, int32 out) at its defaults and at 40 x 64, each
    sequential and pipeline; the one pass bicg_v3 (64 x 32, W = 32) and bicg_v2_1 at its size
    under pipeline; the one pass with the bank, bicg_v3_1, at 64 x 64 and 40 x 64 (W = 4,
    R = 16) and at W = 8, R = 8; in no other set."""
    names = [d.name for d in fs.SETS["bicg"]]
    pad = "_passes.layoutpadded_rows"
    assert names == [
        *(f"bicg_v1{p}_schedule{s}" for p in ("", pad) for s in ("sequential", "pipeline")),
        *(f"bicg_v2_schedule{s}" for s in ("sequential", "pipeline")),
        f"bicg_v2{pad}_schedulepipeline",
        *(f"bicg_v2_1{m}_schedule{s}" for m in ("", "_M40") for s in ("sequential", "pipeline")),
        "bicg_v2_1_N32_W32_schedulepipeline", "bicg_v3",
        "bicg_v3_1", "bicg_v3_1_M40", "bicg_v3_1_W8_R8",
    ]  # fmt: skip
    others = {d.recipe for k, ds in fs.SETS.items() if k != "bicg" for d in ds}
    assert not {d.recipe for d in fs.SETS["bicg"]} & others
