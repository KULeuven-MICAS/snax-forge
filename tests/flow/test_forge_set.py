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
    stubbed.add("dot_1")
    assert fs.main(["quick", "--sim", "vlt"]) == 1
    err = capsys.readouterr().err
    assert (
        "forge-set quick FAILED: dot_1 (exit 1), its output in out/forge-set/quick/dot_1.txt" in err
    )
    assert not (tmp_path / "out" / "forge-set" / "quick.md").exists()


def test_the_full_set_holds_the_quick_one():
    assert fs.SETS["full"][: len(fs.QUICK)] == fs.QUICK and len(fs.SETS["full"]) == 26
    assert fs.SETS["full"][-3] == fs.Design("relu") and fs.Design("relu") not in fs.QUICK


def test_the_pack_set_runs_every_narrow_recipe_once_at_i8():
    """The narrow types' set: each recipe at i8 and its smallest W, dot_serial at i32 only, and
    dot_3 and matvec_3 again at W = 32, dot_3 at i16 and i32, vecadd at W = 1 and matvec_3 at
    W = 4 (sliced inputs, D180); the names forge's folders take."""
    pack = fs.SETS["pack"]
    assert len(pack) == 30 and len({d.name for d in pack}) == 30
    assert pack[-2:] == [fs.Design("vecadd", ("types=i8", "W=1")),
                         fs.Design("matvec_3", ("types=i8", "W=4"))]  # fmt: skip
    assert fs.Design("matvec_3", ("types=i8", "W=8")) in fs.QUICK
    assert fs.Design("matvec_3", ("types=i8", "W=8")).name == "matvec_3_typesi8_W8"


def test_a_design_carries_its_own_sets():
    """rowsum_2 runs at its defaults (contiguous) and with the padded_rows layout: forge gets
    the --set, and the design is named as forge names its folders (flow/run.py's
    default_name)."""
    flat, padded = fs.SETS["full"][-2:]
    assert flat == fs.Design("rowsum_2") and flat.name == "rowsum_2"
    assert padded == fs.Design("rowsum_2", ("memory.passes.layout=padded_rows",))
    assert padded.name == "rowsum_2_passes.layoutpadded_rows"
    assert fs.Design("relu").name == "relu"


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
    assert "| rowsum_2 | " in text and "| rowsum_2_passes.layoutpadded_rows | " in text
    assert "  rowsum_2_passes.layoutpadded_rows passed" in capsys.readouterr().out


def test_every_sets_recipe_is_found_in_its_folder():
    """forge_set takes its names through ``find_recipe``: each set's recipes are in a folder of
    ``recipes/``, and the sets keep their names."""
    assert [d.recipe for d in fs.QUICK] == ["vecadd_i32", "vsum_tree_i8", "dot_1", "dot_1_enc",
                                            "dot_4", "axpy", "vsum_tree", "matvec_3",
                                            "vecadd_i32"]  # fmt: skip
    assert all(not d.sets for d in fs.QUICK[:6]) and all(d.sets == fs.I8 for d in fs.QUICK[6:8])
    assert fs.QUICK[8] == fs.Design("vecadd_i32", ("W=1",))  # sliced inputs (D180)
    for name in {d.recipe for names in fs.SETS.values() for d in names}:
        path = fs.find_recipe(name)
        assert path.parent.parent == REPO / "recipes" and path.stem == name


def test_the_mvt_set_is_its_own():
    """mvt at its defaults, 64 x 64: mvt_1, mvt_1_dot2 and mvt_1_dot2_shared, then mvt_2 and
    mvt_2_shared, each at W = 4 and 8, contiguous and padded_rows (mvt_2 at 8 has 84 crossbar
    inputs, above 64: a note), mvt_2_reuse at its defaults (the twin check), mvt_3 at its defaults (W = 8, contiguous), mvt_4 at W = 8 and 4,
    mvt_3_split at W = 8, contiguous and padded_rows, under each schedule of D179, and mvt_4,
    mvt_4_i64_dma_AT and mvt_4_i8_dma_AT (int8 in, int32 out) at W = 8, contiguous and
    padded_rows, sequential (D183); in neither quick nor full."""
    names = [d.name for d in fs.SETS["mvt"]]
    pad = "_passes.layoutpadded_rows"
    two = ["mvt_2", "mvt_2_shared"]
    assert names == [
        *(f"{r}{w}{p}" for r in ("mvt_1", "mvt_1_dot2", "mvt_1_dot2_shared")
          for w in ("", "_W8") for p in ("", pad)),
        *(f"{r}{w}{p}" for r in two for w in ("", "_W_x1-8_W_x2-8") for p in ("", pad)),
        "mvt_2_reuse",
        "mvt_3",
        *(f"mvt_4{w}{p}" for w in ("", "_W_x1-4") for p in ("", pad)),
        *(f"mvt_3_split{p}_schedule{s}" for p in ("", pad)
          for s in ("serial", "pipeline", "pipeline_compute")),
        *(f"{r}{p}_schedulesequential"
          for r in ("mvt_4", "mvt_4_i64_dma_AT", "mvt_4_i8_dma_AT") for p in ("", pad)),
    ]  # fmt: skip
    assert len(names) == 38
    assert not {d.recipe for d in fs.SETS["mvt"]} & {d.recipe for d in fs.SETS["full"]}
