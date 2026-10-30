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

    def run_forge(recipe, output):
        b = tmp_path / "out" / "bundle" / recipe
        b.mkdir(parents=True, exist_ok=True)
        (b / "forge_expected.txt").write_text(
            f"forge counter {recipe}: 19\nforge data: 0 errors (64 words)\n"
        )
        output.write_text(
            f"forge {recipe}: recipes/{recipe}.json on platforms/snax_forge.json\n"
            f"  bundle    out/bundle/{recipe}/: snax_forge_shell_wrapper, snax-forge 0123456789ab\n"
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
    dirs = [tmp_path / "out" / "bundle" / d for d in fs.QUICK]
    for d, b in zip(fs.QUICK, dirs, strict=True):
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
    assert fs.SETS["full"][: len(fs.QUICK)] == fs.QUICK and len(fs.SETS["full"]) == 13
