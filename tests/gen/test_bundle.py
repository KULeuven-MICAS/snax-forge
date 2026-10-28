"""The bundle for snax_cluster (D111, D130, C§18): gen-gvsoc's files at the root, the RTL under
``rtl/`` with its file list, and ``bundle.json``.

No JVM here: ``elaborate`` is replaced by a fake that writes two files and an ``emitted.json``
in Emit's form, and the checkout's commit is stubbed. The real ``Emit --spec`` runs in the
chisel job (``gen-rtl``) and by hand (``pixi run bundle``, docs/RUN.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.gen import GenError, generate
from snax_forge.gen import bundle as bundle_mod
from snax_forge.gen import gvsoc as gvsoc_mod
from snax_forge.gen import rtl as rtl_mod
from snax_forge.gen.bundle import Bundle, write_bundle
from snax_forge.gen.rtl import EMITTED_JSON

from .helpers import COMMIT, SV, fake_emit, files, point


@pytest.fixture
def fake(monkeypatch):
    return fake_emit(monkeypatch)


def test_the_layout(tmp_path, fake):
    out = tmp_path / "bundle"
    b, written = write_bundle(point(tmp_path), out, tmp_path)
    assert files(out) == sorted([
        "snax_forge.hjson", "snax_forge_streamer.hjson", "accelerator.json",
        "include/streamer_csr_addr_map.h", "include/acc_csr.h", "include/snitch_cluster_cfg.h",
        "include/snitch_cluster_addrmap.h", "data.h", "main.c",
        "manifest.json", *(f"rtl/{f}" for f in SV), "rtl/emitted.json",
        "rtl/files.f", "bundle.json",
    ])  # fmt: skip
    assert sorted(f.relative_to(out).as_posix() for f in written) == files(out)
    assert written[-3:] == [out / "rtl" / "emitted.json", out / "rtl" / "files.f",
                            out / "bundle.json"]  # fmt: skip
    assert b.tag == "snax_forge"


def test_the_root_is_gen_gvsocs_output(tmp_path, fake):
    """Byte for byte what ``generate`` writes for the design point, the cluster file's slot
    naming the bundle's file list."""
    path = point(tmp_path)
    out, alone = tmp_path / "bundle", tmp_path / "gvsoc"
    write_bundle(path, out, tmp_path)
    m = generate(path, alone, tmp_path)
    for f in m.files:
        assert (out / f).read_bytes() == (alone / f).read_bytes(), f
    assert '"snax_forge_rtl": "rtl/files.f"' in (out / "snax_forge.hjson").read_text()
    assert gvsoc_mod.RTL_FILE_LIST == "rtl/files.f"


def test_rtl_is_made_again_and_the_rest_left(tmp_path, fake):
    """A file an earlier run left in rtl/sv/ is gone; a file of the user's beside rtl/ stays."""
    out = tmp_path / "bundle"
    (out / "rtl" / "sv").mkdir(parents=True)
    (out / "rtl" / "sv" / "old.sv").write_text("module old;\nendmodule\n")
    (out / "notes.txt").write_text("mine\n")
    write_bundle(point(tmp_path), out, tmp_path)
    assert not (out / "rtl" / "sv" / "old.sv").exists()
    assert sorted(f.name for f in (out / "rtl" / "sv").iterdir()) == sorted(
        Path(f).name for f in SV
    )
    assert (out / "notes.txt").read_text() == "mine\n"


def test_files_f_is_emitted_jsons_files_in_order(tmp_path, fake):
    out = tmp_path / "bundle"
    write_bundle(point(tmp_path), out, tmp_path)
    assert (out / "rtl" / "files.f").read_text() == "".join(f"{f}\n" for f in SV)
    assert all((out / "rtl" / f).is_file() for f in SV)


def test_bundle_json(tmp_path, fake):
    out = tmp_path / "bundle"
    b, _ = write_bundle(point(tmp_path), out, tmp_path)
    d = json.loads((out / "bundle.json").read_text())
    assert d == {"design_point": "design/vecadd/design_point.json", "tag": "snax_forge",
                 "snax_forge_commit": COMMIT, "dirty": False}  # fmt: skip
    assert Bundle.load(out / "bundle.json") == b == Bundle.from_dict(b.to_dict())
    assert d["design_point"] == json.loads((out / "manifest.json").read_text())["design_point"]


def test_provenance_outside_git_and_in_this_checkout(tmp_path):
    assert bundle_mod.provenance(tmp_path) == (None, False)
    commit, dirty = bundle_mod.provenance(gvsoc_mod.ROOT)
    assert commit is None or len(commit) == 40
    assert isinstance(dirty, bool)


@pytest.mark.parametrize(
    ("listed", "code"),
    [
        ([], "bundle.empty"),
        ([EMITTED_JSON], "bundle.empty"),
    ],
)
def test_a_file_list_that_cannot_be_written(tmp_path, fake, listed, code):
    fake.listed = listed
    out = tmp_path / "bundle"
    with pytest.raises(GenError) as e:
        write_bundle(point(tmp_path), out, tmp_path)
    assert e.value.code == code
    assert not (out / "rtl" / "files.f").exists() and not (out / "bundle.json").exists()


def test_the_command(tmp_path, fake, monkeypatch, capsys):
    monkeypatch.setattr(gvsoc_mod, "ROOT", tmp_path)
    monkeypatch.setattr(rtl_mod, "ROOT", tmp_path)
    monkeypatch.setattr(bundle_mod, "OUT", tmp_path / "out" / "bundle")
    path = point(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert bundle_mod.main([str(path)]) == 0
    text = capsys.readouterr().out
    assert text.startswith(
        f"bundle out/bundle/vecadd: snax_forge_shell_wrapper from "
        f"design/vecadd/design_point.json, snax-forge {COMMIT}\n"
    )
    assert "wrote out/bundle/vecadd/rtl/files.f\n" in text
    assert text.endswith("wrote out/bundle/vecadd/bundle.json\n")
    fake.listed = []
    assert bundle_mod.main([str(path), "--out", "elsewhere"]) == 1
    assert "[bundle.empty]" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        bundle_mod.main([])
    assert e.value.code == 2


def test_dot_1s_bundle_holds_both_groups(tmp_path, fake):
    """An accelerator of two groups (D148): the bundle's accelerator.json, header and program
    are gen-gvsoc's; the RTL is elaborated from the same file."""
    from .helpers import bound_dot1

    out = tmp_path / "bundle"
    b, _ = write_bundle(point(tmp_path, name="dot_1", graph=bound_dot1()), out, tmp_path)
    assert b.tag == "snax_forge" and "include/dot_csr.h" in files(out)
    acc = json.loads((out / "accelerator.json").read_text())
    assert [g["name"] for g in acc["groups"]] == ["mul", "sum"]
