"""The bundle for snax_cluster: gen-gvsoc's files at the root, the RTL under
``rtl/`` with its file list, and ``bundle.json``.

No JVM here: ``elaborate`` is replaced by a fake that writes two files and an ``emitted.json``
in Emit's form, and the checkout's commit is stubbed. The real ``Emit --spec`` runs in the
chisel job (``gen-rtl``) and by hand (``pixi run bundle``, docs/TOOL.md).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from snax_forge.design import DesignPoint
from snax_forge.gen import GenError, generate
from snax_forge.gen import bundle as bundle_mod
from snax_forge.gen import gvsoc as gvsoc_mod
from snax_forge.gen import rtl as rtl_mod
from snax_forge.gen.bundle import Bundle, write_bundle
from snax_forge.gen.rtl import EMITTED_JSON
from snax_forge.lower.derive import task_list

from .helpers import COMMIT, SV, bound_dot1, fake_emit, files, point, with_groups


@pytest.fixture
def fake(monkeypatch):
    return fake_emit(monkeypatch)


def test_the_layout(tmp_path, fake):
    out = tmp_path / "bundle"
    b, written = write_bundle(point(tmp_path), out, tmp_path)
    assert files(out) == sorted([
        "snax_forge.hjson", "snax_forge_vecadd_streamer_acc.hjson", "accelerator.json",
        "include/streamer_acc_csr_addr_map.h", "include/acc_csr.h", "include/snitch_cluster_cfg.h",
        "include/snitch_cluster_addrmap.h", "data.h", "main.c", "obs_marks.json",
        "manifest.json", *(f"rtl/{f}" for f in SV), "rtl/emitted.json",
        "rtl/files.f", "bundle.json",
    ])  # fmt: skip
    assert sorted(f.relative_to(out).as_posix() for f in written) == files(out)
    assert written[-3:] == [out / "rtl" / "emitted.json", out / "rtl" / "files.f",
                            out / "bundle.json"]  # fmt: skip
    assert b.tag == "snax_forge_vecadd"


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
    t = "snax_forge_vecadd"  # the tag, <accelerator_slot>_<design>
    assert d == {"design_point": "design/vecadd/design_point.json", "tag": t,
                 "top": f"{t}_shell_wrapper",
                 "modules": {f"{t}_csr": [f"{t}_peripherals"], f"{t}_ctrl": [f"{t}_acc_ctrl"],
                             f"{t}_datapath": [f"{t}_acc"]},
                 "snax_forge_commit": COMMIT, "dirty": False, "expected": None}  # fmt: skip
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
        f"bundle out/bundle/vecadd: snax_forge_vecadd_shell_wrapper from "
        f"design/vecadd/design_point.json, snax-forge {COMMIT}\n"
    )
    assert "wrote out/bundle/vecadd/rtl/files.f\n" in text
    assert text.endswith("wrote out/bundle/vecadd/bundle.json\nno forge_expected.txt: a bundle alone "
                         "has no SNAX-GVSoC run (pixi run forge writes it)\n")  # fmt: skip
    assert not (tmp_path / "out" / "bundle" / "vecadd" / "forge_expected.txt").exists()
    fake.listed = []
    assert bundle_mod.main([str(path), "--out", "elsewhere"]) == 1
    assert "[bundle.empty]" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        bundle_mod.main([])
    assert e.value.code == 2


def test_dot_1_encs_bundle_holds_both_groups(tmp_path, fake):
    """An accelerator of two groups (dot_1_enc): the bundle's accelerator.json, header and
    program are gen-gvsoc's; the RTL is elaborated from the same file."""
    from .helpers import bound_dot1_enc

    out = tmp_path / "bundle"
    b, _ = write_bundle(point(tmp_path, name="dot_1_enc", graph=bound_dot1_enc()), out, tmp_path)
    assert b.tag == "snax_forge_dot_1_enc" and "include/dot_csr.h" in files(out)
    acc = json.loads((out / "accelerator.json").read_text())
    assert [g["name"] for a in acc["accelerators"] for g in a["groups"]] == ["vmul", "vsum"]
    assert b.top == "snax_forge_dot_1_enc_shell_wrapper"


def test_dot_1s_bundle_holds_both_accelerators(tmp_path, fake):
    """dot 1, two accelerators: one shell, a header pair per set."""
    from .helpers import bound_dot1

    out = tmp_path / "bundle"
    b, _ = write_bundle(point(tmp_path, name="dot_1", graph=bound_dot1()), out, tmp_path)
    assert b.top == "snax_forge_dot_1_shell_wrapper"
    assert {"include/vmul_csr.h", "include/vsum_csr.h"} <= set(files(out))
    assert b.modules["snax_forge_dot_1_ctrl"] == ["snax_forge_dot_1_vmul_ctrl",
                                                  "snax_forge_dot_1_vsum_ctrl"]  # fmt: skip


def test_the_module_tree_names_every_group_unit_and_buffer():
    """From the description, not Emit, under the tag: dot_1_enc's two groups and units,
    and dot_serial's the same over its two accelerators (one shell); a linked group's units
    and its buffer (a wire is no module)."""
    from tests.gen.test_accelerator import dot, dot4, dot_serial

    t = "snax_forge_dot_1_enc"
    assert bundle_mod.module_tree(dot()) == {
        f"{t}_csr": [f"{t}_peripherals"], f"{t}_ctrl": [f"{t}_vmul_ctrl", f"{t}_vsum_ctrl"],
        f"{t}_datapath": [f"{t}_vmul", f"{t}_vsum"],
    }  # fmt: skip
    t = "snax_forge_dot_serial"
    assert bundle_mod.module_tree(dot_serial()) == {
        f"{t}_csr": [f"{t}_peripherals"], f"{t}_ctrl": [f"{t}_mul_ctrl", f"{t}_sum_ctrl"],
        f"{t}_datapath": [f"{t}_mul", f"{t}_sum"],
    }  # fmt: skip
    wire = dot4()
    t = "snax_forge_dot_4"
    assert bundle_mod.module_tree(wire)[f"{t}_datapath"] == [f"{t}_mac", f"{t}_sum"]
    g = wire.groups[0]
    buffer = with_groups(wire, [replace(g, links=[replace(g.links[0], depth=1)])])
    assert bundle_mod.module_tree(buffer) == {
        f"{t}_csr": [f"{t}_peripherals"], f"{t}_ctrl": [f"{t}_dot_ctrl"],
        f"{t}_datapath": [f"{t}_mac", f"{t}_sum", f"{t}_out_partial"],
    }  # fmt: skip


def test_the_expected_lines_and_their_refusal(tmp_path, fake):
    """write_expected keeps the run's counter lines in order, then its data line, and names the
    file in bundle.json; a run without a counter or with data errors is ``bundle.expected``."""
    out = tmp_path / "bundle"
    write_bundle(point(tmp_path), out, tmp_path)
    run = ("forge cycles: 304\nforge data: 0 errors (1 words)\nforge counter mult_mul: 19\n"
           "forge counter sum_sum: 19\nforge errors: 0\n")  # fmt: skip
    fills = {"mult_mul": 3, "sum_sum": 2}
    path = bundle_mod.write_expected(out, run, fills)
    assert path.read_text() == (
        "forge counter mult_mul: 19\nforge counter sum_sum: 19\nforge data: 0 errors (1 words)\n"
        "forge fill mult_mul: 3\nforge fill sum_sum: 2\n"
    )
    assert Bundle.load(out / "bundle.json").expected == "forge_expected.txt"
    for bad, fill in (
        ("forge data: 0 errors (1 words)\n", fills),
        ("forge counter a: 3\nforge data: 2 errors (8 words)\n", {"a": 3}),
        (run, {"mult_mul": 3}),  # sum_sum's counter has no fill
    ):
        with pytest.raises(GenError) as e:
            bundle_mod.write_expected(out, bad, fill)
        assert e.value.code == "bundle.expected"
    write_bundle(point(tmp_path), out, tmp_path)  # a bundle alone: the earlier run's file goes
    assert not path.exists() and Bundle.load(out / "bundle.json").expected is None


def test_each_counter_s_fill_is_its_group_s_from_the_estimate(tmp_path):
    """counter_fills: per accelerator task the estimate's fill of the group it runs, 3
    for vmul's two inputs and 2 for vsum's one, in dot 1's two accelerators."""
    path = point(tmp_path, name="dot_1", graph=bound_dot1())
    tasks = path.with_name("tasks.json")
    task_list(DesignPoint.load(path)).save(tasks)
    assert bundle_mod.counter_fills(path, tasks) == {"mult_vmul": 3, "sum_vsum": 2}
