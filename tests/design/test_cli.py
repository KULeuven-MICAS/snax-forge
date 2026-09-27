"""The design command line (DP1a): run, check, save."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.design import Platform
from snax_forge.design import __main__ as cli

from .helpers import BOUND, FIXTURES, SMALL16


def run(*argv: str) -> int:
    return cli.main([str(a) for a in argv])


def test_run_writes_the_working_copy_and_prints_the_shell(tmp_path, capsys):
    out = tmp_path / "d"
    assert run(BOUND, "--platform", SMALL16, "--out", out) == 0
    w = Platform.load(out / "platform.json")
    assert (w.name, w.base, w.changes) == ("small16", "small16", {})
    text = capsys.readouterr().out
    assert "checks passed" in text and "acc_out      acc.out    writer  4 lanes [4]" in text
    assert text.rstrip().endswith("platform.json")


def test_run_with_sets_then_continue_from_the_working_copy(tmp_path):
    out = tmp_path / "d"
    sets = ["--set", "platform.l1.n_banks=32", "--set", "platform.streamers.acc_out.fifo_depth=4"]
    assert run(BOUND, "--platform", SMALL16, "--out", out, *sets) == 0
    more = ["--set", "platform.streamers.acc_a.temporal_dims=2"]
    assert run(BOUND, "--platform", out / "platform.json", "--out", out, *more) == 0
    d = json.loads((out / "platform.json").read_text())
    assert d["base"] == "small16" and d["l1"]["n_banks"] == 32
    assert list(d["changes"]) == [
        "l1.n_banks", "streamers.acc_out.fifo_depth", "streamers.acc_a.temporal_dims"
    ]  # fmt: skip
    assert d["streamers"] == {
        "default": {"temporal_dims": 1, "fifo_depth": 2, "addr_depth": 8, "prio": 0},
        "acc_out": {"fifo_depth": 4},
        "acc_a": {"temporal_dims": 2},
    }


def test_problems_exit_1_write_nothing_and_name_the_fix(tmp_path, capsys):
    out = tmp_path / "d"
    sets = ["--set", "platform.l1.n_banks=12", "--set", "platform.streamers.acc_ou.prio=1"]
    assert run(BOUND, "--platform", SMALL16, "--out", out, *sets) == 1
    assert not out.exists()
    err = capsys.readouterr().err
    assert "1 problem\n [platform.banks]" in err and "fix: --set platform.l1.n_banks=16" in err


def test_check_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "OUT", tmp_path / "out")
    assert run("check", BOUND, "--platform", SMALL16) == 0
    assert not (tmp_path / "out").exists() and "checks passed" in capsys.readouterr().out
    assert run("check", FIXTURES / "vecadd_split.snaxdfg", "--platform", SMALL16) == 1
    assert "[graph.unbound] tasklet add" in capsys.readouterr().err


def test_a_bad_set_argument_exits_2():
    with pytest.raises(SystemExit) as e:
        run(BOUND, "--platform", SMALL16, "--set", "l1.n_banks=32")
    assert e.value.code == 2


def test_the_design_name_is_the_sandbox_folder():
    assert cli.design_name(Path("out/sandbox/vecadd_w8/2_bind.snaxdfg")) == "vecadd_w8"
    assert cli.design_name(Path("tests/dfg/fixtures/vecadd_accelerated.snaxdfg")) == (
        "vecadd_accelerated"
    )


def test_save_keeps_a_working_copy_under_a_new_name(tmp_path, monkeypatch):
    out = tmp_path / "d"
    assert run(BOUND, "--platform", SMALL16, "--out", out, "--set", "platform.l1.rows=128") == 0
    monkeypatch.setattr(cli, "PLATFORMS", tmp_path / "platforms")
    assert run("save", out / "platform.json", "small16_r128") == 0
    saved = Platform.load(tmp_path / "platforms" / "small16_r128.json")
    assert (saved.name, saved.base, saved.changes) == ("small16_r128", "small16", {"l1.rows": 128})
    assert saved.l1.rows == 128
    assert run("save", out / "platform.json", "small16_r128") == 1  # exists
    assert run("save", out / "platform.json", "small16_r128", "--force") == 0
    assert run("save", out / "platform.json", tmp_path / "x" / "mine.json") == 0
    assert Platform.load(tmp_path / "x" / "mine.json").name == "mine"


def test_save_refuses_a_broken_platform(tmp_path, capsys):
    d = json.loads(SMALL16.read_text())
    d["l1"]["n_banks"] = 12
    src = tmp_path / "p.json"
    src.write_text(json.dumps(d))
    assert run("save", src, tmp_path / "q.json") == 1
    assert "[platform.banks]" in capsys.readouterr().err and not (tmp_path / "q.json").exists()
