"""The design command line: run, check, save."""

from __future__ import annotations

import json

import pytest

from snax_forge.design import Platform
from snax_forge.design import __main__ as cli
from snax_forge.design.point import DesignPoint

from .helpers import BOUND, FIXTURES, SMALL16


def run(*argv: str) -> int:
    return cli.main([str(a) for a in argv])


@pytest.mark.parametrize("w", [1, 4])
def test_serial_accumulate_at_one_lane_is_a_note_not_a_refusal(tmp_path, capsys, w):
    """dot_v5 at W = 1 binds SerialAccumulate at lanes 1, the work of TreeAccumulate at lanes 1:
    the design step passes (exit 0, design_point.json written) and prints the note with its fix,
    and char.missing, since the table holds SerialAccumulate from lanes 2; at W = 4 it prints
    none."""
    from tests.gen.helpers import designed

    rel = designed(tmp_path, "dot_v5", f"dot_v5_w{w}", ["N=16", f"W={w}"])  # asserts exit 0
    out = capsys.readouterr().out
    note = ("note [design.serial_one_lane] sum: SerialAccumulate at lanes 1 reads one lane per "
            "beat, the same work as TreeAccumulate at lanes 1\n    fix: bind chisel_adder_tree")  # fmt: skip
    assert (tmp_path / rel).is_file()
    assert (note in out) == (w == 1)
    missing = "note [char.missing] sum: not characterized: SerialAccumulate at lanes 1 on nangate45"
    assert (missing in out) == (w == 1)
    assert out.count("note [") == (2 if w == 1 else 0)


def test_check_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "OUT", tmp_path / "out")
    assert run("check", BOUND, "--platform", SMALL16) == 0
    assert not (tmp_path / "out").exists() and "checks passed" in capsys.readouterr().out
    assert run("check", FIXTURES / "vecadd_split.snaxdfg", "--platform", SMALL16) == 1
    assert "[graph.unbound] tasklet add" in capsys.readouterr().err


def test_the_schedule_is_set_kept_and_checked(tmp_path, capsys):
    """--set design.schedule: kept in design_point.json; an unknown value or another design.
    path is a design error that names the schedules."""
    out = tmp_path / "d"
    assert run(BOUND, "--platform", SMALL16, "--out", out) == 0
    assert DesignPoint.load(out / "design_point.json").schedule == "serial"
    assert run(BOUND, "--platform", SMALL16, "--out", out, "--set", "design.schedule=pipeline") == 0
    assert DesignPoint.load(out / "design_point.json").schedule == "pipeline"
    capsys.readouterr()
    assert run(BOUND, "--platform", SMALL16, "--set", "design.schedule=overlap") == 1
    err = capsys.readouterr().err
    assert "[design.schedule] design.schedule: 'overlap' is no schedule; give one of serial, " \
        "pipeline, pipeline_compute, sequential" in err  # fmt: skip
    assert run(BOUND, "--platform", SMALL16, "--set", "design.order=pipeline") == 1
    assert "[design.schedule] design.order: no such field" in capsys.readouterr().err


@pytest.mark.parametrize("arg", ["l1.n_banks=32", "mem.A.l1.base=0", "platform.l1.n_banks"])
def test_a_bad_set_argument_exits_2(arg):
    with pytest.raises(SystemExit) as e:
        run(BOUND, "--platform", SMALL16, "--set", arg)
    assert e.value.code == 2


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


# =============================================================================
# Memory plan and design point
# =============================================================================
