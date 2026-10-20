"""run/outputs.py: a run directory written and read back."""

import json

import pytest

from snax_forge.run import Profile, RunError, Trace, read_run, write_run


def test_write_and_read(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"trace_level": "task"}))
    p = Profile(total_cycles=226, functional_check={"passed": True})
    t = Trace("task", 226, 2799, ["core0"], [
        {"t": 0, "k": "func", "src": "core0", "name": "main", "pc": 16},
        {"t": 3, "k": "state", "src": "core0", "mode": "run"},
    ])  # fmt: skip
    write_run(tmp_path, p, t)
    assert len((tmp_path / "trace.jsonl").read_text().splitlines()) == 2
    got = read_run(tmp_path)
    assert (got.run, got.profile, got.trace) == ({"trace_level": "task"}, p, t)


def test_no_trace_removes_an_earlier_one(tmp_path):
    (tmp_path / "run.json").write_text("{}")
    write_run(tmp_path, Profile(1), Trace("task", 1, 0, [], []))
    write_run(tmp_path, Profile(1), None)
    assert not (tmp_path / "trace.jsonl").exists() and not (tmp_path / "trace_meta.json").exists()
    assert read_run(tmp_path).trace is None


def test_not_a_run_directory(tmp_path):
    with pytest.raises(RunError) as e:
        read_run(tmp_path)
    assert e.value.code == "outputs.missing"
