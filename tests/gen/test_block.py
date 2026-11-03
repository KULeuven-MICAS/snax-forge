"""gen-block (``snax_forge/gen/block.py``): a catalogue block alone, checked against
``hw/chisel/blocks.json`` before ``Emit --block``. The refusals need no JVM; one java call is
checked by its command line (a fake ``subprocess.run``), and a real one runs where sbt is."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from snax_forge.gen import GenError
from snax_forge.gen import block as block_mod
from snax_forge.gen.block import check_entry, emit_blocks, main
from snax_forge.gen.rtl import EMITTED_JSON

TREE = {"inType": "int64", "outType": "int64", "lanes": 8, "op": "add"}
MAC = {"inTypes": ["int64", "int64"], "opType": "int64", "outType": "int64", "lanes": 1,
       "accumulate": True, "regOut": False}  # fmt: skip


def entry(tmp_path: Path, datapath: str = "TreeAccumulate", **params) -> dict:
    return {"datapath": datapath, "params": params or dict(TREE), "out": str(tmp_path / datapath)}


def test_a_good_entry_passes(tmp_path):
    check_entry(entry(tmp_path))
    check_entry(entry(tmp_path, "Mac", **MAC))


@pytest.mark.parametrize(
    ("code", "change"),
    [
        ("gen.block", lambda e: [e]),
        ("gen.block", lambda e: {k: v for k, v in e.items() if k != "out"}),
        ("gen.block", lambda e: {**e, "extra": 1}),
        ("gen.catalogue", lambda e: {**e, "datapath": "Adder"}),
        ("gen.params", lambda e: {**e, "params": {k: v for k, v in TREE.items() if k != "op"}}),
        ("gen.params", lambda e: {**e, "params": {**TREE, "T": 4}}),
        ("gen.params", lambda e: {**e, "params": {**TREE, "inType": "int128"}}),
        ("gen.params", lambda e: {**e, "params": {**TREE, "op": "mul"}}),
        ("gen.params", lambda e: {**e, "params": {**TREE, "lanes": 0}}),
        ("gen.params", lambda e: {**e, "params": {**TREE, "lanes": True}}),
    ],
    ids=["not-an-object", "no-out", "extra-key", "bad-name", "missing-param", "extra-param",
         "bad-dtype", "bad-op", "lanes-below-min", "lanes-a-bool"],
)  # fmt: skip
def test_an_entry_the_catalogue_refuses(tmp_path, code, change):
    with pytest.raises(GenError) as e:
        check_entry(change(entry(tmp_path)))
    assert e.value.code == code


def test_a_bad_entry_stops_the_list_before_java(tmp_path, monkeypatch):
    monkeypatch.setattr(block_mod, "emit_classpath", lambda: pytest.fail("java called"))
    bad = entry(tmp_path, "Mac", **{**MAC, "accumulate": 1})
    with pytest.raises(GenError) as e:
        emit_blocks([entry(tmp_path), bad])
    assert e.value.code == "gen.params"
    lst = tmp_path / "list.json"
    lst.write_text(json.dumps({"datapath": "Mac"}))
    assert main([str(lst)]) == 1


def test_every_entry_goes_in_one_java_call(tmp_path, monkeypatch):
    """One ``Emit --block NAME --params JSON --out DIR`` per entry, in order, in one call; each
    folder's emitted.json read back (here written by the fake)."""
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        args = cmd[cmd.index("snax.forge.Emit") + 1 :]
        for i in range(0, len(args), 6):
            out, name = Path(args[i + 5]), args[i + 1]
            (out / "sv").mkdir(parents=True)
            (out / "sv" / f"{name}.sv").write_text(f"module {name};\nendmodule\n")
            (out / EMITTED_JSON).write_text(json.dumps({"block": name, "params": json.loads(args[i + 3]),
                                                        "top": name, "modules": [name], "files": [f"sv/{name}.sv"]}))  # fmt: skip
        return type("Done", (), {"returncode": 0})()

    monkeypatch.setattr(block_mod, "emit_classpath", lambda: "cp")
    monkeypatch.setattr(block_mod.subprocess, "run", run)
    entries = [entry(tmp_path), entry(tmp_path, "Mac", **MAC)]
    listed = emit_blocks(entries)
    (cmd,) = calls
    assert cmd[cmd.index("snax.forge.Emit") + 1:] == [
        a for e in entries for a in ("--block", e["datapath"], "--params", json.dumps(e["params"]),
                                     "--out", str(Path(e["out"]).resolve()))]  # fmt: skip
    assert listed == [[Path(e["out"]) / "sv" / f"{e['datapath']}.sv", Path(e["out"]) / EMITTED_JSON]
                      for e in entries]  # fmt: skip


@pytest.mark.skipif(shutil.which("sbt") is None, reason="no sbt: the hw environment's")
def test_gen_block_writes_emitted_json(tmp_path, capsys):
    """The real call, where sbt is: TreeAccumulate and dot 5's Mac, each its emitted.json."""
    entries = [entry(tmp_path), entry(tmp_path, "Mac", **MAC)]
    lst = tmp_path / "list.json"
    lst.write_text(json.dumps(entries))
    assert main([str(lst)]) == 0
    printed = capsys.readouterr().out
    for e, top in zip(
        entries, ("TreeAccumulate_i64_i64_n8", "Mac_i64xi64_i64_i64_n1_acc"), strict=True
    ):
        emitted = json.loads((Path(e["out"]) / EMITTED_JSON).read_text())
        assert (emitted["block"], emitted["params"], emitted["top"]) == (
            e["datapath"],
            e["params"],
            top,
        )
        assert emitted["files"] == [f"sv/{top}.sv"]
        assert (
            "always @(posedge clock or posedge reset)"
            in (Path(e["out"]) / "sv" / f"{top}.sv").read_text()
        )
        assert f": {top}, 1 file(s) in sv/" in printed
