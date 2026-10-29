"""``pixi run forge`` (flow/forge.py): the flow, the bundle and the bundle's program, each
step's output in ``out/forge/<name>/forge.log``, one line per step on the terminal.

SNAX-GVSoC is stubbed as in test_flow.py (helpers.stub), the bundle's ``Emit --spec`` as in
tests/gen (``fake_emit``, with ``tmp_path`` as the root), and the kernel's import is the vecadd fixture (no DaCe); the bundle's
``make sw`` gives the flow's ELF, or another one for the refusal.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from snax_forge.dfg import Graph
from snax_forge.flow import forge as forge_mod
from snax_forge.flow import run as flow_run
from snax_forge.gen import bundle as bundle_mod
from tests.design.helpers import REPO
from tests.gen.helpers import COMMIT, fake_emit, root

from .helpers import PLAINS, stub

RECIPE = REPO / "recipes" / "vecadd.json"


@pytest.fixture
def forged(tmp_path, monkeypatch):
    """Everything forge calls stubbed, its folders under ``tmp_path``; the fake make, whose
    ``elf`` is what the bundle's ``make sw`` writes (None: the flow's ELF)."""
    fake = stub(monkeypatch, tmp_path)
    fake_emit(monkeypatch)
    # the flow's folder stands for snax-forge's root, which a bundle's design point is inside
    write_bundle = bundle_mod.write_bundle
    monkeypatch.setattr(bundle_mod, "write_bundle",
                        lambda point, out: write_bundle(point, out, root(tmp_path)))  # fmt: skip
    imports = types.ModuleType("snax_forge.dfg.import_sdfg")
    imports.import_kernel = lambda kernel, dtypes=None: Graph.load(PLAINS[kernel])
    monkeypatch.setitem(sys.modules, "snax_forge.dfg.import_sdfg", imports)
    monkeypatch.setattr(flow_run, "OUT", tmp_path / "flow")
    monkeypatch.setattr(bundle_mod, "OUT", tmp_path / "bundle")
    monkeypatch.setattr(forge_mod, "OUT", tmp_path / "forge")
    flow_make = fake.__call__
    fake.elf = None

    def make(goal, variables, log, err=None, live=False):
        forge_dir = variables.get("FORGE", "")
        if goal != "sw" or not forge_dir.startswith(str(tmp_path / "bundle")):
            return flow_make(goal, variables, log, err, live)
        fake.calls.append((goal, dict(variables)))
        elf = Path(variables["FORGE_BUILD"]) / flow_run.ELF
        elf.parent.mkdir(parents=True)
        flows = tmp_path / "flow" / Path(forge_dir).name / "sw" / flow_run.ELF
        elf.write_bytes(fake.elf if fake.elf is not None else flows.read_bytes())
        log.write("make sw\n")  # forge.log, open: make writes straight into it
        return 0

    monkeypatch.setattr(flow_run, "run_gvsoc", make)
    return fake


def test_forge_prints_each_step_and_snax_clusters_commands(tmp_path, forged, capsys):
    assert forge_mod.main([str(RECIPE)]) == 0
    out = capsys.readouterr().out.splitlines()
    flow, bundle = tmp_path / "flow" / "vecadd", tmp_path / "bundle" / "vecadd"
    forge = tmp_path / "forge" / "vecadd"
    platform = REPO / "platforms" / "snax_forge.json"
    sha = json.loads((flow / "run" / "run.json").read_text())["elf_sha256"]
    assert out == [
        (
            f"forge vecadd: recipes/vecadd.json on platforms/snax_forge.json; every step's output "
            f"in {forge / 'forge.log'}"
        ),
        f"  flow      passed: design point {flow / 'design' / 'design_point.json'}, run {flow}/run/",
        f"  bundle    {bundle}/: snax_forge_shell_wrapper, snax-forge {COMMIT[:12]}",
        f"  sw        {forge / 'sw' / 'forge.elf'} from the bundle (sha256 {sha[:16]}, the flow's)",
        f"SNAX-GVSoC ({flow / 'run' / 'stdout.txt'}):",
        "  forge cycles: 226",
        "  forge data: 0 errors (64 words)",
        "  forge counter add_acc: 19",
        "  forge errors: 0",
        "snax_cluster on Verilator, from snax_cluster in its pixi environment:",
        f"  B={bundle / 'snax_forge.hjson'}",
        f"  E={forge / 'sw' / 'forge.elf'}",
        "  pixi run make -C target/snitch_cluster rtl-gen CFG_OVERRIDE=$B",
        "  pixi run make -C target/snitch_cluster snax-forge-sim SIM=vlt CFG_OVERRIDE=$B",
        "  pixi run make -C target/snitch_cluster snax-forge-run SIM=vlt CFG_OVERRIDE=$B ELF=$E",
        (
            "Questa: rtl-gen and vsim_preparation in snax_cluster's development image, then "
            "snax-forge-sim and snax-forge-run with SIM=vsim and the same B and E: docs/RUN.md "
            "section 8."
        ),
    ]
    # The log: each step's command and its whole output, as its own command line prints it
    log = (forge / "forge.log").read_text()
    lines = log.splitlines()
    assert lines[0] == f"== flow: {RECIPE} --platform {platform} --trace off"
    assert lines[1].startswith("  build     building SNAX-GVSoC: ")  # the live build's lines too
    assert lines[2:4] == [
        "make build",
        f"flow vecadd (vecadd: W=4, N=64; N=64) on snax_forge -> {flow}/",
    ]
    assert f"== bundle: {flow / 'design' / 'design_point.json'} --out {bundle}\n" in log
    assert f"bundle {bundle}: snax_forge_shell_wrapper from " in log
    assert f"wrote {bundle / 'bundle.json'}\n" in log
    assert (
        log.endswith("make sw\n") and not (forge / "sw.log").exists()
    )  # make's output in forge.log
    sw = forged.calls[-1][1]
    assert sw == {"APPS": "forge", "FORGE": str(bundle), "DESIGN": "vecadd",
                  "FORGE_BUILD": str(forge / "sw")}  # fmt: skip


def test_the_sets_and_the_trace_reach_the_flow_and_name_the_folders(tmp_path, forged, capsys):
    assert forge_mod.main([str(RECIPE), "--set", "N=128", "--trace", "task"]) == 0
    assert (tmp_path / "flow" / "vecadd_N128_task" / "run" / "trace.jsonl").is_file()
    assert (tmp_path / "bundle" / "vecadd_N128_task" / "bundle.json").is_file()
    log = (tmp_path / "forge" / "vecadd_N128_task" / "forge.log").read_text()
    assert log.splitlines()[0].endswith(" --set N=128 --trace task")


@pytest.mark.parametrize("failing", ["flow", "sw"])
def test_a_failing_step_is_named_with_its_logs_tail(tmp_path, forged, capsys, failing):
    if failing == "flow":
        forged.rc["run"] = 2
        forged.stderr["run"] = "make: *** [Makefile:1: run] Error 3\n"
    else:
        forged.elf = b"another ELF"
    assert forge_mod.main([str(RECIPE)]) == 1
    captured = capsys.readouterr()
    log_path = tmp_path / "forge" / "vecadd" / "forge.log"
    err = captured.err.splitlines()
    tail = log_path.read_text().splitlines()[-30:]
    assert err[0].startswith(f"forge vecadd FAILED at {failing}")
    assert err[0].endswith(f"; the last {len(tail)} lines of {log_path}:")
    assert err[1:] == tail
    if failing == "flow":
        assert "  bundle    " not in captured.out and "check     FAILED" in "\n".join(tail)
    else:
        assert err[0].endswith(f"the bundle's program is not the flow's; the last {len(tail)} "
                               f"lines of {log_path}:")  # fmt: skip
        assert "  bundle    " in captured.out and "  sw  " not in captured.out


@pytest.mark.parametrize("argv", [[str(RECIPE), "--set", "W"], [str(RECIPE), "--trace", "loud"],
                                  ["out/sandbox/vecadd/2_bind.snaxdfg"]])  # fmt: skip
def test_bad_arguments_exit_2(argv):
    with pytest.raises(SystemExit) as e:
        forge_mod.main(argv)
    assert e.value.code == 2
