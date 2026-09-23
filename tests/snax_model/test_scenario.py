"""Tests for the SNAX-MODEL scenario runner (MOD9, D41-D44).

MOD9 acceptance (docs/STATUS.md): the elementwise, reduce and DMA scenarios
run from the CLI; final memory checked against NumPy. Plus: the vecadd
scenario gives exactly the profile and trace of test_profile's hand-built
run, round trips, byte-identical output files, output files reload,
registration order kept, and every named error on a small broken scenario.

Sections:
  1. helpers
  2. checked-in scenarios from the CLI, against NumPy (and vecadd_conflict's
     pinned conflicts)
  3. vecadd equals the hand-built run
  4. round trips
  5. deterministic output files, and reloading them
  6. registration order
  7. registries
  8. named errors
"""

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_profile import run_vecadd

from snax_forge.snax_model import ControllerConfig, DmaPattern, SimulationTimeout
from snax_forge.snax_model.__main__ import main
from snax_forge.snax_model.profile import build_profile
from snax_forge.snax_model.scenario import (
    OPS,
    ClusterConfig,
    MemoryInitError,
    PortAttachError,
    Scenario,
    ScenarioError,
    UnknownAccelKind,
    UnknownBlockError,
    UnknownComponentKind,
    UnknownOpError,
    UnknownRegisterError,
    build,
    read_outputs,
    register_op,
    run,
    to_json,
    write_outputs,
)

REPO = Path(__file__).resolve().parents[2]
SCEN = REPO / "scenarios"
NAMES = ("vecadd", "vecadd_conflict", "vecadd_tiled", "reduce", "dma")
OUT_FILES = ("run.json", "profile.json", "trace.jsonl", "trace_meta.json", "l1.npy", "l2.npy")


# =============================================================================
# 1. Helpers
# =============================================================================


def _load_make():
    spec = importlib.util.spec_from_file_location("scenarios_make", SCEN / "make.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MAKE = _load_make()


def path_of(name):
    return SCEN / name / "scenario.json"


def load(name):
    return Scenario.load(path_of(name))


def inline_dict(name="vecadd"):
    """The scenario as a dict with its cluster inline, for breaking it in tests."""
    sc = load(name)
    d = sc.to_dict()
    d["cluster"] = sc.cluster.to_dict()
    return d


def from_dict(d, name="vecadd"):
    return Scenario.from_dict(d, base_dir=SCEN / name)


def npy(name, file):
    return np.load(SCEN / name / file)


def ctl_config(sc):
    """The controller's config dict in a scenario's cluster."""
    return next(c.config for c in sc.cluster.components if c.kind == "controller")


def cli(name, out, *extra):
    return main(["run", str(path_of(name)), "--out", str(out), *extra])


def files(out):
    return {f: (out / f).read_bytes() for f in OUT_FILES if (out / f).is_file()}


# =============================================================================
# 2. Checked-in scenarios from the CLI, against NumPy (MOD9 acceptance)
# =============================================================================


def test_vecadd_from_cli(tmp_path):
    assert cli("vecadd", tmp_path) == 0
    l2 = np.load(tmp_path / "l2.npy")
    a, b = npy("vecadd", "a.npy"), npy("vecadd", "b.npy")
    assert l2.shape == (4096, 1) and l2.dtype == np.int64
    assert np.array_equal(l2[256:320, 0], a + b)  # L2 byte 2048
    assert np.array_equal(l2[:64, 0], a) and np.array_equal(l2[128:192, 0], b)


def test_vecadd_tiled_from_cli(tmp_path):
    """Three tiles of 192 run one after another; c = a + b over all 576 elements.

    The cycle count is pinned from the generator (1-cycle writes and reads):
    the scenario exists to give the viewer a run of several hundred cycles.
    """
    assert cli("vecadd_tiled", tmp_path) == 0
    a, b = npy("vecadd_tiled", "a.npy"), npy("vecadd_tiled", "b.npy")
    l2 = np.load(tmp_path / "l2.npy")[:, 0]
    assert np.array_equal(l2[1152:1728], a + b)  # L2 byte 2 * 576 * 8
    prof = read_outputs(tmp_path).profile
    assert prof.total_cycles == 471
    assert not any(p.stalls for p in prof.ports.values())  # b is 8 banks after a
    assert prof.dmas["dma"].beats_written == 3 * (2 * 24 + 24)  # per tile: a, b in; c out


def test_reduce_from_cli(tmp_path):
    assert cli("reduce", tmp_path) == 0
    x = npy("reduce", "x.npy")
    l1 = np.load(tmp_path / "l1.npy")
    assert l1.shape == (1024, 1)
    assert np.array_equal(l1[128:132, 0], x.reshape(4, 16).sum(axis=1))
    assert not (tmp_path / "l2.npy").exists()  # red4 has no L2
    # The csr_read at the end is reported in run.json by name.
    reads = json.loads((tmp_path / "run.json").read_text())["reads"]
    assert [r["reg"] for r in reads] == ["acc.busy_cycles"] and reads[0]["value"] > 0


def test_dma_from_cli(tmp_path):
    assert cli("dma", tmp_path) == 0
    src = npy("dma", "src.npy").reshape(16, 8)  # 16 beats of 8 words
    l1, l2 = np.load(tmp_path / "l1.npy")[:, 0], np.load(tmp_path / "l2.npy")[:, 0]
    assert np.array_equal(l2[512:640], src.ravel())  # copied back to L2 byte 4096
    # In L1, beat k sits at the k-th address of the 2D pattern.
    for k, addr in enumerate(DmaPattern(0, (8, 2), (1024, 64)).addresses()):
        assert np.array_equal(l1[addr // 8 : addr // 8 + 8], src[k])


def test_vecadd_conflict_from_cli(tmp_path):
    """b in a's banks (VIS3's conflict case): only ra and rb collide, rb loses.

    Numbers pinned from the generator with 1-cycle writes and reads: 8
    cycles more than vecadd's 77 (rb's 7 stall cycles, plus one because the
    controller polls wr every 4 cycles), conflicts only on the banks a and b
    share while both readers run, every stall on rb's ports, and the data
    unchanged.
    """
    assert cli("vecadd_conflict", tmp_path) == 0
    a, b = npy("vecadd_conflict", "a.npy"), npy("vecadd_conflict", "b.npy")
    assert np.array_equal(np.load(tmp_path / "l2.npy")[256:320, 0], a + b)
    prof = read_outputs(tmp_path).profile
    assert prof.total_cycles == 85
    assert [i for i, c in enumerate(prof.banks.conflicts) if c] == [0, 1, 2, 3, 8, 9, 10, 11]
    stalls = {name: p.stalls for name, p in prof.ports.items() if p.stalls}
    assert stalls == {f"rb.{i}": 7 for i in range(4)}


def test_cli_as_a_process(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    cmd = [sys.executable, "-m", "snax_forge.snax_model", "run", str(path_of("vecadd")),
           "--out", str(tmp_path), "--trace", "beat", "--no-skip"]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO, check=False)
    assert r.returncode == 0, r.stderr
    assert "77 cycles (skip off, trace beat)" in r.stdout
    assert set(files(tmp_path)) == set(OUT_FILES)


def test_checked_in_files_equal_the_generator():
    for rel, data in MAKE.generate().items():
        assert (SCEN / rel).read_bytes() == data, f"scenarios/{rel} is stale: run make.py"


# =============================================================================
# 3. vecadd equals the hand-built run of test_profile, with the scenario's costs
# =============================================================================


@pytest.mark.parametrize("skip", [True, False])
@pytest.mark.parametrize("level", ["off", "task", "beat"])
def test_vecadd_equals_hand_built_run(level, skip):
    sc = load("vecadd")
    # The hand-built run gets the scenario's controller costs: test_profile keeps
    # its own non-default ones (VECADD_CFG) to exercise the D37 formulas.
    cfg = ControllerConfig.from_dict(ctl_config(sc))
    res = run(sc, skip_idle=skip, trace_level=level)
    ref = run_vecadd(mode="poll", skip=skip, level=None if level == "off" else level, cfg=cfg)
    assert res.total_cycles == ref["total"] == 77
    assert res.profile.to_dict() == build_profile(ref["cl"]).to_dict()
    if level == "off":
        assert res.trace is None and ref["trace"] is None
    else:
        assert res.trace.to_dict() == ref["trace"].to_dict()
    assert np.array_equal(res.l2, ref["l2"].dump(0, ref["l2"].cfg.n_words))


def test_named_and_raw_addresses_run_the_same():
    sc = load("vecadd")
    raw = copy.deepcopy(sc)
    b = build(sc)
    raw.program = [b.controller.program[i] for i in range(len(sc.program))]  # resolved CsrWrite
    assert all("addr" in c.to_dict() or c.to_dict()["op"] == "wait" for c in raw.program)
    assert (
        run(raw, trace_level="task").trace.to_dict() == run(sc, trace_level="task").trace.to_dict()
    )


# =============================================================================
# 4. Round trips
# =============================================================================


@pytest.mark.parametrize("name", NAMES)
def test_scenario_round_trip(name):
    sc = load(name)
    assert Scenario.from_dict(sc.to_dict(), base_dir=sc.base_dir) == sc
    through = json.loads(json.dumps(sc.to_dict()))
    assert Scenario.from_dict(through, base_dir=sc.base_dir) == sc
    # File level: loading and writing back gives the same bytes.
    assert to_json(sc.to_dict()) == path_of(name).read_text()
    cl_path = (SCEN / name / sc.cluster_ref).resolve()
    assert to_json(ClusterConfig.load(cl_path).to_dict()) == cl_path.read_text()


def test_inline_cluster_and_mixed_forms_round_trip():
    d = inline_dict()
    d["program"][0] = {"op": "csr_write", "addr": 3, "value": 0}  # raw form, same register
    d["program"].insert(1, {"op": "csr_read", "reg": "dma.busy"})
    d["memory"].append(
        {"mem": "l1", "addr": 4096, "random": {"seed": 1, "n": 8, "low": 0, "high": 9}}
    )
    d["memory"].append({"mem": "l1", "addr": 8000, "data": [1, 2, 3]})
    sc = from_dict(d)
    assert sc.cluster_ref is None
    assert sc.to_dict() == d  # each command and fill keeps its form
    assert from_dict(json.loads(to_json(d))) == sc
    res = run(sc)
    assert np.array_equal(res.l1[1000:1003, 0], [1, 2, 3])
    assert np.array_equal(res.l1[512:520, 0], np.random.default_rng(1).integers(0, 9, 8))


def test_defaults_may_be_left_out_of_a_file():
    d = inline_dict()
    d["cluster"]["l1"] = {"n_banks": 16, "rows": 64}
    for c in d["cluster"]["components"]:
        if c["kind"] == "streamer":
            c["config"] = {
                k: v for k, v in c["config"].items() if k in ("write", "n_ports", "fifo_depth")
            }
    assert run(from_dict(d)).total_cycles == 77


# =============================================================================
# 5. Deterministic output files, and reloading them
# =============================================================================


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("level", ["task", "beat"])
def test_output_files_byte_identical(tmp_path, name, level):
    runs = [("on", []), ("off", ["--no-skip"]), ("again", [])]
    got = []
    for tag, extra in runs:
        assert cli(name, tmp_path / tag, "--trace", level, *extra) == 0
        got.append(files(tmp_path / tag))
    assert got[0] == got[1] == got[2]
    assert {"run.json", "profile.json", "trace.jsonl", "trace_meta.json", "l1.npy"} <= set(got[0])


def test_output_files_reload(tmp_path):
    res = run(load("vecadd"), trace_level="beat")
    write_outputs(res, tmp_path)
    back = read_outputs(tmp_path)
    assert back.profile == res.profile
    assert back.trace.to_dict() == res.trace.to_dict()
    assert np.array_equal(back.l1, res.l1) and np.array_equal(back.l2, res.l2)
    assert back.run["total_cycles"] == 77 and back.run["trace_level"] == "beat"
    assert back.run["register_map"] == res.regmap.to_dict()
    assert "skip" not in json.dumps(back.run)  # nothing depends on the skip mode
    # One event per line.
    lines = (tmp_path / "trace.jsonl").read_text().splitlines()
    assert len(lines) == len(res.trace.events)


def test_trace_off_removes_stale_trace_files(tmp_path):
    assert cli("vecadd", tmp_path, "--trace", "task") == 0
    assert (tmp_path / "trace.jsonl").exists()
    assert cli("vecadd", tmp_path) == 0
    assert not (tmp_path / "trace.jsonl").exists() and not (tmp_path / "trace_meta.json").exists()
    assert read_outputs(tmp_path).trace is None


# =============================================================================
# 6. Registration order
# =============================================================================


def test_registration_order_is_file_order():
    sc = load("vecadd")
    b = build(sc, trace=None)
    order = [c.name for c in sc.cluster.components]
    assert [c.name for c in b.cluster] == order
    ports = [p.name for p in b.cluster["xbar"].ports]
    assert ports == ["dma.wide", *(f"{s}.{i}" for s in ("ra", "rb", "wr") for i in range(4))]
    sources = run(sc, trace_level="task").trace.sources
    assert [s for s in sources if not s.endswith(".fifo")] == order


def test_swapped_components_swap_ports_and_sources():
    d = inline_dict()
    comps = d["cluster"]["components"]
    comps[2], comps[3] = comps[3], comps[2]  # rb before ra
    sc = from_dict(d)
    res = run(sc, trace_level="task")
    assert [p.name for p in build(sc).cluster["xbar"].ports][1:5] == [f"rb.{i}" for i in range(4)]
    assert res.trace.sources.index("rb") < res.trace.sources.index("ra")
    a, b = npy("vecadd", "a.npy"), npy("vecadd", "b.npy")
    assert np.array_equal(res.l2[256:320, 0], a + b)


# =============================================================================
# 7. Registries
# =============================================================================


def test_registered_op_is_usable():
    register_op("absdiff", lambda x, y: np.abs(x - y))
    try:
        d = inline_dict()
        d["cluster"]["components"][5]["params"]["op"] = "absdiff"
        res = run(from_dict(d))
        a, b = npy("vecadd", "a.npy"), npy("vecadd", "b.npy")
        assert np.array_equal(res.l2[256:320, 0], np.abs(a - b))
    finally:
        del OPS["absdiff"]
    with pytest.raises(ValueError):
        register_op("add", np.add)


# =============================================================================
# 8. Named errors
# =============================================================================


def _comp(d, name):
    return next(c for c in d["cluster"]["components"] if c["name"] == name)


def _set(d, path, value):
    *head, last = path
    for k in head:
        d = d[k]
    d[last] = value


def _without_l2(d):
    """vecadd with the DMA and L2 removed but its L2 fills kept."""
    cl = d["cluster"]
    del cl["l2"]
    cl["components"] = [c for c in cl["components"] if c["kind"] != "dma"]
    cl["register_map"]["blocks"].remove("dma")
    d["program"] = []


BROKEN = {
    "component kind": (UnknownComponentKind, lambda d: _comp(d, "wr").update(kind="writer")),
    "accel kind": (UnknownAccelKind, lambda d: _comp(d, "acc").update(accel="conv")),
    "op": (UnknownOpError, lambda d: _comp(d, "acc")["params"].update(op="div")),
    "block in reg": (UnknownBlockError, lambda d: _set(d, ["program", 0, "reg"], "dmx.direction")),
    "register": (UnknownRegisterError, lambda d: _set(d, ["program", 0, "reg"], "dma.nope")),
    "raw address": (
        UnknownRegisterError,
        lambda d: _set(d, ["program", 0], {"op": "csr_write", "addr": 31, "value": 0}),
    ),
    "wait block": (UnknownBlockError, lambda d: _set(d, ["program", 23, "block"], "dmx")),
    "map block": (UnknownBlockError, lambda d: d["cluster"]["register_map"]["blocks"].append("x")),
    "port missing": (PortAttachError, lambda d: _comp(d, "acc")["attach"].pop("b")),
    "port unknown": (PortAttachError, lambda d: _comp(d, "acc")["attach"].update(c="rb")),
    "port target": (PortAttachError, lambda d: _comp(d, "acc")["attach"].update(b="dma")),
    "port side": (PortAttachError, lambda d: _comp(d, "acc")["attach"].update(b="wr")),
    "port lanes": (PortAttachError, lambda d: _comp(d, "acc")["params"].update(lanes=2)),
    "mem outside": (MemoryInitError, lambda d: _set(d, ["memory", 1, "addr"], 32768 - 8)),
    "mem misaligned": (MemoryInitError, lambda d: _set(d, ["memory", 1, "addr"], 1028)),
    "mem two sources": (MemoryInitError, lambda d: d["memory"][0].update(data=[1])),
    "mem npy missing": (MemoryInitError, lambda d: d["memory"][0].update(npy="nope.npy")),
    "mem no l2": (MemoryInitError, lambda d: _without_l2(d)),
    "unknown key": (ScenarioError, lambda d: _comp(d, "ra")["config"].update(fifo_dpeth=4)),
    "no controller": (ScenarioError, lambda d: d["cluster"]["components"].pop()),
}


@pytest.mark.parametrize("case", list(BROKEN))
def test_named_error(case):
    err, breaks = BROKEN[case]
    d = inline_dict()
    breaks(d)
    with pytest.raises(err):
        run(from_dict(d))
    assert issubclass(err, ScenarioError)


def test_program_that_does_not_finish(tmp_path):
    with pytest.raises(SimulationTimeout):
        run(load("vecadd"), max_cycles=50)
    d = inline_dict()
    d["max_cycles"] = 50  # the scenario's own limit
    with pytest.raises(SimulationTimeout):
        run(from_dict(d))
    assert cli("vecadd", tmp_path, "--max-cycles", "50") == 1


def test_cli_rejects_a_bad_scenario(tmp_path):
    d = inline_dict()
    _comp(d, "acc")["params"]["op"] = "div"
    d["memory"] = []  # the .npy paths are relative to scenarios/vecadd
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(d))
    assert main(["run", str(bad), "--out", str(tmp_path / "o")]) == 2
    assert main(["run", str(tmp_path / "missing.json"), "--out", str(tmp_path / "o")]) == 2
