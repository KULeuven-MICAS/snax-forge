"""Tests for the skipping machinery of SNAX-MODEL (MOD10, D29, D40, D47).

A component that sleeps wrongly does not crash: it reports wrong statistics.
These tests are the guard against that, for the cases the per-module tests do
not reach. The rules they check are written down in docs/CONTRACTS.md
(section 8, R1-R5).

Sections:
  1. helpers
  2. whole-output equivalence with skipping on and off, over random scenarios
  3. next_wake asked twice gives the same answer (R1)
  4. DMA class intervals over a timing sweep (R3)
  5. gaps at the edges: idle tail, never started, zero cycles, zero work
  6. small gaps found while writing this: named registers, default blocks
"""

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest
from helpers import assert_cycles_add_up, build, contiguous

from snax_forge.snax_model import (
    CsrRead,
    CsrWrite,
    Dma,
    DmaConfig,
    DmaDescriptor,
    DmaPattern,
    L1Config,
    L2Config,
    SimulationError,
    Streamer,
    StreamerConfig,
    Trace,
    Wait,
)
from snax_forge.snax_model.profile import build_profile
from snax_forge.snax_model.scenario import (
    ClusterConfig,
    ComponentSpec,
    MemInit,
    NamedRead,
    NamedWrite,
    RegisterMapSpec,
    Scenario,
    named,
    register_map_of,
    run,
    write_outputs,
)
from snax_forge.snax_model.scenario import (
    build as build_scenario,
)

REPO = Path(__file__).resolve().parents[2]


def _load_make():
    spec = importlib.util.spec_from_file_location("scenarios_make", REPO / "scenarios" / "make.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MAKE = _load_make()
WORD, BEAT, LANES = MAKE.WORD, MAKE.BEAT, MAKE.LANES


# =============================================================================
# 1. Helpers
# =============================================================================


def random_scenario(seed):
    """A vecadd scenario with random sizes, latencies, depths and wait modes.

    Same shape as scenarios/vecadd (DMA in, compute, DMA out) so it is always
    a valid program, but every number that changes when a component sleeps is
    drawn: L1 read latency, FIFO depth, DMA timing, controller costs, the
    number of beats and poll vs signal. The cluster is inline and the memory
    is a seeded random fill, so the scenario needs no files.
    """
    r = random.Random(seed)
    cl = MAKE.alu4()
    cl.cluster_ref = None
    cl.l1 = L1Config(n_banks=16, rows=64, read_latency=r.choice([0, 1, 2]))
    depth = r.choice([1, 2, 4])
    for spec in cl.components:
        if spec.kind == "streamer":
            spec.config["fifo_depth"] = depth
        elif spec.kind == "dma":
            spec.config |= {
                "startup": r.choice([1, 2, 4]),
                "beat_interval": r.choice([1, 2, 3]),
                "l1_read_extra": r.choice([0, 2]),
                "done_latency": r.choice([0, 3]),
            }
        elif spec.kind == "controller":
            spec.config |= {
                "read_cost": r.choice([1, 2]),
                "poll_interval": r.choice([2, 4]),
                "signal_latency": r.choice([1, 3]),
            }

    n_elems = r.choice([8, 16, 32])  # multiple of 8: whole wide beats
    nb, n_dma = n_elems // LANES, n_elems // 8
    wa, wb, wc = 0, 72, 144  # L1 word addresses, beat aligned
    l2a, l2b, l2c = 0, 1024, 2048

    def mode():
        return r.choice(["poll", "signal"])

    p = MAKE.Program(cl)
    for src, word in ((l2a, wa), (l2b, wb)):
        p.config("dma", DmaDescriptor("l2_to_l1", contiguous(src, n_dma),
                                      contiguous(word * WORD, n_dma)))  # fmt: skip
        p.start("dma")
        p.wait("dma", mode())
    for blk, word in (("ra", wa), ("rb", wb), ("wr", wc)):
        p.config(blk, MAKE.unit(word, nb, LANES))
    p.config("acc", {"n": nb})
    for blk in ("ra", "rb", "wr", "acc"):
        p.start(blk)
    p.wait("wr", mode())
    p.config("dma", DmaDescriptor("l1_to_l2", contiguous(wc * WORD, n_dma),
                                  contiguous(l2c, n_dma)))  # fmt: skip
    p.start("dma")
    p.wait("dma", mode())
    p.read("acc.busy_cycles")
    fill = {"low": -1000, "high": 1000, "n": n_elems}
    return Scenario(
        name=f"random{seed}",
        cluster=cl,
        memory=[
            MemInit("l2", l2a, random={"seed": seed, **fill}),
            MemInit("l2", l2b, random={"seed": seed + 100, **fill}),
        ],
        program=p.cmds,
        max_cycles=20000,
    )


def output_bytes(sc, skip, level, out_dir):
    """Every file the run writes, by name: what two runs must agree on (D44)."""
    d = write_outputs(run(sc, skip_idle=skip, trace_level=level), out_dir)
    return {f.name: f.read_bytes() for f in sorted(d.iterdir())}


# =============================================================================
# 2. Whole-output equivalence, over random scenarios
# =============================================================================


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("level", ["task", "beat"])
def test_random_scenario_skip_on_off_identical(seed, level, tmp_path):
    """Skipping changes nothing that is written (D29, D38, D40, D44).

    The per-module random tests compare chosen counters; this compares every
    output file byte for byte, which is what a later component (a view, the
    anchor, cosim) actually reads.
    """
    sc = random_scenario(seed)
    on = output_bytes(sc, True, level, tmp_path / "on")
    off = output_bytes(sc, False, level, tmp_path / "off")
    assert list(on) == list(off)
    for name in on:
        assert on[name] == off[name], name


# =============================================================================
# 3. next_wake asked twice gives the same answer (R1)
# =============================================================================


def _ask_twice(cluster):
    """Wrap every component's next_wake so that each call asks the real one twice.

    The xbar and the streamers ask their neighbours' next_wake as well, so
    the wrapper also covers those calls. Any answer that depends on when (or
    how often) it is asked fails here instead of silently changing a gap.
    """
    for comp in list(cluster):
        original = comp.next_wake

        def wrapped(cycle, _f=original, _n=comp.name):
            first = _f(cycle)
            second = _f(cycle)
            assert first == second, f"{_n}.next_wake({cycle}): {first} then {second}"
            return second

        comp.next_wake = wrapped


@pytest.mark.parametrize("seed", range(4))
def test_next_wake_is_idempotent(seed, tmp_path):
    """Asking twice per cycle must not change the answer or the run (R1)."""
    sc = random_scenario(seed)
    plain = output_bytes(sc, True, "beat", tmp_path / "plain")
    trace = Trace("beat")
    b = build_scenario(sc, skip_idle=True, trace=trace)
    _ask_twice(b.cluster)
    total = b.cluster.run(max_cycles=20000)
    prof = build_profile(b.cluster)
    assert prof.total_cycles == total
    assert total == json.loads(plain["run.json"])["total_cycles"]


# =============================================================================
# 4. DMA class intervals over a timing sweep (R3)
# =============================================================================

# The DMA wakes at max(head visible, last write + beat_interval), so the
# moment a buffered beat becomes visible is not itself a wake. That is only
# safe because the DMA is always ticked in the cycle its data arrives. This
# sweep is what says so; it covers intervals and latencies the random test of
# test_dma.py does not reach.
DMA_TIMINGS = [
    (k, extra, done, startup)
    for k in (1, 2, 4, 5)
    for extra, done, startup in ((0, 0, 1), (2, 0, 2), (3, 2, 1), (4, 3, 4))
]


@pytest.mark.parametrize("timing", DMA_TIMINGS)
@pytest.mark.parametrize("direction", ["l2_to_l1", "l1_to_l2"])
def test_dma_classes_match_over_timings(timing, direction):
    """One class per gap, whatever the beat interval and the latencies (R3)."""
    k, extra, done, startup = timing
    cfg = DmaConfig(startup=startup, beat_interval=k, l1_read_extra=extra, done_latency=done)
    out = {}
    for skip in (True, False):
        cl, mem, xb, l2 = build(skip, rows=32, l2_size=1 << 13)
        mem.load(0, np.arange(16 * 32))
        l2.load(0, np.arange((1 << 13) // WORD))
        dma = cl.add(Dma("dma", xb, l2, cfg))
        n = 5
        l1p, l2p = contiguous(0, n), contiguous(0, n)
        desc = (
            DmaDescriptor(direction, l2p, l1p)
            if direction == "l2_to_l1"
            else DmaDescriptor(direction, l1p, l2p)
        )
        dma.start(desc)  # before the run: the DMA is busy from cycle 0
        total = cl.run(max_cycles=500)
        assert_cycles_add_up(dma, total)
        out[skip] = (total, dict(dma.cycles), dma.done_cycle, mem.data.copy(), l2.data.copy())
    a, b = out[True], out[False]
    assert a[:3] == b[:3]
    assert np.array_equal(a[3], b[3]) and np.array_equal(a[4], b[4])


# =============================================================================
# 5. Gaps at the edges
# =============================================================================


def small_cluster(blocks=None, dma=True):
    """A cluster config with one reader, one writer, an accelerator and a DMA."""
    comps = [ComponentSpec("xbar", "xbar", {"check_hold": True})]
    if dma:
        comps.append(ComponentSpec("dma", "dma", DmaConfig().to_dict()))
    comps += [
        ComponentSpec("ra", "streamer", StreamerConfig(n_ports=2, fifo_depth=2).to_dict()),
        ComponentSpec(
            "wr", "streamer", StreamerConfig(write=True, n_ports=2, fifo_depth=2).to_dict()
        ),
        ComponentSpec(
            "acc",
            "accel",
            accel="elementwise",
            params={"lanes": 2, "n_inputs": 1, "op": "add"},
            attach={"a": "ra", "out": "wr"},
        ),
        ComponentSpec("ctl", "controller", {}),
    ]
    return ClusterConfig(
        l1=L1Config(n_banks=8, rows=32),
        l2=L2Config(size_bytes=1 << 13) if dma else None,
        components=comps,
        register_map=RegisterMapSpec(blocks=blocks),
    )


def test_controller_idle_tail_is_credited():
    """A program that ends before the hardware does: the tail is controller idle."""
    cl = small_cluster()
    m = register_map_of(cl)
    prog = [named(c, m) for c in m.start_writes("ra", MAKE.unit(0, 8, 2))]
    sc = Scenario(name="tail", cluster=cl, program=prog, max_cycles=500)
    res = run(sc, trace_level="task")
    ctl = res.profile.controller
    assert ctl.cycles["idle"] > 0  # the reader ran on after the last command
    assert sum(ctl.cycles.values()) == res.total_cycles


@pytest.mark.parametrize("skip", [True, False])
def test_component_never_started(skip):
    """A component that never wakes is idle over the whole run, in both modes."""
    cl, _mem, xb, l2 = build(skip, rows=32, l2_size=1 << 13)
    trace = Trace("task")
    cl.trace = trace
    idle = cl.add(Streamer("never", xb, StreamerConfig(n_ports=1)))
    dma = cl.add(Dma("dma", xb, l2, DmaConfig()))
    dma.start(DmaDescriptor("l2_to_l1", contiguous(0, 3), contiguous(0, 3)))
    total = cl.run(max_cycles=500)
    assert total > 0
    assert dict(idle.cycles) == {"busy": 0, "stall_xbar": 0, "stall_fifo": 0, "idle": total}
    assert trace.intervals["never"] == [("idle", 0, total)]
    build_profile(cl)  # the R4 check passes for a component that never ran


@pytest.mark.parametrize("skip", [True, False])
def test_zero_cycle_run(skip):
    """Nothing to do: no cycles, no class runs, and the profile still builds."""
    cl, _mem, xb, _l2 = build(skip, rows=32, l2_size=1 << 13)
    cl.trace = Trace("task")
    s = cl.add(Streamer("ra", xb, StreamerConfig(n_ports=1)))
    total = cl.run(max_cycles=500)
    assert total == 0
    assert sum(s.cycles.values()) == 0
    assert cl.trace.intervals["ra"] == []
    assert build_profile(cl).total_cycles == 0


def test_zero_work_starts_are_traced():
    """A task with no beats, firings or DMA beats: busy never rises, done is t+1."""
    cl = small_cluster()
    m = register_map_of(cl)
    regs = MAKE.unit(0, 0, 2)  # temporal bound 0: no beats
    prog = [named(c, m) for c in m.start_writes("ra", regs)]
    prog += [named(c, m) for c in m.start_writes("acc", {"n": 0})]
    zero = DmaDescriptor("l2_to_l1", DmaPattern(0, (0,), (BEAT,)), DmaPattern(0, (0,), (BEAT,)))
    prog += [named(c, m) for c in m.start_writes("dma", zero)]
    prog += [Wait("ra", "signal"), Wait("acc", "poll"), Wait("dma", "signal")]
    prog += [NamedRead("acc.busy_cycles")]
    sc = Scenario(name="zero", cluster=cl, program=prog, max_cycles=500)
    res = run(sc, trace_level="task")
    starts = {e.src: e.t for e in res.trace.of_kind("start")}
    dones = {e.src: e.t for e in res.trace.of_kind("done")}
    assert set(starts) == set(dones) == {"ra", "acc", "dma"}
    for name, t in starts.items():
        assert dones[name] == t + 1, name  # done the cycle after the start landed
    assert res.reads[-1][2] == 0  # busy_cycles of a task with no work
    assert sum(res.profile.accelerators["acc"].cycles.values()) == res.total_cycles


def test_class_total_mismatch_is_caught():
    """build_profile refuses a component whose classes do not cover the run (R4)."""
    cl, _mem, xb, l2 = build(True, rows=32, l2_size=1 << 13)
    s = cl.add(Streamer("ra", xb, StreamerConfig(n_ports=1)))
    dma = cl.add(Dma("dma", xb, l2, DmaConfig()))
    dma.start(DmaDescriptor("l2_to_l1", contiguous(0, 2), contiguous(0, 2)))
    cl.run(max_cycles=500)
    s.cycles["idle"] -= 1  # as a component that slept through a class change
    with pytest.raises(SimulationError, match="cycle classes add up"):
        build_profile(cl)


# =============================================================================
# 6. Small gaps found while writing these
# =============================================================================


def test_named_covers_every_command():
    """``named`` (used by the generators) covers reads and waits, not only writes."""
    m = register_map_of(small_cluster())
    addr = m.addr("ra.base")
    assert named(CsrWrite(addr, 7), m) == NamedWrite("ra.base", 7)
    assert named(CsrRead(addr), m) == NamedRead("ra.base")
    assert named(Wait("ra", "poll"), m) == Wait("ra", "poll")  # a wait names its block already


def test_default_blocks_are_every_component_but_xbar_and_controller():
    """``register_map.blocks`` left out: the documented default, used by no cluster file."""
    m = register_map_of(small_cluster(blocks=None))
    assert list(m.blocks) == ["dma", "ra", "wr", "acc"]
    assert [b.base for b in m.blocks.values()] == [0, 32, 64, 96]
