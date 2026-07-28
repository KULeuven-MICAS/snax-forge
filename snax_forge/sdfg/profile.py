"""
Wall-clock profiling of compiled SDFGs.

Two independent measurements per size:
  - wall clock (perf_counter) -- honest end-to-end cost, includes Python/ctypes
  - DaCe Timer instrumentation (optional) -- per-state cost inside generated C++

The gap between them IS the host offload overhead. Neither number predicts SNAX
cluster performance; they exist for roofline positioning and as a sanity oracle
for the analytical model later.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from collections import defaultdict

import dace
import numpy as np

from .build import OUT as _SDFG_OUT
from .build import build
from .spec import KernelSpec

# Derived from build.OUT, so fixing that path fixes this one too.
OUT = _SDFG_OUT.parent / "profile"

DEFAULT_SIZES = (1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18)

# Report files are named by timestamp, so fast back-to-back calls overwrite each
# other. Expect FEWER events than this -- check "samples" in the output.
INSTRUMENT_REPS = 10


def as_size_kwargs(size) -> dict:
    """Normalise a sweep entry to make_inputs kwargs.

    Accepts a bare int (1-D kernels -> n=) or a mapping for multi-symbol
    kernels, e.g. {"m": 256, "n": 512}.
    """
    return {"n": int(size)} if isinstance(size, int) else dict(size)


def size_label(kw: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in kw.items())


def _auto_bytes(args: dict) -> int:
    """Compulsory traffic: every array touched once.

    Exact for elementwise ops and single-pass reductions. WRONG for kernels
    with reuse (matmul, stencils with halo) -- override via spec.bytes_moved.
    """
    return sum(v.nbytes for v in args.values() if isinstance(v, np.ndarray))


def attach_timers(sdfg: dace.SDFG) -> None:
    """Timer on the whole SDFG and on every state (tier 1).

    Must run BEFORE compile() -- instrumentation changes the emitted C++.
    """
    sdfg.instrument = dace.InstrumentationType.Timer
    for state in sdfg.states():
        state.instrument = dace.InstrumentationType.Timer


def collect_durations(sdfg: dace.SDFG) -> dict:
    """Aggregate every available report into per-state and whole-SDFG stats.

    Report keys are (sdfg_id, state_id, node_id); node_id == -1 means the
    entry covers a whole state, and state_id == -1 as well means the whole
    SDFG. Raw values are MILLISECONDS.
    """
    per_state: dict[str, list[float]] = defaultdict(list)
    whole: list[float] = []
    n_events = 0

    # Reports accumulate across calls; each holds one invocation's events.
    for report in sdfg.get_instrumentation_reports():
        for uuid, inner in report.durations.items():
            _, state_id, node_id = uuid
            if node_id != -1:
                continue  # node-level timers are tier 2; not requested
            for label, vals in inner.items():
                # Inner dict is keyed by thread id -> list of durations.
                flat = [x for v in (vals.values() if hasattr(vals, "items") else [vals]) for x in v]
                n_events += len(flat)
                # state_id == -1 is the SDFG-wide timer, everything else a state.
                target = whole if state_id == -1 else per_state[str(label)]
                target.extend(flat)

    def stats(xs: list[float]) -> dict:
        a = np.asarray(xs, dtype=float) * 1e3  # ms -> us
        return {
            "min_us": float(a.min()),
            "median_us": float(np.median(a)),
            # Low sample count => low trust. Timestamp collisions drop events.
            "samples": int(a.size),
        }

    # A state can execute many times per SDFG call (loops). Normalise the
    # per-state total by how many SDFG invocations we actually captured.
    n_calls = len(whole) or 1
    states = {k: stats(v) for k, v in per_state.items() if v}
    for k, v in per_state.items():
        states[k]["per_call_us"] = round(float(np.sum(v)) * 1e3 / n_calls, 3)

    out = {
        "n_events": n_events,
        "n_calls": n_calls,
        "states": states,
        "states_sum_us": round(sum(s["per_call_us"] for s in states.values()), 3),
    }

    if whole:
        out["sdfg_total_us"] = stats(whole)["min_us"]
        # Whole-SDFG minus sum-of-states = inter-state edges and control flow.
        # ~0 for single-state kernels; the cost state fusion would remove.
        out["inter_state_us"] = round(out["sdfg_total_us"] - out["states_sum_us"], 3)
    return out


def time_kernel(
    csdfg, spec: KernelSpec, warmup: int, reps: int, instr: tuple | None = None, **sizes
) -> dict:
    """Time one compiled SDFG at one problem size.

    `sizes` are forwarded to make_inputs (n=, or m=/n= for 2D kernels); the
    matching symbol values are recovered from the resulting array shapes.
    """
    rng = np.random.default_rng(0)  # fixed seed: same data every run
    args = spec.make_inputs(rng, **sizes)
    symbols = spec.bind_symbols(args)  # e.g. {"N": 4096} from A.shape

    # Warm up: first-touch page faults, cold caches, lazy loader work.
    for _ in range(warmup):
        csdfg(**args, **symbols)

    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        csdfg(**args, **symbols)
        samples.append(time.perf_counter() - t0)

    # min, not mean: every source of noise on a shared machine ADDS time.
    # stdev is kept only as a trust signal -- large stdev => contended node.
    best = min(samples)
    point = {
        "sizes": sizes,
        "symbols": symbols,
        "reps": reps,
        "min_s": best,
        "median_s": statistics.median(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }

    # Explicit model if the kernel supplies one, else assume no reuse.
    nbytes = spec.bytes_moved(**sizes) if spec.bytes_moved else _auto_bytes(args)
    point["bytes"] = nbytes
    point["gbytes_s"] = nbytes / best / 1e9

    # AI = ops/byte -> where this kernel sits on the roofline.
    if spec.flops:
        f = spec.flops(**sizes)
        point["flops"] = f
        point["arithmetic_intensity"] = f / nbytes
        point["gflops_s"] = f / best / 1e9

    if instr is not None:
        isdfg, icsdfg = instr
        for _ in range(2):  # warm the instrumented build too
            icsdfg(**args, **symbols)
        # Essential: the same SDFG object accumulates reports across sizes.
        isdfg.clear_instrumentation_reports()
        for _ in range(INSTRUMENT_REPS):
            icsdfg(**args, **symbols)
        point["instrumentation"] = collect_durations(isdfg)

    return point


def sweep(
    spec: KernelSpec,
    sizes=None,
    warmup: int = 10,
    reps: int = 50,
    instrument: bool = False,
) -> dict:
    """Build once, compile once, time across sizes.

    Symbolic descriptors are what make one compile serve the whole sweep --
    concrete shapes would force a rebuild per size.
    """
    sizes = sizes or spec.sweep_sizes or DEFAULT_SIZES
    sdfg = build(spec, simplify=True)
    csdfg = sdfg.compile()  # once; symbolic bounds serve every size

    # Separate build so timer overhead never contaminates the wall-clock numbers.
    # The rename gives it its own .dacecache dir so the two don't collide.
    instr = None
    if instrument:
        isdfg = build(spec, simplify=True)
        isdfg.name = f"{isdfg.name}_instr"
        attach_timers(isdfg)  # before compile()
        instr = (isdfg, isdfg.compile())

    # Provenance: a timing baseline without its thread/CPU config isn't reproducible.
    report = {
        "name": spec.name,
        "domain": spec.domain,
        "dace": dace.__version__,
        "omp_threads": os.environ.get("OMP_NUM_THREADS", "<unset>"),
        "cpu": platform.processor() or platform.machine(),
        "points": [],
    }

    print(f"{spec.name}  (threads={report['omp_threads']})")
    for size in sizes:
        kw = as_size_kwargs(size)
        p = time_kernel(csdfg, spec, warmup, reps, instr=instr, **kw)
        report["points"].append(p)
        ai = p.get("arithmetic_intensity")
        print(
            f"  {size_label(kw):<18} {p['min_s'] * 1e6:9.2f} us  "
            f"+/-{p['stdev_s'] * 1e6:7.2f}  {p['gbytes_s']:6.2f} GB/s"
            + (f"  AI={ai:.4f}" if ai is not None else "")
        )
        # Indented block: in-kernel time. Compare against the wall clock above --
        # the difference is Python + ctypes marshalling.
        if "instrumentation" in p:
            ins = p["instrumentation"]
            for label, st in sorted(ins["states"].items()):
                print(f"      {label:34} {st['min_us']:9.2f} us  (x{st['samples']})")
            print(
                f"      {'= states sum':34} {ins['states_sum_us']:9.2f} us"
                + (
                    f"   sdfg total {ins['sdfg_total_us']:9.2f} us"
                    f"   inter-state {ins['inter_state_us']:.2f} us"
                    if "sdfg_total_us" in ins
                    else ""
                )
            )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{spec.name}.json").write_text(json.dumps(report, indent=2, default=str))
    return report
