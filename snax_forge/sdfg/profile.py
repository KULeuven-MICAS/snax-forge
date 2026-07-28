"""
Wall-clock profiling of compiled SDFGs.

Two independent measurements per size:
  - wall clock (perf_counter) -- honest end-to-end cost, includes Python/ctypes
  - DaCe Timer instrumentation (optional) -- per-state cost inside generated C++

The gap between them IS the host offload overhead. Neither number predicts SNAX
cluster performance; they exist for roofline positioning and as a sanity oracle
for the analytical model later.

Two entry points:
  - sweep / compare            -- driven by a KernelSpec (shapes, dtypes, models)
  - sweep_sdfg / compare_sdfgs -- driven by a stored .sdfg alone, no spec
"""

from __future__ import annotations

import copy
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


# ---------------------------------------------------------------------------
# Sizes and derived metrics
# ---------------------------------------------------------------------------


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


def _add_metrics(point: dict, args: dict, flops, bytes_moved, kwargs: dict) -> None:
    """Fill bytes / bandwidth / FLOPs / arithmetic intensity into `point`.

    `kwargs` are whatever the models expect: make_inputs sizes on the spec
    path (n=, m=), concrete symbols on the spec-free path (N=, M=).
    """
    nbytes = bytes_moved(**kwargs) if bytes_moved else _auto_bytes(args)
    point["bytes"] = nbytes
    point["gbytes_s"] = nbytes / point["min_s"] / 1e9
    if flops:
        f = flops(**kwargs)
        point["flops"] = f
        point["arithmetic_intensity"] = f / nbytes  # roofline position
        point["gflops_s"] = f / point["min_s"] / 1e9


def _provenance(**extra) -> dict:
    """A timing baseline without its thread/CPU config isn't reproducible."""
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        affinity = None
    return {
        "dace": dace.__version__,
        "omp_threads": os.environ.get("OMP_NUM_THREADS", "<unset>"),
        "affinity": affinity,
        "cpu": platform.processor() or platform.machine(),
        **extra,
    }


# ---------------------------------------------------------------------------
# Tier-1 instrumentation
# ---------------------------------------------------------------------------


def attach_timers(sdfg: dace.SDFG) -> None:
    """Timer on the whole SDFG and on every state (tier 1).

    Must run BEFORE compile() -- instrumentation changes the emitted C++.
    """
    sdfg.instrument = dace.InstrumentationType.Timer
    for state in sdfg.states():
        state.instrument = dace.InstrumentationType.Timer


def make_instrumented(sdfg: dace.SDFG):
    """Compile a timered twin of `sdfg`.

    deepcopy, not rebuild -- a transformed variant must stay transformed. The
    rename gives it its own .dacecache dir so the two binaries don't collide.
    """
    twin = copy.deepcopy(sdfg)
    twin.name = f"{sdfg.name}_instr"
    attach_timers(twin)
    return twin, twin.compile()


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


def run_instrumented(instr: tuple, args: dict, symbols: dict) -> dict:
    """Short burst on the timered twin, then harvest its reports."""
    isdfg, icsdfg = instr
    for _ in range(2):  # warm the instrumented build too
        icsdfg(**args, **symbols)
    # Essential: the same SDFG object accumulates reports across sizes.
    isdfg.clear_instrumentation_reports()
    for _ in range(INSTRUMENT_REPS):
        icsdfg(**args, **symbols)
    return collect_durations(isdfg)


# ---------------------------------------------------------------------------
# Timing core and printing, shared by both entry points
# ---------------------------------------------------------------------------


def time_calls(csdfg, args: dict, symbols: dict, warmup: int, reps: int) -> dict:
    """Warm up, then time `reps` invocations."""
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
    return {
        "symbols": symbols,
        "reps": reps,
        "min_s": min(samples),
        "median_s": statistics.median(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def print_point(point: dict) -> None:
    """One result line, plus the in-kernel breakdown when instrumented."""
    ai = point.get("arithmetic_intensity")
    print(
        f"  {size_label(point['sizes']):<18} {point['min_s'] * 1e6:9.2f} us  "
        f"+/-{point['stdev_s'] * 1e6:7.2f}  {point['gbytes_s']:6.2f} GB/s"
        + (f"  AI={ai:.4f}" if ai is not None else "")
    )
    # Indented block: in-kernel time. Compare against the wall clock above --
    # the difference is Python + ctypes marshalling.
    if "instrumentation" not in point:
        return
    ins = point["instrumentation"]
    for state_label, st in sorted(ins["states"].items()):
        print(f"      {state_label:34} {st['per_call_us']:9.2f} us  (x{st['samples']})")
    print(
        f"      {'= states sum':34} {ins['states_sum_us']:9.2f} us"
        + (
            f"   sdfg total {ins['sdfg_total_us']:9.2f} us"
            f"   inter-state {ins['inter_state_us']:.2f} us"
            if "sdfg_total_us" in ins
            else ""
        )
    )


def print_comparison(reports: dict, title: str) -> None:
    """Side-by-side table, speedups relative to the first variant."""
    base = next(iter(reports))
    print(f"\n=== {title} (min us, vs {base}) ===")
    cols = [size_label(p["sizes"]) for p in reports[base]["points"]]
    print(f"{'variant':22}" + "  ".join(f"{c:>16}" for c in cols))
    ref = [p["min_s"] for p in reports[base]["points"]]
    for name, rep in reports.items():
        cells = [
            f"{p['min_s'] * 1e6:9.1f} ({r / p['min_s']:.2f}x)" for p, r in zip(rep["points"], ref)
        ]
        print(f"{name:22}" + "  ".join(f"{c:>16}" for c in cells))


def _write(report: dict, label: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{label}.json").write_text(json.dumps(report, indent=2, default=str))


# ---------------------------------------------------------------------------
# Spec-driven path
# ---------------------------------------------------------------------------


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

    point = {"sizes": sizes, **time_calls(csdfg, args, symbols, warmup, reps)}
    _add_metrics(point, args, spec.flops, spec.bytes_moved, sizes)
    if instr is not None:
        point["instrumentation"] = run_instrumented(instr, args, symbols)
    return point


def sweep(
    spec: KernelSpec,
    sizes=None,
    warmup: int = 10,
    reps: int = 50,
    instrument: bool = False,
    sdfg: dace.SDFG | None = None,
    label: str | None = None,
) -> dict:
    """Build once, compile once, time across sizes.

    Symbolic descriptors are what make one compile serve the whole sweep --
    concrete shapes would force a rebuild per size. Pass `sdfg` to profile a
    transformed variant instead of rebuilding from the spec.
    """
    sizes = sizes or spec.sweep_sizes or DEFAULT_SIZES
    sdfg = sdfg if sdfg is not None else build(spec, simplify=True)
    label = label or sdfg.name
    csdfg = sdfg.compile()

    # Separate copy so timer overhead never contaminates the wall-clock numbers.
    instr = make_instrumented(sdfg) if instrument else None

    report = _provenance(name=spec.name, label=label, domain=spec.domain, points=[])
    print(f"{label}  (threads={report['omp_threads']})")
    for size in sizes:
        point = time_kernel(csdfg, spec, warmup, reps, instr=instr, **as_size_kwargs(size))
        report["points"].append(point)
        print_point(point)

    _write(report, label)
    return report


def compare(spec: KernelSpec, variants: dict[str, dace.SDFG], **kw) -> dict:
    """Profile several SDFG variants of one kernel side by side.

    Every variant must be semantically identical -- apply_recipe already
    verifies bit-exactness, so pass its output directly.
    """
    reports = {
        name: sweep(spec, sdfg=v, label=f"{spec.name}.{name}", **kw) for name, v in variants.items()
    }
    print_comparison(reports, f"{spec.name}: variant comparison")
    return reports


# ---------------------------------------------------------------------------
# Spec-free path: profile a stored .sdfg directly
# ---------------------------------------------------------------------------


def sdfg_inputs(sdfg: dace.SDFG, symbols: dict, rng) -> dict:
    """Allocate concrete arrays for every non-transient array in the SDFG.

    Recovers shape and dtype from the descriptors, so no KernelSpec is needed.
    Values are positive: some kernels floor-divide, where numpy and C agree
    only for non-negative operands.
    """
    args = {}
    for name, desc in sdfg.arrays.items():
        if desc.transient:
            continue
        shape = tuple(int(dace.symbolic.evaluate(d, symbols)) for d in desc.shape)
        dtype = desc.dtype.as_numpy_dtype()
        if np.issubdtype(dtype, np.integer):
            args[name] = rng.integers(1, 100, size=shape, dtype=dtype)
        else:
            args[name] = rng.random(shape).astype(dtype)
    return args


def sweep_sdfg(
    source,
    symbol_sets,
    warmup: int = 10,
    reps: int = 50,
    instrument: bool = False,
    label: str | None = None,
    flops=None,
    bytes_moved=None,
) -> dict:
    """Profile a stored .sdfg (path or SDFG object) with no KernelSpec.

    symbol_sets: iterable of concrete bindings, e.g. [{"N": 4096}, {"N": 65536}].
    sdfg.free_symbols tells you WHICH symbols are needed, never which values
    are interesting -- so they must be supplied.

    Correctness is NOT checked: without a spec there is no golden reference.
    Use apply_recipe (which verifies) to produce transformed SDFGs; this is
    timing only. Without flops/bytes_moved, bandwidth falls back to
    _auto_bytes and is not comparable to the spec path's figures.
    """
    sdfg = source if isinstance(source, dace.SDFG) else dace.SDFG.from_file(str(source))
    label = label or sdfg.name
    csdfg = sdfg.compile()
    instr = make_instrumented(sdfg) if instrument else None

    report = _provenance(label=label, sdfg_name=sdfg.name, points=[])
    print(f"{label}  (threads={report['omp_threads']})")

    rng = np.random.default_rng(0)
    for symbols in symbol_sets:
        symbols = dict(symbols)
        args = sdfg_inputs(sdfg, symbols, rng)
        point = {"sizes": symbols, **time_calls(csdfg, args, symbols, warmup, reps)}
        _add_metrics(point, args, flops, bytes_moved, symbols)
        if instr is not None:
            point["instrumentation"] = run_instrumented(instr, args, symbols)
        report["points"].append(point)
        print_point(point)

    _write(report, label)
    return report


def compare_sdfgs(variants: dict, symbol_sets, **kw) -> dict:
    """Side-by-side timing of several stored SDFGs. variants: label -> path or SDFG."""
    reports = {name: sweep_sdfg(v, symbol_sets, label=name, **kw) for name, v in variants.items()}
    print_comparison(reports, "stored SDFG comparison")
    return reports
