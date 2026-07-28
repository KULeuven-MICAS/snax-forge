"""KernelSpec -> SDFG on disk, plus a correctness check against NumPy."""

from __future__ import annotations

import json
import time
from pathlib import Path

import dace
import numpy as np

from .spec import KernelSpec

OUT = Path("out/sdfg")


def build(spec: KernelSpec, simplify: bool = False) -> dace.SDFG:
    prog = dace.program(auto_optimize=False)(spec.func)
    sdfg = prog.to_sdfg(**spec.descriptors, simplify=simplify)
    sdfg.name = f"{spec.name}_{'simp' if simplify else 'raw'}"
    return sdfg


def structural_summary(sdfg: dace.SDFG) -> dict:
    """Seed of the W2 census. Deliberately shallow for now."""
    states = list(sdfg.states())
    scopes, maps = 0, 0
    for state in states:
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                maps += 1
                if state.entry_node(node) is None:
                    scopes += 1
    return {
        "states": len(states),
        "map_entries": maps,
        "top_level_map_scopes": scopes,
        "arrays": len([n for n, d in sdfg.arrays.items() if not d.transient]),
        "transients": len([n for n, d in sdfg.arrays.items() if d.transient]),
        "symbols": sorted(str(s) for s in sdfg.free_symbols),
    }


def verify(spec: KernelSpec, sdfg: dace.SDFG, n: int | None = None) -> bool:
    rng = np.random.default_rng(0)
    ref_args = spec.make_inputs(rng) if n is None else spec.make_inputs(rng, n=n)
    dut_args = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ref_args.items()}

    spec.reference(**ref_args)

    csdfg = sdfg.compile()
    csdfg(**dut_args, **spec.bind_symbols(dut_args))

    return all(np.array_equal(ref_args[k], dut_args[k]) for k in spec.inout)


def run(spec: KernelSpec) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"name": spec.name, "domain": spec.domain, "dace": dace.__version__}

    for simplify in (False, True):
        tag = "simplified" if simplify else "raw"
        t0 = time.perf_counter()
        sdfg = build(spec, simplify=simplify)
        path = OUT / f"{spec.name}.{tag}.sdfg"
        sdfg.save(path)
        report[tag] = {
            "path": str(path),
            "build_s": round(time.perf_counter() - t0, 2),
            **structural_summary(sdfg),
        }
        if simplify:
            report[tag]["matches_numpy"] = verify(spec, sdfg)

    (OUT / f"{spec.name}.json").write_text(json.dumps(report, indent=2))
    return report
