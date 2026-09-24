"""Generate the checked-in SNAX-MODEL clusters and scenarios (MOD9, D41, D42, D65).

    python scenarios/make.py            write every generated file below scenarios/
    python scenarios/make.py --check    exit 1 if a generated file is missing or differs

Every scenario lives in its own folder with the script that makes it
(D65): ``<name>/scenario.py`` defines ``make()``, which returns the
``Scenario`` and its input arrays by ``.npy`` file name. This driver writes
``<name>/scenario.json`` and the arrays, and ``clusters/<stem>.json`` for
every cluster in ``clusters/clusters.py``. A folder with a hand-written
``tasks.json`` is written as a task list: its ``scenario.py`` lowers it into
the scenario's program (``lower_program``, D64), and this driver never
writes it. Every scenario is a task list now (D66).

The generated files are not in git (D67): they are written again by
``pixi run scenarios``, by ``pixi run model-run`` before it runs, and by
the test session before any test (tests/conftest.py). The sources are the
``scenario.py`` and ``tasks.json`` files, ``common.py`` and
``clusters/clusters.py``.

Shared helpers live in ``common.py``; ``scenarios/`` is put on the import
path so every ``scenario.py`` can use it and ``clusters.clusters``.

Scenarios:

    vecadd/              the MOD7 vecadd (test_profile.run_vecadd), the M3 target; task list
    vecadd_conflict/     vecadd with b in the same banks as a (VIS3's conflict case); task list
    vecadd_tiled/        vecadd over 576 elements in 3 tiles of 192 (471 cycles); task list
    fmul/                a * b in 5 tiles of 16 on mul1, double buffered; task list
    reduce/              64 elements in L1 summed in groups of 16; task list
    dma/                 L2 -> L1 with a 2D pattern and back, on alu4; task list
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for common and clusters.clusters

from clusters.clusters import CLUSTERS, CTL, alu4, mul1, red4
from common import BEAT, LANES, WORD, contiguous, unit

from snax_forge.lower import Program
from snax_forge.snax_model.scenario import Scenario, to_json

ROOT = Path(__file__).resolve().parent
SCENARIO_FILE = "scenario.py"

# Re-exported for the tests that build programs with the scenario helpers
# (test_gaps: MAKE.alu4, MAKE.unit, MAKE.Program, MAKE.WORD, ...).
__all__ = [
    "BEAT", "CTL", "LANES", "WORD", "Program", "alu4", "contiguous", "generate", "main",
    "mul1", "red4", "unit", "write",
]  # fmt: skip


def scenario_dirs() -> list[Path]:
    """Every folder with a ``scenario.py``, by name."""
    return sorted(p.parent for p in ROOT.glob(f"*/{SCENARIO_FILE}"))


def load(folder: Path) -> ModuleType:
    """A folder's ``scenario.py``, as a module of its own name."""
    spec = importlib.util.spec_from_file_location(f"scenario_{folder.name}", folder / SCENARIO_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make(folder: Path) -> tuple[Scenario, dict[str, np.ndarray]]:
    sc, arrays = load(folder).make()
    if sc.name != folder.name:
        raise ValueError(f"{folder.name}/{SCENARIO_FILE} makes a scenario named {sc.name!r}")
    return sc, arrays


def _npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr, dtype=np.int64), allow_pickle=False)
    return buf.getvalue()


def generate() -> dict[str, bytes]:
    """Every generated file, by path relative to scenarios/, with its exact contents."""
    files = {
        f"clusters/{stem}.json": to_json(b().to_dict()).encode() for stem, b in CLUSTERS.items()
    }
    for folder in scenario_dirs():
        sc, arrays = make(folder)
        files[f"{sc.name}/scenario.json"] = to_json(sc.to_dict()).encode()
        for name, arr in arrays.items():
            files[f"{sc.name}/{name}"] = _npy(arr)
    return files


def write(check: bool = False) -> list[str]:
    """Write every generated file that is missing or differs; returns their paths.

    With ``check`` nothing is written. A file is replaced in one step (a
    temporary file, then ``os.replace``), so a reader never sees half of it.
    """
    stale = []
    for rel, data in generate().items():
        path = ROOT / rel
        if path.is_file() and path.read_bytes() == data:
            continue
        stale.append(rel)
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
    return stale


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="only compare, write nothing")
    args = ap.parse_args(argv)
    stale = write(check=args.check)
    if args.check and stale:
        print("out of date: " + ", ".join(stale), file=sys.stderr)
        return 1
    print(("stale: " if args.check else "written: ") + (", ".join(stale) or "nothing"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
