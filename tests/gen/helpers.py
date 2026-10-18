"""Shared helpers for the GVSoC generator's tests (GEN3 step 3).

``point`` writes a design point of vecadd (bound by the checked-in recipe,
W = 4 by default), or of a given graph, on ``platforms/snax_alu.json``; ``fake_install`` makes an install folder holding every module the
generated files import, as empty files. The snax-gvsoc submodule (``SUB``)
is read only where it is checked out (CI never fetches it, D128).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from snax_forge.design import Design, DesignPoint, Platform, run_checks
from snax_forge.dfg import Graph
from snax_forge.gen import IMPORTS
from tests.design.helpers import REPO, SNAX_ALU, bound_vecadd

SUB = REPO / "gvsoc" / "snax-gvsoc"
HJSON = SUB / "sw" / "cfg" / "snax_alu_cluster.hjson"


def point(tmp_path: Path, w: int = 4, name: str = "vecadd", graph: Graph | None = None) -> Path:
    """A design point file of vecadd at ``w`` (or of ``graph``) on snax_alu, named ``name``."""
    design = Design(Platform.load(SNAX_ALU), graph or bound_vecadd(w))
    assert run_checks(design) == []
    path = tmp_path / "design" / name / "design_point.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    DesignPoint.of(design, name).save(path)
    return path


def fake_install(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    """An install folder with every module of IMPORTS (empty), and ``extra`` (relative path -> text)."""
    root = tmp_path / "install"
    for d in ("targets", "generators", "python"):
        (root / d).mkdir(parents=True, exist_ok=True)
    places = {"gvsoc": "python", "interco": "generators"}
    for imp in IMPORTS:
        parts = imp.split(".")
        where = root / places.get(parts[0], "generators")
        path = where.joinpath(*parts).with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for rel, text in (extra or {}).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def snax_streamer() -> types.ModuleType:
    """snax-gvsoc's snax/snax_streamer.py, imported with a stub gvsoc.systree (its helpers only)."""
    stub = types.ModuleType("gvsoc.systree")
    stub.Component = stub.SlaveItf = object
    pkg = types.ModuleType("gvsoc")
    pkg.systree = stub
    saved = {k: sys.modules.get(k) for k in ("gvsoc", "gvsoc.systree")}
    sys.modules.update({"gvsoc": pkg, "gvsoc.systree": stub})
    try:
        spec = importlib.util.spec_from_file_location(
            "snax_streamer", SUB / "snax" / "snax_streamer.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
