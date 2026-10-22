"""Shared helpers for the generator's tests.

``point`` writes a design point of vecadd (bound by the checked-in recipe,
W = 4 by default), or of a given graph (``bound_vecmul``: vecmul bound by
``recipes/vecmul.json``; ``bound_vsum``: vsum by ``recipes/vsum.json``, or at
another W and implementation), on
``platforms/snax_forge.json`` (or a given platform), in a folder that stands
for snax-forge's root: the platform's source hjson is copied into its
``platforms/``. The snax-gvsoc submodule (``SUB``) is read only where it is
checked out (only CI's gvsoc job fetches it, D128); ``snax_cfg`` imports its reader.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import types
from collections.abc import Sequence
from pathlib import Path

import pytest

from snax_forge.design import Design, DesignPoint, Platform, run_checks
from snax_forge.dfg import Graph
from snax_forge.sandbox import Recipe, apply_recipe
from tests.design.helpers import FIXTURES, REPO, SNAX_FORGE, bound_vecadd

SUB = REPO / "gvsoc" / "snax-gvsoc"
HJSON = SUB / "sw" / "cfg" / "snax_alu_cluster.hjson"
# snax_forge's source hjson (platforms/snax_forge.json's hjson)
SOURCE = REPO / "platforms" / "snax_alu_cluster.hjson"


def root(tmp_path: Path) -> Path:
    """``tmp_path`` as snax-forge's root: snax_forge's source hjson in its platforms/."""
    (tmp_path / "platforms").mkdir(parents=True, exist_ok=True)
    shutil.copy(SOURCE, tmp_path / "platforms" / SOURCE.name)
    return tmp_path


def point(
    tmp_path: Path, w: int = 4, name: str = "vecadd", graph: Graph | None = None,
    platform: Platform | None = None,
) -> Path:  # fmt: skip
    """A design point file of vecadd at ``w`` (or of ``graph``) on snax_forge (or ``platform``),
    named ``name``, under ``tmp_path`` as the root (``root``)."""
    root(tmp_path)
    design = Design(platform or Platform.load(SNAX_FORGE), graph or bound_vecadd(w))
    assert run_checks(design) == []
    path = tmp_path / "design" / name / "design_point.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    DesignPoint.of(design, name).save(path)
    return path


def bound_vecmul(w: int = 4) -> Graph:
    """vecmul bound with W = ``w`` by the checked-in recipe (elementwise_mul, instance acc)."""
    recipe = Recipe.load(REPO / "recipes" / "vecmul.json").with_params({"W": w})
    return apply_recipe(recipe, Graph.load(FIXTURES / "vecmul.snaxdfg"))[-1].graph


def bound_vsum(w: int = 1, impl: str = "chisel_accumulator") -> Graph:
    """vsum bound by the checked-in recipe (accumulate's chisel_accumulator, instance acc), or
    with W = ``w`` to ``impl``."""
    d = Recipe.load(REPO / "recipes" / "vsum.json").with_params({"W": w}).to_dict()
    d["steps"][-1]["params"]["implementation"] = impl
    return apply_recipe(Recipe.from_dict(d), Graph.load(FIXTURES / "vsum.snaxdfg"))[-1].graph


def designed(tmp: Path, recipe: str, name: str, sets: Sequence[str] = ()) -> str:
    """The design point of ``recipes/<recipe>.json`` (``--set`` each of ``sets``) on snax_forge,
    made by the sandbox's and the design step's commands under ``tmp`` as the root (``root``),
    as ``out/design/<name>/design_point.json``; that path, from ``tmp``."""
    from snax_forge.design import __main__ as design_cli
    from snax_forge.sandbox import __main__ as sandbox_cli

    root(tmp)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        args = [str(REPO / "recipes" / f"{recipe}.json"), "--out", f"out/sandbox/{name}"]
        assert sandbox_cli.main([*args, *(a for s in sets for a in ("--set", s))]) == 0
        assert design_cli.main([f"out/sandbox/{name}/2_bind.snaxdfg", "--platform",
                                str(SNAX_FORGE), "--out", f"out/design/{name}"]) == 0  # fmt: skip
    return f"out/design/{name}/design_point.json"


def snax_cfg() -> types.ModuleType | None:
    """snax-gvsoc's snax/snax_cfg.py (the reader of the cluster hjson), or None where the
    submodule is not checked out."""
    path = SUB / "snax" / "snax_cfg.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("snax_cfg", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("snax_cfg", mod)  # its dataclasses look their module up
    spec.loader.exec_module(mod)
    return mod


def snax_streamer() -> types.ModuleType:
    """snax-gvsoc's snax/snax_streamer.py, imported with a stub gvsoc.systree (its helpers only),
    after its snax_cfg (``snax_cfg``), which it imports."""
    snax_cfg()
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
