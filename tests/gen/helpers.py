"""Shared helpers for the generator's tests.

``point`` writes a design point of vecadd (bound by the checked-in recipe,
W = 4 by default), or of a given graph (``bound_vecmul``: vecmul bound by
``recipes/vecmul.json``; ``bound_vsum``: vsum by ``recipes/vsum.json``, or at
another W and implementation), on
``platforms/snax_forge.json`` (or a given platform), in a folder that stands
for snax-forge's root: the platform's source hjson is copied into its
``platforms/``. The snax-gvsoc submodule (``SUB``) is read only where it is
checked out (CI's python, chisel and gvsoc jobs fetch it, D134); ``snax_cfg`` imports its
reader.
``fake_emit`` stands in for the bundle's ``Emit --spec`` (no JVM).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import hjson
import pytest
from jsonschema import Draft202012Validator

from snax_forge.design import Design, DesignPoint, Platform, run_checks
from snax_forge.dfg import Graph
from snax_forge.gen import Accelerator, Csr, Group, Port, Unit
from snax_forge.gen import bundle as bundle_mod
from snax_forge.gen.rtl import EMITTED_JSON
from snax_forge.gen.schemas import schema
from snax_forge.sandbox import Recipe, apply_recipe
from tests.design.helpers import SNAX_FORGE, bound_vecadd
from tests.helpers import FIXTURES, REPO, SUB

# snax_forge's source hjson (platforms/snax_forge.json's hjson), snax-forge's base cluster
SOURCE = REPO / "platforms" / "snax_forge_cluster.hjson"
# The fake Emit's files, in emitted.json's order (not sorted, so files.f's order is its)
SV = ["sv/snax_forge_shell_wrapper.sv", "sv/ElementwiseAlu_w64_add.sv"]
COMMIT = "0123456789abcdef0123456789abcdef01234567"


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


def one_unit(k: int = 0, name: str = "acc") -> Accelerator:
    """An accelerator of one group of one unit with ``k`` R/W CSRs of its own (``F0``, ...),
    offsets in CSR order."""
    rw = [Csr(n, "rw", i, 32) for i, n in enumerate(["START", "JOBS"])]
    own = [Csr(f"F{i}", "rw", 2 + i, 32) for i in range(k)]
    ro = [Csr(n, "ro", 2 + k + i) for i, n in enumerate(["STATUS", "COUNTER"])]
    unit = Unit(name, "elementwise_add", "ElementwiseLanes",
                {p: Port(p, "out" if p == "out" else "in", "int64", 4) for p in ("a", "b", "out")},
                {"op": "add"}, 0, 1, own,
                "int64")  # fmt: skip
    return Accelerator(name, "design/x/design_point.json", "snax_forge", [f"{name}.a", f"{name}.b"],
                       [f"{name}.out"], [Group(name, [*rw, *ro], [unit])])  # fmt: skip


def bound_vecmul(w: int = 4) -> Graph:
    """vecmul bound with W = ``w`` by the checked-in recipe (elementwise_mul, instance acc)."""
    recipe = Recipe.load(REPO / "recipes" / "vecmul.json").with_params({"W": w})
    return apply_recipe(recipe, Graph.load(FIXTURES / "vecmul.snaxdfg"))[-1].graph


def bound_dot1() -> Graph:
    """dot 1 bound by the checked-in recipe: groups mul and sum in the accelerator dot (D148)."""
    recipe = Recipe.load(REPO / "recipes" / "dot_1.json")
    return apply_recipe(recipe, Graph.load(FIXTURES / "dot.snaxdfg"))[-1].graph


def designed(tmp: Path, recipe: str, name: str, sets: Sequence[str] = ()) -> str:
    """The design point of ``recipes/<recipe>.json`` (``--set`` each of ``sets``) on snax_forge,
    made by the sandbox's and the design step's commands under ``tmp`` as the root (``root``),
    as ``out/design/<name>/design_point.json``; that path, from ``tmp``."""
    from snax_forge.design import __main__ as design_cli
    from snax_forge.sandbox import Recipe
    from snax_forge.sandbox import __main__ as sandbox_cli

    root(tmp)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        args = [str(REPO / "recipes" / f"{recipe}.json"), "--out", f"out/sandbox/{name}"]
        assert sandbox_cli.main([*args, *(a for s in sets for a in ("--set", s))]) == 0
        steps = Recipe.load(REPO / "recipes" / f"{recipe}.json").steps  # the last: the bound graph
        last = f"out/sandbox/{name}/{len(steps)}_{steps[-1].transform}.snaxdfg"
        assert design_cli.main([last, "--platform",
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


def files(folder: Path) -> list[str]:
    """Every file below ``folder``, as paths from it, sorted."""
    return sorted(f.relative_to(folder).as_posix() for f in folder.rglob("*") if f.is_file())


def parsed(path: Path):
    """An hjson file as plain dicts and lists."""
    return json.loads(json.dumps(hjson.loads(path.read_text())))


def fake_emit(monkeypatch) -> SimpleNamespace:
    """The bundle's ``elaborate`` writes ``SV`` and lists the returned ``.listed`` in
    emitted.json (``SV`` unless a test changes it); the checkout is ``COMMIT``, clean."""
    fake = SimpleNamespace(listed=list(SV))

    def elaborate(spec_json: Path, out_dir: Path) -> list[Path]:
        tag = json.loads(spec_json.read_text())["tag"]
        for f in SV:
            (out_dir / f).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / f).write_text(f"module {Path(f).stem};\nendmodule\n")
        emitted = {"rtl": str(spec_json), "top": f"{tag}_shell_wrapper",
                   "modules": [Path(f).stem for f in fake.listed], "files": fake.listed}  # fmt: skip
        (out_dir / EMITTED_JSON).write_text(json.dumps(emitted))
        return [*(out_dir / f for f in fake.listed), out_dir / EMITTED_JSON]

    monkeypatch.setattr(bundle_mod, "elaborate", elaborate)
    monkeypatch.setattr(bundle_mod, "provenance", lambda checkout: (COMMIT, False))
    return fake


def validate(name: str, value: object) -> None:
    """Raise jsonschema.ValidationError unless ``value`` is a ``name`` file's content (the
    generator's schemas, gen/schemas.py)."""
    Draft202012Validator(schema(name)).validate(value)
