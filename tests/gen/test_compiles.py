"""Every full-set recipe's compile (mvt's y1 clash among them): the generator's data.h and main.c
against the runtime's headers, as ``make sw`` includes them, so a container named after a
libc or runtime symbol (mvt's ``y1``, math.h's Bessel function) is caught here and not at a
flow's ``sw`` step. Syntax only (clang's ``-fsyntax-only`` for riscv32): no runtime library,
no link. Skipped without a clang that targets riscv32 or the snax-gvsoc submodule."""

from __future__ import annotations

import shutil
import subprocess
from functools import cache
from pathlib import Path

import pytest

from snax_forge.flow.forge_set import SETS
from tests.helpers import REPO

from .helpers import designed

SW = REPO / "gvsoc" / "snax-gvsoc" / "sw"
INCLUDES = [
    "generated/snax_forge",
    "runtime/rtl-generic/src",
    "runtime/common",
    "snRuntime/api",
    "snRuntime/api/omp",
    "snRuntime/src",
    "snRuntime/src/omp",
    "deps/riscv-opcodes",
    "math/include",
]  # fmt: skip (common.mk's INCDIRS)
FLAGS = [
    "--target=riscv32-unknown-elf",
    "-march=rv32imafdzfh",
    "-mabi=ilp32d",
    "-ffast-math",
    "-fno-builtin-printf",
    "-fopenmp",
    "-fsyntax-only",
]  # fmt: skip (toolchain.mk's)
RECIPES = list(dict.fromkeys(d.recipe for d in SETS["full"]))  # mvt's y1 among them


def _generated(tmp: Path, recipe: str) -> Path:
    """gen-gvsoc's folder for the recipe's design point, made under ``tmp`` as the root."""
    from snax_forge.gen.gvsoc import generate

    out = tmp / "out" / "gvsoc" / recipe
    generate(tmp / designed(tmp, recipe, recipe), out, root=tmp)
    return out


@cache
def _clang() -> str | None:
    clang = shutil.which("clang")
    if clang is None or not (SW / "math" / "include" / "math.h").is_file():
        return None
    out = subprocess.run([clang, "--print-targets"], capture_output=True, text=True, check=False)
    return clang if "riscv32" in out.stdout else None


@pytest.mark.slow
@pytest.mark.parametrize("recipe", RECIPES)
def test_the_generated_program_compiles(tmp_path, recipe):
    clang = _clang()
    if clang is None:
        pytest.skip("no clang for riscv32, or no snax-gvsoc submodule")
    out = _generated(tmp_path, recipe)
    cmd = [clang, *FLAGS, f"-I{out}", f"-I{out / 'include'}",
           *(f"-I{SW / d}" for d in INCLUDES), str(out / "main.c")]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr[-2000:]
