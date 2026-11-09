"""Every forge-set recipe and mvt's compile (MVT1, the y1 clash): the generator's data.h and main.c
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

from snax_forge.common import RESERVED_C, c_name
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
MVT = ["mvt_1", "mvt_1_dot2", "mvt_2", "mvt_1_reuse", "mvt_1_dot2_reuse", "mvt_2_reuse"]
RECIPES = list(dict.fromkeys([d.recipe for d in SETS["full"]] + MVT))


def test_a_container_named_as_a_runtime_symbol_gets_a_c_name_of_its_own():
    assert c_name("y1") == "y1_l2" and c_name("A") == "A" and c_name("x1") == "x1"
    assert {"y0", "y1", "yn", "j0", "j1", "jn", "int", "main"} <= RESERVED_C


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


@pytest.mark.slow
def test_mvt_names_y1_as_its_own(tmp_path):
    """mvt's data.h declares y1 as ``y1_l2``, and main.c moves it by that name."""
    out = _generated(tmp_path, "mvt_2")
    data, main = (out / "data.h").read_text(), (out / "main.c").read_text()
    assert "const int64_t y1_l2[16]" in data and " y1[" not in data
    assert "y1_l2, 128);" in main
