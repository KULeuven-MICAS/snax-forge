"""snax-forge reproduces snax-gvsoc's fixtures.

Each fixture is a folder of the snax-gvsoc submodule's ``forge/``, the generator's output for
one design point (``DESIGNS``, each from its recipe unless ``FROM`` says otherwise), checked in
untouched (``forge/README.md`` there). Its refresh commands are run here as written, from a
temporary folder that stands for the repo root (the checked-in recipe and platform, the
platform's source hjson copied into its ``platforms/``), so every header comment names
``out/design/<design>/design_point.json`` as the fixture's do. The generated folder must equal
the fixture byte for byte, file list included, and every folder of ``forge/`` must be a design.
Skipped where the submodule is not checked out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from snax_forge.gen import generate
from tests.helpers import requires_submodule

from .helpers import SUB, designed, files

FORGE = SUB / "forge"
# Each fixture's design: its recipe, on platforms/snax_forge.json
DESIGNS = [
    "vecadd", "vecmul", "vsum", "vsum_tree", "matadd", "rowsum", "rowsum_serial",
    "vecadd_i32", "vecmul_i32", "vsum_tree_i32", "vecadd_i8", "vsum_tree_i8",
    "vecadd_n7", "vsum_tree_n21", "rowsum_3x7", "vecmul_i32_i64", "dot_1", "dot_1_enc",
    "dot_2", "dot_3", "dot_4", "dot_5", "dot_6",
    "matvec_1", "matvec_2", "matvec_3", "matvec_4", "matvec_5", "matvec_6", "relu",
    "axpy", "axpy_baked", "rowsum_2", "rowsum_2_padded", "matvec_seed", "matvec_seed_serial",
    "dot_1_shared", "mvt_1_dot2_shared",
]  # fmt: skip
# A fixture made from another recipe, or with --set values: its recipe and its sets
FROM = {"rowsum_2_padded": ("rowsum_2", ("memory.passes.layout=padded_rows",)),
        "mvt_1_dot2_shared": ("mvt_1_dot2_shared", ("W=8",))}  # fmt: skip

pytestmark = requires_submodule


def regenerate(tmp: Path, design: str) -> Path:
    """The fixture's refresh commands under ``tmp`` as the root; the generated folder."""
    recipe, sets = FROM.get(design, (design, ()))
    point = designed(tmp, recipe, design, sets)
    out = tmp / "gvsoc" / design
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)  # the files name the design point by its path from the root
        generate(point, out, root=tmp)
    return out


def test_every_folder_is_a_design():
    assert sorted(p.name for p in FORGE.iterdir() if p.is_dir()) == sorted(DESIGNS)


@pytest.mark.parametrize("design", DESIGNS)
def test_the_fixture_is_reproduced(tmp_path, design):
    fixture = FORGE / design
    out = regenerate(tmp_path, design)
    assert files(out) == files(fixture)
    for f in files(fixture):
        assert (out / f).read_bytes() == (fixture / f).read_bytes(), f
