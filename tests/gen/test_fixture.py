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
    "vecadd_v1", "vecmul_v1", "vsum_v1", "vsum_v1_1", "matadd_v1", "rowsum_v1", "rowsum_v3",
    "vecadd_v1_1", "vecmul_v1_1", "vsum_v1_1_1", "vecadd_v1_2", "vsum_v1_1_2",
    "vecadd_v1_3", "vsum_v1_1_3", "rowsum_v1_1", "vecmul_v1_1_1", "dot_v1", "dot_v1_1",
    "dot_v2", "dot_v3", "dot_v4", "dot_v5", "dot_v6",
    "matvec_v1", "matvec_v2", "matvec_v3", "matvec_v4", "matvec_v5", "matvec_v6", "relu_v1",
    "axpy_v1", "axpy_v1_1", "rowsum_v2", "rowsum_v2_padded", "matvec_acc_v1", "matvec_acc_v2",
    "dot_v1_2", "mvt_v1_2_2", "vecadd_v1_1_w1", "mvt_v4_1_sequential",
    "mvt_v4_1_1_sequential", "mvt_v5_sequential", "mvt_v5_2_sequential",
]  # fmt: skip
# A fixture made from another recipe, or with --set values: its recipe and its sets
FROM = {"rowsum_v2_padded": ("rowsum_v2", ("memory.passes.layout=padded_rows",)),
        "mvt_v1_2_2": ("mvt_v1_2_2", ("W=8",)),
        "vecadd_v1_1_w1": ("vecadd_v1_1", ("W=1",)),
        "mvt_v4_1_sequential": ("mvt_v4_1", ("design.schedule=sequential",)),
        "mvt_v4_1_1_sequential": ("mvt_v4_1_1", ("design.schedule=sequential",)),
        "mvt_v5_sequential": ("mvt_v5", ("design.schedule=sequential",)),
        "mvt_v5_2_sequential": ("mvt_v5_2", ("design.schedule=sequential",))}  # fmt: skip

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
