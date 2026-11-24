"""snax-forge reproduces snax-gvsoc's fixtures.

Each fixture is a folder of the snax-gvsoc submodule's ``forge/``, the generator's output for
one design point (``FORGE_FIXTURES`` in tests/helpers.py: its recipe and ``--set`` values), checked in
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
from tests.helpers import FORGE_FIXTURES, requires_submodule

from .helpers import SUB, designed, files

FORGE = SUB / "forge"
DESIGNS = list(FORGE_FIXTURES)

pytestmark = requires_submodule


def regenerate(tmp: Path, design: str) -> Path:
    """The fixture's refresh commands under ``tmp`` as the root; the generated folder."""
    recipe, sets = FORGE_FIXTURES[design]
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
