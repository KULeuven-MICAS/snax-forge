"""snax-forge reproduces snax-gvsoc's fixtures (D134).

Each fixture is a folder of the snax-gvsoc submodule's ``forge/``, the
generator's output for one design point, checked in untouched
(``forge/README.md`` there): vecadd, vecmul, vsum, vsum_tree, matadd (int32,
the first 2-D bind, two temporal loops per streamer), rowsum and rowsum_serial
(a restart every T = N / W beats); the narrow ones (vecadd_i32, vecmul_i32,
vsum_tree_i32, vecadd_i8, vsum_tree_i8: D142, D143) and the padded tails
(vecadd_n7, vsum_tree_n21, rowsum_3x7: D144). Its refresh commands are run
here as written, from a temporary folder that stands for the repo root (the
recipe and the platform the checked-in ones, the platform's source hjson
copied into its ``platforms/``), so the design point is
``out/design/<design>/design_point.json`` and every header comment names it
as the fixture's do. The generated folder must equal the fixture byte for
byte, file list included, and every folder of ``forge/`` must be one of them.
Skipped where the submodule is not checked out (only CI's gvsoc job fetches
it).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from snax_forge.design import __main__ as design_cli
from snax_forge.gen import generate
from snax_forge.sandbox import Recipe
from snax_forge.sandbox import __main__ as sandbox_cli
from tests.design.helpers import REPO, SNAX_FORGE

from .helpers import SUB, files, root

FORGE = SUB / "forge"
# Each fixture's design: its recipe, on platforms/snax_forge.json
DESIGNS = [
    "vecadd", "vecmul", "vsum", "vsum_tree", "matadd", "rowsum", "rowsum_serial",
    "vecadd_i32", "vecmul_i32", "vsum_tree_i32", "vecadd_i8", "vsum_tree_i8",
    "vecadd_n7", "vsum_tree_n21", "rowsum_3x7",
]  # fmt: skip

pytestmark = pytest.mark.skipif(
    not (SUB / "snax").is_dir(),
    reason="the snax-gvsoc submodule is not checked out (only CI's gvsoc job fetches it)",
)


def regenerate(tmp: Path, design: str) -> Path:
    """The fixture's refresh commands under ``tmp`` as the root; the generated folder."""
    root(tmp)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        recipe = REPO / "recipes" / f"{design}.json"
        bound = f"out/sandbox/{design}/{len(Recipe.load(recipe).steps)}_bind.snaxdfg"  # last step
        assert sandbox_cli.main([str(recipe), "--out", f"out/sandbox/{design}"]) == 0
        assert design_cli.main([bound, "--platform", str(SNAX_FORGE),
                                "--out", f"out/design/{design}"]) == 0  # fmt: skip
        out = tmp / "gvsoc" / design
        generate(f"out/design/{design}/design_point.json", out, root=tmp)
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
