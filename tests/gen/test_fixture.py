"""snax-forge reproduces snax-gvsoc's fixtures (GEN3 step 5, D128).

Each fixture is a folder of the snax-gvsoc submodule's ``forge/``, the
generator's output for one design point, checked in untouched
(``forge/README.md`` there). Its refresh commands are run here as written,
from a temporary folder that stands for the repo root (the recipe and the
platform the checked-in ones), so the design point is
``out/design/<design>/design_point.json`` and every header comment names it
as the fixture's do. The generated folder must equal the fixture byte for
byte, file list included. Skipped where the submodule is not checked out
(CI never fetches it) or predates ``forge/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from snax_forge.design import __main__ as design_cli
from snax_forge.gen import generate
from snax_forge.sandbox import __main__ as sandbox_cli
from tests.design.helpers import REPO, SNAX_ALU

from .helpers import SUB

FORGE = SUB / "forge"
# Each fixture's design: its recipe, on platforms/snax_alu.json
DESIGNS = ["vecadd", "vecmul"]

pytestmark = [
    pytest.mark.skipif(
        not (SUB / "snax").is_dir(),
        reason="the snax-gvsoc submodule is not checked out (CI never fetches it)",
    ),
    pytest.mark.skipif(
        (SUB / "snax").is_dir() and not FORGE.is_dir(),
        reason="the snax-gvsoc submodule has no forge/: bump the submodule",
    ),
]


def files(folder: Path) -> list[str]:
    """The folder's files; gvrun's ``__pycache__`` (ignored by snax-gvsoc's git) left out."""
    return sorted(f.relative_to(folder).as_posix() for f in folder.rglob("*")
                  if f.is_file() and "__pycache__" not in f.parts)  # fmt: skip


def regenerate(root: Path, design: str) -> Path:
    """The fixture's refresh commands under ``root``; the generated folder."""
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        recipe = REPO / "recipes" / f"{design}.json"
        assert sandbox_cli.main([str(recipe), "--out", f"out/sandbox/{design}"]) == 0
        assert design_cli.main([f"out/sandbox/{design}/2_bind.snaxdfg", "--platform",
                                str(SNAX_ALU), "--out", f"out/design/{design}"]) == 0  # fmt: skip
        out = root / "gvsoc" / design
        generate(f"out/design/{design}/design_point.json", out, root=root)
    return out


@pytest.mark.parametrize("design", DESIGNS)
def test_the_fixture_is_reproduced(tmp_path, design):
    fixture = FORGE / design
    if not fixture.is_dir():
        pytest.skip(f"the snax-gvsoc submodule has no forge/{design}: bump the submodule")
    out = regenerate(tmp_path, design)
    assert files(out) == files(fixture)
    for f in files(fixture):
        assert (out / f).read_bytes() == (fixture / f).read_bytes(), f
