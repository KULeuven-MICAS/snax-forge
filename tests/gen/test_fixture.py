"""snax-forge reproduces snax-gvsoc's fixtures (D134).

Each fixture is a folder of the snax-gvsoc submodule's ``forge/``, the
generator's output for one design point, checked in untouched
(``forge/README.md`` there): vecadd, vecmul, vsum, vsum_tree, matadd (the first
2-D bind, two temporal loops per streamer), rowsum and rowsum_serial (a restart
every T = N / W beats). Its refresh commands are run
here as written, from a temporary folder that stands for the repo root (the
recipe and the platform the checked-in ones, the platform's source hjson
copied into its ``platforms/``), so the design point is
``out/design/<design>/design_point.json`` and every header comment names it
as the fixture's do. The generated folder must equal the fixture byte for
byte, file list included. Skipped where the submodule is not checked out
(only CI's gvsoc job fetches it).

Until snax-gvsoc's PACK0 R2 checks in the fixtures PACK0 R3 generates, the
checked-in ones are R2b's: each file must still be byte-identical but the
accelerator file, which must be the old one with ``data_width`` and ``signed``
replaced by the four type lists in their place (``in_widths``, ``in_signed``,
``out_widths``, ``out_signed``; every port the old type); and matadd, int32
since PACK0 R3, differs throughout, so it waits for those fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.design import __main__ as design_cli
from snax_forge.gen import generate
from snax_forge.sandbox import Recipe
from snax_forge.sandbox import __main__ as sandbox_cli
from tests.design.helpers import REPO, SNAX_FORGE

from .helpers import SUB, files, parsed, root

FORGE = SUB / "forge"
# Each fixture's design: its recipe, on platforms/snax_forge.json
DESIGNS = ["vecadd", "vecmul", "vsum", "vsum_tree", "matadd", "rowsum", "rowsum_serial"]
ACC = "snax_forge_acc.hjson"
NARROW_SINCE_R3 = {"matadd"}  # int32 again (PACK0 R3): its fixture comes with snax-gvsoc R2

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


def with_type_lists(old: dict) -> dict:
    """An R2b accelerator file with ``data_width`` and ``signed`` as the four type lists, in
    the place ``data_width`` had, every port of that one type."""
    width, signed = old["data_width"], old["signed"]
    out = {}
    for k, v in old.items():
        if k == "data_width":
            out |= {"in_widths": [width] * len(old["inputs"]),
                    "in_signed": [signed] * len(old["inputs"]),
                    "out_widths": [width] * len(old["outputs"]),
                    "out_signed": [signed] * len(old["outputs"])}  # fmt: skip
        elif k != "signed":
            out[k] = v
    return out


@pytest.mark.parametrize(
    "design",
    [pytest.param(d, marks=pytest.mark.skip(reason="int32 since PACK0 R3; snax-gvsoc R2"))
     if d in NARROW_SINCE_R3 else d for d in DESIGNS],
)  # fmt: skip
def test_the_fixture_is_reproduced(tmp_path, design):
    fixture = FORGE / design
    out = regenerate(tmp_path, design)
    assert files(out) == files(fixture)
    for f in files(fixture):
        if f == ACC and "data_width" in parsed(fixture / f):  # an R2b fixture (module doc)
            new = parsed(out / f)
            assert json.dumps(new) == json.dumps(with_type_lists(parsed(fixture / f))), f
            continue
        assert (out / f).read_bytes() == (fixture / f).read_bytes(), f
