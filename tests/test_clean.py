"""util/clean.py: out/ goes but the GVSoC build and the editor's sessions, which --all takes
too."""

from __future__ import annotations

import importlib.util

from tests.helpers import REPO

spec = importlib.util.spec_from_file_location("clean", REPO / "util" / "clean.py")
clean = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean)


def test_the_sessions_and_the_gvsoc_build_stay_without_all(tmp_path):
    for d in (
        "out/flow/vecadd",
        "out/sandbox/vecadd",
        "out/snax-gvsoc/install",
        "out/sessions/vecadd",
    ):
        (tmp_path / d).mkdir(parents=True)
    rel = lambda ps: [str(p.relative_to(tmp_path)) for p in ps]
    assert rel(clean.targets(tmp_path)) == ["out/flow", "out/sandbox"]
    assert rel(clean.targets(tmp_path, everything=True)) == ["out"]
