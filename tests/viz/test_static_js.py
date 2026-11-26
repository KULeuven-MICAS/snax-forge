"""The viewers' scripts parse: each module under ``snax_forge/viz/static/`` through ``node --check``
as an ES module, so a syntax slip fails here rather than as a blank page (skipped without node)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[2] / "snax_forge" / "viz" / "static"
SCRIPTS = sorted(STATIC.glob("*.js"))


@pytest.mark.skipif(shutil.which("node") is None, reason="no node on PATH")
@pytest.mark.parametrize("script", SCRIPTS, ids=[s.name for s in SCRIPTS])
def test_each_script_parses_as_a_module(script: Path, tmp_path: Path) -> None:
    module = tmp_path / f"{script.stem}.mjs"
    module.write_text(script.read_text())
    run = subprocess.run(
        ["node", "--check", str(module)], capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stderr
