"""bicg_v4 on the generator: s's sums in a bank of R rows, r read once a row and held,
A read once.

accelerator.json's banked unit and held port, the program's ROWS and job writes, r's reader once
a row, the Chisel specs' vectors, the control's row register and the estimate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.design.point import DesignPoint
from snax_forge.estimate.cycles import groups
from snax_forge.gen import generate

from .helpers import designed


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> tuple[Path, str]:
    tmp = tmp_path_factory.mktemp("bicg_v4")
    path = designed(tmp, "bicg_v4", "bicg_v4")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        generate(path, tmp / "gvsoc", tmp)
    return tmp, path


def _mac(tmp: Path) -> dict:
    d = json.loads((tmp / "gvsoc" / "accelerator.json").read_text())
    (acc,) = d["accelerators"]
    (g,) = acc["groups"]
    return next(u for u in g["units"] if u["name"] == "mac")


def test_the_program_writes_rows_and_each_chains_jobs(generated):
    tmp, _ = generated
    main = (tmp / "gvsoc" / "main.c").read_text()
    for line in ("csrw_ss(BICG_TYPES_RW_JOBS_SUM, 64);", "csrw_ss(BICG_TYPES_RW_JOBS_MAC, 1);",
                 "csrw_ss(BICG_TYPES_RW_SUM_JOB_BEATS, 16);",
                 "csrw_ss(BICG_TYPES_RW_MAC_JOB_BEATS, 1024);",
                 "csrw_ss(BICG_TYPES_RW_MAC_ROWS, 16);"):  # fmt: skip
        assert line in main


@pytest.fixture(scope="module")
def strips(tmp_path_factory) -> tuple[Path, str]:
    """bicg_v4_1: two strips of 32 columns, q's sum carried across them."""
    tmp = tmp_path_factory.mktemp("bicg_v4_1")
    path = designed(tmp, "bicg_v4_1", "bicg_v4_1")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        generate(path, tmp / "gvsoc", tmp)
    return tmp, path


def test_the_strips_cost_no_cycle(strips):
    """Two strips fire the heads 1 024 times, as one strip does; the bank is half as deep."""
    tmp, path = strips
    (g,) = groups(DesignPoint.load(tmp / path).graph)
    assert g.cycles == 1027 and g.bits["mac"] == 8 * 4 * 32 + 8


@pytest.fixture(scope="module")
def seeded(tmp_path_factory) -> tuple[Path, str]:
    """mvt_v6: x2's sums in a seeded bank of 8 rows of 8, x1's tree seeded per row."""
    tmp = tmp_path_factory.mktemp("mvt_v6")
    path = designed(tmp, "mvt_v6", "mvt_v6")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        generate(path, tmp / "gvsoc", tmp)
    return tmp, path
