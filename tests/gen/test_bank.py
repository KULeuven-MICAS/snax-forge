"""bicg_v4 on the generator: s's sums in a bank of R rows (D191), r read once a row and held
(D190), A read once (D189).

accelerator.json's banked unit and held port, the program's ROWS and job writes, r's reader once
a row, the Chisel specs' vectors, the control's row register and the estimate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.design.point import DesignPoint
from snax_forge.estimate.bits import accelerator_registers
from snax_forge.estimate.cycles import groups
from snax_forge.gen import generate
from snax_forge.gen.rtl import vectors
from snax_forge.lower.derive import task_list

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


def test_the_mac_is_a_bank_of_16_rows_and_r_is_held(generated):
    tmp, _ = generated
    u = _mac(tmp)
    assert u["params"] == {"accumulate": True, "regOut": False, "rows": 16}
    assert u["ports"]["b"] == {"port": "b", "dir": "in", "dtype": "int8", "lanes": 4, "d_in": 8,
                               "broadcast": True, "hold": True}  # fmt: skip
    assert "hold" not in u["ports"]["a"]
    assert [(c["name"], c["width"]) for c in u["csrs"]] == [("JOB_BEATS", 32), ("ROWS", 5)]


def test_the_program_writes_rows_and_each_chains_jobs(generated):
    tmp, _ = generated
    main = (tmp / "gvsoc" / "main.c").read_text()
    for line in ("csrw_ss(BICG_TYPES_RW_JOBS_SUM, 64);", "csrw_ss(BICG_TYPES_RW_JOBS_MAC, 1);",
                 "csrw_ss(BICG_TYPES_RW_SUM_JOB_BEATS, 16);",
                 "csrw_ss(BICG_TYPES_RW_MAC_JOB_BEATS, 1024);",
                 "csrw_ss(BICG_TYPES_RW_MAC_ROWS, 16);"):  # fmt: skip
        assert line in main


def test_r_is_read_once_a_row(generated):
    """r's reader skips the bank's loop: 64 elements, 8 words through the slicer, not 1 024."""
    tmp, path = generated
    tl = task_list(DesignPoint.load(tmp / path))
    (r,) = [s for s in tl.steps if getattr(s, "task_name", "").endswith("_mac_b")]
    assert r.values["temporal_bounds"] == [8, 1] and r.values["temporal_strides"][0] == 8


def test_the_specs_vectors_count_the_bank(generated):
    tmp, path = generated
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        v = vectors(path, tmp)
    (task,) = v["tasks"]
    assert task["csrs"] == {"JOBS_SUM": 64, "JOBS_MAC": 1, "SUM_JOB_BEATS": 16,
                            "MAC_JOB_BEATS": 1024, "MAC_ROWS": 16}  # fmt: skip
    assert task["counter"] == 1027  # 1 024 firings of both heads, then the pipelines


def test_the_control_counts_the_row(generated):
    tmp, _ = generated
    regs = accelerator_registers(json.loads((tmp / "gvsoc" / "accelerator.json").read_text()))
    (ctrl,) = regs.values()
    assert ("mac_row", 4) in ctrl


def test_the_estimate_prices_the_bank_and_the_held_register(generated):
    tmp, path = generated
    (g,) = groups(DesignPoint.load(tmp / path).graph)
    assert g.cycles == 1027
    assert g.bits["mac"] == 16 * 4 * 32 + 8  # 16 rows of 4 int32 sums, r's int8 held


@pytest.fixture(scope="module")
def strips(tmp_path_factory) -> tuple[Path, str]:
    """bicg_v4_1: two strips of 32 columns, q's sum carried across them (D192)."""
    tmp = tmp_path_factory.mktemp("bicg_v4_1")
    path = designed(tmp, "bicg_v4_1", "bicg_v4_1")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        generate(path, tmp / "gvsoc", tmp)
    return tmp, path


def test_q_is_carried_across_the_strips(strips):
    tmp, path = strips
    d = json.loads((tmp / "gvsoc" / "accelerator.json").read_text())
    (g,) = d["accelerators"][0]["groups"]
    s = next(u for u in g["units"] if u["name"] == "sum")
    assert s["ports"]["seed"]["carry"] is True
    assert [c["name"] for c in s["csrs"]] == ["CARRY"]  # T = 8 fixed
    main = (tmp / "gvsoc" / "main.c").read_text()
    for line in ("csrw_ss(BICG_TYPES_RW_JOBS_SUM, 128);", "csrw_ss(BICG_TYPES_RW_JOBS_MAC, 2);",
                 "csrw_ss(BICG_TYPES_RW_MAC_JOB_BEATS, 512);", "csrw_ss(BICG_TYPES_RW_MAC_ROWS, 8);",
                 "csrw_ss(BICG_TYPES_RW_SUM_CARRY, 64);"):  # fmt: skip
        assert line in main
    tl = task_list(DesignPoint.load(tmp / path))
    (seed,) = [s for s in tl.steps if getattr(s, "task_name", "").endswith("_sum_seed")]
    assert seed.values["temporal_bounds"] == [32, 2]  # q's 32 words, once a strip


def test_the_strips_cost_no_cycle(strips):
    """Two strips fire the heads 1 024 times, as one strip does; the bank is half as deep."""
    tmp, path = strips
    (g,) = groups(DesignPoint.load(tmp / path).graph)
    assert g.cycles == 1027 and g.bits["mac"] == 8 * 4 * 32 + 8
