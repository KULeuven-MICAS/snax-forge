"""bicg_v4 on the generator: s's sums in a bank of R rows (D191), r read once a row and held
(D190), A read once (D189).

accelerator.json's banked unit and held port, the program's ROWS and job writes, r's reader once
a row, the Chisel specs' vectors, the control's row register and the estimate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.design.point import DesignPoint
from snax_forge.estimate import estimate
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


def test_the_bank_is_priced_by_its_forms(generated):
    """D193: the rows-1 Mac entry plus X97's forms, one row loading a fire; X97 measured the same
    bank alone at 16 515 um2 and 11.43 pJ a fire."""
    tmp, path = generated
    p = DesignPoint.load(tmp / path)
    (g,) = estimate(p.graph, p.platform).groups
    b = g.cost.units["mac"]
    assert b.found == "bank"
    assert 0.98 < (b.comb_um2 + b.flops_um2) / 16515 < 1.0
    assert 0.95 < b.e_fire_pj / 11.43 < 1.0


@pytest.fixture(scope="module")
def seeded(tmp_path_factory) -> tuple[Path, str]:
    """mvt_v6: x2's sums in a seeded bank of 8 rows of 8 (D194), x1's tree seeded per row."""
    tmp = tmp_path_factory.mktemp("mvt_v6")
    path = designed(tmp, "mvt_v6", "mvt_v6")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(tmp)
        generate(path, tmp / "gvsoc", tmp)
    return tmp, path


def test_a_seeded_bank_reads_its_seed_a_beat_a_row(seeded):
    """The seed is neither held nor carried: x2's 64 sums read as 8 beats of 8 at the job's start,
    the same words the bank writes back at its end."""
    tmp, path = seeded
    d = json.loads((tmp / "gvsoc" / "accelerator.json").read_text())
    (g,) = d["accelerators"][0]["groups"]
    m = next(u for u in g["units"] if u["name"] == "mac_x2")
    assert m["params"] == {"accumulate": True, "regOut": False, "rows": 8}
    assert m["ports"]["seed"] == {"port": "seed", "dir": "in", "dtype": "int32", "lanes": 8}
    assert m["ports"]["b"]["hold"] is True
    main = (tmp / "gvsoc" / "main.c").read_text()
    for line in ("csrw_ss(MVT_RW_JOBS_VSUM_X1, 64);", "csrw_ss(MVT_RW_JOBS_MAC_X2, 1);",
                 "csrw_ss(MVT_RW_MAC_X2_JOB_BEATS, 512);", "csrw_ss(MVT_RW_MAC_X2_ROWS, 8);"):  # fmt: skip
        assert line in main
    tl = task_list(DesignPoint.load(tmp / path))
    (seed,) = [s for s in tl.steps if getattr(s, "task_name", "").endswith("_mac_x2_seed")]
    assert seed.values["temporal_bounds"] == [8, 1]  # one beat a row, one strip


def test_a_seeded_bank_costs_the_one_pass_no_cycle(seeded):
    """512 firings and the ramp, as bicg_v4 at W = 8; the rule's vectors count the same."""
    tmp, path = seeded
    p = DesignPoint.load(tmp / path)
    (g,) = groups(p.graph)
    assert g.cycles == 515 and g.bits["mac_x2"] == 8 * 8 * 32 + 8
    (t,) = vectors(tmp / path, tmp)["tasks"]
    assert t["counter"] == 515
    (e,) = estimate(p.graph, p.platform).groups
    assert e.cost.units["mac_x2"].found == "bank"  # D194: the rows-1 Mac plus D193's forms
