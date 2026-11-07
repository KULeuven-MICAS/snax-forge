"""axpy's K1 through design, LOWER, the generator and the program (C§9, C§17).

axpy bound with its slot k1 read from ``alpha`` (a kernel's scalar argument): the design gives
alpha no streamer and no L1 layout, only the core's memory (L2 on snax_forge); the accelerator
task's values hold ``{"k1": {"from": "alpha"}}``; accelerator.json's unit has ``scalars``
``{"k1": <opType>}`` and an R/W CSR after JOB_BEATS (``K1``, or ``K1_LO`` and ``K1_HI`` at 64
bits) with ``pin`` k1; the header names it ``ACC_RW_K1``; data.h holds alpha and main.c writes
``alpha[0]`` into K1 before START. Baked (``bake_scalar``), k1 is a param and none of it
appears. The worked example: W = 2, N = 4, JOBS 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from snax_forge.design import Design, Platform, run_checks
from snax_forge.design.point import DesignPoint
from snax_forge.dfg import Graph
from snax_forge.gen import Description, generate
from snax_forge.lower import TaskListError, check_values, task_list
from snax_forge.sandbox import bake_scalar, bind, fuse_map, split_map
from tests.design.helpers import SNAX_FORGE
from tests.helpers import FIXTURES

from .helpers import point, validate

BIND = {"brm": "simd", "implementation": "chisel_tiled_spatial", "instance": "acc"}


def axpy(dtype: str = "int32", w: int = 2, n: int = 4, baked: int | None = None) -> Graph:
    """axpy split by ``w``, fused and bound (k1 from alpha, or baked at ``baked``), every
    container at ``dtype``, N = ``n``."""
    d = Graph.load(FIXTURES / "axpy.snaxdfg").to_dict()
    for c in d["containers"].values():
        c["dtype"] = dtype
    d["symbols"]["N"] = n
    g = Graph.from_dict(d)
    g = split_map(split_map(g, "mult_map", w), "augassign_19_4_map", w)
    g = fuse_map(g, "mult_map", "augassign_19_4_map")
    if baked is not None:
        g = bake_scalar(g, "alpha", baked)
    return bind(g, node="mult_map_s", **BIND)


@pytest.fixture(scope="module")
def made(tmp_path_factory) -> dict[str, tuple[Path, Path]]:
    """Each variant's design point and gen-gvsoc folder: int32 and int64 with k1 from alpha,
    and int32 baked at 3."""
    tmp = tmp_path_factory.mktemp("axpy")
    out = {}
    for name, graph in (("axpy", axpy()), ("axpy_i64", axpy("int64")),
                        ("axpy_baked", axpy(baked=3))):  # fmt: skip
        path = point(tmp, name=name, graph=graph)
        folder = tmp / "out" / "gvsoc" / name
        generate(path, folder, root=tmp)
        out[name] = (path, folder)
    return out


def test_the_scalar_has_no_streamer_and_lives_in_l2_only(made):
    p = DesignPoint.load(made["axpy"][0])
    assert sorted(p.streamers) == ["acc_a", "acc_b", "acc_out"]
    assert sorted(p.memory.layouts["alpha"]) == ["l2"]
    assert sorted(p.memory.layouts["x"]) == ["l1", "l2"]


def test_the_design_checks_accept_the_slot_connector():
    assert run_checks(Design(Platform.load(SNAX_FORGE), axpy())) == []


def test_the_accelerator_task_reads_k1_from_alpha(made):
    for name, k1 in (("axpy", {"k1": {"from": "alpha"}}), ("axpy_baked", {})):
        p = DesignPoint.load(made[name][0])
        tasks = task_list(p)
        (acc,) = [s for s in tasks.steps if getattr(s, "type", None) == "accel"]
        assert acc.values == {"n": 2} | k1
        check_values(tasks, p)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"n": 2}, r"parameters \['n'\], registers \['n', 'k1'\]"),
        ({"n": 2, "k1": 3}, r"k1 is 3; a slot read from a container is \{'from': <container>\}"),
        ({"n": 2, "k1": {"from": "x"}}, r"k1 from 'x', which is no kernel's scalar argument"),
    ],
)
def test_check_values_refuses_a_slot_without_its_container(made, values, message):
    p = DesignPoint.load(made["axpy"][0])
    tasks = task_list(p)
    (acc,) = [s for s in tasks.steps if getattr(s, "type", None) == "accel"]
    acc.values = values
    with pytest.raises(TaskListError, match=message):
        check_values(tasks, p)


@pytest.mark.parametrize(
    ("name", "csrs", "scalars"),
    [
        ("axpy", [("START", 0, 1), ("JOBS", 1, 32), ("K1", 2, 32), ("STATUS", 3, 2),
                  ("COUNTER", 4, 32)], {"k1": "int32"}),
        ("axpy_i64", [("START", 0, 1), ("JOBS", 1, 32), ("K1_LO", 2, 32), ("K1_HI", 3, 32),
                      ("STATUS", 4, 2), ("COUNTER", 5, 32)], {"k1": "int64"}),
        ("axpy_baked", [("START", 0, 1), ("JOBS", 1, 32), ("STATUS", 2, 2),
                        ("COUNTER", 3, 32)], None),
    ],
)  # fmt: skip
def test_the_unit_has_its_scalars_and_their_csrs(made, name, csrs, scalars):
    d = json.loads((made[name][1] / "accelerator.json").read_text())
    validate("accelerator", d)
    desc = Description.from_dict(d)
    (unit,) = desc.units
    assert [(c.name, c.offset, c.width) for _, _, c in desc.csrs()] == csrs
    assert [c.pin for c in unit.csrs] == [k for k in (scalars or {}) for _ in range(
        2 if scalars[k] == "int64" else 1)]  # fmt: skip
    assert d["accelerators"][0]["groups"][0]["units"][0].get("scalars") == scalars
    assert unit.params == {"op": "axpby"} | ({} if scalars else {"k1": 3})
    (acc,) = desc.accelerators
    assert (acc.rw, acc.ro) == (2 + len(csrs) - 4, 2)


def test_a_units_scalars_must_be_its_pinned_csrs(made):
    d = json.loads((made["axpy"][1] / "accelerator.json").read_text())
    d["accelerators"][0]["groups"][0]["units"][0]["scalars"] = {"k1": "int64"}
    with pytest.raises(ValueError, match=r"unit acc's scalars \{'k1': 'int64'\} are the R/W CSRs "
                       r"\[\('K1_LO', 'k1'\), \('K1_HI', 'k1'\)\] \(name, pin\), not "
                       r"\[\('K1', 'k1'\)\]"):  # fmt: skip
        Description.from_dict(d)


def test_the_program_writes_k1_from_data_h_before_start(made):
    folder = made["axpy"][1]
    main = (folder / "main.c").read_text()
    assert main.index("csrw_ss(ACC_RW_JOBS, 2);") < main.index("csrw_ss(ACC_RW_K1, alpha[0]);")
    assert main.index("csrw_ss(ACC_RW_K1, alpha[0]);") < main.index("csrw_ss(ACC_RW_START, 1);")
    assert "#define ACC_RW_K1 " in next((folder / "include").glob("acc_csr.h")).read_text()
    data = (folder / "data.h").read_text()
    assert "const int32_t alpha[16] __attribute__((aligned(64))) = {\n    3, 0," in data
    assert "int32_t y[16]" in data and "const int32_t y_golden[16]" in data


def test_at_int64_the_program_writes_two_words(made):
    main = (made["axpy_i64"][1] / "main.c").read_text()
    assert (
        "csrw_ss(ACC_RW_K1_LO, (uint32_t)(uint64_t)(int64_t)alpha[0]);\n"
        "    csrw_ss(ACC_RW_K1_HI, (uint32_t)((uint64_t)(int64_t)alpha[0] >> 32));"
    ) in main


def test_baked_the_program_writes_no_k1_and_data_h_has_no_alpha(made):
    folder = made["axpy_baked"][1]
    assert "K1" not in (folder / "main.c").read_text()
    assert "alpha" not in (folder / "data.h").read_text()


@pytest.mark.parametrize(
    ("params", "read", "dtypes", "fill_params", "scalars"),
    [
        ({"k1": 3}, (), ("int32", "int32"), {"k1": 3}, {}),
        ({}, ("k1",), ("int32", "int32"), {}, {"k1": "int32"}),
        ({"k1": -2, "k2": 5}, (), ("int64", "int64"), {"k1": -2, "k2": 5}, {}),
        ({}, ("k1", "k2"), ("int16", "int32"), {}, {"k1": "int32", "k2": "int32"}),
        ({"k2": 7}, ("k1",), ("int64", "int64"), {"k2": 7}, {"k1": "int64"}),
    ],
)
def test_simd_lanes_takes_axpby_with_each_slot_baked_read_or_k2_unset(
    params, read, dtypes, fill_params, scalars
):
    """A baked slot is a const in the params, one read from a container a scalar at opType
    (its CSR, the datapath's pin); k2 unset is ``+ b``."""
    from snax_forge.brm import load_brm
    from snax_forge.gen.blocks import simd_lanes

    inst = load_brm("simd").resolve("chisel_tiled_spatial", {"op": "axpby"} | params, read)
    fill = simd_lanes(inst, dict(zip(("a", "b", "out"), (*dtypes, dtypes[1]), strict=True)))
    assert fill.params == {"op": "axpby"} | fill_params and fill.scalars == scalars


def test_one_stream_reads_k_from_a_container_too():
    from snax_forge.brm import load_brm
    from snax_forge.gen.blocks import simd_lanes

    inst = load_brm("simd").resolve("chisel_tiled_spatial", {"op": "mul", "streams": 1}, ["k"])
    fill = simd_lanes(inst, {"a": "int16", "out": "int16"})
    assert fill.params == {"op": "mul"} and fill.scalars == {"k": "int16"}
