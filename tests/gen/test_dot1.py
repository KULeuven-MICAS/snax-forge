"""dot 1 through the generator (D148, C§17): one accelerator ``dot`` of two groups, ``mul``
and ``sum``, in encapsulate's order, each with its own CSRs and a task of its own.

The description is C§17's (tests/gen/test_accelerator.py's ``dot``): MUL at offsets 0-3, SUM
at 4-8 (rw 5, ro 4); the header names every CSR with its group; the streamer holds every
unit's movers, readers mul.a, mul.b, sum.a and writers mul.out, sum.out; the manifest lists
both tasks with their group's values and their unit's trace path. Without encapsulate the
two groups are two accelerators, which the platform's one slot refuses (gen.instances).
"""

from __future__ import annotations

import json
from dataclasses import replace

import hjson
import pytest

from snax_forge.design import DesignPoint
from snax_forge.gen import GenError, describe, generate
from snax_forge.gen.rtl import write_fixture
from tests.design.helpers import FIXTURES

from .helpers import bound_dot1, point
from .test_accelerator import dot

POINT = "design/dot_1/design_point.json"


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dot_1")
    path = point(tmp, name="dot_1", graph=bound_dot1())
    return tmp, path, generate(path, tmp / "out", root=tmp)


def test_the_description_is_c17s(made):
    tmp, path, _ = made
    assert describe(DesignPoint.load(path), POINT) == replace(dot(), design_point=POINT)
    assert (tmp / "out" / "accelerator.json").read_text() == replace(
        dot(), design_point=POINT
    ).to_json()


def test_the_header_names_each_csr_with_its_group(made):
    _, _, m = made
    first = m.csrs.first
    names = ["MUL_START", "MUL_JOBS", "MUL_STATUS", "MUL_COUNTER", "SUM_START", "SUM_JOBS",
             "SUM_JOB_BEATS", "SUM_STATUS", "SUM_COUNTER"]  # fmt: skip
    access = ["RW", "RW", "RO", "RO", "RW", "RW", "RW", "RO", "RO"]
    want = {f"DOT_{a}_{n}": first + i for i, (a, n) in enumerate(zip(access, names, strict=True))}
    assert m.csrs.accelerator == want


def test_the_streamer_holds_every_units_movers(made):
    tmp, _, _ = made
    text = (tmp / "out" / "snax_forge_streamer.hjson").read_text()
    s = hjson.loads(text)
    assert len(s["data_reader_params"]["num_channel"]) == 3  # mul.a, mul.b, sum.a
    assert len(s["data_writer_params"]["num_channel"]) == 2  # mul.out, sum.out
    assert s["data_writer_params"]["spatial_bounds"] == [[4], [1]]


def test_the_slot_counts_both_groups_csrs(made):
    tmp, _, m = made
    cluster = json.loads("\n".join(
        x for x in (tmp / "out" / m.cluster).read_text().splitlines() if not x.startswith("//")
    ))  # fmt: skip
    (slot,) = cluster["snax_forge_core_template"]["snax_acc_cfg"]
    assert (slot["snax_num_rw_csr"], slot["snax_num_ro_csr"]) == (5, 4)


def test_the_manifest_lists_both_tasks_with_their_units_trace(made):
    _, _, m = made
    assert m.trace == "snax_acc0"
    assert [t.to_dict() for t in m.tasks] == [
        {"name": "mult_mul", "beats": 16, "csrs": {"JOBS": 16}, "trace": "snax_acc0/unit_mul"},
        {"name": "sum_sum", "beats": 16, "csrs": {"JOBS": 1, "JOB_BEATS": 16},
         "trace": "snax_acc0/unit_sum"},
    ]  # fmt: skip


def test_main_c_includes_dots_header(made):
    tmp, _, _ = made
    assert '#include "dot_csr.h"' in (tmp / "out" / "main.c").read_text()


def test_the_rtl_fixture_has_no_vectors(made, tmp_path, monkeypatch):
    """vectors.json is a one-task fixture's; dot 1's spec drives each group itself."""
    tmp, path, _ = made
    monkeypatch.chdir(tmp)
    files = write_fixture(path, tmp_path / "fx", tmp)
    assert files == ["accelerator.json", "snax_forge_streamer.hjson"]
    assert sorted(f.name for f in (tmp_path / "fx").iterdir()) == sorted(files)


def test_without_encapsulate_the_groups_are_two_accelerators(tmp_path):
    from snax_forge.dfg import Graph

    g = Graph.load(FIXTURES / "dot_accelerated.snaxdfg")
    with pytest.raises(GenError) as e:
        generate(point(tmp_path, name="dot", graph=g), tmp_path / "out", root=tmp_path)
    assert e.value.code == "gen.instances" and "2 accelerators ['mul', 'sum']" in str(e.value)
