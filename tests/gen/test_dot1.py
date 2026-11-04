"""dot 1 through the generator (D148, D157, C§16, C§17): ``recipes/dot_1_enc.json``, one
accelerator ``dot`` of two groups, ``vmul`` and ``vsum``, in encapsulate's order, each with its
own CSRs and a task of its own; ``recipes/dot_1.json``, the same groups as two accelerators,
two streamer sets of the one entry; and dot_serial, two accelerators ``mul`` and ``sum``.

dot_1_enc's description is C§17's (tests/gen/test_accelerator.py's ``dot``): VMUL at offsets
0-3, VSUM at 4-8 (rw 5, ro 4); the header names every CSR with its group; the set's streamer
holds every unit's movers, readers vmul.a, vmul.b, vsum.a and writers vmul.out, vsum.out; the
manifest lists both tasks with their group's values and their unit's trace path. dot 1's and
dot_serial's slots carry two sets each, its own streamer file, CSR counts and interconnect
group.
"""

from __future__ import annotations

import json
from dataclasses import replace

import hjson
import pytest

from snax_forge.design import DesignPoint
from snax_forge.gen import describe, generate
from snax_forge.gen.rtl import write_fixture

from .helpers import bound_dot1, bound_dot1_enc, bound_recipe, parsed, point
from .test_accelerator import dot, dot1, dot_serial

POINT = "design/dot_1_enc/design_point.json"


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dot_1_enc")
    path = point(tmp, name="dot_1_enc", graph=bound_dot1_enc())
    return tmp, path, generate(path, tmp / "out", root=tmp)


def test_the_description_is_c17s(made):
    tmp, path, _ = made
    assert describe(DesignPoint.load(path), POINT) == replace(dot(), design_point=POINT)
    assert dot().tag == "snax_forge_dot_1_enc"
    assert (tmp / "out" / "accelerator.json").read_text() == replace(
        dot(), design_point=POINT
    ).to_json()


def test_the_header_names_each_csr_with_its_group(made):
    _, _, m = made
    (streamer,) = m.csrs.streamers
    assert (streamer.name, streamer.first, streamer.last) == ("dot", 960, 987)  # 5 movers: 3 + 25
    first = streamer.last + 1
    names = ["VMUL_START", "VMUL_JOBS", "VMUL_STATUS", "VMUL_COUNTER", "VSUM_START", "VSUM_JOBS",
             "VSUM_JOB_BEATS", "VSUM_STATUS", "VSUM_COUNTER"]  # fmt: skip
    access = ["RW", "RW", "RO", "RO", "RW", "RW", "RW", "RO", "RO"]
    want = {f"DOT_{a}_{n}": first + i for i, (a, n) in enumerate(zip(access, names, strict=True))}
    assert m.csrs.accelerator == want


def test_the_streamer_holds_every_units_movers(made):
    tmp, _, _ = made
    text = (tmp / "out" / "snax_forge_dot_1_enc_streamer_dot.hjson").read_text()
    s = hjson.loads(text)
    assert len(s["data_reader_params"]["num_channel"]) == 3  # vmul.a, vmul.b, vsum.a
    assert len(s["data_writer_params"]["num_channel"]) == 2  # vmul.out, vsum.out
    assert s["data_writer_params"]["spatial_bounds"] == [[4], [1]]


def test_the_slot_counts_both_groups_csrs(made):
    tmp, _, m = made
    cluster = json.loads("\n".join(
        x for x in (tmp / "out" / m.cluster).read_text().splitlines() if not x.startswith("//")
    ))  # fmt: skip
    (slot,) = cluster["snax_forge_core_template"]["snax_acc_cfg"]
    (the_set,) = slot["snax_streamer_sets"]
    assert (the_set["name"], the_set["snax_num_rw_csr"], the_set["snax_num_ro_csr"]) == (
        "dot",
        5,
        4,
    )


def test_the_manifest_lists_both_tasks_with_their_units_trace(made):
    _, _, m = made
    assert m.trace == "snax_acc0"
    assert [t.to_dict() for t in m.tasks] == [
        {"name": "mult_vmul", "beats": 16, "csrs": {"JOBS": 16}, "trace": "snax_acc0/unit_vmul"},
        {"name": "sum_vsum", "beats": 16, "csrs": {"JOBS": 1, "JOB_BEATS": 16},
         "trace": "snax_acc0/unit_vsum"},
    ]  # fmt: skip


def test_main_c_includes_dots_header(made):
    tmp, _, _ = made
    assert '#include "dot_csr.h"' in (tmp / "out" / "main.c").read_text()


def test_the_rtl_fixture_has_no_vectors(made, tmp_path, monkeypatch):
    """vectors.json is a one-task fixture's; dot_1_enc's spec drives each group itself: nothing
    is written."""
    tmp, path, _ = made
    monkeypatch.chdir(tmp)
    assert write_fixture(path, tmp_path / "fx", tmp) == []
    assert not (tmp_path / "fx").exists()


@pytest.fixture(scope="module")
def serial(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dot_serial")
    path = point(tmp, name="dot_serial", graph=bound_recipe("dot_serial", "dot"))
    return tmp, path, generate(path, tmp / "out", root=tmp)


def test_dot_serials_description_is_two_accelerators(serial):
    tmp, path, _ = serial
    want = replace(dot_serial(), design_point="design/dot_serial/design_point.json")
    got = describe(DesignPoint.load(path), want.design_point)
    assert [a.name for a in got.accelerators] == ["mul", "sum"]
    assert got.defines() == want.defines()
    assert (
        json.loads((tmp / "out" / "accelerator.json").read_text())["tag"] == "snax_forge_dot_serial"
    )


def test_dot_serials_slot_has_two_sets(serial):
    """D157, C§16: one set per accelerator in CSR order, each its streamer file and counts; the
    TCDM ports their sum, the interconnect one group per set."""
    tmp, _, m = serial
    (slot,) = parsed(tmp / "out" / m.cluster)["snax_forge_core_template"]["snax_acc_cfg"]
    assert slot["snax_acc_name"] == "snax_forge_dot_serial"
    assert slot["snax_streamer_sets"] == [
        {"name": "mul", "snax_streamer_cfg": {"$ref": "snax_forge_dot_serial_streamer_mul.hjson"},
         "snax_num_rw_csr": 2, "snax_num_ro_csr": 2},
        {"name": "sum", "snax_streamer_cfg": {"$ref": "snax_forge_dot_serial_streamer_sum.hjson"},
         "snax_num_rw_csr": 3, "snax_num_ro_csr": 2},
    ]  # fmt: skip
    assert (slot["snax_tcdm_ports"], slot["sparse_interconnect_config"]) == (14, [[12, 1], [2, 1]])
    assert slot["snax_disable_csr_manager"] is True
    for k in ("snax_streamer_cfg", "snax_num_rw_csr", "snax_num_ro_csr"):
        assert k not in slot
    mul = parsed(tmp / "out" / "snax_forge_dot_serial_streamer_mul.hjson")
    total = parsed(tmp / "out" / "snax_forge_dot_serial_streamer_sum.hjson")
    assert len(mul["data_reader_params"]["num_channel"]) == 2  # mul.a, mul.b
    assert total["data_reader_params"]["num_channel"] == [1]  # sum.a at W = 1


def test_dot_serials_main_c_includes_each_sets_headers(serial):
    tmp, _, m = serial
    text = (tmp / "out" / "main.c").read_text()
    includes = [x for x in text.splitlines() if x.startswith("#include")]
    assert includes == ['#include "snrt.h"', '#include "data.h"', '#include "mul_csr.h"',
                        '#include "streamer_mul_csr_addr_map.h"', '#include "sum_csr.h"',
                        '#include "streamer_sum_csr_addr_map.h"']  # fmt: skip
    assert [t.name for t in m.tasks] == ["mult_mul", "sum_sum"]


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dot_1")
    path = point(tmp, name="dot_1", graph=bound_dot1())
    return tmp, path, generate(path, tmp / "out", root=tmp)


def test_dot_1_is_c16s_two_sets(plain):
    """dot 1 (F2b, no encapsulate): C§16's table, vmul's streamer 960-977 and own 978-981, vsum's
    982-994 and 995-999; its slot two sets, a streamer file and a header pair each."""
    tmp, path, m = plain
    want = replace(dot1(), design_point="design/dot_1/design_point.json")
    assert describe(DesignPoint.load(path), want.design_point) == want
    assert [(r.name, r.first, r.last) for r in m.csrs.streamers] == [("vmul", 960, 977),
                                                                     ("vsum", 982, 994)]  # fmt: skip
    assert (m.csrs.accelerator["VMUL_RW_START"], m.csrs.accelerator["VSUM_RO_COUNTER"]) == (
        978,
        999,
    )
    (slot,) = parsed(tmp / "out" / m.cluster)["snax_forge_core_template"]["snax_acc_cfg"]
    assert [(x["name"], x["snax_num_rw_csr"], x["snax_num_ro_csr"])
            for x in slot["snax_streamer_sets"]] == [("vmul", 2, 2), ("vsum", 3, 2)]  # fmt: skip
    for f in ("snax_forge_dot_1_streamer_vmul.hjson", "snax_forge_dot_1_streamer_vsum.hjson",
              "include/vmul_csr.h", "include/streamer_vmul_csr_addr_map.h", "include/vsum_csr.h",
              "include/streamer_vsum_csr_addr_map.h"):  # fmt: skip
        assert f in m.files, f
    assert [t.name for t in m.tasks] == ["mult_vmul", "sum_vsum"]
