"""dot 1 through the generator: ``recipes/dot/dot_v1_1.json``, one
accelerator ``dot`` of two groups, ``vmul`` and ``vsum``, in encapsulate's order, each with its
own CSRs and a task of its own; ``recipes/dot/dot_v1.json``, the same groups as two accelerators,
two streamer sets of the one entry; and dot_v7, two accelerators ``mul`` and ``sum``.

dot_v1_1's description is the (tests/gen/test_accelerator.py's ``dot``): VMUL at offsets
0-3, VSUM at 4-8 (rw 5, ro 4); the header names every CSR with its group; the set's streamer
holds every unit's movers, readers vmul.a, vmul.b, vsum.a and writers vmul.out, vsum.out; the
manifest lists both tasks with their group's values and their unit's trace path. dot 1's and
dot_v7's slots carry two sets each, its own streamer file, CSR counts and interconnect
group.
"""

from __future__ import annotations

import json

import hjson
import pytest

from snax_forge.gen import generate
from snax_forge.gen.rtl import write_fixture

from .helpers import bound_dot1, bound_dot1_enc, bound_recipe, point

POINT = "design/dot_v1_1/design_point.json"


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dot_v1_1")
    path = point(tmp, name="dot_v1_1", graph=bound_dot1_enc())
    return tmp, path, generate(path, tmp / "out", root=tmp)


def test_the_streamer_holds_every_units_movers(made):
    tmp, _, _ = made
    text = (tmp / "out" / "snax_forge_dot_v1_1_streamer_dot.hjson").read_text()
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


def test_main_c_includes_dots_header(made):
    tmp, _, _ = made
    assert '#include "dot_csr.h"' in (tmp / "out" / "main.c").read_text()


def test_the_rtl_fixture_has_no_vectors(made, tmp_path, monkeypatch):
    """vectors.json is a one-task fixture's; dot_v1_1's spec drives each group itself: nothing
    is written."""
    tmp, path, _ = made
    monkeypatch.chdir(tmp)
    assert write_fixture(path, tmp_path / "fx", tmp) == []
    assert not (tmp_path / "fx").exists()


@pytest.fixture(scope="module")
def serial(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dot_v7")
    path = point(tmp, name="dot_v7", graph=bound_recipe("dot_v7", "dot"))
    return tmp, path, generate(path, tmp / "out", root=tmp)


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dot_v1")
    path = point(tmp, name="dot_v1", graph=bound_dot1())
    return tmp, path, generate(path, tmp / "out", root=tmp)
