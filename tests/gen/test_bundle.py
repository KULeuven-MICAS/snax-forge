"""The bundle for snax_cluster: gen-gvsoc's files at the root, the RTL under
``rtl/`` with its file list, and ``bundle.json``.

No JVM here: ``elaborate`` is replaced by a fake that writes two files and an ``emitted.json``
in Emit's form, and the checkout's commit is stubbed. The real ``Emit --spec`` runs in the
chisel job (``gen-rtl``) and by hand (``pixi run bundle``, docs/TOOL.md).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from snax_forge.gen import GenError
from snax_forge.gen import bundle as bundle_mod
from snax_forge.gen import gvsoc as gvsoc_mod
from snax_forge.gen.bundle import Bundle, write_bundle
from snax_forge.gen.rtl import EMITTED_JSON

from .helpers import fake_emit, point, with_groups


@pytest.fixture
def fake(monkeypatch):
    return fake_emit(monkeypatch)


def test_provenance_outside_git_and_in_this_checkout(tmp_path):
    assert bundle_mod.provenance(tmp_path) == (None, False)
    commit, dirty = bundle_mod.provenance(gvsoc_mod.ROOT)
    assert commit is None or len(commit) == 40
    assert isinstance(dirty, bool)


@pytest.mark.parametrize(
    ("listed", "code"),
    [
        ([], "bundle.empty"),
        ([EMITTED_JSON], "bundle.empty"),
    ],
)
def test_a_file_list_that_cannot_be_written(tmp_path, fake, listed, code):
    fake.listed = listed
    out = tmp_path / "bundle"
    with pytest.raises(GenError) as e:
        write_bundle(point(tmp_path), out, tmp_path)
    assert e.value.code == code
    assert not (out / "rtl" / "files.f").exists() and not (out / "bundle.json").exists()


def test_the_module_tree_names_every_group_unit_and_buffer():
    """From the description, not Emit, under the tag: dot_v1_1's two groups and units,
    and dot_v7's the same over its two accelerators (one shell); a linked group's units
    and its buffer (a wire is no module)."""
    from tests.gen.test_accelerator import dot, dot4, dot_v7

    t = "snax_forge_dot_v1_1"
    assert bundle_mod.module_tree(dot()) == {
        f"{t}_csr": [f"{t}_peripherals"], f"{t}_ctrl": [f"{t}_vmul_ctrl", f"{t}_vsum_ctrl"],
        f"{t}_datapath": [f"{t}_vmul", f"{t}_vsum"],
    }  # fmt: skip
    t = "snax_forge_dot_v7"
    assert bundle_mod.module_tree(dot_v7()) == {
        f"{t}_csr": [f"{t}_peripherals"], f"{t}_ctrl": [f"{t}_mul_ctrl", f"{t}_sum_ctrl"],
        f"{t}_datapath": [f"{t}_mul", f"{t}_sum"],
    }  # fmt: skip
    wire = dot4()
    t = "snax_forge_dot_v4"
    assert bundle_mod.module_tree(wire)[f"{t}_datapath"] == [f"{t}_mac", f"{t}_sum"]
    g = wire.groups[0]
    buffer = with_groups(wire, [replace(g, links=[replace(g.links[0], depth=1)])])
    assert bundle_mod.module_tree(buffer) == {
        f"{t}_csr": [f"{t}_peripherals"], f"{t}_ctrl": [f"{t}_dot_ctrl"],
        f"{t}_datapath": [f"{t}_mac", f"{t}_sum", f"{t}_out_partial"],
    }  # fmt: skip


def test_the_module_tree_ends_with_each_stream_block_once():
    """dot_v7 with mul's a and b at int32, one lane, sliced 2, and sum's out at int32
    packed 2: one FifoSlicer module for both inputs, then the packer, after the shell's."""
    from snax_forge.gen.blocks import Port
    from tests.gen.test_accelerator import dot_v7

    desc = dot_v7()
    mul, total = (a.groups[0].units[0] for a in desc.accelerators)
    mul = replace(mul, ports={p: Port(p, "in", "int32", 1, d_in=2) for p in ("a", "b")}
                  | {"out": Port("out", "out", "int32", 1)})  # fmt: skip
    total = replace(total, ports={**total.ports, "out": Port("out", "out", "int32", 1, d_out=2)})
    accs = [replace(a, groups=[replace(a.groups[0], units=[u])])
            for a, u in zip(desc.accelerators, (mul, total), strict=True)]  # fmt: skip
    t = "snax_forge_dot_v7"
    tree = bundle_mod.module_tree(replace(desc, accelerators=accs))
    assert list(tree)[-2:] == ["FifoSlicer_w1_i32_n1", "FifoPacker_n1_i32_w1"]
    assert list(tree)[:3] == [f"{t}_csr", f"{t}_ctrl", f"{t}_datapath"]
    assert tree["FifoSlicer_w1_i32_n1"] == tree["FifoPacker_n1_i32_w1"] == []


def test_the_expected_lines_and_their_refusal(tmp_path, fake):
    """write_expected keeps the run's counter lines in order, then its data line, and names the
    file in bundle.json; a run without a counter or with data errors is ``bundle.expected``."""
    out = tmp_path / "bundle"
    write_bundle(point(tmp_path), out, tmp_path)
    run = ("forge cycles: 304\nforge data: 0 errors (1 words)\nforge counter mult_mul: 19\n"
           "forge counter sum_sum: 19\nforge errors: 0\n")  # fmt: skip
    fills = {"mult_mul": 3, "sum_sum": 2}
    path = bundle_mod.write_expected(out, run, fills)
    assert path.read_text() == (
        "forge counter mult_mul: 19\nforge counter sum_sum: 19\nforge data: 0 errors (1 words)\n"
        "forge fill mult_mul: 3\nforge fill sum_sum: 2\n"
    )
    assert Bundle.load(out / "bundle.json").expected == "forge_expected.txt"
    for bad, fill in (
        ("forge data: 0 errors (1 words)\n", fills),
        ("forge counter a: 3\nforge data: 2 errors (8 words)\n", {"a": 3}),
        (run, {"mult_mul": 3}),  # sum_sum's counter has no fill
    ):
        with pytest.raises(GenError) as e:
            bundle_mod.write_expected(out, bad, fill)
        assert e.value.code == "bundle.expected"
    write_bundle(point(tmp_path), out, tmp_path)  # a bundle alone: the earlier run's file goes
    assert not path.exists() and Bundle.load(out / "bundle.json").expected is None
