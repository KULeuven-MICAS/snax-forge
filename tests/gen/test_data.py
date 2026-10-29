"""data.h: vecadd, vecmul and vsum's inputs, outputs and goldens, padded and
aligned as the memory plan's L2 layouts, and every refusal."""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest

from snax_forge.common import generated_header
from snax_forge.design import DesignPoint
from snax_forge.dfg.reference import kernel_inputs, kernel_spec
from snax_forge.gen import GenError
from snax_forge.gen import data as data_mod
from snax_forge.gen.data import arrays, data_h, literal
from snax_forge.lower.layout import Layout, moved_bytes

from .helpers import bound_vecmul, bound_vsum, point

GRAPHS = {"vecadd": None, "vecmul": bound_vecmul, "vsum": bound_vsum}


def load(tmp_path, design: str) -> DesignPoint:
    graph = GRAPHS[design]
    return DesignPoint.load(point(tmp_path, name=design, graph=graph() if graph else None))


@pytest.mark.parametrize("design", list(GRAPHS))
def test_the_golden_is_the_kernels_own_reference(tmp_path, design):
    p = load(tmp_path, design)
    spec = kernel_spec(design, "test")
    ref = {k: np.copy(v) for k, v in kernel_inputs(spec, p.graph, 0).items()}
    spec.reference(**ref)
    got = {a.name: a for a in arrays(p)}
    for c in spec.inout:
        g = got[f"{c}_golden"]
        assert np.array_equal(np.array(g.values[: g.elements]), ref[c].reshape(-1)), c
        assert got[c].role == "output" and not any(got[c].values)
    for c, v in kernel_inputs(spec, p.graph, 0).items():
        if c not in spec.inout:
            a = got[c]
            assert a.role == "input" and list(a.values[: a.elements]) == v.tolist()


@pytest.mark.parametrize("design", list(GRAPHS))
def test_padding_and_alignment_are_the_memory_plans(tmp_path, design):
    p = load(tmp_path, design)
    beat = p.platform.l2.beat_bits // 8
    text = data_h(p, "out/design/x/design_point.json")
    for a in arrays(p):
        lay = p.memory.layouts[a.container]["l2"]
        assert len(a.values) * 8 == moved_bytes(lay, 8, beat)
        assert a.elements == int(np.prod(lay.shape)) and not any(a.values[a.elements :])
        assert a.align == beat == 64
        decl = re.search(
            rf"\bint64_t {a.name}\[(\d+)\] __attribute__\(\(aligned\((\d+)\)\)\)", text
        )
        assert decl and (int(decl[1]), int(decl[2])) == (len(a.values), beat), a.name


def test_vecadd_text(tmp_path):
    p = load(tmp_path, "vecadd")
    text = data_h(p, "out/design/vecadd/design_point.json")
    assert text.startswith(generated_header("out/design/vecadd/design_point.json", ""))
    names = re.findall(r"int64_t (\w+)\[", text)
    assert names == ["A", "B", "C", "C_golden"]
    assert (
        "const int64_t A[64]" in text
        and "\nint64_t C[64] __attribute__((aligned(64))) = {0};" in text
    )


def test_vsum_pads_one_element_to_a_beat(tmp_path):
    got = {a.name: a for a in arrays(load(tmp_path, "vsum"))}
    assert (got["out"].elements, len(got["out"].values)) == (1, 8)
    assert len(got["out_golden"].values) == 8


def test_same_point_same_text(tmp_path):
    p = load(tmp_path, "vecadd")
    assert data_h(p, "x") == data_h(p, "x")
    assert data_h(p, "x", seed=1) != data_h(p, "x")


def test_literals():
    assert literal(-(2**63), "int64_t") == "(-INT64_C(9223372036854775807) - 1)"
    assert literal(2**40, "int64_t") == "INT64_C(1099511627776)"
    assert literal(-3, "int64_t") == "-3"


def test_refused_an_inplace_output(tmp_path, monkeypatch):
    p = load(tmp_path, "vecadd")
    spec = kernel_spec("vecadd", "test")

    def ones(rng, n=1024):
        d = spec.make_inputs(rng, n)
        d["C"] = np.ones(n, dtype=np.int64)
        return d

    monkeypatch.setattr(data_mod, "kernel_spec", lambda *_: replace(spec, make_inputs=ones))
    with pytest.raises(GenError, match=r"\[gen.inout\] output C"):
        arrays(p)


def test_refused_a_missing_input(tmp_path, monkeypatch):
    p = load(tmp_path, "vecadd")
    spec = kernel_spec("vecadd", "test")

    def no_b(rng, n=1024):
        return {k: v for k, v in spec.make_inputs(rng, n).items() if k != "B"}

    monkeypatch.setattr(data_mod, "kernel_spec", lambda *_: replace(spec, make_inputs=no_b))
    monkeypatch.setattr(data_mod, "execute", lambda g, i: {"C": np.zeros(64, np.int64)})
    with pytest.raises(GenError, match=r"\[gen.inputs\] B"):
        arrays(p)


def test_refused_a_strided_layout(tmp_path):
    p = load(tmp_path, "vecadd")
    lay = p.memory.layouts["A"]["l2"]
    p.memory.layouts["A"]["l2"] = Layout(lay.base, lay.shape, (16,))
    with pytest.raises(GenError, match=r"\[gen.layout\] A's L2 layout"):
        arrays(p)


def test_refused_a_float_dtype():
    with pytest.raises(GenError, match=r"\[gen.dtype\] container A"):
        data_mod.ctype("float64", "container A")


# =============================================================================
# Narrow containers (D142): refused until R3, the data path ready for it
# =============================================================================


def _int32_vecadd(tmp_path) -> DesignPoint:
    from snax_forge.dfg import Graph

    from .helpers import bound_vecadd

    d = bound_vecadd(4).to_dict()
    for c in d["containers"].values():
        c["dtype"] = "int32"
    return DesignPoint.load(point(tmp_path, graph=Graph.from_dict(d)))


def test_int32_vecadd_packs_its_data_two_per_word(tmp_path):
    """No refusal since PACK0 R3: data.h's int32_t arrays, the kernel's first values packed two
    per 64-bit word (D142)."""
    p = _int32_vecadd(tmp_path)
    a = {x.name: x for x in arrays(p)}
    assert a["A"].ctype == "int32_t" and a["A"].elements == 64 and len(a["A"].values) == 64
    first = np.array(a["A"].values[:2], dtype=np.int32).view(np.uint64)[0]
    want = kernel_inputs(kernel_spec("vecadd", "test"), p.graph, 0)["A"][:2]
    assert int(first) == int(want.view(np.uint64)[0])


def _packed(values: np.ndarray) -> tuple[list[int], int]:
    """``values`` as data.h's array helper pads them (whole 64-byte DMA beats), read back as
    the 64-bit little-endian words the DMA moves; and the padded length."""
    elem = values.dtype.itemsize
    padded = moved_bytes(Layout.contiguous(0, values.shape, elem), elem, 64) // elem
    t = data_mod.ctype(values.dtype.name, "x")
    a = data_mod._array("x", "x", "input", t, values, padded, 64)
    words = np.array(a.values, dtype=values.dtype).view(np.uint64)
    return [int(w) for w in words], len(a.values)


def test_e1_int32_packs_two_per_word_padded_to_a_beat():
    x = np.array([-3, 5, 7, -1, 100, 2**31 - 1, -(2**31), 0], dtype=np.int32)
    words, n = _packed(x)
    assert n == 16  # one 64-byte DMA beat
    assert words == [0x00000005FFFFFFFD, 0xFFFFFFFF00000007, 0x7FFFFFFF00000064,
                     0x0000000080000000, 0, 0, 0, 0]  # fmt: skip


def test_e2_int8_packs_eight_per_word_padded_to_a_beat():
    e = np.arange(32)
    y = ((9 * e - 128 + 128) % 256 - 128).astype(np.int8)  # 9e - 128, wrapped
    words, n = _packed(y)
    assert n == 64 and words[0] == 0xBFB6ADA49B928980
    assert words[4:] == [0, 0, 0, 0]


def test_vecadd_n7_writes_its_pad_and_a_whole_beat(tmp_path):
    """N = 7 at W = 4 (D144): each array 7 elements, a pad element of 0, one 64-byte beat."""
    from snax_forge.dfg.import_sdfg import import_kernel
    from snax_forge.sandbox import Recipe, apply_recipe
    from tests.design.helpers import REPO

    recipe = Recipe.load(REPO / "recipes" / "vecadd.json").with_params({"W": 4, "N": 7})
    g = apply_recipe(recipe, import_kernel("vecadd"))[-1].graph
    p = DesignPoint.load(point(tmp_path, graph=g))
    a = {x.name: x for x in arrays(p)}
    assert (a["A"].elements, len(a["A"].values), a["A"].values[7]) == (7, 8, 0)
    assert a["A"].shape == (7,) and a["A"].padded == (8,)
