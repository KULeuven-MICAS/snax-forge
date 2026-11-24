"""data.h: vecadd, vecmul and vsum's inputs, outputs and goldens, padded and
aligned as the memory plan's L2 layouts, and every refusal."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from snax_forge.design import DesignPoint
from snax_forge.dfg.reference import kernel_inputs, kernel_spec
from snax_forge.gen import GenError
from snax_forge.gen import data as data_mod
from snax_forge.gen.data import arrays, literal
from snax_forge.lower.layout import Layout, moved_bytes
from tests.helpers import bound_vsum

from .helpers import bound_vecmul, point

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


def test_literals():
    assert literal(-(2**63), "int64_t") == "(-INT64_C(9223372036854775807) - 1)"
    assert literal(2**40, "int64_t") == "INT64_C(1099511627776)"
    assert literal(-3, "int64_t") == "-3"


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


# =============================================================================
# Narrow containers: packed in data.h
# =============================================================================


def _packed(values: np.ndarray) -> tuple[list[int], int]:
    """``values`` as data.h's array helper pads them (whole 64-byte DMA beats), read back as
    the 64-bit little-endian words the DMA moves; and the padded length."""
    elem = values.dtype.itemsize
    padded = moved_bytes(Layout.contiguous(0, values.shape, elem), elem, 64) // elem
    t = data_mod.ctype(values.dtype.name, "x")
    a = data_mod._array("x", "x", "input", t, values, padded, 64)
    words = np.array(a.values, dtype=values.dtype).view(np.uint64)
    return [int(w) for w in words], len(a.values)
