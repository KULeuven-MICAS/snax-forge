"""mlp_int and mlp_int_col3: bit for bit against a loop over Python ints at both sizes, the
same logits from both layouts, the biases inside their range on full-range data, and the
import (slow): a1 and a2 stay int8, and the graph's executor equals the reference."""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.dfg.graph import Graph
from snax_forge.dfg.reference import full_range
from snax_forge.sdfg.loader import load
from snax_forge.sdfg.spec import KernelSpec

KERNELS = ("mlp_int", "mlp_int_col3")
SIZES = load("mlp_int").sweep_sizes


def _loop(args: dict, col3: bool) -> list[int]:
    """The MLP over Python ints: no wrap, floor division, the clamp to [0, 127]."""

    def layer(w, a, b, t=False):
        at = (lambda o, i: int(w[i][o])) if t else (lambda o, i: int(w[o][i]))
        return [int(b[o]) + sum(at(o, i) * int(a[i]) for i in range(len(a))) for o in range(len(b))]

    def requant(z):
        return [min(max(v // 1024, 0), 127) for v in z]

    a1 = requant(layer(args["w1"], args["x"], args["b1"]))
    a2 = requant(layer(args["w2"], a1, args["b2"]))
    return layer(args["w3"], a2, args["b3"], col3)


@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("sizes", SIZES, ids=lambda s: "-".join(map(str, s.values())))
def test_the_kernel_is_the_integer_mlp_bit_for_bit(name, sizes):
    spec = load(name)
    args = spec.make_inputs(np.random.default_rng(1), **sizes)
    want = _loop(args, name.endswith("col3"))
    spec.reference(**args)
    assert args["out"].dtype == np.int32 and args["out"].tolist() == want


@pytest.mark.parametrize("sizes", SIZES, ids=lambda s: "-".join(map(str, s.values())))
def test_both_layouts_give_the_same_logits(sizes):
    rows, col3 = load("mlp_int"), load("mlp_int_col3")
    a, b = (s.make_inputs(np.random.default_rng(2), **sizes) for s in (rows, col3))
    assert np.array_equal(a["w3"].T, b["w3"])
    rows.reference(**a)
    col3.reference(**b)
    assert np.array_equal(a["out"], b["out"])


def test_the_shift_keeps_the_activations_off_the_clamp():
    """At 1024 about half of each layer's outputs are 0 after the relu and at most 9 % clamp
    at 127, at both sizes (the shift rule on full-range data)."""
    for sizes in SIZES:
        args = load("mlp_int").make_inputs(np.random.default_rng(0), **sizes)
        z1 = args["w1"].astype(np.int64) @ args["x"] + args["b1"]
        a1 = np.clip(z1 // 1024, 0, 127)
        z2 = args["w2"].astype(np.int64) @ a1 + args["b2"]
        for a in (a1, np.clip(z2 // 1024, 0, 127)):
            assert 0.3 < np.mean(a == 0) < 0.65 and np.mean(a == 127) <= 0.09


def test_full_range_data_keeps_a_kernels_range():
    """A container with a range in the spec is drawn inside it, the others over their dtype."""
    g = Graph.from_dict({"name": "t", "symbols": {"N": 4096},
                         "containers": {"x": {"dtype": "int8", "shape": ["N"]},
                                        "b": {"dtype": "int32", "shape": ["N"]}},
                         "body": []})  # fmt: skip
    ins = {"x": np.zeros(4096, np.int8), "b": np.zeros(4096, np.int32)}
    out = full_range(g, ins, 0, {"b": (-(2**15), 2**15 - 1)})
    assert out["b"].min() >= -(2**15) and out["b"].max() < 2**15 and out["b"].max() > 2**14
    assert out["x"].min() == -128 and out["x"].max() == 127
    assert np.abs(full_range(g, ins, 0)["b"]).max() > 2**30


def test_a_range_outside_its_dtype_or_on_no_container_is_refused():
    base = load("mlp_int")
    with pytest.raises(ValueError, match=r"the range of b1, \[0, 2147483648\], is not inside"):
        KernelSpec(**{**base.__dict__, "ranges": {"b1": (0, 2**31)}})
    with pytest.raises(ValueError, match="ranges names 'y', which is not a container"):
        KernelSpec(**{**base.__dict__, "ranges": {"y": (0, 1)}})


@pytest.mark.slow  # each import builds its SDFG in DaCe
@pytest.mark.parametrize("name", KERNELS)
def test_the_import_keeps_the_activations_int8_and_computes_the_reference(name):
    from snax_forge.dfg.import_sdfg import import_kernel
    from snax_forge.dfg.reference import reference_check

    g = import_kernel(name)
    assert g.containers["a1"].dtype == g.containers["a2"].dtype == "int8"
    for sizes in SIZES:
        for s, v in sizes.items():
            g.symbols[s.upper()] = v
        for data in ("kernel", "full"):
            gold = reference_check(load(name), g, seed=3, data=data)  # raises on a difference
            assert np.abs(gold.inputs["b1"]).max() < 2**15
