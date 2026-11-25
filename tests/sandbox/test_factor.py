"""factor_out and unfactor_out (factor.py) on gesummv at N = 8, int64, and on gesummv_types at
``i8``: each half's scale nest moved past its sum, the outputs bit for bit as before (also over
the whole int8 and int32 ranges, where the sums wrap), and back to the import."""

from __future__ import annotations

import re

import numpy as np
import pytest

from snax_forge.dfg import Graph
from snax_forge.dfg.execute import execute
from snax_forge.dfg.graph import FACTORED
from snax_forge.dfg.reference import kernel_inputs, kernel_spec
from snax_forge.sandbox import Recipe, SandboxError, bind_recipe_symbols, factor_out, unfactor_out
from snax_forge.sandbox.transforms import TRANSFORMS
from tests.helpers import imported

N = 8


def gesummv(kernel: str = "gesummv", types: str | None = None) -> Graph:
    r = Recipe.from_dict({"name": kernel, "kernel": kernel, "symbols": {"N": N}, "steps": []})
    return bind_recipe_symbols(r, imported(kernel, types=types))


def both(g: Graph) -> Graph:
    return factor_out(factor_out(g, "mult_map"), "mult_map_2")


def nest(g: Graph, top: str) -> list[tuple[str, str]]:
    """``top``'s nest as (id, the tasklet's code or the map's range) in walk order."""
    out, inside = [], False
    for n, outer in g.walk():
        inside = n.id == top or any(o.id == top for o in outer)
        if inside:
            out.append((n.id, n.attrs.get("code", n.attrs.get("range"))))
    return out


def test_each_half_reads_its_matrix_and_scales_its_sum():
    h = both(gesummv())
    assert "tmp0" not in h.containers and "tmp2" not in h.containers
    assert h.containers["tmp1_sum"].shape == h.containers["tmp1"].shape == ["N"]
    assert h.node("tmp1_mult").inputs["in1"].data == "A"
    assert h.node("tmp3_mult").inputs["in1"].data == "B"
    assert h.node("sum").outputs["out"].data == "tmp1_sum"
    assert nest(h, "tmp1_scale_map") == [("tmp1_scale_map", "0:N"),
                                         ("tmp1_scale", "out = in1 * in2")]  # fmt: skip
    scale = h.node("tmp3_scale")
    assert (scale.inputs["in1"].data, scale.inputs["in2"].data) == ("beta", "tmp3_sum")
    assert scale.outputs["out"].data == "tmp3"
    order = [n.id for n in h.body]
    assert order == ["tmp1_mult_map", "sum_map", "tmp1_scale_map", "tmp3_mult_map", "sum_map_2",
                     "tmp3_scale_map", "add_map"]  # fmt: skip


@pytest.mark.parametrize(("kernel", "types"), [("gesummv", None), ("gesummv_types", "i8")])
def test_the_outputs_are_bit_for_bit_and_unfactor_out_gives_the_import(kernel, types):
    g = gesummv(kernel, types)
    h = both(g)
    ins = kernel_inputs(kernel_spec(kernel, "test"), g, 0)
    assert np.array_equal(execute(g, ins)["out"], execute(h, ins)["out"])
    back = unfactor_out(unfactor_out(h, "tmp3_scale_map"), "tmp1_scale")
    assert back.to_json() == g.to_json()
    other = unfactor_out(unfactor_out(h, "tmp1_scale_map"), "tmp3_scale")  # the other order
    assert other.to_json() == g.to_json()


def test_exact_where_the_sums_wrap():
    g = gesummv("gesummv_types", "i8")
    rng = np.random.default_rng(7)
    ins = kernel_inputs(kernel_spec("gesummv_types", "test"), g, 0)
    for c in ("A", "B", "x"):
        ins[c] = rng.integers(-128, 128, size=ins[c].shape).astype(np.int8)
    for c in ("alpha", "beta"):
        ins[c] = rng.integers(-(2**31), 2**31, size=np.shape(ins[c])).astype(np.int32)
    want = execute(g, ins)["out"]
    assert np.array_equal(want, execute(both(g), ins)["out"])
    exact = (ins["alpha"].astype(object).ravel()[0] * (ins["A"].astype(object) @ ins["x"].astype(object))
             + ins["beta"].astype(object).ravel()[0] * (ins["B"].astype(object) @ ins["x"].astype(object)))  # fmt: skip
    assert any(int(v) != int(e) for v, e in zip(want, exact, strict=True))  # it wraps


def test_the_record_holds_what_went():
    rec = both(gesummv()).node("tmp1_scale").attrs[FACTORED]
    assert (rec["container"], rec["product"], rec["connector"], rec["fold"], rec["output"]) == (
        "tmp0",
        "tmp1_mult",
        "in1",
        "sum",
        "tmp1",
    )
    assert rec["nest"]["id"] == "mult_map" and rec["after"] is None


@pytest.mark.parametrize("site", ["mult_map", "mult_map_1", "mult"])
def test_any_map_of_the_scale_nest_or_its_tasklet(site):
    assert TRANSFORMS["factor_out"].site(gesummv(), site) == {"map": site}
    assert factor_out(gesummv(), site).to_json() == factor_out(gesummv(), "mult_map").to_json()


@pytest.mark.parametrize(("site", "message"), [
    ("tmp1_mult_map", "tmp1_mult computes 'out = in1 * in2'; factor_out takes a scale nest out = k * a with k a kernel scalar argument"),
    ("sum_map", "sum computes 'out = in1'; factor_out takes a scale nest out = k * a with k a kernel scalar argument"),
    ("add_map", "add computes 'out = in1 + in2'; factor_out takes a scale nest out = k * a with k a kernel scalar argument"),
])  # fmt: skip
def test_not_a_scale_nest(site, message):
    with pytest.raises(SandboxError, match=f"^{re.escape(f'factor_out {site!r}: {message}')}$"):
        factor_out(gesummv(), site)


def test_a_second_reader_of_the_scaled_matrix():
    d = gesummv().to_dict()
    for n in d["body"]:  # half 2's product reads tmp0 instead of tmp2
        if n["id"] == "tmp3_mult_map":
            n["body"][0]["body"][0]["inputs"]["in1"]["data"] = "tmp0"
    with pytest.raises(SandboxError, match=r"tmp0 is named by \['tmp1_mult', 'tmp3_mult'\]"):
        factor_out(Graph.from_dict(d), "mult_map")


def test_a_product_narrower_than_the_sum():
    d = gesummv("gesummv_types", "i8").to_dict()
    for n in d["body"]:  # x no longer cast: A (int8) times x (int8) runs at int8
        if n["id"] == "tmp2_mult_map":
            n["body"][0]["body"][0]["attrs"]["code"] = "out = in1 * in2"
    with pytest.raises(SandboxError, match="tmp2_mult's product of A is int8, narrower than tmp1's "
                                           "int32"):  # fmt: skip
        factor_out(Graph.from_dict(d), "mult_map")


def test_floats():
    d = gesummv().to_dict()
    for c in d["containers"].values():
        c["dtype"] = "float64"
    with pytest.raises(SandboxError, match="a float type"):
        factor_out(Graph.from_dict(d), "mult_map")


def test_unfactor_out_takes_only_factor_outs_scale():
    with pytest.raises(SandboxError, match="add is not a scale factor_out made"):
        unfactor_out(both(gesummv()), "add_map")
