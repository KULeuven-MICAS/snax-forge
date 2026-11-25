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
    return factor_out(factor_out(g, "alpha"), "beta")


def nest(g: Graph, top: str) -> list[tuple[str, str]]:
    """``top``'s nest as (id, the tasklet's code or the map's range) in walk order."""
    out, inside = [], False
    for n, outer in g.walk():
        inside = n.id == top or any(o.id == top for o in outer)
        if inside:
            out.append((n.id, n.attrs.get("code", n.attrs.get("range"))))
    return out


def test_each_half_reads_its_matrix_and_the_scales_land_in_the_add():
    h = both(gesummv())
    for c in ("tmp0", "tmp2", "tmp1", "tmp3"):
        assert c not in h.containers
    assert h.containers["tmp1_sum"].shape == ["N"] and h.containers["tmp1_sum"].transient
    assert h.node("tmp1_mult").inputs["in1"].data == "A"
    assert h.node("tmp3_mult").inputs["in1"].data == "B"
    assert h.node("sum").outputs["out"].data == "tmp1_sum"
    add = h.node("add")
    assert add.attrs["code"] == "out = in3 * in1 + in4 * in2"
    assert {c: m.data for c, m in add.inputs.items()} == {
        "in1": "tmp1_sum",
        "in2": "tmp3_sum",
        "in3": "alpha",
        "in4": "beta",
    }
    assert [n.id for n in h.body] == ["tmp1_mult_map", "sum_map", "tmp3_mult_map", "sum_map_2",
                                      "add_map"]  # fmt: skip


def test_a_nest_of_its_own_when_the_sum_has_no_elementwise_reader():
    d = gesummv().to_dict()
    d["containers"]["tmp1"]["transient"] = False  # tmp1 an output: the scale cannot land in add
    h = factor_out(Graph.from_dict(d), "alpha")
    assert nest(h, "tmp1_scale_map") == [("tmp1_scale_map", "0:N"),
                                         ("tmp1_scale", "out = in1 * in2")]  # fmt: skip
    scale = h.node("tmp1_scale")
    assert (scale.inputs["in1"].data, scale.inputs["in2"].data) == ("alpha", "tmp1_sum")
    assert scale.outputs["out"].data == "tmp1"
    assert h.node("add").attrs["code"] == "out = in1 + in2"
    assert [n.id for n in h.body][:3] == ["tmp1_mult_map", "sum_map", "tmp1_scale_map"]
    assert unfactor_out(h, "alpha").to_json() == Graph.from_dict(d).to_json()


@pytest.mark.parametrize(("kernel", "types"), [("gesummv", None), ("gesummv_types", "i8")])
def test_the_outputs_are_bit_for_bit_and_unfactor_out_gives_the_import(kernel, types):
    g = gesummv(kernel, types)
    h = both(g)
    ins = kernel_inputs(kernel_spec(kernel, "test"), g, 0)
    assert np.array_equal(execute(g, ins)["out"], execute(h, ins)["out"])
    back = unfactor_out(unfactor_out(h, "beta"), "alpha")
    assert back.to_json() == g.to_json()
    other = unfactor_out(unfactor_out(h, "alpha"), "beta")  # the other order
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
    recs = both(gesummv()).node("add").attrs[FACTORED]
    assert sorted(recs) == ["alpha", "beta"]
    rec = recs["alpha"]
    got = tuple(rec[k] for k in ("container", "product", "connector", "fold", "output"))
    assert got == ("tmp0", "tmp1_mult", "in1", "sum", "tmp1")
    got = tuple(rec[k] for k in ("into", "k", "read", "code"))
    assert got == ("add", "in3", "in1", "out = in1 + in2")
    assert rec["nest"]["id"] == "mult_map" and rec["after"] is None


def test_a_baked_scalar():
    g = TRANSFORMS["bake_scalar"].fn(gesummv(), container="alpha", value="csr")
    with pytest.raises(SandboxError, match="alpha is baked already"):
        factor_out(g, "alpha")


def test_the_site_is_the_scalar():
    assert TRANSFORMS["factor_out"].site(gesummv(), "alpha") == {"container": "alpha"}
    assert TRANSFORMS["unfactor_out"].site(both(gesummv()), "beta") == {"container": "beta"}


@pytest.mark.parametrize(("selected", "message"), [
    ("mult_map", "'mult_map' is not a kernel scalar argument; the step takes the scalar a nest scales by (gesummv's alpha)"),
    ("A", "'A' is not a kernel scalar argument; the step takes the scalar a nest scales by (gesummv's alpha)"),
])  # fmt: skip
def test_not_a_kernel_scalar(selected, message):
    with pytest.raises(SandboxError, match=f"^{re.escape(f'factor_out {selected!r}: {message}')}$"):
        factor_out(gesummv(), selected)


def test_a_scalar_that_scales_two_nests():
    d = gesummv().to_dict()
    for n in d["body"]:  # half 2 scaled by alpha too
        if n["id"] == "mult_map_2":
            n["body"][0]["body"][0]["inputs"]["in1"]["data"] = "alpha"
    with pytest.raises(SandboxError, match=r"alpha is read by \['mult', 'mult_1'\]; factor_out moves "
                                           "a scalar that scales one nest"):  # fmt: skip
        factor_out(Graph.from_dict(d), "alpha")


def test_a_second_reader_of_the_scaled_matrix():
    d = gesummv().to_dict()
    for n in d["body"]:  # half 2's product reads tmp0 instead of tmp2
        if n["id"] == "tmp3_mult_map":
            n["body"][0]["body"][0]["inputs"]["in1"]["data"] = "tmp0"
    with pytest.raises(SandboxError, match=r"tmp0 is named by \['tmp1_mult', 'tmp3_mult'\]"):
        factor_out(Graph.from_dict(d), "alpha")


def test_a_product_narrower_than_the_sum():
    d = gesummv("gesummv_types", "i8").to_dict()
    for n in d["body"]:  # x no longer cast: A (int8) times x (int8) runs at int8
        if n["id"] == "tmp2_mult_map":
            n["body"][0]["body"][0]["attrs"]["code"] = "out = in1 * in2"
    with pytest.raises(SandboxError, match="tmp2_mult's product of A is int8, narrower than tmp1's "
                                           "int32"):  # fmt: skip
        factor_out(Graph.from_dict(d), "alpha")


def test_floats():
    d = gesummv().to_dict()
    for c in d["containers"].values():
        c["dtype"] = "float64"
    with pytest.raises(SandboxError, match="a float type"):
        factor_out(Graph.from_dict(d), "alpha")


def test_unfactor_out_takes_only_factor_outs_scale():
    with pytest.raises(SandboxError, match="alpha is read by 0 scales factor_out made, not one"):
        unfactor_out(gesummv(), "alpha")


def test_a_scalar_that_does_not_scale():
    d = gesummv().to_dict()
    d["body"][0]["body"][0]["body"][0]["attrs"]["code"] = "out = in1 + in2"  # alpha + A
    with pytest.raises(SandboxError, match=re.escape("mult computes 'out = in1 + in2'; factor_out "
                                                     "takes a scale nest out = alpha * a")):  # fmt: skip
        factor_out(Graph.from_dict(d), "alpha")
