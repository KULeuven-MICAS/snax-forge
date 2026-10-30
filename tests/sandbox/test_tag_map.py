"""tag_map and untag_map, and bind on its firing loops: matadd, the first 2-D bind (D73).

matadd imports as two nested maps (``add_map`` over i, ``add_map_1`` over j).
The row map is tagged temporal as it is, the column map split by W, and
``add`` bound to ``elementwise_add``: every map around the spatial map is a
firing loop, and each memlet keeps its row index and runs its lanes along
the columns. The bound graph passes the reference check against the kernel.
A split and its tags are separate steps: an untagged split joins, a map
tagged spatial unsplit binds at W = N, and no tag comes off around a unit.
"""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.dfg import Graph, execute
from snax_forge.dfg.execute import against_reference
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.sandbox import SandboxError, bind, join_map, split, split_map
from snax_forge.sandbox.transforms import tag_map, untag_map
from snax_forge.sdfg.loader import load
from tests.design.helpers import MATADD_RECIPE, bound_matadd
from tests.helpers import graph_dict

BIND = MATADD_RECIPE["steps"][-1]["params"]


@pytest.fixture(scope="module")
def matadd() -> Graph:
    """matadd as imported, M = 8 and N = 16 bound."""
    d = import_kernel("matadd").to_dict()
    d["symbols"] = {"M": 8, "N": 16}
    return Graph.from_dict(d)


# =============================================================================
# tag_map and untag_map
# =============================================================================


@pytest.mark.parametrize("kind", ["temporal", "tile", "spatial"])
def test_tag_and_untag_round_trip(matadd, kind):
    """The map keeps its var and range; untag_map gives back the graph it was."""
    g = tag_map(matadd, "add_map", kind)
    m = g.node("add_map")
    assert (m.attrs["var"], m.attrs["range"], m.attrs["loop.kind"]) == ("i", "0:M", kind)
    assert "loop.kind" not in matadd.node("add_map").attrs  # the input is not changed
    assert untag_map(g, "add_map") == matadd


def test_tag_map_refuses_a_tagged_map(matadd):
    g = tag_map(matadd, "add_map", "temporal")
    with pytest.raises(SandboxError, match="tag_map 'add_map': already temporal"):
        tag_map(g, "add_map", "tile")
    with pytest.raises(SandboxError, match="kind 'diagonal' is not one of"):
        tag_map(matadd, "add_map", "diagonal")
    with pytest.raises(SandboxError, match="not a tagged map"):
        untag_map(matadd, "add_map")


def test_untag_map_takes_any_tag_but_around_a_bound_unit(matadd):
    """A split map's tags come off as any; around a bound unit they are its firing loops."""
    g = split_map(matadd, "add_map_1", 4)
    assert (
        untag_map(untag_map(g, "add_map_1"), "add_map_1_s").to_json()
        == split(matadd, "add_map_1", 4).to_json()
    )
    with pytest.raises(SandboxError, match="untag_map 'add_map': the bound unit 'add' is inside "
                       r"it \(unbind first\)"):  # fmt: skip
        untag_map(bound_matadd(), "add_map")


def vecadd64() -> Graph:
    d = graph_dict("vecadd")
    d["symbols"] = {"N": 64}
    return Graph.from_dict(d)


def test_join_map_joins_an_untagged_split():
    g = vecadd64()
    assert join_map(split(g, "add_map", 4), "add_map").to_json() == g.to_json()


def test_tag_map_spatial_unsplit_binds_at_w_n():
    """vecadd's add_map tagged spatial as it is: one firing of N lanes, W read off N."""
    acc = bind(tag_map(vecadd64(), "add_map", "spatial"), "add", "elementwise_add",
               "chisel_tiled_spatial", "acc").node("add")  # fmt: skip
    assert acc.attrs["params"]["W"] == 64
    assert acc.inputs["a"].subset == ["0:64"]


def test_a_graph_with_the_old_loop_tagged_still_loads():
    """loop.tagged, which tag_map wrote before D73's split, is a namespaced attr: it passes
    through, and untag_map takes the tag off as any."""
    d = graph_dict("vecadd")
    d["body"][0]["attrs"] |= {"loop.kind": "temporal", "loop.tagged": True}
    g = Graph.from_dict(d)
    assert g.node("add_map").attrs["loop.tagged"] is True
    assert "loop.kind" not in untag_map(g, "add_map").node("add_map").attrs


# =============================================================================
# bind on the firing loops
# =============================================================================


def test_bind_under_an_untagged_map_is_refused(matadd):
    g = split_map(matadd, "add_map_1", 4)
    with pytest.raises(SandboxError, match="map 'add_map' around it is untagged.*tag it temporal"):
        bind(g, **BIND)


def test_bind_under_a_tile_map_is_refused(matadd):
    g = split_map(tag_map(matadd, "add_map", "tile"), "add_map_1", 4)
    with pytest.raises(SandboxError, match="map 'add_map' is a tile; tiles are not lowered yet"):
        bind(g, **BIND)


def test_matadd_binds_in_2d_and_passes_the_reference_check():
    """Tag the row map, split the column map, bind: two firing loops, lanes along a row."""
    g = bound_matadd()
    loops = [(n.id, n.attrs["var"], n.attrs["range"], n.attrs.get("loop.kind"))
             for n, _ in g.walk() if n.kind == "map"]  # fmt: skip
    assert loops == [
        ("add_map", "i", "0:M", "temporal"),
        ("add_map_1", "j_t", "0:N // 4", "temporal"),
    ]
    acc = g.node("add")
    assert acc.kind == "accelerated" and acc.attrs["params"]["W"] == 4
    lanes = ["i", "4 * j_t:4 * j_t + 4"]
    assert {c: (m.data, m.subset) for c, m in (acc.inputs | acc.outputs).items()} == {
        "a": ("A", lanes), "b": ("B", lanes), "out": ("C", lanes)
    }  # fmt: skip
    spec = load("matadd")
    inputs = spec.make_inputs(np.random.default_rng(0), m=8, n=16)
    for got, ref in against_reference(spec, inputs, lambda x: execute(g, x)).values():
        assert np.array_equal(got, ref)
