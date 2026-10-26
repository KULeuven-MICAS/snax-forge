"""tag_map and untag_map, and bind on its firing loops: matadd, the first 2-D bind (D73).

matadd imports as two nested maps (``add_map`` over i, ``add_map_1`` over j).
The row map is tagged temporal as it is, the column map split by W, and
``add`` bound to ``elementwise_add``: every map around the spatial map is a
firing loop, and each memlet keeps its row index and runs its lanes along
the columns. The bound graph passes the reference check against the kernel.
"""

from __future__ import annotations

import numpy as np
import pytest

from snax_forge.dfg import Graph, execute
from snax_forge.dfg.execute import against_reference
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.sandbox import SandboxError, bind, split_map
from snax_forge.sandbox.transforms import tag_map, untag_map
from snax_forge.sdfg.loader import load
from tests.design.helpers import MATADD_RECIPE, bound_matadd

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


@pytest.mark.parametrize("kind", ["temporal", "tile"])
def test_tag_and_untag_round_trip(matadd, kind):
    """The map keeps its var and range; untag_map gives back the graph it was."""
    g = tag_map(matadd, "add_map", kind)
    m = g.node("add_map")
    assert (m.attrs["var"], m.attrs["range"], m.attrs["loop.kind"]) == ("i", "0:M", kind)
    assert m.attrs["loop.tagged"] is True
    assert "loop.kind" not in matadd.node("add_map").attrs  # the input is not changed
    assert untag_map(g, "add_map") == matadd


def test_tag_map_refuses_a_tagged_map(matadd):
    g = tag_map(matadd, "add_map", "temporal")
    with pytest.raises(SandboxError, match="tag_map 'add_map': already temporal"):
        tag_map(g, "add_map", "tile")
    with pytest.raises(SandboxError, match="kind 'spatial' is not one of"):
        tag_map(matadd, "add_map", "spatial")


def test_untag_map_refuses_a_split_map(matadd):
    g = split_map(matadd, "add_map_1", 4)
    with pytest.raises(SandboxError, match="split_map tagged it; join_map undoes a split"):
        untag_map(g, "add_map_1")
    with pytest.raises(SandboxError, match="not a tagged map"):
        untag_map(matadd, "add_map")


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
