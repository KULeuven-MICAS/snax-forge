"""tag_map and untag_map, and bind on its firing loops: matadd, the first 2-D bind.

matadd imports as two nested maps (``add_map`` over i, ``add_map_1`` over j).
The row map is tagged temporal as it is, the column map split by W, and
``add`` bound to ``simd``: every map around the spatial map is a
firing loop, and each memlet keeps its row index and runs its lanes along
the columns. The bound graph passes the reference check against the kernel.
A split and its tags are separate steps: an untagged split joins, a map
tagged spatial unsplit binds at W = N, and no tag comes off around a unit.
"""

from __future__ import annotations

import pytest

from snax_forge.dfg import Graph
from snax_forge.dfg.import_sdfg import import_kernel
from snax_forge.dfg.loops import SUFFIX
from snax_forge.sandbox import SandboxError, bind, split_map
from snax_forge.sandbox.transforms import tag_map, untag_map
from tests.design.helpers import MATADD_RECIPE

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
    """The map keeps its range, its variable takes the tag's suffix (``m_t``, ``m_tile``,
    ``m_s``); untag_map gives back the graph it was."""
    g = tag_map(matadd, "add_map", kind)
    m = g.node("add_map")
    assert (m.attrs["var"], m.attrs["range"], m.attrs["loop.kind"]) == (
        f"m{SUFFIX[kind]}",
        "0:M",
        kind,
    )
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
